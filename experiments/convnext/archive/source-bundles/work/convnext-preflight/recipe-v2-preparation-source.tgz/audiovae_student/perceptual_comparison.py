"""Paired finite-data perceptual trial with an explicit review boundary.

The comparison changes only the declared training objective.
Each arm inherits the same parent optimizer history and normalization. A paired
exposure journal prevents replay after an uncertain partial update or restart.
"""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import time

import torch

from .cache import sample_training_crop
from .comparison_data import verify_comparison_windows, assert_comparison_disjoint
from .discriminators import AudioDiscriminators, DiscriminatorConfig
from .distillation_training import DistillationEngine, DistillationTrainingConfig, evaluate_crops, waveform_gate
from .gradient_balancer import GradientBalancerConfig
from .perceptual_trial_engine import PerceptualTrialEngine, PerceptualTrialConfig
from .perceptual_readiness import evaluation_evidence
from .losses_distillation import DistillationLossConfig
from .model import StudentConfig, StudentDecoder
from .objective_comparison import state_fingerprint
from .preflight_distillation import _crop_identity
from .representative_pilot import (_append, _atomic_json, _checkpoint_reserve, _exclusive_run,
    _merge_coverage, _record, _verify_panel_targets, _writer, runtime_identity, summarize_evaluation, teacher_coverage)
from .restart_data import FixedWindowSampler, canonical, file_sha
from .teacher import CHECKPOINT_SHA256
from .training import _atomic_save, _rng_state, _restore_rng


@dataclass(frozen=True)
class PerceptualComparisonConfig:
    total_steps: int = 1000
    batch_size: int = 32
    context_frames: int = 29
    evaluation_interval: int = 250
    checkpoint_interval: int = 50
    teacher_batch_size: int = 8
    teacher_batch_samples: int = 1_920_000
    expected_parent_step: int = 250
    min_free_bytes: int = 4 * 1024**3

    def __post_init__(self):
        for key, value in asdict(self).items():
            if type(value) is not int or value < (0 if key == "min_free_bytes" else 1):
                raise ValueError(f"Invalid positive integer comparison setting: {key}")
        if self.total_steps > 1000 or self.evaluation_interval >= self.total_steps or self.evaluation_interval > 250:
            raise ValueError("Perceptual trial requires at most1000 updates and review by250")

    @property
    def arms(self):
        return ("control", "perceptual")


def new_arm(parent_state, config, device, *, capped, readiness_inputs, crops):
    saved=deepcopy(parent_state)
    if saved.get('quiet_phase') is not None or saved.get('perceptual_trial') is not None:
        raise ValueError('Perceptual trial requires an explicitly selected plain reconstruction parent')
    model=StudentDecoder(StudentConfig(**saved['model_config'])).to(device)
    cls=PerceptualTrialEngine if capped else DistillationEngine
    kwargs={'trial_config': PerceptualTrialConfig(max_updates=config.total_steps,review_update=config.evaluation_interval)} if capped else {}
    engine=cls(model,config=DistillationTrainingConfig(**saved['config']),
        loss_config=DistillationLossConfig(**saved['loss_config']),
        balancer_config=GradientBalancerConfig(**saved['balancer']['config']),
        discriminators=AudioDiscriminators(DiscriminatorConfig(**saved['discriminator_config'])),**kwargs)
    engine.load_state_dict(saved)
    if capped:
        provenance=engine.enable_trial(readiness_inputs['report'],readiness_inputs['policy'],crops=crops,
            current=readiness_inputs['current'],previous=readiness_inputs['previous'],evidence=readiness_inputs['evidence'])
    else:
        fork=replace(engine.config,total_steps=config.total_steps,warmup_steps=0,freeze_normalization_step=0,
            learning_rate_schedule='constant_after_warmup',warmup_start_learning_rate=None,
            final_learning_rate=engine.config.learning_rate)
        provenance=engine.fork_reconstruction(fork)
    return engine,provenance


def start_fingerprints(engine):
    state = engine.state_dict()
    return {key: state_fingerprint(state[key]) for key in
            ("model", "optimizer", "discriminators", "crop_rng")} | {
        "balancer_ema": state_fingerprint(state["balancer"]["ema"]),
        "balancer_updates": state["balancer"]["updates"],
        "balancer_weights": state["balancer"]["weights"]}


def _quantile(values, q):
    return float(torch.tensor(values, dtype=torch.float64).quantile(q)) if values else None


def evaluation_summary(rows, metadata):
    groups = summarize_evaluation(rows, metadata)
    for group, result in groups.items():
        if group == "all":
            selected = rows
        else:
            field, value = group.split("/", 1)
            selected = [r for r in rows if str(metadata[r["source_id"]].get(field) or "unverified") == value]
        active = [r for r in selected if r["teacher_rms"] >= .001]
        quiet = [w for r in selected for w in r.get("quiet_windows", {}).get("windows", []) if w["is_quiet"]]
        residuals = [w["residual_rms"] for w in quiet]
        phases = [r["quiet_phase480"]["residual_template_rms"] for r in selected
                  if "residual_template_rms" in r.get("quiet_phase480", {})]
        result.update({"nonquiet_level_error_db_median": statistics.median(r["rms_db_error"] for r in active) if active else None,
            "nonquiet_absolute_level_error_db_median": statistics.median(abs(r["rms_db_error"]) for r in active) if active else None,
            "quiet_residual_rms_mean": statistics.mean(residuals) if residuals else None,
            "quiet_residual_rms_p95": _quantile(residuals, .95),
            "active_nearzero_count": sum(r["student_rms"] < 1e-6 for r in active),
            "student_peak_abs": max((r.get("student_peak_abs", 0) for r in selected), default=0),
            "student_clipped_samples": sum(r.get("student_clipped_samples", 0) for r in selected),
            "student_exact_zero_samples": sum(r.get("student_exact_zero_samples", 0) for r in selected),
            "quiet_phase480_residual_rms_median": statistics.median(phases) if phases else None})
    return groups


def compare_evaluations(reports, previous=None):
    control,candidate=(reports[a]['groups']['all'] for a in ('control','perceptual'))
    checks={}
    for key in ('teacher_waveform_mean','teacher_mel_mean','quiet_residual_rms_mean','quiet_residual_rms_p95'):
        checks[key+'_within_10_percent_of_control']=candidate[key] is not None and control[key] is not None and candidate[key]<=1.10*control[key]
    checks['no_added_clipping']=candidate['student_clipped_samples']<=control['student_clipped_samples']
    checks['no_added_active_collapse']=candidate['active_nearzero_count']<=control['active_nearzero_count']
    checks['level_error_increase_at_most_1db']=candidate['nonquiet_absolute_level_error_db_median']<=control['nonquiet_absolute_level_error_db_median']+1
    conditions={}
    for group,c in reports['control']['groups'].items():
        if group.startswith('condition/'):
            v=reports['perceptual']['groups'][group]
            ratio=v['teacher_waveform_mean']/max(c['teacher_waveform_mean'],1e-12)
            delta=(v['nonquiet_cosine_mean']-c['nonquiet_cosine_mean']) if v['nonquiet_cosine_mean'] is not None and c['nonquiet_cosine_mean'] is not None else None
            conditions[group]={'waveform_error_ratio':ratio,'cosine_delta':delta}
    checks['no_condition_collapse']=all(v['waveform_error_ratio']<=1.10 and (v['cosine_delta'] is None or v['cosine_delta']>=-.05) for v in conditions.values())
    return {'checks':checks,'safety_checks_passed':all(checks.values()),'conditions':conditions,
        'mel_improvement_fraction':1-candidate['teacher_mel_mean']/control['teacher_mel_mean'],
        'automatic_promotion':False,'interpretation':'Controlled perceptual experiment. Passing rollback checks is not quality acceptance.'}


def _implementation():
    root = Path(__file__).parent
    names = ("perceptual_comparison", "perceptual_trial_engine", "perceptual_readiness", "balance_comparison", "comparison_data", "gradient_balancer", "distillation_training", "model",
             "source_corpus", "batched_teacher", "teacher", "quiet_audio", "representative_pilot", "restart_data", "training",
             "losses_distillation", "optimizers", "discriminators", "cache", "data", "preflight_distillation", "objective_comparison", "batching")
    return {n: file_sha(root / (n + ".py")) for n in names}



@_exclusive_run
def run_perceptual_comparison(corpus, windows, training_rows, sample_counts, ledger, heldout_crops,
        heldout_rows, output_dir, *, parent, parent_checkpoint_sha256, data_identity,
        heldout_metadata, config=PerceptualComparisonConfig(), device="cuda", reserved_rows=(),
        resume_from=None, max_updates=None, log_dir=None, run_prefix="perceptual-v1", heldout_corpus=None,
        readiness_inputs=None, review_before_continuation=False):
    """Advance both arms on each unique batch; never silently replay one arm."""
    if not isinstance(readiness_inputs,dict):
        raise ValueError('Exact parent readiness artifacts are required')
    if readiness_inputs['policy']['parent_checkpoint_sha256'] != parent_checkpoint_sha256:
        raise ValueError('Readiness policy refers to another parent')
    if readiness_inputs['policy']['data_plan_sha256'] != data_identity.get('plan',{}).get('identity_sha256'):
        raise ValueError('Readiness policy refers to another data plan')
    windows, training_rows = tuple(windows), tuple(training_rows)
    required=parent['engine']['config']['adversarial_samples']
    if any(w.valid_output_samples48k<required for w in windows):
        raise ValueError('Every planned scored crop must support the adversarial window')
    arms = config.arms
    candidate_name = arms[1]
    heldout_crops, heldout_rows = tuple(heldout_crops), tuple(heldout_rows)
    verify_comparison_windows(windows, {r.source_id: r for r in training_rows}, sample_counts, ledger,
                              reserved_rows=tuple(reserved_rows) + heldout_rows)
    assert_comparison_disjoint(training_rows, heldout_rows)
    if len(windows) != config.total_steps * config.batch_size:
        raise ValueError("Comparison needs exactly one finite full-batch plan")
    if (parent.get("engine", {}).get("step") != config.expected_parent_step or
            parent.get("identity", {}).get("data", {}).get("teacher_checkpoint_sha256") != CHECKPOINT_SHA256):
        raise ValueError("Wrong parent update or frozen teacher")
    if config.context_frames * 4 < StudentConfig(**parent["engine"]["model_config"]).history_frames:
        raise ValueError("Comparison context is shorter than the actual decoder history")
    if not re.fullmatch(r"[a-f0-9]{64}", parent_checkpoint_sha256) or data_identity.get("teacher_checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("Pinned parent and teacher hashes are required")
    if data_identity.get("source_corpus") != corpus.identity or not data_identity.get("teacher_batch_qualification"):
        raise ValueError("Source corpus and qualified batching evidence are required")
    if not heldout_crops or {c.source_id for c in heldout_crops} != {r.source_id for r in heldout_rows}:
        raise ValueError("Fixed panel crops must cover exactly the development sources")
    if any(r.split != "dev" for r in heldout_rows):
        raise ValueError("Evaluation rows must be development records")
    if heldout_corpus is None:
        raise ValueError("A qualified held-out corpus is required to verify exact targets")
    _verify_panel_targets(heldout_corpus, heldout_crops, heldout_rows, sample_counts, config.context_frames)
    if max_updates is not None and (type(max_updates) is not int or max_updates < 1):
        raise ValueError("Invalid explicit update bound")
    metadata = {r.source_id: {"dataset": r.dataset, "language": r.language, "condition": None} for r in heldout_rows}
    for key, value in heldout_metadata.items():
        if key not in metadata or value.get("dataset") != metadata[key]["dataset"]:
            raise ValueError("Evaluation metadata changes dataset identity")
        metadata[key].update(value)
    sampler = FixedWindowSampler(windows)
    identity = json.loads(canonical({"kind": "paired_unique_perceptual_comparison", "format_version": 1,
        "config": asdict(config), "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "selected_parent_engine_sha256": state_fingerprint(parent["engine"]),
        "selected_parent_rng_sha256": state_fingerprint(parent["rng"]),
        "parent_step": config.expected_parent_step, "data": data_identity, "window_identity": sampler.identity_sha256,
        "heldout": _crop_identity(heldout_crops), "metadata": metadata, "implementation": _implementation(),
        "runtime": runtime_identity(device), "readiness_artifacts":readiness_inputs,
        "comparison":{"control":"unchanged reconstruction", "perceptual":"500-step ramp into MPD/MRD adversarial and feature matching"},
        "rollback_policy":{"waveform_mel_quiet_ratio_max":1.10,"level_error_increase_db_max":1.,
            "condition_waveform_ratio_max":1.10,"condition_cosine_delta_min":-.05,"automatic_promotion":False}}))
    directory = Path(output_dir)
    if resume_from is None:
        directory.mkdir(parents=True, exist_ok=False)
    elif not directory.is_dir():
        raise ValueError("Resume requires its original journal directory")
    if shutil.disk_usage(directory).free < config.min_free_bytes:
        raise OSError("Insufficient free disk reserve")
    engines = {}
    for name in arms:
        engines[name], _ = new_arm(parent["engine"], config, device, capped=name == candidate_name, readiness_inputs=readiness_inputs, crops=heldout_crops)
    starts = {name: start_fingerprints(engine) for name, engine in engines.items()}
    if starts["control"] != starts[candidate_name]:
        raise ValueError("Paired student, generator optimizer, discriminator weights or EMA starting values differ")
    _restore_rng(parent["rng"])
    journal, inflight, latest = (directory / name for name in ("exposure.jsonl", "inflight.json", "latest.pt"))
    evaluation, midpoint = None, None
    stop_reason = None
    coverage = teacher_coverage(())
    seconds = 0.0
    if resume_from is not None:
        saved = torch.load(resume_from, map_location="cpu", weights_only=True)
        if saved.get('stop_reason') is not None:
            raise ValueError('A failed safety review requires a new declared experiment, not continuation')
        if saved.get("identity") != identity or saved.get("paired_start") != starts or inflight.exists():
            raise ValueError("Changed identity or uncertain in-flight exposure prevents resume")
        for name in arms:
            engines[name].load_state_dict(saved["engines"][name])
        sampler.load_state_dict(saved["sampler"])
        records = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
        step = engines["control"].step
        if engines[candidate_name].step != step or sampler.cursor != step * config.batch_size or len(records) != step:
            raise ValueError("Paired model steps, journal and sampler exposure disagree")
        for index, record in enumerate(records, 1):
            if record["step"] != index or record["cursor"] != index * config.batch_size or record["window_identity"] != sampler.identity_sha256:
                raise ValueError("Uncheckpointed exposure or damaged journal cannot be replayed")
        evaluation, midpoint, seconds, coverage = (saved[key] for key in ("evaluation", "midpoint", "seconds", "teacher_coverage"))
        if coverage["valid_samples"] != sum(w.valid_output_samples48k for w in windows[:sampler.cursor]):
            raise ValueError("Teacher coverage and exposure cursor differ")
        _restore_rng(saved["rng"])
        if review_before_continuation:
            if step != config.evaluation_interval or not evaluation['comparison']['safety_checks_passed']:
                raise ValueError('Matched-control safety checks do not permit continuation')
            evidence=evaluation_evidence(heldout_crops,evaluation['arms']['perceptual']['rows'],
                run_identity=identity,model_config=engines['perceptual'].model.config.to_dict())
            review=engines['perceptual'].review_trial(evidence,crops=heldout_crops)
            _atomic_json(directory/'safety-review.json',review)
            if not review['continuation_authorized']:
                raise ValueError('Perceptual trial failed its baseline safety review')
    else:
        _atomic_json(directory / "identity.json", identity)
        _atomic_json(directory / "paired-start.json", starts)

    def checkpoint():
        corpus.flush()
        payload = {"format_version": 1, "identity": identity, "engines": {a: e.state_dict() for a, e in engines.items()},
            "sampler": sampler.state_dict(), "rng": _rng_state(), "evaluation": evaluation, "midpoint": midpoint,
            "seconds": seconds, "teacher_coverage": coverage, "paired_start": starts, "stop_reason": stop_reason}
        if shutil.disk_usage(directory).free < config.min_free_bytes + _checkpoint_reserve(payload):
            raise OSError("Insufficient atomic checkpoint disk reserve")
        _atomic_save(payload, latest)

    step = engines["control"].step
    first_step = step
    stop_step = min(config.total_steps, step + max_updates) if max_updates else config.total_steps
    if engines['perceptual'].trial_review is None:
        stop_step=min(stop_step,config.evaluation_interval)
    by_id = {r.source_id: r for r in training_rows}
    with ExitStack() as stack:
        writers = {a: stack.enter_context(_writer(log_dir, run_prefix + "-" + a, identity, step, resume_from is not None)) for a in arms}

        def evaluate():
            nonlocal evaluation, midpoint
            rng = _rng_state()
            reports = {}
            try:
                for name, engine in engines.items():
                    rows = evaluate_crops(engine, heldout_crops, include_signal_checks=True)
                    reports[name] = {"rows": rows, "groups": evaluation_summary(rows, metadata), "gate": waveform_gate(engine, rows)}
                evaluation = {"step": engines["control"].step, "arms": reports,
                              "comparison": compare_evaluations(reports, midpoint)}
                if evaluation["step"] == config.total_steps and midpoint is None:
                    raise ValueError("Final comparison is missing its required midpoint evidence")
                if evaluation["step"] == config.evaluation_interval:
                    midpoint = deepcopy(evaluation)
            finally:
                _restore_rng(rng)
            _atomic_json(directory / f"evaluation-step{evaluation['step']:06d}.json", evaluation)
            for name in arms:
                if writers[name]:
                    for group, values in reports[name]["groups"].items():
                        for key, value in values.items():
                            if isinstance(value, (float, int)):
                                writers[name].add_scalar(f"validation/{group}/{key}", value, evaluation["step"])
                    writers[name].flush()

        try:
            if resume_from is None:
                evaluate()
                if reports_differ_at_start(evaluation):
                    raise ValueError("Initial paired evaluation differs despite equal model state")
                checkpoint()
            while engines["control"].step < stop_step:
                started = time.monotonic()
                chosen = windows[sampler.cursor:sampler.cursor + config.batch_size]
                corpus.prefetch(list(dict.fromkeys(w.source_id for w in chosen)), max_batch_size=config.teacher_batch_size,
                                max_total_input_samples=config.teacher_batch_samples)
                crops = []
                for window in chosen:
                    record = _record(corpus, by_id[window.source_id], sample_counts[window.source_id])
                    crop = sample_training_crop(record, window.start_frame, window.scored_frames, config.context_frames)
                    if crop.valid_scored_samples < window.valid_output_samples48k:
                        raise ValueError("Target shorter than planned scored interval")
                    crops.append(replace(crop, valid_scored_samples=window.valid_output_samples48k))
                target_identity = state_fingerprint(_crop_identity(crops))
                _atomic_json(inflight, {"step": engines["control"].step + 1, "cursor": sampler.cursor,
                    "windows": [w.to_dict() for w in chosen], "targets_sha256": target_identity})
                rng = _rng_state()
                metrics, rng_after = {}, None
                for name in arms:
                    _restore_rng(rng)
                    metrics[name] = engines[name].train_step(crops)
                    current_rng = _rng_state()
                    if rng_after is not None and state_fingerprint(current_rng) != state_fingerprint(rng_after):
                        raise RuntimeError("Paired arms consumed different random operations")
                    rng_after = current_rng
                    norms=[metrics[name].get(k+'/scaled_norm',0.) for k in ('teacher_waveform','teacher_mel','feature_matching','adversarial')]
                    metrics[name]['actual_teacher_mel_share']=norms[1]/sum(norms) if sum(norms) else 0.
                _restore_rng(rng_after)
                if state_fingerprint(_crop_identity(crops)) != target_identity:
                    raise RuntimeError("Teacher targets or raw latent values changed during the paired update")
                sampler.take_batch(len(chosen))
                step = engines["control"].step
                if engines[candidate_name].step != step:
                    raise RuntimeError("Paired update counters diverged")
                _append(journal, {"step": step, "cursor": sampler.cursor,
                    "window_identity": sampler.identity_sha256, "targets_sha256": target_identity})
                inflight.unlink()
                coverage = _merge_coverage(coverage, teacher_coverage(crops))
                seconds += time.monotonic() - started
                for name in arms:
                    metrics[name].update({"unique_scored_hours": coverage["valid_samples"] / 172_800_000,
                                          "paired_training_loop_seconds": seconds, "consumed_windows": sampler.cursor})
                    _append(directory / (name + "-metrics.jsonl"), metrics[name])
                    if writers[name]:
                        for key, value in metrics[name].items():
                            if isinstance(value, (float, int)):
                                writers[name].add_scalar("train/" + key, value, step)
                if step % config.evaluation_interval == 0 or step == stop_step:
                    evaluate()
                    if step > config.evaluation_interval and not evaluation['comparison']['safety_checks_passed']:
                        stop_reason = 'matched_control_rollback_limit'
                if step % config.checkpoint_interval == 0 or step == stop_step or stop_reason is not None:
                    checkpoint()
                _atomic_json(directory / "status.json", {"state": "training", "step": step,
                    "total_steps_per_arm": config.total_steps, "consumed_windows_per_arm": sampler.cursor,
                    "unique_scored_hours_per_arm": coverage["valid_samples"] / 172_800_000,
                    "metrics": metrics, "last_evaluation_step": evaluation["step"], "perceptual_training_started": True})
                if stop_reason is not None:
                    break
            result = {"state": "stopped_for_safety" if stop_reason is not None else "completed_awaiting_review" if step == config.total_steps else "paused_for_safety_review",
                "stop_reason": stop_reason,
                "step": step, "new_updates_per_arm": step - first_step, "total_steps_per_arm": config.total_steps,
                "consumed_windows_per_arm": sampler.cursor, "remaining_windows": sampler.remaining_segments,
                "unique_scored_hours_per_arm": coverage["valid_samples"] / 172_800_000,
                "seconds": seconds, "paired_start_verified": True, "perceptual_training_started": True,
                "comparison": evaluation["comparison"], "groups": {a: evaluation["arms"][a]["groups"] for a in arms},
                "checkpoint": str(latest)}
            _atomic_json(directory / "status.json", result)
            return result
        except BaseException as error:
            _atomic_json(directory / "status.json", {"state": "stopped_on_error", "step": engines["control"].step,
                "arm_steps": {a: e.step for a, e in engines.items()}, "cursor": sampler.cursor,
                "inflight_reserved": inflight.exists(), "error": f"{type(error).__name__}: {error}", "automatic_resume": False})
            raise


def reports_differ_at_start(evaluation):
    rows = [value["rows"] for value in evaluation["arms"].values()]
    return rows[0] != rows[1]
