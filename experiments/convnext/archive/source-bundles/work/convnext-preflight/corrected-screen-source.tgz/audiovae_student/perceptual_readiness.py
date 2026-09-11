"""Read-only eligibility checks for a bounded perceptual-training experiment.

This module never enables a discriminator, changes a loss, or modifies final
acceptance. ``ready_for_bounded_trial`` is deliberately distinct from the
``passed`` key consumed by the historical final-quality gate. Numeric defaults
are declared engineering safeguards for an experiment, not perceptual scores
or published Supertonic thresholds.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics

import torch

from .distillation_training import (_quiet_evidence_passed, model_state_fingerprint,
                                    reconstruction_gradient_report, waveform_gate)
from .objective_comparison import state_fingerprint
from .preflight_distillation import _crop_identity, fold_streaming_check
from .representative_pilot import _verify_panel_targets
from .restart_data import file_sha
from .teacher import CHECKPOINT_SHA256, SOURCE_SHA256
from .training import _rng_state, _restore_rng


@dataclass(frozen=True)
class PerceptualReadinessPolicy:
    max_trial_updates: int = 1000
    safety_review_update: int = 250
    maximum_reconstruction_regression: float = 1.10
    minimum_active_median_level_db: float = -20.0
    maximum_active_median_level_db: float = 6.0
    maximum_clipped_fraction: float = .001
    active_teacher_rms_min: float = .001
    nearzero_student_rms: float = 1e-6

    def __post_init__(self):
        if (type(self.max_trial_updates) is not int or not 1 <= self.max_trial_updates <= 1000
                or type(self.safety_review_update) is not int
                or not 1 <= self.safety_review_update <= min(250, self.max_trial_updates)):
            raise ValueError("Readiness only authorizes a bounded trial with an early review")
        if any(not math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Readiness thresholds must be finite")
        if (self.maximum_reconstruction_regression < 1
                or self.minimum_active_median_level_db >= self.maximum_active_median_level_db
                or not 0 <= self.maximum_clipped_fraction < 1
                or self.active_teacher_rms_min <= 0 or self.nearzero_student_rms <= 0):
            raise ValueError("Invalid readiness safeguard thresholds")


def discriminator_optimizer_proposal():
    """Explicit proposed next-trial settings; no optimizer is instantiated."""
    return {
        "optimizer": "AdamW", "learning_rate": 2e-4,
        "betas": [.8, .9], "weight_decay": 0.0,
        "current_engine_defaults": {"betas": [.9, .999], "weight_decay": .01},
        "status": "Proposal requiring explicit next-stage integration; current engine unchanged",
        "source": "https://github.com/gemelo-ai/vocos/blob/eb39abfc42c1dee4854d9b10d44dd7d4fd3b0e53/vocos/experiment.py#L74",
        "scope": "Vocos informs betas; zero decay is our declared initial choice. Supertonic's exact settings are unpublished.",
    }


def _seal(body):
    body = json.loads(json.dumps(body, allow_nan=False))
    return {**body, "sha256": state_fingerprint(body)}


def _unseal(value):
    if not isinstance(value, dict):
        raise ValueError("Readiness requires a complete evidence record")
    body = {k: v for k, v in value.items() if k != "sha256"}
    if value.get("sha256") != state_fingerprint(body):
        raise ValueError("Readiness evidence hash changed")
    return body


def readiness_binding(engine, crops):
    """Bind a diagnostic to actual current training state and exact raw targets."""
    modules = ("perceptual_readiness", "distillation_training", "preflight_distillation", "representative_pilot",
               "model", "gradient_balancer", "losses_distillation", "quiet_audio", "optimizers",
               "source_corpus", "cache", "teacher")
    return json.loads(json.dumps({
        "teacher_checkpoint_sha256": CHECKPOINT_SHA256,
        "teacher_source_sha256": SOURCE_SHA256,
        "model_state_sha256": model_state_fingerprint(engine.model),
        "model_config": engine.model.config.to_dict(),
        "training_config": asdict(engine.config),
        "loss_config": asdict(engine.criterion.config),
        "optimizer_sha256": state_fingerprint(engine.optimizer.state_dict()),
        "balancer_sha256": state_fingerprint(engine.balancer.state_dict()),
        "step": engine.step,
        "panel_sha256": state_fingerprint(_crop_identity(crops)),
        "implementation": {name: file_sha(Path(__file__).with_name(name + ".py")) for name in modules},
    }, allow_nan=False))


def evaluation_evidence(crops, rows, *, run_identity, model_config):
    """Bind existing signal-check evaluations without inventing new measurements.

    Use the immutable comparison run identity for both saved evaluations.
    Historical rows need not be rerun on a new model: their existing model hash
    and step are checked. Signal checks must have been recorded at evaluation.
    """
    crops, rows = tuple(crops), tuple(rows)
    if not crops or len(crops) != len(rows) or not run_identity:
        raise ValueError("A complete fixed-panel evaluation and run identity are required")
    expected = {(c.source_id, c.start_frame): c.valid_scored_samples for c in crops}
    if len(expected) != len(crops):
        raise ValueError("Duplicate panel crop identity")
    seen, steps, fingerprints = set(), set(), set()
    numeric = ("teacher_waveform", "teacher_mel", "teacher_rms", "student_rms",
               "waveform_to_silence_error_ratio", "rms_db_error", "waveform_cosine", "student_peak_abs")
    for row in rows:
        key = (row.get("source_id"), row.get("start_frame"))
        if key in seen or key not in expected or row.get("samples") != expected[key]:
            raise ValueError("Evaluation crop identity or scored-sample accounting changed")
        seen.add(key)
        if row.get("fixed_statistics") is not True:
            raise ValueError("Evaluation needs fixed normalization")
        if any(type(row.get(k)) not in (int, float) or not math.isfinite(row[k]) for k in numeric):
            raise ValueError("Evaluation measurements must be complete and finite")
        if (any(row[k] < 0 for k in ("teacher_waveform", "teacher_mel", "teacher_rms", "student_rms",
                                    "waveform_to_silence_error_ratio", "student_peak_abs"))
                or type(row.get("student_clipped_samples")) is not int
                or not 0 <= row["student_clipped_samples"] <= row["samples"]
                or type(row.get("evaluated_step")) is not int or row["evaluated_step"] < 0):
            raise ValueError("Invalid evaluation counts or magnitudes")
        fingerprint = row.get("model_state_sha256")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError("Evaluation needs its actual model-state fingerprint")
        steps.add(row["evaluated_step"])
        fingerprints.add(fingerprint)
    if len(steps) != 1 or len(fingerprints) != 1:
        raise ValueError("Evaluation mixes model checkpoints")
    return _seal({"format_version": 1, "kind": "fixed_panel_evaluation",
        "teacher_checkpoint_sha256": CHECKPOINT_SHA256,
        "panel_sha256": state_fingerprint(_crop_identity(crops)),
        "model_config": deepcopy(model_config), "run_identity": deepcopy(run_identity),
        "step": next(iter(steps)), "model_state_sha256": next(iter(fingerprints)),
        "rows": deepcopy(list(rows))})


def collect_readiness_evidence(engine, crops, *, corpus, panel_rows, sample_counts, context_frames=29):
    """Collect actual cache, gradient, and CPU streaming checks without updates.

    The caller supplies an already-qualified teacher corpus and complete panel.
    No datasets are downloaded, no optimizer step runs, and no RTF is measured.
    Gradients and streaming are probed on one beginning and one interior crop;
    target alignment/provenance is checked for every panel crop.
    """
    crops, panel_rows = tuple(crops), tuple(panel_rows)
    if not crops or any(r.split != "dev" for r in panel_rows):
        raise ValueError("Readiness requires a development panel")
    if {c.source_id for c in crops} != {r.source_id for r in panel_rows}:
        raise ValueError("Readiness panel rows and crops differ")
    probes = []
    for beginning in (True, False):
        choices = [c for c in crops if (c.start_frame == 0) == beginning and c.latents.shape[-1] >= 2]
        if not choices:
            raise ValueError("Readiness needs beginning and interior probes with multiple latent frames")
        probes.append(choices[0])
    teacher = getattr(corpus, "_teacher", None)
    if teacher is None:
        raise ValueError("Readiness requires the live frozen teacher used by the corpus")
    provenance = getattr(corpus, "teacher_identity", {})
    if (provenance.get("checkpoint_sha256") != CHECKPOINT_SHA256
            or provenance.get("source_sha256") != SOURCE_SHA256
            or provenance.get("dtype") != "float32"
            or teacher.training or any(p.requires_grad for p in teacher.parameters())):
        raise ValueError("Teacher must be the pinned frozen FP32 inference model")
    binding = readiness_binding(engine, crops)
    before = state_fingerprint(engine.state_dict())
    teacher_before = state_fingerprint(teacher.state_dict())
    modes = {module: module.training for module in engine.model.modules()}
    rng = _rng_state()
    try:
        _verify_panel_targets(corpus, crops, panel_rows, sample_counts, context_frames)
        streams = [{"crop": _crop_identity([c])[0], "report": fold_streaming_check(engine.model, c)} for c in probes]
        gradients = reconstruction_gradient_report(engine, probes)
        if state_fingerprint(engine.state_dict()) != before or state_fingerprint(teacher.state_dict()) != teacher_before:
            raise RuntimeError("Readiness collection mutated student or teacher state")
        if readiness_binding(engine, crops) != binding:
            raise RuntimeError("Model, raw latents or teacher targets changed during readiness collection")
    finally:
        # The existing probe restores the root mode; retain deliberately mixed
        # child modes as well without cascading Module.train() into children.
        for module, mode in modes.items():
            module.training = mode
        _restore_rng(rng)
    return _seal({"format_version": 1, "kind": "bounded_perceptual_readiness_checks", "binding": binding,
        "teacher": {"checkpoint_sha256": CHECKPOINT_SHA256, "source_sha256": SOURCE_SHA256,
                    "frozen": True, "dtype": "float32", "state_unchanged": True,
                    "state_sha256": teacher_before},
        "alignment": {"passed": True, "crops": len(crops), "context_frames": context_frames,
                      "latent_channels": 64, "output_samples_per_latent": 1920, "posterior": "raw_mu"},
        "streaming": streams, "gradients": gradients,
        "scope": "All target crops checked; gradient and CPU streaming probes cover one beginning and one interior crop."})


def _summary(rows, policy):
    active = [r for r in rows if r["teacher_rms"] >= policy.active_teacher_rms_min]
    if not active:
        raise ValueError("Readiness panel has no active audio")
    quiet = [w for r in rows for w in r.get("quiet_windows", {}).get("windows", []) if w["is_quiet"]]
    if not quiet:
        raise ValueError("Readiness panel requires measured quiet intervals")
    residuals = [w["residual_rms"] for w in quiet]
    if any(not math.isfinite(v) or v < 0 for v in residuals):
        raise ValueError("Quiet measurements must be finite")
    return {
        "waveform_error": statistics.mean(r["teacher_waveform"] for r in rows),
        "mel_error": statistics.mean(r["teacher_mel"] for r in rows),
        "quiet_residual_mean": statistics.mean(residuals),
        "active_silence_error_ratio": statistics.mean(r["waveform_to_silence_error_ratio"] for r in active),
        "active_median_level_db": statistics.median(r["rms_db_error"] for r in active),
        "active_nearzero_count": sum(r["student_rms"] <= policy.nearzero_student_rms for r in active),
        "clipped_fraction": sum(r["student_clipped_samples"] for r in rows) / sum(r["samples"] for r in rows),
    }


def assess_perceptual_readiness(engine, crops, *, current, previous, evidence,
                               policy=PerceptualReadinessPolicy()):
    """Fail closed on absent/stale evidence; return eligibility, never activation.

    ``current`` and ``previous`` are evaluation_evidence records from the same
    fixed-panel run, with strictly increasing steps. ``evidence`` must come from
    collect_readiness_evidence at this exact current model/optimizer state.
    """
    crops = tuple(crops)
    binding = readiness_binding(engine, crops)
    checked = _unseal(evidence)
    if (checked.get("format_version") != 1 or checked.get("kind") != "bounded_perceptual_readiness_checks"
            or checked.get("binding") != binding):
        raise ValueError("Missing or stale readiness evidence")
    evaluations = []
    for item in (current, previous):
        body = _unseal(item)
        rebuilt = evaluation_evidence(crops, body.get("rows", ()), run_identity=body.get("run_identity"),
                                      model_config=engine.model.config.to_dict())
        if rebuilt != item:
            raise ValueError("Evaluation identity is stale or inconsistent")
        evaluations.append(body)
    now, prior = evaluations
    if (now["model_state_sha256"] != binding["model_state_sha256"] or now["step"] != engine.step
            or now["run_identity"] != prior["run_identity"] or prior["step"] >= now["step"]):
        raise ValueError("Readiness needs the current model and an earlier matched-run evaluation")
    teacher, alignment = checked.get("teacher", {}), checked.get("alignment", {})
    if ({k: v for k, v in teacher.items() if k != "state_sha256"} !=
                   {"checkpoint_sha256": CHECKPOINT_SHA256, "source_sha256": SOURCE_SHA256,
                    "frozen": True, "dtype": "float32", "state_unchanged": True}
            or not isinstance(teacher.get("state_sha256"), str) or len(teacher["state_sha256"]) != 64
            or alignment.get("passed") is not True or alignment.get("crops") != len(crops)
            or alignment.get("latent_channels") != 64 or alignment.get("output_samples_per_latent") != 1920
            or alignment.get("posterior") != "raw_mu"):
        raise ValueError("Missing frozen-teacher or aligned raw-latent evidence")
    streams = checked.get("streaming", ())
    panel = _crop_identity(crops)
    positions = set()
    for stream in streams:
        c, report = stream.get("crop"), stream.get("report", {})
        if c not in panel:
            raise ValueError("Streaming probe is outside the bound panel")
        positions.add(c["start_frame"] == 0)
        expected_samples = min(16, c["context_frames"] + c["scored_frames"]) * 1920
        if (report.get("passed") is not True or report.get("device") != "cpu" or report.get("threads") != 1
                or report.get("rtf_measured") is not False or report.get("samples") != expected_samples
                or type(report.get("chunks")) is not int or not 2 <= report["chunks"] <= 8
                or any(type(report.get(k)) not in (int, float) or not math.isfinite(report[k]) or report[k] < 0
                       for k in ("fold_max_abs", "stream_max_abs"))):
            raise ValueError("Incomplete streaming parity or sample-accounting evidence")
    if positions != {True, False}:
        raise ValueError("Streaming evidence needs beginning and interior coverage")
    gradients = checked.get("gradients", {})
    if (gradients.get("evaluated_step") != engine.step or gradients.get("normalization_mode") != "fixed_statistics"
            or gradients.get("probe_examples") != len(streams) or not gradients.get("parameters")):
        raise ValueError("Missing current gradient-route evidence")
    expected_names = set(engine.gradient_probe_names())
    if {p.get("parameter") for p in gradients["parameters"]} != expected_names:
        raise ValueError("Gradient evidence must cover adapter, hidden block, and output head")
    for row in gradients["parameters"]:
        values = [v for k, v in row.items() if k != "parameter"]
        if not values or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError("Parameter gradient evidence is not finite")
        if row.get("waveform_scaled_norm", 0) <= 0:
            raise ValueError("Waveform anchor has no gradient route through a required parameter")
    output_gradients = gradients.get("output_gradients", {})
    if (not output_gradients or any(type(v) not in (int, float) or not math.isfinite(v) for v in output_gradients.values())
            or output_gradients.get("combined_norm", 0) <= 0):
        raise ValueError("Output gradient evidence is missing, zero, or nonfinite")
    before, after = _summary(prior["rows"], policy), _summary(now["rows"], policy)
    for row in (*prior["rows"], *now["rows"]):
        # False is an expected quality failure; malformed evidence raises.
        _quiet_evidence_passed(row, engine.config.quiet_audio)
    checks = {
        "reconstruction_stage_only": engine.perceptual_start is None and engine.gate is None,
        "frozen_normalization": all(bool(getattr(engine.model, n).statistics_frozen) for n in ("stem_norm", "affine")),
        "strict_local_quiet_gate_retained": engine.config.quiet_window_gate,
        "no_active_nearzero_collapse": after["active_nearzero_count"] == 0,
        "better_than_silence_overall": after["active_silence_error_ratio"] < 1.0,
        "active_volume_sanity": policy.minimum_active_median_level_db <= after["active_median_level_db"] <= policy.maximum_active_median_level_db,
        "clipping_sanity": after["clipped_fraction"] <= policy.maximum_clipped_fraction,
    }
    for name in ("waveform_error", "mel_error", "quiet_residual_mean"):
        checks[name + "_stable"] = after[name] <= policy.maximum_reconstruction_regression * before[name]
    # This invokes the original final gate, retaining its 0.99 and quiet rules.
    # A failed final gate is expected during an early bounded experiment.
    final = waveform_gate(engine, now["rows"])
    return _seal({"format_version": 1, "kind": "perceptual_trial_readiness", "binding": binding,
        "policy": asdict(policy), "checks": checks, "ready_for_bounded_trial": all(checks.values()),
        "baseline": before, "current": after, "evidence_sha256": evidence["sha256"],
        "evaluation_sha256": {"current": current["sha256"], "previous": previous["sha256"]},
        "final_acceptance": final, "automatic_activation": False,
        "interpretation": "Eligibility for one controlled perceptual experiment, not decoder acceptance or a perceptual quality score.",
        "discriminator_optimizer_proposal": discriminator_optimizer_proposal()})
