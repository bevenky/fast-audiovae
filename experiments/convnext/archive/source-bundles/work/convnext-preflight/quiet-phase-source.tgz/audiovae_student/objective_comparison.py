"""Paired reconstruction continuations from one immutable teacher diagnostic.

Both arms retain the parent's latents, model, optimizer history and frozen
normalization. The waveform share is the sole difference between arms. This
runner never enables a discriminator or launches representative training.
Interrupted comparisons are preserved for inspection and are not auto-resumed.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
from typing import Sequence

import torch

from .cache import TrainingCrop
from .corpus_training import _atomic_json
from .discriminators import AudioDiscriminators, DiscriminatorConfig
from .distillation_training import (DistillationEngine, DistillationTrainingConfig,
    reconstruction_gradient_report)
from .gradient_balancer import GradientBalancerConfig
from .losses_distillation import DistillationLossConfig
from .model import StudentConfig, StudentDecoder
from .preflight_distillation import (_crop_identity, _evaluate, _implementation_identity,
    _log_evaluation, _plain, _save_audio, _sha, _writer, fold_streaming_check)
from .teacher import CHECKPOINT_SHA256
from .training import _atomic_save, _restore_rng, _rng_state, _runtime_spec


@dataclass(frozen=True)
class ObjectiveComparisonConfig:
    total_steps: int = 2000
    evaluation_interval: int = 100
    checkpoint_interval: int = 500
    warmup_steps: int = 50
    learning_rate: float = 2e-4
    warmup_start_learning_rate: float = 2e-5
    parameter_update_metrics_interval: int = 100
    expected_parent_step: int = 500
    expected_crops_per_role: int = 32
    consecutive_acceptances: int = 2
    save_audio: bool = True
    check_fold_streaming: bool = True
    min_free_bytes: int = 4 * 1024**3

    def __post_init__(self):
        for name in ("total_steps", "evaluation_interval", "checkpoint_interval",
                     "parameter_update_metrics_interval", "expected_parent_step", "expected_crops_per_role"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.total_steps > 2000:
            raise ValueError("Each objective arm is bounded to 2000 new updates")
        if self.consecutive_acceptances != 2:
            raise ValueError("Acceptance requires exactly two consecutive evaluations")
        if type(self.warmup_steps) is not int or not 2 <= self.warmup_steps < self.total_steps:
            raise ValueError("Explicit warmup needs at least two steps and must finish within the update budget")
        if any(not math.isfinite(v) or v <= 0 for v in (self.learning_rate, self.warmup_start_learning_rate)):
            raise ValueError("Learning rates must be positive and finite")
        if self.warmup_start_learning_rate > self.learning_rate:
            raise ValueError("Warmup cannot start above the held learning rate")
        if type(self.min_free_bytes) is not int or self.min_free_bytes < 0:
            raise ValueError("Disk reserve must be a nonnegative integer")


def state_fingerprint(value) -> str:
    """Stable fingerprint of nested state, including exact tensor bytes."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor")
            digest.update(json.dumps([str(tensor.dtype), list(tensor.shape)]).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            digest.update(str(len(item)).encode())
            for child in item:
                visit(child)
        elif item is None or type(item) in (bool, int, float, str):
            digest.update(type(item).__name__.encode())
            digest.update(json.dumps(item, allow_nan=False).encode())
        else:
            raise TypeError(f"Unsupported checkpoint state type {type(item).__name__}")
    visit(value)
    return digest.hexdigest()


def _common_start(engine):
    state = engine.state_dict()
    return {"model": state_fingerprint(state["model"]),
            "optimizer": state_fingerprint(state["optimizer"]),
            "discriminators": state_fingerprint(state["discriminators"]),
            "discriminator_optimizer": state_fingerprint(state["discriminator_optimizer"]),
            "crop_rng": state_fingerprint(state["crop_rng"]),
            "balancer_ema": state_fingerprint({k: v for k, v in state["balancer"].items() if k != "weights"}),
            "global_rng": state_fingerprint(_rng_state())}


def _checkpoint_bytes(state):
    """Conservative tensor size, with space for metadata/serialization."""
    def count(item):
        if isinstance(item, torch.Tensor):
            return item.numel() * item.element_size()
        if isinstance(item, dict):
            return sum(count(v) for v in item.values())
        if isinstance(item, (tuple, list)):
            return sum(count(v) for v in item)
        return 0
    return count(state) + 2 * 1024**2


def _new_arm(parent_engine, waveform_share, config, device):
    # Optimizer loading may retain references to CPU state. Deep-copy first so
    # neither arm nor a CPU test can mutate the verified parent's payload.
    saved = deepcopy(parent_engine)
    model = StudentDecoder(StudentConfig(**saved["model_config"])).to(device)
    engine = DistillationEngine(model, config=DistillationTrainingConfig(**saved["config"]),
        loss_config=DistillationLossConfig(**saved["loss_config"]),
        balancer_config=GradientBalancerConfig(**saved["balancer"]["config"]),
        discriminators=AudioDiscriminators(DiscriminatorConfig(**saved["discriminator_config"])))
    engine.load_state_dict(saved)
    fork_config = replace(engine.config, total_steps=config.total_steps,
        warmup_steps=config.warmup_steps, learning_rate=config.learning_rate,
        final_learning_rate=config.learning_rate,
        freeze_normalization_step=0,
        reconstruction_waveform_share=waveform_share,
        learning_rate_schedule="constant_after_warmup",
        warmup_start_learning_rate=config.warmup_start_learning_rate,
        parameter_update_metrics_interval=config.parameter_update_metrics_interval)
    provenance = engine.fork_reconstruction(fork_config)
    return engine, provenance


def _validate_inputs(parent, diagnostic, sentinel, data_identity, config):
    if (parent.get("format_version") != 1 or not isinstance(parent.get("identity"), dict)
            or parent.get("engine", {}).get("step") != config.expected_parent_step
            or "rng" not in parent):
        raise ValueError("A complete verified parent checkpoint at the declared step is required")
    if parent["engine"].get("perceptual_start") is not None:
        raise ValueError("Objective comparison requires a reconstruction-only parent")
    if data_identity.get("teacher_checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("Data identity must bind the original frozen teacher")
    if parent["identity"].get("data", {}).get("teacher_checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("Parent identity must bind the original frozen teacher")
    for role, crops in (("diagnostic", diagnostic), ("sentinel", sentinel)):
        if len(crops) != config.expected_crops_per_role:
            raise ValueError(f"Expected exactly {config.expected_crops_per_role} {role} crops")
        if _crop_identity(crops) != parent["identity"].get(role):
            raise ValueError(f"{role} crop values or order differ from the verified parent")
        if not any(c.start_frame == 0 for c in crops) or not any(c.start_frame > 0 for c in crops):
            raise ValueError(f"{role} must contain beginning and interior crops")
    if {c.source_id for c in diagnostic} & {c.source_id for c in sentinel}:
        raise ValueError("Held-out sentinel identities must never enter training")
    for name in ("stem_norm", "affine"):
        if not bool(parent["engine"]["model"].get(f"{name}.statistics_frozen", False)):
            raise ValueError("Parent normalization statistics must already be frozen")


@contextmanager
def _fresh_directory(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".run.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This comparison already has an active writer") from error
        if any(p.name != ".run.lock" for p in path.iterdir()):
            raise ValueError("Existing comparison outputs are immutable; automatic resume is disabled")
        yield


def _evaluation_safety(report, baseline_report=None):
    """Stop only on nonfinite output or unmistakable amplitude explosion.

    Ordinary sentinel quality failures are reported without selecting or
    stopping training. This guard is deliberately far above the 1 dB gate.
    """
    problems = []
    baseline = {(role, row["source_id"], row["start_frame"]): row["student_rms"]
                for role, rows in (baseline_report or {"rows": {}})["rows"].items() for row in rows}
    for role, rows in report["rows"].items():
        for row in rows:
            numeric = (v for v in row.values() if type(v) in (int, float))
            if any(not math.isfinite(v) for v in numeric):
                problems.append({"role": role, "source_id": row["source_id"],
                                 "start_frame": row["start_frame"], "reason": "nonfinite_evaluation"})
            elif row["student_rms"] > max(1.0, 100 * row["teacher_rms"],
                    10 * baseline.get((role, row["source_id"], row["start_frame"]), 0.0)):
                problems.append({"role": role, "source_id": row["source_id"],
                                 "start_frame": row["start_frame"], "reason": "catastrophic_amplitude"})
    return problems


def _run_arm(engine, diagnostic, sentinel, directory, *, identity, config, log_dir, run_name):
    directory.mkdir()
    _atomic_json(directory / "identity.json", identity)
    started = time.monotonic()
    latest_metrics, latest_checkpoint = {}, None
    exposure, consecutive, evaluated_step = 0, 0, -1
    milestones = {"first_095_all_clips_step": None, "first_099_all_clips_step": None}
    state = "running"
    safety_problems = []
    report = None
    baseline_report = None
    def status(state_name, **extra):
        _atomic_json(directory / "status.json", {"state": state_name, "step": engine.step,
            "parent_step": config.expected_parent_step, "total_steps": config.total_steps,
            "phase": "paired_reconstruction_diagnostic", "pid": os.getpid(),
            "elapsed_seconds": time.monotonic() - started, "latest_metrics": latest_metrics,
            "latest_checkpoint": latest_checkpoint, "consecutive_acceptances": consecutive,
            "repeated_scored_hours": exposure / 48000 / 3600,
            "main_exposure_hours": 0, **extra})
    def checkpoint():
        nonlocal latest_checkpoint
        name = f"checkpoint-step{engine.step:06d}.pt"
        path = directory / name
        if path.exists():
            if latest_checkpoint is None or latest_checkpoint["path"] != name or _sha(path) != latest_checkpoint["sha256"]:
                raise ValueError("Cannot overwrite an immutable comparison checkpoint")
            return
        payload = {"format_version": 1, "identity": identity, "engine": engine.state_dict(),
            "rng": _rng_state(), "evaluated_step": evaluated_step,
            "repeated_scored_samples": exposure, "latest_metrics": latest_metrics,
            "consecutive_acceptances": consecutive, "milestones": milestones,
            "automatic_resume_supported": False}
        if shutil.disk_usage(directory).free < config.min_free_bytes + _checkpoint_bytes(payload):
            raise OSError("Insufficient checkpoint space while preserving the declared free-disk reserve")
        _atomic_save(payload, path)
        latest_checkpoint = {"path": name, "sha256": _sha(path), "step": engine.step}
        _atomic_json(directory / "latest.json", latest_checkpoint)
    def evaluate(writer):
        nonlocal report, baseline_report, consecutive, evaluated_step, safety_problems
        report = _evaluate(engine, diagnostic, sentinel)
        gate = report["gates"]["diagnostic"]["all"]
        if gate["criteria"]["waveform_cosine_min"] != 0.99:
            raise ValueError("Comparison acceptance must use the selected 0.99 waveform target")
        milestone = gate.get("milestone_095_passed", False)
        if milestone and milestones["first_095_all_clips_step"] is None:
            milestones["first_095_all_clips_step"] = engine.step
        if report["diagnostic_passed"] and milestones["first_099_all_clips_step"] is None:
            milestones["first_099_all_clips_step"] = engine.step
        # Initial state is evidence, but is not one of the two consecutive
        # post-update evaluations required to stop an arm successfully.
        if engine.step > 0:
            consecutive = consecutive + 1 if report["diagnostic_passed"] else 0
        if baseline_report is None:
            baseline_report = report
        safety_problems = _evaluation_safety(report, baseline_report)
        report.update({"milestones": dict(milestones), "consecutive_acceptances": consecutive,
                       "sentinel_used_for_acceptance": False, "safety_problems": safety_problems,
                       "amplitude_stop_rms_exceeds": "max(1, 100 * teacher RMS, 10 * parent student RMS)"})
        _atomic_json(directory / f"evaluation-step{engine.step:06d}.json", report)
        _log_evaluation(writer, report)
        if writer is not None:
            for role, rows in report["rows"].items():
                nonquiet = [r for r in rows if r["teacher_rms"] >= 1e-3]
                if nonquiet:
                    cosines = [r["waveform_cosine"] for r in nonquiet]
                    writer.add_scalar(f"evaluation/{role}/nonquiet_cosine_min", min(cosines), engine.step)
                    writer.add_scalar(f"evaluation/{role}/nonquiet_cosine_mean", sum(cosines) / len(cosines), engine.step)
                    for threshold, name in ((0.95, "095"), (0.99, "099")):
                        passing = sum(r["waveform_cosine"] >= threshold and abs(r["rms_db_error"]) <= 1
                                      and r["waveform_to_silence_error_ratio"] <= 0.5 for r in nonquiet)
                        writer.add_scalar(f"evaluation/{role}/nonquiet_fraction_passing_{name}",
                                          passing / len(nonquiet), engine.step)
                writer.add_scalar(f"evaluation/{role}/waveform_cosine_target", 0.99, engine.step)
            writer.add_scalar("evaluation/diagnostic/milestone_095_passed", float(milestone), engine.step)
            writer.add_scalar("evaluation/diagnostic/consecutive_acceptances", consecutive, engine.step)
            writer.flush()
        evaluated_step = engine.step
    try:
        with _writer(log_dir, run_name, identity, 0, False) as writer:
            _atomic_json(directory / "gradient-initial.json", reconstruction_gradient_report(engine, diagnostic))
            evaluate(writer)
            checkpoint()
            status(state)
            while engine.step < config.total_steps and not safety_problems:
                before = time.monotonic()
                latest_metrics = engine.train_step(diagnostic)
                latest_metrics["step_seconds"] = time.monotonic() - before
                exposure += latest_metrics["scored_samples"]
                if engine.perceptual_start is not None:
                    raise RuntimeError("Comparison must not enter perceptual training")
                if writer is not None:
                    for key, value in latest_metrics.items():
                        writer.add_scalar(f"train/{key}", value, engine.step)
                    writer.add_scalar("exposure/repeated_diagnostic_hours", exposure / 48000 / 3600, engine.step)
                    writer.flush()
                if engine.step % config.evaluation_interval == 0 or engine.step == config.total_steps:
                    evaluate(writer)
                accepted = consecutive >= config.consecutive_acceptances
                if (engine.step % config.checkpoint_interval == 0 or engine.step == config.total_steps
                        or accepted or safety_problems):
                    checkpoint()
                status("running")
                if accepted:
                    break
            if safety_problems:
                state = "stopped_instability"
            elif consecutive >= config.consecutive_acceptances:
                state = "gate_passed_awaiting_review"
            else:
                state = "budget_exhausted_below_gate"
            _atomic_json(directory / "gradient-final.json", reconstruction_gradient_report(engine, diagnostic))
            streaming = (fold_streaming_check(engine.model, diagnostic[0])
                         if config.check_fold_streaming else {"performed": False})
            audio = (_save_audio(engine, {"diagnostic": diagnostic, "sentinel": sentinel}, directory)
                     if config.save_audio else [])
            summary = {"state": state, "step": engine.step, "parent_step": config.expected_parent_step,
                "diagnostic_passed": consecutive >= config.consecutive_acceptances and not safety_problems,
                "sentinel_passed": report["sentinel_passed"], "sentinel_used_for_acceptance": False,
                "gates": report["gates"], "milestones": milestones,
                "consecutive_acceptances": consecutive, "safety_problems": safety_problems,
                "fold_streaming": streaming, "audio_items": len(audio), "latest_checkpoint": latest_checkpoint,
                "elapsed_seconds": time.monotonic() - started, "latest_metrics": latest_metrics,
                "repeated_scored_hours": exposure / 48000 / 3600, "main_exposure_hours": 0,
                "automatically_launched_training": False, "perceptual_training_started": False,
                "automatic_resume_supported": False}
            _atomic_json(directory / "summary.json", summary)
            status(state)
            return summary
    except BaseException as error:
        status("error", error_type=type(error).__name__, error=str(error))
        raise


def run_comparison(parent_payload: dict, diagnostic_crops: Sequence[TrainingCrop],
                   sentinel_crops: Sequence[TrainingCrop], output_dir: str | Path, *,
                   parent_checkpoint_sha256: str, data_identity: dict, device="cuda",
                   config=ObjectiveComparisonConfig(), log_dir=None, run_name="objective-comparison-v1"):
    """Run two fresh, sequential arms from a caller-verified parent payload.

    The caller must verify the source checkpoint file against
    ``parent_checkpoint_sha256`` before loading it with ``weights_only=True``.
    Exact crop hashes/order are checked against that checkpoint, independently
    of any new cache-reader artifact. No teacher or cache writes occur here.
    Existing outputs always refuse; checkpoints are inspectable continuations,
    not an implicit resume contract under a changed objective.
    """
    diagnostic, sentinel = tuple(diagnostic_crops), tuple(sentinel_crops)
    if not re.fullmatch(r"[0-9a-f]{64}", parent_checkpoint_sha256):
        raise ValueError("Parent checkpoint requires its verified SHA256")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_name):
        raise ValueError("Comparison run name must be a single safe directory name")
    _validate_inputs(parent_payload, diagnostic, sentinel, data_identity, config)
    parent_fingerprint = state_fingerprint(parent_payload)
    directory = Path(output_dir).resolve()
    device = torch.device(device)
    runtime = {**_runtime_spec(), "device": str(device),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
    if device.type == "cuda":
        runtime.update({"gpu_name": torch.cuda.get_device_name(device), "cuda_version": torch.version.cuda})
    implementation = {**_implementation_identity(), **{
        name: _sha(Path(__file__).with_name(name)) for name in (
            "objective_comparison.py", "cached_comparison_data.py", "run_objective_comparison.py")}}
    identity = _plain({"format_version": 1, "kind": "paired_objective_comparison",
        "config": asdict(config), "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "parent_payload_state_sha256": parent_fingerprint,
        "parent_identity": parent_payload["identity"], "data": data_identity,
        "diagnostic": _crop_identity(diagnostic), "sentinel": _crop_identity(sentinel),
        "runtime": runtime, "implementation": implementation, "run_name": run_name,
        "arm_order": ["waveform-mel", "waveform-only"], "waveform_cosine_target": 0.99,
        "milestone_cosine": 0.95, "main_exposure_hours": 0, "repeated_diagnostic": True,
        "automatic_resume_supported": False})
    results, paired_start = {}, None
    with _fresh_directory(directory):
        copies = 2 * (1 + math.ceil(config.total_steps / config.checkpoint_interval))
        forecast = copies * _checkpoint_bytes(parent_payload)
        if shutil.disk_usage(directory).free < config.min_free_bytes + forecast:
            raise OSError("Insufficient space for both bounded arms and the declared free-disk reserve")
        _atomic_json(directory / "identity.json", identity)
        def status(state, **extra):
            _atomic_json(directory / "status.json", {"state": state, "pid": os.getpid(),
                "completed_arms": list(results), "main_exposure_hours": 0,
                "automatically_launched_training": False, **extra})
        status("starting")
        try:
            for arm, share in (("waveform-mel", 0.5), ("waveform-only", 1.0)):
                status("running", active_arm=arm)
                engine, provenance = _new_arm(parent_payload["engine"], share, config, device)
                _restore_rng(parent_payload["rng"])
                start = _common_start(engine)
                if paired_start is None:
                    paired_start = start
                elif start != paired_start:
                    raise ValueError("Paired arms do not have identical model, optimizer, EMA and RNG states")
                arm_identity = _plain({**identity, "arm": arm, "waveform_share": share,
                                      "fork": provenance, "starting_state_sha256": start})
                result = _run_arm(engine, diagnostic, sentinel, directory / arm,
                    identity=arm_identity, config=config, log_dir=log_dir, run_name=f"{run_name}-{arm}")
                results[arm] = result
                del engine
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            if state_fingerprint(parent_payload) != parent_fingerprint:
                raise RuntimeError("The immutable parent checkpoint payload was modified")
            summary = {"state": "comparison_complete_awaiting_review", "arms": results,
                "paired_start_verified": True, "starting_state_sha256": paired_start,
                "parent_payload_unchanged": True, "parent_checkpoint_sha256": parent_checkpoint_sha256,
                "waveform_cosine_target": 0.99, "main_exposure_hours": 0,
                "automatically_launched_training": False, "perceptual_training_started": False,
                "automatic_resume_supported": False}
            _atomic_json(directory / "summary.json", summary)
            status(summary["state"])
            return summary
        except BaseException as error:
            status("error", error_type=type(error).__name__, error=str(error))
            raise
