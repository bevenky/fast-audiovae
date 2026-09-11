"""A bounded, resumable fixed-batch reconstruction preflight.

This is not the corpus/GAN trainer. Its CLI uses synthetic audio on CPU and
does not establish speech quality, trained-model RTF, or a training budget.
"""

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from .losses import WarmupLossConfig, WarmupReconstructionLoss
from .optimizers import build_optimizer_bundle


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    grad_clip_norm: float = 1.0
    seed: int = 0
    optimizer: str = "adamw"
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_adjust_lr_fn: str = "match_rms_adamw"

    def __post_init__(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if len(self.betas) != 2 or any(not 0 <= b < 1 for b in self.betas):
            raise ValueError("betas must contain two values in [0, 1)")
        if not math.isfinite(self.grad_clip_norm) or self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be finite and positive")
        if self.optimizer not in {"adamw", "muon_adamw"}:
            raise ValueError("optimizer must be 'adamw' or 'muon_adamw'")
        if not math.isfinite(self.muon_momentum) or not 0 < self.muon_momentum < 1:
            raise ValueError("muon_momentum must be between zero and one")
        if type(self.muon_ns_steps) is not int or self.muon_ns_steps < 1:
            raise ValueError("muon_ns_steps must be a positive integer")
        if self.muon_adjust_lr_fn != "match_rms_adamw":
            raise ValueError("This experiment requires muon_adjust_lr_fn='match_rms_adamw'")


@dataclass(frozen=True)
class TrainingBatch:
    latents: Tensor
    teacher_audio: Tensor
    reference_audio_16k: Tensor | None = None


@dataclass
class TrainingResult:
    step: int
    metrics: list[dict[str, float | int]]
    checkpoint: Path | None
    step_elapsed_seconds: list[float] = field(default_factory=list)
    log_path: Path | None = None


def _model_spec(model: nn.Module, model_config: Mapping[str, Any] | None) -> dict[str, Any]:
    if model_config is None:
        config = getattr(model, "config", None)
        if is_dataclass(config) and not isinstance(config, type):
            model_config = asdict(config)
        elif isinstance(config, Mapping):
            model_config = config
        else:
            raise ValueError("Provide model_config or a model.config dataclass/mapping")
    # JSON round-tripping rejects tensors/objects and canonicalizes tuples.
    return {"class": f"{type(model).__module__}.{type(model).__qualname__}",
            "config": json.loads(json.dumps(dict(model_config), allow_nan=False))}


def _batch_digest(batch: TrainingBatch) -> str:
    digest = hashlib.sha256()
    for name in ("latents", "teacher_audio", "reference_audio_16k"):
        value = getattr(batch, name)
        digest.update(name.encode())
        if value is None:
            digest.update(b"none")
        else:
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
            "numpy": {"name": numpy_state[0],
                      "keys": torch.tensor(numpy_state[1].astype(np.int64)),
                      "position": numpy_state[2], "has_gauss": numpy_state[3],
                      "cached_gaussian": numpy_state[4]},
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def _runtime_spec() -> dict[str, Any]:
    return {"torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "default_dtype": str(torch.get_default_dtype()),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["name"], numpy_state["keys"].numpy().astype(np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"],
                         numpy_state["cached_gaussian"]))
    if state["torch_cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("A CUDA checkpoint cannot exactly resume on a CPU-only host")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _atomic_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = stream.name
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


@contextmanager
def _tensorboard_logger(log_dir: str | Path | None, run_name: str, *, step: int,
                        resuming: bool, manifest: dict[str, Any]):
    if log_dir is None:
        yield None
        return
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_name):
        raise ValueError("run_name must be a single directory name using letters, numbers, '.', '_' or '-'")
    path = Path(log_dir) / run_name
    manifest = json.loads(json.dumps(manifest, allow_nan=False))
    record = path / "run.json"
    if record.exists() and json.loads(record.read_text()) != manifest:
        raise ValueError("TensorBoard run metadata differs; choose a new run_name")
    has_events = path.exists() and any(path.glob("events.out.tfevents.*"))
    if has_events and not resuming:
        raise ValueError("TensorBoard run already has events; resume its checkpoint or choose a new run_name")
    if has_events and not record.exists():
        raise ValueError("TensorBoard run has no verifiable metadata; choose a new run_name")

    writer = None
    try:
        # Importing optional logging dependencies must not change the model's
        # seeded update sequence, including when logging starts after resume.
        rng = _rng_state()
        try:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as error:
                raise RuntimeError("TensorBoard logging was requested but tensorboard is not installed. "
                                   "Install tensorboard in the training environment or omit log_dir.") from error
            path.mkdir(parents=True, exist_ok=True)
            record.write_text(json.dumps(manifest, indent=2) + "\n")
            # Resume from step N hides any stale events at N+1 or later,
            # including events emitted before a crash but after checkpoint N.
            writer = SummaryWriter(log_dir=str(path), max_queue=1, flush_secs=1,
                                   purge_step=step + 1 if resuming else None)
            writer.add_text("run/qualification", "Fixed-batch reconstruction preflight. "
                            "Training curves only; no validation or speech-quality result.", step)
            writer.add_text("run/configuration", "```json\n" + json.dumps(manifest, indent=2)
                            + "\n```", step)
            writer.flush()
        finally:
            _restore_rng(rng)
        yield writer
    finally:
        if writer is not None:
            writer.close()


def _write_tensorboard_step(writer, metric: dict[str, float | int], elapsed: float,
                            optimizer) -> None:
    if writer is None:
        return
    step = int(metric["step"])
    for component in ("total", "teacher_spectral", "teacher_waveform", "reference_spectral"):
        writer.add_scalar(f"loss/{component}", metric[component], step)
    writer.add_scalar("gradient/global_norm", metric["gradient_norm"], step)
    writer.add_scalar("timing/step_seconds", elapsed, step)
    for name, member in optimizer.optimizers.items():
        for index, group in enumerate(member.param_groups):
            suffix = "" if len(member.param_groups) == 1 else f"/group_{index}"
            writer.add_scalar(f"learning_rate/{name}{suffix}", float(group["lr"]), step)
    writer.flush()


def train_fixed_batch(model: nn.Module, batch: TrainingBatch, *, steps: int,
                      config: TrainingConfig = TrainingConfig(),
                      loss_config: WarmupLossConfig = WarmupLossConfig(),
                      model_config: Mapping[str, Any] | None = None,
                      checkpoint_path: str | Path | None = None,
                      resume_from: str | Path | None = None,
                      log_dir: str | Path | None = None,
                      run_name: str = "preflight") -> TrainingResult:
    """Run exactly ``steps`` additional updates, optionally resuming a run.

    Resume requires the same architecture, batch contents, loss, optimizer
    settings, device type and recorded runtime settings. It restores RNG after
    loading model/optimizer. Exact replay is qualified on CPU in one fixed
    environment; CUDA replay and cross-hardware equivalence remain unqualified.
    The caller controls model initialization; ``seed`` seeds update-time RNG.
    Checkpoints are written atomically after successful updates, at run end.
    Optional TensorBoard events contain measured training quantities only.
    Resumed logs purge stale later steps and reject unrelated run metadata.
    """
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be an explicit positive integer")
    if batch.latents.ndim != 3 or not batch.latents.is_floating_point():
        raise ValueError("latents must be a floating-point [batch, channels, frames] tensor")
    if not torch.isfinite(batch.latents).all():
        raise ValueError("latents must be finite")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("The student must have trainable parameters")
    device = parameters[0].device
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("This preflight supports CPU and CUDA, with CPU-only CLI execution")
    if any(parameter.device != device for parameter in parameters) or batch.latents.device != device:
        raise ValueError("Student parameters and latents must use the same device")
    specification = _model_spec(model, model_config)
    batch_digest = _batch_digest(batch)
    runtime = _runtime_spec()
    optimizer = build_optimizer_bundle(model, optimizer=config.optimizer, lr=config.learning_rate,
                                        betas=config.betas, weight_decay=config.weight_decay,
                                        muon_momentum=config.muon_momentum,
                                        muon_ns_steps=config.muon_ns_steps,
                                        muon_adjust_lr_fn=config.muon_adjust_lr_fn)
    criterion = WarmupReconstructionLoss(loss_config)
    step = 0
    if resume_from is not None:
        payload = torch.load(resume_from, map_location="cpu", weights_only=True)
        expected = {"format_version": 2, "model_spec": specification,
                    "training_config": asdict(config), "loss_config": asdict(loss_config),
                    "batch_sha256": batch_digest, "device_type": device.type,
                    "runtime": runtime,
                    "optimizer_group_fingerprint": optimizer.group_fingerprint}
        for name, value in expected.items():
            if payload.get(name) != value:
                raise ValueError(f"Checkpoint {name} does not match this preflight")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        _restore_rng(payload["rng"])
        step = int(payload["step"])
    else:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    metrics: list[dict[str, float | int]] = []
    elapsed_times: list[float] = []
    destination = Path(checkpoint_path) if checkpoint_path is not None else None
    logging_manifest = {"qualification": "fixed-batch reconstruction preflight",
                        "model_spec": specification, "training_config": asdict(config),
                        "loss_config": asdict(loss_config), "batch_sha256": batch_digest,
                        "optimizer_group_fingerprint": optimizer.group_fingerprint,
                        "device_type": device.type, "runtime": runtime}
    with _tensorboard_logger(log_dir, run_name, step=step, resuming=resume_from is not None,
                              manifest=logging_manifest) as writer:
        model.train()
        for _ in range(steps):
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch.latents.detach())
            components = criterion(prediction, batch.teacher_audio, batch.reference_audio_16k)
            loss = components["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite reconstruction loss; no checkpoint was written")
            loss.backward()
            gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
            if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
                raise FloatingPointError("Missing or nonfinite student gradients")
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip_norm,
                                                       error_if_nonfinite=True)
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in parameters):
                raise FloatingPointError("Nonfinite student parameters; no checkpoint was written")
            step += 1
            metric = {"step": step, **{name: float(value.detach())
                                       for name, value in components.items()},
                       "gradient_norm": float(grad_norm)}
            # Scalar reads and finite checks have synchronized this step's
            # device work. This excludes event-file I/O and checkpoint saving.
            elapsed = time.perf_counter() - started
            metrics.append(metric)
            elapsed_times.append(elapsed)
            _write_tensorboard_step(writer, metric, elapsed, optimizer)

        if destination is not None:
            _atomic_save({"format_version": 2, "model_spec": specification,
                          "training_config": asdict(config), "loss_config": asdict(loss_config),
                          "batch_sha256": batch_digest, "device_type": device.type,
                          "runtime": runtime,
                          "optimizer_group_fingerprint": optimizer.group_fingerprint,
                          "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "rng": _rng_state(), "step": step}, destination)
    return TrainingResult(step=step, metrics=metrics, checkpoint=destination,
                           step_elapsed_seconds=elapsed_times,
                           log_path=Path(log_dir) / run_name if log_dir is not None else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12, help="Additional synthetic updates, default 12")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--optimizer", choices=("adamw", "muon_adamw"), default="adamw")
    parser.add_argument("--log-dir", type=Path, help="Optional TensorBoard log root")
    parser.add_argument("--run-name", default="preflight-synthetic",
                        help="TensorBoard run name; synthetic runs must start with 'preflight'")
    arguments = parser.parse_args()
    if not 1 <= arguments.steps <= 500:
        parser.error("Synthetic preflight requires 1 to 500 updates")
    if arguments.threads < 1:
        parser.error("--threads must be positive")
    if not arguments.run_name.startswith("preflight"):
        parser.error("Synthetic run names must start with 'preflight'")
    torch.set_num_threads(arguments.threads)
    torch.manual_seed(arguments.seed)
    # The reduced width is solely for a fast wiring/gradient/resume check.
    # Production architecture capacity is set independently in pilot.json.
    from .model import StudentConfig, StudentDecoder

    model_config = StudentConfig(hidden_channels=16, expansion_channels=32, head_channels=32)
    model = StudentDecoder(model_config).cpu()
    generator = torch.Generator(device="cpu").manual_seed(arguments.seed + 1)
    latents = torch.randn(1, 64, 3, generator=generator)
    time = torch.arange(3 * 1920) / 48000.0
    target = (0.15 * torch.sin(2 * torch.pi * 220 * time)
              + 0.04 * torch.sin(2 * torch.pi * 3300 * time)).reshape(1, 1, -1)
    result = train_fixed_batch(model, TrainingBatch(latents, target), steps=arguments.steps,
                               config=TrainingConfig(learning_rate=1e-3, seed=arguments.seed,
                                                     optimizer=arguments.optimizer),
                               loss_config=WarmupLossConfig(teacher_fft_sizes=(128, 256, 512),
                                                           reference_fft_sizes_16k=(64, 128)),
                               checkpoint_path=arguments.checkpoint, resume_from=arguments.resume,
                               log_dir=arguments.log_dir, run_name=arguments.run_name)
    print(json.dumps({"qualification": "synthetic fixed-batch preflight only",
                      "device": "cpu", "threads": arguments.threads, "step": result.step,
                      "optimizer": arguments.optimizer,
                      "initial_loss": result.metrics[0]["total"],
                      "last_update_loss": result.metrics[-1]["total"],
                      "mean_step_seconds": sum(result.step_elapsed_seconds) / len(result.step_elapsed_seconds),
                      "tensorboard_log_path": str(result.log_path) if result.log_path is not None else None,
                      "checkpoint": str(result.checkpoint)}, indent=2))


if __name__ == "__main__":
    main()
