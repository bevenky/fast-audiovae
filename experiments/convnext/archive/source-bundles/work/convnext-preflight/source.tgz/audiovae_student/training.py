"""A bounded, resumable fixed-batch reconstruction preflight.

This is not the corpus/GAN trainer. Its CLI uses synthetic audio on CPU and
does not establish speech quality, trained-model RTF, or a training budget.
"""

import argparse
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from .losses import WarmupLossConfig, WarmupReconstructionLoss


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    grad_clip_norm: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if len(self.betas) != 2 or any(not 0 <= b < 1 for b in self.betas):
            raise ValueError("betas must contain two values in [0, 1)")
        if not math.isfinite(self.grad_clip_norm) or self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be finite and positive")


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


def train_fixed_batch(model: nn.Module, batch: TrainingBatch, *, steps: int,
                      config: TrainingConfig = TrainingConfig(),
                      loss_config: WarmupLossConfig = WarmupLossConfig(),
                      model_config: Mapping[str, Any] | None = None,
                      checkpoint_path: str | Path | None = None,
                      resume_from: str | Path | None = None) -> TrainingResult:
    """Run exactly ``steps`` additional updates, optionally resuming a run.

    Resume requires the same architecture, batch contents, loss, optimizer
    settings, device type and recorded runtime settings. It restores RNG after
    loading model/optimizer. Exact replay is qualified on CPU in one fixed
    environment; CUDA replay and cross-hardware equivalence remain unqualified.
    The caller controls model initialization; ``seed`` seeds update-time RNG.
    Checkpoints are written atomically after successful updates, at run end.
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
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, betas=config.betas,
                                  weight_decay=config.weight_decay)
    criterion = WarmupReconstructionLoss(loss_config)
    step = 0
    if resume_from is not None:
        payload = torch.load(resume_from, map_location="cpu", weights_only=True)
        expected = {"format_version": 1, "model_spec": specification,
                    "training_config": asdict(config), "loss_config": asdict(loss_config),
                    "batch_sha256": batch_digest, "device_type": device.type,
                    "runtime": runtime}
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
    model.train()
    for _ in range(steps):
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
        metrics.append({"step": step, **{name: float(value.detach())
                                        for name, value in components.items()},
                        "gradient_norm": float(grad_norm)})

    destination = Path(checkpoint_path) if checkpoint_path is not None else None
    if destination is not None:
        _atomic_save({"format_version": 1, "model_spec": specification,
                      "training_config": asdict(config), "loss_config": asdict(loss_config),
                      "batch_sha256": batch_digest, "device_type": device.type,
                      "runtime": runtime,
                      "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "rng": _rng_state(), "step": step}, destination)
    return TrainingResult(step=step, metrics=metrics, checkpoint=destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12, help="Additional synthetic updates, default 12")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    arguments = parser.parse_args()
    if not 1 <= arguments.steps <= 500:
        parser.error("Synthetic preflight requires 1 to 500 updates")
    if arguments.threads < 1:
        parser.error("--threads must be positive")
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
                               config=TrainingConfig(learning_rate=1e-3, seed=arguments.seed),
                               loss_config=WarmupLossConfig(teacher_fft_sizes=(128, 256, 512),
                                                           reference_fft_sizes_16k=(64, 128)),
                               checkpoint_path=arguments.checkpoint, resume_from=arguments.resume)
    print(json.dumps({"qualification": "synthetic fixed-batch preflight only",
                      "device": "cpu", "threads": arguments.threads, "step": result.step,
                      "initial_loss": result.metrics[0]["total"],
                      "last_update_loss": result.metrics[-1]["total"],
                      "checkpoint": str(result.checkpoint)}, indent=2))


if __name__ == "__main__":
    main()
