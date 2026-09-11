"""Explicit, pinned loading of the original AudioVAE2 distillation teacher.

There are no downloads or implicit resampling. Callers supply trusted local
assets and mono 16 kHz FP32 audio. Targets are generated over whole utterances;
resetting this encoder on arbitrary training crops changes its causal context.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import io
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import torch
from torch import Tensor, nn


MODEL_REVISION = "32279effe8c19989596f05d353d1447f51d9e915"
SOURCE_REVISION = "f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69"
SOURCE_SHA256 = "2efdff1708d8ec1471624aae6f232d0f933de26b788b6901c434246847e2d3a8"
CHECKPOINT_SHA256 = "94b5d51e107e0507d4acc976cfdadb64edd6fd06d1f751dadbf2fd1594274bf1"


def _verified_bytes(path: Path, expected: str, label: str) -> bytes:
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return payload


def _load_source(path: Path, payload: bytes) -> ModuleType:
    """Execute the bytes just verified rather than reopening unverified code."""
    name = f"_audiovae2_teacher_{SOURCE_SHA256}"
    module = ModuleType(name)
    module.__file__ = str(path)
    previous = sys.modules.get(name)
    sys.modules[name] = module  # TorchScript resolves source through this module.
    try:
        exec(compile(payload, str(path), "exec"), module.__dict__)
    except BaseException:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module


def _check_cuda_precision(device: torch.device) -> None:
    # Do not silently alter process-global backend policy in a library call.
    if device.type == "cuda" and (
        torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
    ):
        raise ValueError(
            "FP32 teacher requires both CUDA matmul and cuDNN allow_tf32=False; "
            "set these explicitly before loading or running the teacher."
        )


class FrozenAudioVAE2(nn.Module):
    """Frozen FP32 posterior-mean teacher with explicit sample accounting.

    ``encode`` accepts [B, 1, N] at 16 kHz and returns ceil(N / 640)
    unscaled posterior-mean frames. ``decode`` returns 1,920 samples per
    latent frame at 48 kHz. Only ``reconstruct`` knows the original length
    and trims right padding to exactly 3N samples. No fitted alignment,
    loudness normalization, clipping, or latent sampling is performed.

    Returned tensors use no-grad rather than inference mode, so they can be
    inputs and targets in student autograd operations without an extra clone.
    Use ``from_files`` to load the pinned teacher. This wrapper intentionally
    does not provide chunk-by-chunk encoding: generate targets continuously.
    """

    sample_rate_in = 16_000
    sample_rate_out = 48_000
    latent_channels = 64
    latent_rate = 25
    encoder_hop = 640
    decoder_hop = 1920
    sr_cond = 48_000

    def __init__(self, model: nn.Module, provenance: dict[str, Any]):
        super().__init__()
        self.model = model.float().requires_grad_(False)
        self._provenance = deepcopy(provenance)
        self.train(False)

    @classmethod
    def from_files(
        cls,
        source_path: str | Path,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
    ) -> "FrozenAudioVAE2":
        """Verify both complete assets before executing source or loading weights.

        CPU is the default. CUDA is supported for subsequent teacher-target
        preparation with explicit TF32-off policy; MPS is not a teacher path.
        The original upstream weight normalization and strict state keys stay
        intact. No optimized inference checkpoint can enter through this API.
        """
        device = torch.device(device)
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("Teacher device must be cpu or cuda; MPS is not validated")
        if device.type == "cpu" and device.index is not None:
            device = torch.device("cpu")
        _check_cuda_precision(device)
        source = Path(source_path).expanduser().resolve(strict=True)
        checkpoint = Path(checkpoint_path).expanduser().resolve(strict=True)
        source_bytes = _verified_bytes(source, SOURCE_SHA256, "AudioVAE2 source")
        checkpoint_bytes = _verified_bytes(checkpoint, CHECKPOINT_SHA256, "AudioVAE2 checkpoint")
        module = _load_source(source, source_bytes)
        config = module.AudioVAEConfig()
        config_values = config.model_dump(mode="json")
        expected_contract = {
            "sample_rate": cls.sample_rate_in,
            "out_sample_rate": cls.sample_rate_out,
            "latent_dim": cls.latent_channels,
            "encoder_rates": [2, 5, 8, 8],
            "decoder_rates": [8, 6, 5, 2, 2, 2],
            "use_noise_block": False,
        }
        for name, expected in expected_contract.items():
            if config_values.get(name) != expected:
                raise ValueError(f"Pinned AudioVAE2 configuration violates {name}={expected!r}")
        state = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
        del checkpoint_bytes
        model = module.AudioVAE(config)
        model.load_state_dict(state.get("state_dict", state), strict=True)
        del state
        model = model.to(device=device, dtype=torch.float32)
        # Resolve an unspecified CUDA index to the actual parameter device.
        actual_device = next(model.parameters()).device
        provenance = {
            "model_repo": "openbmb/VoxCPM2",
            "model_revision": MODEL_REVISION,
            "source_repo": "OpenBMB/VoxCPM",
            "source_revision": SOURCE_REVISION,
            "source_sha256": SOURCE_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "config": config_values,
            "posterior": "raw_mu",
            "sample_rate_in": cls.sample_rate_in,
            "sample_rate_out": cls.sample_rate_out,
            "latent_channels": cls.latent_channels,
            "latent_rate": cls.latent_rate,
            "encoder_hop": cls.encoder_hop,
            "decoder_hop": cls.decoder_hop,
            "sr_cond": cls.sr_cond,
            "input_padding": "right_zero_to_multiple_of_640",
            "reconstruct_trim": "right_trim_to_3_times_original_input_samples",
            "weight_norm": "original_retained",
            "dtype": "float32",
            "device": str(actual_device),
            "autocast": False,
            "tf32": False,
            "torch_version": str(torch.__version__),
        }
        return cls(model, provenance)

    @property
    def provenance(self) -> dict[str, Any]:
        """Serializable identity without machine-local paths or credentials."""
        result = deepcopy(self._provenance)
        result["device"] = str(self.device)
        return result

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def train(self, mode: bool = True) -> "FrozenAudioVAE2":
        # A parent training module's .train() must never unfreeze this teacher.
        super().train(False)
        return self

    def _validate(self, tensor: Tensor, channels: int, label: str) -> None:
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{label} must be a torch.Tensor")
        if tensor.ndim != 3 or tensor.shape[1] != channels or tensor.shape[0] < 1:
            raise ValueError(f"{label} must have shape [B, {channels}, T] with B >= 1")
        if tensor.dtype != torch.float32:
            raise TypeError(f"{label} must be float32, got {tensor.dtype}")
        if tensor.device != self.device:
            raise ValueError(f"{label} is on {tensor.device}; teacher is on {self.device}")
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Teacher must remain on cpu or cuda")
        if any(m.training for m in self.model.modules()):
            raise RuntimeError("Teacher must remain in eval mode")
        if any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("Teacher parameters must remain frozen")
        if any(
            x.is_floating_point() and x.dtype != torch.float32
            for x in (*self.model.parameters(), *self.model.buffers())
        ):
            raise RuntimeError("Teacher parameters and floating buffers must remain float32")
        _check_cuda_precision(self.device)
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{label} contains nonfinite values")

    def _validate_output(self, tensor: Tensor, shape: tuple[int, ...], label: str) -> Tensor:
        if not isinstance(tensor, Tensor) or tuple(tensor.shape) != shape:
            raise RuntimeError(f"Teacher {label} violated expected shape {shape}")
        if tensor.dtype != torch.float32 or tensor.device != self.device:
            raise RuntimeError(f"Teacher {label} violated FP32/device contract")
        if not torch.isfinite(tensor).all().item():
            raise RuntimeError(f"Teacher {label} contains nonfinite values")
        return tensor

    @torch.no_grad()
    def encode(self, audio: Tensor) -> Tensor:
        """Encode 16 kHz audio; no normalization or resampling is implicit."""
        self._validate(audio, 1, "audio")
        frames = (audio.shape[-1] + self.encoder_hop - 1) // self.encoder_hop
        if frames == 0:
            return audio.new_empty((audio.shape[0], self.latent_channels, 0))
        with torch.autocast(device_type=self.device.type, enabled=False):
            latent = self.model.encode(audio, self.sample_rate_in)
        return self._validate_output(latent, (audio.shape[0], self.latent_channels, frames), "mu")

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        """Decode raw mu with explicit 48 kHz bandwidth conditioning."""
        self._validate(latents, self.latent_channels, "latents")
        length = latents.shape[-1] * self.decoder_hop
        if length == 0:
            return latents.new_empty((latents.shape[0], 1, 0))
        cond = torch.full((latents.shape[0],), self.sr_cond, dtype=torch.int32, device=self.device)
        with torch.autocast(device_type=self.device.type, enabled=False):
            audio = self.model.decode(latents, sr_cond=cond)
        return self._validate_output(audio, (latents.shape[0], 1, length), "waveform")

    @torch.no_grad()
    def reconstruct(self, audio: Tensor) -> Tensor:
        """Return exactly 3N samples; raw decode alone cannot recover N."""
        latent = self.encode(audio)
        return self.decode(latent)[..., : audio.shape[-1] * 3]

    def forward(self, audio: Tensor) -> Tensor:
        return self.reconstruct(audio)
