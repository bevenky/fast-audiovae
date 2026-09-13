"""Explicit Apple MPS inference. Importing this module does not import Torch."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import threading

import numpy as np

from .assets import MODEL_FILES, MODEL_REVISION, fetch_model, verify_model


def _policy():
    # Both settings are opt-in in PyTorch. Reject a pre-enabled setting instead
    # of pretending a change after Torch initialization can undo it.
    for name in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH"):
        if os.environ.get(name) not in (None, "0"):
            raise RuntimeError(name + " must be 0 before importing Torch for GPU decoding")
    for name in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH"):
        os.environ[name] = "0"


def _torch():
    if platform.system() != "Darwin" or platform.machine().lower() not in ("arm64", "aarch64"):
        raise RuntimeError("device='gpu' requires Apple Silicon macOS with MPS; CPU fallback is disabled")
    _policy()
    try:
        torch = importlib.import_module("torch")
    except ImportError as error:
        raise RuntimeError("GPU decoding requires the optional dependency: pip install 'fast-audiovae[gpu]'") from error
    if torch.__version__.split("+")[0].split(".")[:2] != ["2", "14"]:
        raise RuntimeError("GPU decoding requires PyTorch2.14.x from fast-audiovae[gpu]")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; an explicit GPU request never falls back to CPU")
    return torch


def _options(mode, threads, offline, prefer_custom, build_native=False):
    if mode not in ("batch", "streaming", "both"):
        raise ValueError("mode must be batch, streaming or both")
    if type(threads) is not int or threads != 1:
        raise ValueError("GPU decoding requires threads=1; CPU worker counts do not control MPS")
    if any(type(value) is not bool for value in (offline, prefer_custom, build_native)):
        raise ValueError("offline, prefer_custom and build_native must be booleans")
    if build_native:
        raise ValueError("build_native is a CPU option; the GPU route builds no CPU kernels")


def setup_gpu(*, mode="streaming", threads=1, cache_dir=None, source=None, offline=False,
              prefer_custom=True, build_native=False):
    """Verify/cache the original ONNX tensors without preparing a CPU recipe.

    No decoder is instantiated and no model inference occurs. The cache hit
    refers to verified source files. CPU custom-kernel preference is unused.
    """
    _options(mode, threads, offline, prefer_custom, build_native)
    torch = _torch()
    from .automatic import cache_directory, _lock
    root = cache_directory(cache_dir)
    path = Path(source).expanduser().resolve() if source else root / "models/audio_vae_decoder.onnx"
    # CPU and GPU setup share the pinned model files and download temporaries.
    with _lock(root / ".setup.lock"):
        cached = path.is_file() and (path.parent / "audio_vae_decoder.onnx.data").is_file()
        if source or offline:
            try:
                verify_model(path)
            except FileNotFoundError as error:
                raise RuntimeError("Pinned model files are not cached; set source or run GPU setup without offline") from error
        else:
            fetch_model(path.parent)
            verify_model(path)
    common = dict(device="gpu", backend="mps", selected="torch_mps", recipe="apple_mps",
                  precision="FP32", mode=mode, threads=1, cpu_only=False, fallback=False,
                  reason="Explicit Apple GPU request", model_path=str(path), model_dir=str(path.parent),
                  source_sha256=MODEL_FILES["audio_vae_decoder.onnx"], source_files=dict(MODEL_FILES),
                  model_revision=MODEL_REVISION, torch=torch.__version__, cache_hit=cached,
                  cache_kind="verified_source", sample_rate=48000, latent_rate=25,
                  latent_channels=64, samples_per_latent=1920,
                  mps_cpu_fallback=False, mps_fast_math=False)
    results = {}
    for selected_mode in (("batch", "streaming") if mode == "both" else (mode,)):
        identity = dict(device="gpu", mode=selected_mode, torch=torch.__version__, model_files=MODEL_FILES)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        results[selected_mode] = {**common, "mode": selected_mode, "cache_key": key}
    return results if mode == "both" else results[mode]


def load_gpu(*, mode="streaming", threads=1, cache_dir=None, source=None, offline=False,
             prefer_custom=True):
    if mode not in ("batch", "streaming"):
        raise ValueError("mode must be batch or streaming")
    info = setup_gpu(mode=mode, threads=threads, cache_dir=cache_dir, source=source,
                     offline=offline, prefer_custom=prefer_custom)
    torch = _torch()
    from .mps_decoder import MPSModel
    model = MPSModel.from_onnx(info["model_path"], device="mps")
    return GPUDecoder(model, torch, info, mode)


def _latents(value):
    if (not isinstance(value, np.ndarray) or value.dtype != np.float32
            or value.ndim != 3 or value.shape[:2] != (1, 64)):
        raise ValueError("Latents must be a float32 NumPy array with shape [1,64,L]")
    if not np.isfinite(value).all():
        raise ValueError("Latents must contain only finite values")
    # Own upload storage, including when callers supplied noncontiguous arrays.
    return np.array(value, copy=True, order="C") if value.shape[2] else value


class GPUDecoder:
    """Immutable MPS weights shared by independent, serialized GPU streams."""

    def __init__(self, model, torch, info, mode):
        if mode not in ("batch", "streaming"):
            raise ValueError("mode must be batch or streaming")
        if (model.sample_rate, model.hop_samples, model.latent_channels) != (48000, 1920, 64):
            raise RuntimeError("Unexpected GPU decoder sample/latent contract")
        shapes = model.state_shapes
        if (not isinstance(shapes, dict) or not shapes or any(
                not isinstance(name, str) or not name or not isinstance(shape, (tuple, list))
                or len(shape) != 3 or shape[0] != 1 or any(type(n) is not int or n <= 0 for n in shape)
                for name, shape in shapes.items())):
            raise RuntimeError("Invalid GPU history specification")
        self._state_shapes = {name: tuple(shape) for name, shape in shapes.items()}
        self._model, self._torch, self.mode = model, torch, mode
        self._lock = threading.Lock()
        self.state_bytes = sum(int(np.prod(shape)) * 4 for shape in self._state_shapes.values())
        self.info = {**info, "mode": mode, "state_bytes_per_stream": self.state_bytes,
                     "state_device": "mps", "output_device": "cpu", "output_ready": True}

    def _run(self, latents, history):
        torch = self._torch
        with self._lock:
            _policy()
            with torch.inference_mode(), torch.autocast(device_type="mps", enabled=False):
                z = torch.from_numpy(latents).to(device="mps", dtype=torch.float32, non_blocking=False)
                audio, next_history = self._model.decode(z, dict(history))
                if (not isinstance(audio, torch.Tensor) or audio.dtype != torch.float32
                        or audio.device.type != "mps" or tuple(audio.shape) != (1, 1, latents.shape[2] * 1920)):
                    raise RuntimeError("GPU model returned invalid audio")
                if not isinstance(next_history, dict) or set(next_history) != set(self._state_shapes):
                    raise RuntimeError("GPU model returned incomplete history")
                for name, shape in self._state_shapes.items():
                    value = next_history[name]
                    if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                            or value.device.type != "mps" or tuple(value.shape) != shape):
                        raise RuntimeError("GPU model returned invalid history: " + name)
                # One device-side validation buffer avoids one reduction and
                # scalar synchronization per history tensor. This is validation
                # overhead, separate from the decoder's arithmetic.
                checked = torch.cat([value.reshape(-1) for value in (audio, *next_history.values())])
                if not torch.isfinite(checked).all().item():
                    raise RuntimeError("GPU model returned nonfinite audio or history")
                # Explicitly wait for computation and copy audio to owned CPU
                # storage. Histories remain on MPS; no state advances on failure.
                ready = audio.detach().to(device="cpu", non_blocking=False)
                torch.mps.synchronize()
                result = ready.numpy().copy()
                return result, dict(next_history)

    def decode(self, latents):
        """Decode an independent sequence in batch mode, starting with zero state."""
        if self.mode != "batch":
            raise RuntimeError("Use decoder.stream() for stateful streaming")
        latents = _latents(latents)
        if not latents.shape[2]:
            return np.empty((1, 1, 0), dtype=np.float32)
        return self._run(latents, {})[0]

    def stream(self):
        if self.mode != "streaming":
            raise RuntimeError("Load mode='streaming' to create an audio stream")
        return GPUStream(self)


class GPUStream:
    """Per-utterance MPS history; all returned NumPy audio is ready to use."""

    def __init__(self, decoder):
        self._decoder = decoder
        self._lock = threading.Lock()
        self._history = {}
        self._closed = False
        self.frames_decoded = 0

    def _check_open(self):
        if self._closed:
            raise RuntimeError("Decoder stream is closed; create a new stream")

    def decode_chunk(self, latents):
        latents = _latents(latents)
        with self._lock:
            self._check_open()
            if not latents.shape[2]:
                return np.empty((1, 1, 0), dtype=np.float32)
            audio, history = self._decoder._run(latents, self._history)
            self._history = history
            self.frames_decoded += latents.shape[2]
            return audio

    def reset(self):
        with self._lock:
            self._check_open()
            self._history = {}
            self.frames_decoded = 0

    def flush(self):
        with self._lock:
            self._check_open()
            return np.empty((1, 1, 0), dtype=np.float32)

    def close(self):
        with self._lock:
            self._history.clear()
            self._closed = True

    def __enter__(self):
        with self._lock:
            self._check_open()
        return self

    def __exit__(self, *exc):
        self.close()
