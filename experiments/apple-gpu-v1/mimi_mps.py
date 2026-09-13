"""Experiment-only original Pocket Mimi on MPS; no conversion or RVQ change.

Inputs are device-resident FP32 [1,32,L] raw continuous encoder latents.
The authenticated checkpoint stores BF16 coefficients. Loading them into the
original FP32 modules widens them exactly, matching the prior CPU adapter.
One latent frame emits 1920 samples at 24 kHz (80 ms, not 40 ms).
full(z) uses fresh upstream state; stream(max_frames=128) retains independent
upstream ModelState. The original linear KV cache is capacity-limited, not the
249-position cache rewrite in the ONNX exporter. Its offset.item() synchronizes
with the host and remains part of every measured call. No model hooks are used.

All model calls are serialized; distinct streams have distinct state dictionaries.
The caller must validate finite inputs/outputs and synchronize/copy results when
measuring completed GPU work. Methods do not add per-packet finite-check syncs.
Dependencies: torch==2.14.0, numpy, safetensors, pydantic>=2, PyYAML,
requests, huggingface_hub, typing_extensions. No downloads occur here.
"""
from __future__ import annotations

import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import threading
import types

# Set before importing Torch, and reject a conflicting caller configuration.
for _name in ("PYTORCH_MPS_FAST_MATH", "PYTORCH_ENABLE_MPS_FALLBACK"):
    if os.environ.get(_name, "0") != "0":
        raise RuntimeError(f"{_name} must be 0 for the FP32 GPU comparison")
    os.environ[_name] = "0"

import torch

SOURCE_REVISION = "f544a39b07db56fadeed575d57a33e1eda17c022"
CHECKPOINT_REVISION = "39592ff23c9ef80098bb74895d104c26275fe2c9"
CHECKPOINT_SHA256 = "473f47d99560bd50eb8b4509d3cacfe7f316ab20bdca86505403a2e6a936a6e9"
DEFAULT_SOURCE = Path("/Users/venky/tech/pockettts/research/pocket-tts-source")
DEFAULT_CHECKPOINT = Path("/Users/venky/tech/pockettts/benchmarks/.cache/pocket/model.safetensors")
SOURCE_SHA256 = {
    "models/mimi.py": "93996a84525e41fb204a53f1edd72abdf5925f6bb8f92304e87d29c7a431054b",
    "modules/conv.py": "9fdeb902936ac3e7d0e0abaf2bffdceb7b1e1632804c36b4120e6bad0840faa1",
    "modules/attention.py": "4b1763c527e6bb87c4152c7f17f5aa30a88164f1ed11136110539a53d0dbd296",
    "modules/transformer.py": "c9543c9abb4a6f76173942696d6add83cfc4539000140bb9d6e1238fa0cc96ea",
    "modules/rope.py": "0cf8cac4b5ef0bbb6e025f04d77af627528a8adad69a0792a17a193693ddcbce",
    "modules/stateful_module.py": "2b111910ea55bc39b5810ad95167404d3face9e586d94e7917b8f6cedb3b0836",
    "modules/resample.py": "463ee64b0fadc88d8755316aed9f9037864c5d3d294c0223c150ac6cc8173771",
    "modules/seanet.py": "17cfd00617f8aae225b699c3d66d3bfd463fa717090edb97498936bbdf5080d8",
    "modules/dummy_quantizer.py": "68a5a957e69df72a56aa82727a112ec7c913456fa961733e35adceb1f290133f",
    "modules/layer_scale.py": "72178214ab7fe7a112cfa4797b5bd6c10a377f65e928ee043d422495c17e1ff2",
    "utils/config.py": "fac56a1531ea93223261f0d773a898280fd1ef0281bea43e72f7d0d0ea14f200",
    "utils/utils.py": "f884e980c4b922bf144388d73c8c5faf398709a558c786068f331a572ff8f6de",
    "config/english.yaml": "463605b15840d1c62666d1669b0fc70495c45ef4e8fc8ebda422f7e0fbc0f546",
}


def _sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _upstream_imports(source):
    """Import untouched decoder modules without the unrelated TTS entry point."""
    package = source / "pocket_tts"
    for relative, expected in SOURCE_SHA256.items():
        if _sha256(package / relative) != expected:
            raise ValueError(f"Pinned Pocket source differs: {relative}")
    # Do not let a previously imported different package bypass source checks.
    for name, module in tuple(sys.modules.items()):
        if name == "pocket_tts" or name.startswith("pocket_tts."):
            origin = getattr(module, "__file__", None)
            if origin and not Path(origin).resolve().is_relative_to(package):
                raise RuntimeError(f"A different Pocket package is already loaded: {name}")
    # __init__ imports the full TTS/tokenizer stack and changes CPU thread count.
    # Namespace mounting avoids only that entry point; model source is unchanged.
    for suffix in ("", ".models", ".modules", ".utils"):
        name = "pocket_tts" + suffix
        path = package.joinpath(*suffix.strip(".").split(".")) if suffix else package
        if name in sys.modules:
            paths = [Path(p).resolve() for p in getattr(sys.modules[name], "__path__", [])]
            if paths != [path.resolve()]:
                raise RuntimeError(f"Unexpected Pocket package search path: {name}")
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = module
    from pocket_tts.models.mimi import build_mimi
    from pocket_tts.modules.stateful_module import StatefulModule, init_states, increment_steps
    from pocket_tts.utils.config import MimiConfig
    return build_mimi, StatefulModule, init_states, increment_steps, MimiConfig


class MimiMPSAdapter:
    sample_rate = sample_rate_out = 24000
    hop_samples = hop_length = 1920
    latent_fps = latent_rate = 12.5
    channels = latent_dim = 32

    def __init__(self, source_root=DEFAULT_SOURCE, checkpoint=DEFAULT_CHECKPOINT, max_frames=128):
        if torch.__version__.split("+")[0] != "2.14.0":
            raise RuntimeError("This experiment requires torch==2.14.0")
        if torch.get_default_dtype() != torch.float32:
            raise RuntimeError("Upstream state factories require the FP32 default dtype")
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable in this process")
        if type(max_frames) is not int or max_frames < 1:
            raise ValueError("max_frames must be a positive integer")
        source, checkpoint = Path(source_root).resolve(), Path(checkpoint).resolve()
        if _sha256(checkpoint) != CHECKPOINT_SHA256:
            raise ValueError("Pocket English checkpoint checksum mismatch")
        build, stateful, self._init_states, self._increment, config_type = _upstream_imports(source)
        import yaml
        from safetensors import safe_open
        configuration = yaml.safe_load((source / "pocket_tts/config/english.yaml").read_text())
        config = config_type(**configuration["mimi"])
        if (config.sample_rate, config.frame_rate, config.quantizer.dimension,
                config.quantizer.output_dimension, config.transformer.context) != (24000, 12.5, 32, 512, 250):
            raise ValueError("Unexpected Pocket continuous Mimi configuration")
        with torch.device("cpu"):
            model = build(config).float().eval()
        with safe_open(checkpoint, framework="pt", device="cpu") as handle:
            weights = {key.removeprefix("mimi."): handle.get_tensor(key)
                       for key in handle.keys() if key.startswith("mimi.")}
        if len(weights) != 87 or any(value.dtype != torch.bfloat16 for value in weights.values()):
            raise ValueError("Expected the original 87 BF16 Mimi checkpoint tensors")
        # Same as the historical CPU adapter: load into FP32 modules, an exact
        # widening of stored BF16 values, without quantization or normalization.
        model.load_state_dict(weights, strict=True)
        for name, module in model.named_modules():
            if isinstance(module, stateful):
                module._module_absolute_name = name
        self.model = model.requires_grad_(False).to("mps")
        self.max_frames = max_frames
        self._lock = threading.RLock()
        self._closed = False
        self.metadata = {
            "name": "Pocket English Mimi continuous codec", "backend": "pytorch_mps",
            "torch": torch.__version__, "dtype": "float32", "sample_rate": 24000,
            "latent_dim": 32, "latent_fps": 12.5, "hop_samples": 1920,
            "checkpoint_revision": CHECKPOINT_REVISION, "checkpoint_sha256": CHECKPOINT_SHA256,
            "source_revision": SOURCE_REVISION, "source_sha256": dict(SOURCE_SHA256),
            "mimi_config_sha256": hashlib.sha256(json.dumps(config.model_dump(), sort_keys=True).encode()).hexdigest(),
            "stream_capacity_frames": max_frames, "transformer_positions_per_frame": 16,
            "cache": "Original upstream linear KV cache, capacity max_frames*16+64",
            "context": 250, "upstream_offset_item_sync_preserved": True,
            "stored_checkpoint_dtype": "bfloat16", "execution_dtype": "float32",
            "checkpoint_widening": "BF16 to FP32, exact and identical to the historical CPU adapter",
            "new_quantization": False, "rvq": False, "package_entrypoint_bypassed": True,
            "numerical_qualification": "pending; caller must compare frozen CPU ONNX references",
        }

    def _check(self, z):
        if self._closed:
            raise RuntimeError("Decoder is closed")
        if not isinstance(z, torch.Tensor) or z.dtype != torch.float32 or z.device.type != "mps":
            raise ValueError("Expected an MPS float32 tensor")
        if z.ndim != 3 or tuple(z.shape[:2]) != (1, 32):
            raise ValueError("Expected raw continuous latent shape [1,32,L]")
        return z.shape[-1]

    def _decode(self, z, state):
        y = self.model.decode_from_latent(z.transpose(1, 2), state)
        if y.dtype != torch.float32 or y.device.type != "mps" or tuple(y.shape) != (1, 1, z.shape[-1] * 1920):
            raise RuntimeError("Unexpected Mimi output dtype, device or sample count")
        return y

    @torch.inference_mode()
    def full(self, z):
        with self._lock, torch.autocast("mps", enabled=False):
            frames = self._check(z)
            if not frames:
                return z.new_empty((1, 1, 0))
            # Matches the original CPU MimiAdapter.decode, including fresh state.
            state = self._init_states(self.model, 1, frames * 16 + 64)
            return self._decode(z, state)

    def stream(self, max_frames=None):
        return MimiMPSStream(self, self.max_frames if max_frames is None else max_frames)

    streaming_decode = stream

    def close(self):
        with self._lock:
            self._closed = True


class MimiMPSStream:
    def __init__(self, decoder, max_frames):
        if type(max_frames) is not int or max_frames < 1:
            raise ValueError("max_frames must be a positive integer")
        self.decoder, self.max_frames = decoder, max_frames
        self._lock = threading.RLock()
        self.closed = False
        self.reset()

    @torch.inference_mode()
    def reset(self):
        with self._lock, self.decoder._lock:
            if self.closed or self.decoder._closed:
                raise RuntimeError("Stream is closed")
            self.states = self.decoder._init_states(self.decoder.model, 1, self.max_frames * 16 + 64)
            self.frames_decoded = 0
            self.failed = False

    @torch.inference_mode()
    def decode_chunk(self, z):
        with self._lock, self.decoder._lock, torch.autocast("mps", enabled=False):
            if self.closed or self.failed:
                raise RuntimeError("Stream is closed or requires reset after failure")
            frames = self.decoder._check(z)
            if self.frames_decoded + frames > self.max_frames:
                raise ValueError("Upstream KV capacity exceeded; choose max_frames before starting")
            if not frames:
                return z.new_empty((1, 1, 0))
            try:
                result = self.decoder._decode(z, self.states)
                self.decoder._increment(self.decoder.model, self.states, frames * 16)
                self.frames_decoded += frames
                return result
            except Exception:
                # Upstream mutates state in place; never continue it after failure.
                self.failed = True
                raise

    def flush(self):
        with self._lock, self.decoder._lock:
            if self.closed or self.failed or self.decoder._closed:
                raise RuntimeError("Stream is closed or failed")
            return torch.empty((1, 1, 0), device="mps", dtype=torch.float32)

    @property
    def state_bytes(self):
        return sum(t.numel() * t.element_size() for state in self.states.values() for t in state.values())

    def close(self):
        with self._lock, self.decoder._lock:
            self.closed = True
            self.states = {}

    def __enter__(self):
        if self.closed:
            raise RuntimeError("Stream is closed")
        return self

    def __exit__(self, *args):
        self.close()
