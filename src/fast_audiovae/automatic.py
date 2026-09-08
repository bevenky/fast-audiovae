"""Prepare a verified CPU recipe once and reuse it for later loads."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import warnings

from .assets import MODEL_FILES, MODEL_REVISION, fetch_model, sha256, verify_model

RECIPE_VERSION = 1


def cache_directory(directory=None):
    if directory is not None:
        return Path(directory).expanduser().resolve()
    override = os.environ.get("FAST_AUDIOVAE_CACHE")
    if override:
        return Path(override).expanduser().resolve()
    if platform.system() == "Darwin":
        base = Path.home() / "Library/Caches"
    elif platform.system() == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "fast-audiovae"


@contextmanager
def _lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            handle.write(b"0")
            handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _inventory(bundle):
    answer = {}
    for path in sorted(bundle.rglob("*")):
        if path.is_symlink() and not path.resolve().is_relative_to(bundle.resolve()):
            raise RuntimeError("Prepared bundle has a dependency outside its directory")
        if path.is_file():
            answer[str(path.relative_to(bundle))] = sha256(path)
    if "bundle.json" not in answer:
        raise RuntimeError("Recipe did not produce a model bundle")
    return answer


def _verify_receipt(receipt, root, key):
    if receipt.get("key") != key or receipt.get("status") != "ready":
        raise RuntimeError("Automatic cache receipt does not match this recipe")
    bundle = (root / receipt["bundle"]).resolve()
    if not bundle.is_relative_to(root.resolve()):
        raise RuntimeError("Automatic cache path is outside the cache")
    if _inventory(bundle) != receipt.get("files"):
        raise RuntimeError("Cached decoder files changed; use a fresh cache directory")
    return bundle


def _prerequisites(plan):
    if plan["recipe"].startswith("apple"):
        return [name for name in ("xcrun",) if not shutil.which(name)]
    if plan["recipe"].startswith(("intel", "amd")):
        return [name for name in ("cc", "c++", "cmake", "make") if not shutil.which(name)]
    return []


def _payload_supported(payload, cpu):
    if payload:
        tag = payload["manifest"].get("wheel_platform", "")
        if cpu.get("platform") == "Darwin/arm64":
            if not tag.startswith("macosx_") or not tag.endswith("_arm64"):
                return False
        elif cpu.get("platform") == "Linux/x86_64":
            if not tag.startswith(("linux_", "manylinux_")) or not tag.endswith("_x86_64"):
                return False
            minimum = payload["manifest"].get("minimum_glibc")
            if minimum:
                family, release = platform.libc_ver()
                try:
                    current = tuple(int(x) for x in release.split("."))
                    required = tuple(int(x) for x in minimum.split("."))
                except ValueError:
                    return False
                if family != "glibc" or current < required:
                    return False
        else:
            return False
    minimum = payload["manifest"].get("minimum_macos") if payload else None
    if minimum and cpu.get("system") == "Darwin":
        current = tuple(int(x) for x in platform.mac_ver()[0].split(".")[:2])
        required = tuple(int(x) for x in minimum.split(".")[:2])
        return current >= required
    return True


def _build(work, source, cpu, plan):
    if plan["recipe"].startswith("apple"):
        from .recipes.apple import build_recipe
    elif plan["recipe"].startswith(("intel", "amd")):
        from .recipes.x86 import build_recipe
    else:
        from .recipes.portable import build_recipe
    return Path(build_recipe(work, source, cpu, plan["mode"], plan["threads"])).resolve()


def setup(*, mode="streaming", threads=1, cache_dir=None, source=None, offline=False,
          prefer_custom=True, build_native=False):
    """Download weights and prepare the CPU recipe, or verify and reuse its cache.

    No model inference occurs here. Platform wheels supply native libraries.
    Missing compatible libraries select portable ONNX with an explicit reason.
    Maintainers can explicitly request source compilation with build_native.
    Build failures and corrupt artifacts are never hidden.
    """
    from .platforms import detect_cpu, select_recipe
    from .build_resources import materialize_resources, resource_fingerprint
    from .native_payload import inspect_payload, materialize_payload, probe_payload
    import onnxruntime as ort

    if mode not in ("batch", "streaming", "both"):
        raise ValueError("mode must be batch, streaming or both")
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    if any(type(value) is not bool for value in (offline, prefer_custom, build_native)):
        raise ValueError("offline, prefer_custom and build_native must be booleans")
    cpu = detect_cpu(allow_compile=build_native)
    payload = inspect_payload()
    root = cache_directory(cache_dir)
    results = {}
    modes = ("batch", "streaming") if mode == "both" else (mode,)
    with _lock(root / ".setup.lock"):
        # The installed resources are read-only; builders write only to this cache.
        # A new package revision receives a new workspace, preserving old cache receipts.
        package_root = Path(__file__).resolve().parent
        implementation = {str(p.relative_to(package_root)): sha256(p)
                          for p in sorted(package_root.rglob("*.py"))
                          if "_build_resources" not in p.parts}
        generation = hashlib.sha256(json.dumps({"resources": resource_fingerprint(),
                  "implementation": implementation, "native": payload["identity"] if payload else None,
                  "platform": cpu.get("platform"), "vendor": cpu.get("vendor")},
                  sort_keys=True).encode()).hexdigest()
        work = root / "build" / generation
        resource_identity = materialize_resources(work)
        source_path = Path(source).expanduser().resolve() if source else root / "models/audio_vae_decoder.onnx"
        if source or offline:
            try:
                verify_model(source_path)
            except FileNotFoundError as error:
                raise RuntimeError("Pinned model files are not cached; run setup once without --offline") from error
        else:
            fetch_model(source_path.parent)
        for selected_mode in modes:
            plan = select_recipe(cpu, mode=selected_mode, threads=threads)
            if not prefer_custom or ort.__version__ != "1.29.0":
                plan = {**plan, "recipe": "portable", "fallback": True, "reason": (
                    "Portable ONNX explicitly requested" if not prefer_custom else
                    "Native recipes require ONNX Runtime 1.29.0")}
            if payload and not _payload_supported(payload, cpu):
                plan = {**plan, "recipe": "portable", "fallback": True, "reason":
                        "The native wheel does not support this OS and process architecture; using portable ONNX"}
            prebuilt = materialize_payload(payload, work / "prebuilt", plan["recipe"]) if payload else None
            if prebuilt:
                supported, error = probe_payload(prebuilt)
                if not supported:
                    plan = {**plan, "recipe": "portable", "fallback": True, "reason":
                            "Native runtime dependencies are unavailable on this OS; using portable ONNX",
                            "native_load_error": error}
                    prebuilt = None
            if plan["recipe"] != "portable" and not prebuilt and not build_native:
                plan = {**plan, "recipe": "portable", "fallback": True, "reason":
                        "This installation lacks a compatible native platform wheel; using portable ONNX"}
            missing = [] if prebuilt else _prerequisites(plan)
            if missing:
                plan = {**plan, "recipe": "portable", "fallback": True, "reason":
                        "Native build tools unavailable: " + ", ".join(missing) + "; using portable ONNX"}
            identity = {"recipe_version": RECIPE_VERSION, "resources": resource_identity, "generation": generation,
                        "model_revision": MODEL_REVISION, "model_files": MODEL_FILES,
                        "runtime": ort.__version__, "platform": cpu.get("platform"),
                        "recipe": plan["recipe"], "mode": selected_mode, "threads": plan["threads"]}
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            receipt_path = root / "receipts" / (key + ".json")
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                bundle = _verify_receipt(receipt, root, key)
                hit = True
            else:
                build_cpu = {**cpu, "offline": offline, "recipe": plan["recipe"], "prebuilt": prebuilt}
                bundle = _build(work, source_path, build_cpu, plan)
                if not bundle.is_relative_to(root):
                    raise RuntimeError("Recipe output must remain inside the setup cache")
                receipt = {"status": "ready", "key": key, "identity": identity,
                           "bundle": str(bundle.relative_to(root)), "files": _inventory(bundle)}
                receipt_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = receipt_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(receipt, indent=2) + "\n")
                temporary.replace(receipt_path)
                hit = False
            results[selected_mode] = {**plan, "model_dir": str(bundle), "cache_hit": hit,
                                      "cache_key": key, "cpu": cpu, "cpu_only": True}
    return results if mode == "both" else results[mode]


class AudioVAEDecoder:
    """Small batch/streaming facade over the existing validated runtime."""

    def __init__(self, decoder, info, mode):
        self._decoder, self.info, self.mode = decoder, info, mode

    def decode(self, latents):
        """Decode one complete latent sequence in batch mode."""
        import numpy as np
        if self.mode != "batch":
            raise RuntimeError("Use decoder.stream() for stateful streaming")
        if (not isinstance(latents, np.ndarray) or latents.dtype != np.float32
                or latents.ndim != 3 or latents.shape[:2] != (1, 64)):
            raise ValueError("Latents must be a float32 NumPy array with shape [1,64,L]")
        if not np.isfinite(latents).all():
            raise ValueError("Latents must contain only finite values")
        if latents.shape[2] == 0:
            return np.empty((1, 1, 0), dtype=np.float32)
        return self._decoder.run(None, {self._decoder.get_inputs()[0].name: latents})[0]

    def stream(self):
        """Create independent causal history for a new utterance."""
        if self.mode != "streaming":
            raise RuntimeError("Load mode='streaming' to create an audio stream")
        return self._decoder.streaming_decode()


def load(*, mode="streaming", threads=1, cache_dir=None, source=None, offline=False,
         prefer_custom=True):
    """Select, prepare and load the best validated recipe for this CPU and mode."""
    if mode not in ("batch", "streaming"):
        raise ValueError("mode must be batch or streaming")
    result = setup(mode=mode, threads=threads, cache_dir=cache_dir, source=source,
                   offline=offline, prefer_custom=prefer_custom)
    from .runtime import load_decoder, load_streaming_decoder
    loader = load_streaming_decoder if mode == "streaming" else load_decoder
    decoder, info = loader(result["model_dir"], threads=result["threads"], prefer_custom=prefer_custom)
    automatic = {"recipe": result["recipe"], "selection_reason": result["reason"],
                 "mode": mode, "cache_hit": result["cache_hit"], "cache_key": result["cache_key"]}
    if info["selected"] == "portable_onnx":
        automatic["recipe"] = "portable"
        warnings.warn("Using portable ONNX: " + result["reason"] + "; " + info["reason"],
                      RuntimeWarning, stacklevel=2)
    return AudioVAEDecoder(decoder, {**info, **automatic}, mode)
