"""GPU facade contracts using NumPy-backed fake tensors; no GPU or Torch runs."""
from contextlib import contextmanager, nullcontext
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import sys
import threading

import numpy as np
import pytest

from fast_audiovae import automatic, gpu


class Tensor:
    def __init__(self, data, device="mps"):
        self.data = np.array(data, copy=True)
        self.device = SimpleNamespace(type=device)
        self.dtype = self.data.dtype
        self.shape = self.data.shape

    def to(self, device, dtype=None, non_blocking=False):
        assert non_blocking is False
        return Tensor(self.data.astype(dtype) if dtype is not None else self.data, device)

    def detach(self):
        return self

    def numpy(self):
        assert self.device.type == "cpu"
        return self.data

    def all(self):
        return Tensor(self.data.all(), self.device.type)

    def reshape(self, *shape):
        return Tensor(self.data.reshape(*shape), self.device.type)

    def item(self):
        return self.data.item()


class Torch:
    __version__ = "2.14.0"
    float32 = np.float32
    Tensor = Tensor

    def __init__(self):
        self.syncs = 0
        self.uploads = 0
        self.backends = SimpleNamespace(mps=SimpleNamespace(is_built=lambda: True, is_available=lambda: True))
        self.mps = SimpleNamespace(synchronize=self.sync)

    def sync(self):
        self.syncs += 1

    def from_numpy(self, value):
        assert value.flags.c_contiguous
        self.uploads += 1
        return Tensor(value, "cpu")

    def inference_mode(self):
        return nullcontext()

    def autocast(self, *, device_type, enabled):
        assert device_type == "mps" and enabled is False
        return nullcontext()

    def isfinite(self, value):
        return Tensor(np.isfinite(value.data), value.device.type)

    def stack(self, values):
        return Tensor(np.stack([value.data for value in values]))

    def cat(self, values):
        return Tensor(np.concatenate([value.data for value in values]))


class Model:
    state_shapes = {"history": (1, 1, 1)}
    sample_rate, hop_samples, latent_channels = 48000, 1920, 64

    def __init__(self):
        self.calls = []
        self.bad = None
        self.active = self.maximum_active = 0
        self.entered = self.release = None

    def decode(self, z, states):
        assert z.device.type == "mps" and z.dtype == np.float32
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            self.calls.append((z.data.copy(), dict(states)))
            if self.entered is not None and len(self.calls) == 1:
                self.entered.set()
                assert self.release.wait(3)
            prior = states["history"].data.item() if states else 0
            values = np.cumsum(z.data.mean(axis=1, keepdims=True), axis=-1) + prior
            audio = Tensor(np.repeat(values, 1920, axis=-1).astype("f"))
            next_states = {"history": Tensor(values[..., -1:].astype("f"))}
            if self.bad == "raise":
                states.clear()
                raise RuntimeError("GPU operation failed")
            if self.bad == "missing":
                next_states = {}
            if self.bad == "cpu_state":
                next_states["history"].device.type = "cpu"
            if self.bad == "dtype_state":
                next_states["history"] = Tensor(np.zeros((1, 1, 1), np.float64))
            if self.bad == "shape_state":
                next_states["history"] = Tensor(np.zeros((1, 1, 2), np.float32))
            if self.bad == "nan_state":
                next_states["history"].data.fill(np.nan)
            if self.bad == "cpu_audio":
                audio.device.type = "cpu"
            if self.bad == "shape_audio":
                audio = Tensor(np.zeros((1, 1, 1), np.float32))
            if self.bad == "nan_audio":
                audio.data.fill(np.nan)
            return audio, next_states
        finally:
            self.active -= 1


@pytest.fixture
def fake(monkeypatch):
    for name in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH",
                 "TORCHINDUCTOR_USE_FAST_MATH", "PYTORCH_MPS_PREFER_METAL"):
        monkeypatch.delenv(name, raising=False)
    torch = Torch()
    monkeypatch.setattr(gpu.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(gpu.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(gpu, "_torch", lambda: torch)
    model = Model()
    return torch, model


def decoder(fake, mode="streaming"):
    torch, model = fake
    return gpu.GPUDecoder(model, torch, {"device": "gpu", "backend": "mps", "precision": "FP32"}, mode)


def test_cpu_default_and_explicit_cpu_never_enter_gpu(monkeypatch):
    def unexpected(**kw):
        raise AssertionError("GPU route called")
    monkeypatch.setattr(gpu, "load_gpu", unexpected)
    plans = []
    def setup(**kw):
        plans.append(kw)
        return dict(model_dir="verified", threads=1, recipe="cpu", reason="cpu", cache_hit=True, cache_key="key")
    monkeypatch.setattr(automatic, "setup", setup)
    import fast_audiovae.runtime as runtime
    existing = SimpleNamespace(streaming_decode=lambda: "cpu stream")
    monkeypatch.setattr(runtime, "load_streaming_decoder", lambda *a, **kw: (existing, {"selected": "native"}))
    for options in ({}, {"device": "cpu"}):
        assert automatic.load(**options).stream() == "cpu stream"
    assert plans[0] == plans[1]


@pytest.mark.parametrize("entry", ["load", "setup"])
def test_explicit_gpu_dispatches_before_cpu_setup(monkeypatch, entry):
    called = []
    monkeypatch.setattr(gpu, entry + "_gpu", lambda **kw: called.append(kw) or "GPU")
    monkeypatch.setattr(automatic, "_build", lambda *a: pytest.fail("CPU build requested"))
    assert getattr(automatic, entry)(device="gpu", source="source.onnx", offline=True) == "GPU"
    assert called[0]["mode"] == "streaming" and called[0]["threads"] == 1
    with pytest.raises(ValueError, match="device"):
        getattr(automatic, entry)(device="auto")


@pytest.mark.parametrize("threads", [0, 2, 4, True, None])
def test_gpu_rejects_cpu_worker_counts_before_loading(fake, threads):
    with pytest.raises(ValueError, match="threads=1"):
        gpu.setup_gpu(threads=threads)


def test_setup_and_load_only_verify_original_source(fake, monkeypatch, tmp_path):
    torch, model = fake
    verified = []
    monkeypatch.setattr(gpu, "verify_model", lambda path: verified.append(path))
    monkeypatch.setattr(gpu, "fetch_model", lambda *a: pytest.fail("Unexpected download"))
    monkeypatch.setattr(automatic, "_build", lambda *a: pytest.fail("CPU build requested"))
    from_onnx = []
    class Factory:
        @staticmethod
        def from_onnx(path, device):
            from_onnx.append((path, device)); return model
    monkeypatch.setitem(sys.modules, "fast_audiovae.mps_decoder", SimpleNamespace(MPSModel=Factory))
    wrapped = []
    monkeypatch.setitem(sys.modules, "fast_audiovae.mps_compiled", SimpleNamespace(
        CompiledModel=lambda value: wrapped.append(value) or value))
    source = tmp_path / "source.onnx"
    info = automatic.setup(device="gpu", mode="both", source=source, cache_dir=tmp_path, offline=True)
    assert set(info) == {"streaming", "batch"}
    assert info["streaming"]["cache_key"] != info["batch"]["cache_key"]
    assert not from_onnx and not model.calls
    loaded = automatic.load(device="gpu", source=source, cache_dir=tmp_path, offline=True)
    assert from_onnx == [(str(source), "mps")]
    assert verified == [source, source]
    assert loaded.info["sample_rate"] == 48000 and loaded.info["samples_per_latent"] == 1920
    assert loaded.info["state_device"] == "mps" and not loaded.info["fallback"]
    assert wrapped == [model]
    assert [z.shape[-1] for z, _ in model.calls] == [1, 2]
    assert all(not state for _, state in model.calls)
    batch = automatic.load(device="gpu", mode="batch", source=source, cache_dir=tmp_path, offline=True)
    assert batch.mode == "batch" and len(wrapped) == 1 and len(model.calls) == 2
    assert batch.info["compiled_packet_ms"] == []
    model.bad = "raise"
    with pytest.raises(RuntimeError, match="GPU operation failed"):
        automatic.load(device="gpu", source=source, cache_dir=tmp_path, offline=True)


def test_cpu_and_gpu_model_downloads_share_setup_lock(fake, monkeypatch, tmp_path):
    import fast_audiovae.build_resources as resources
    import fast_audiovae.native_payload as payload
    import fast_audiovae.platforms as platforms

    class DownloadReached(Exception):
        pass

    active = []
    downloads = []

    @contextmanager
    def capture_lock(path):
        active.append(path)
        try:
            yield
        finally:
            active.pop()

    def download(directory):
        assert active == [tmp_path / ".setup.lock"]
        downloads.append(directory)
        # Stop before any model bytes, recipe preparation or inference.
        raise DownloadReached

    monkeypatch.setattr(automatic, "_lock", capture_lock)
    monkeypatch.setattr(platforms, "detect_cpu", lambda **kw: {"platform": "Darwin/arm64"})
    monkeypatch.setattr(payload, "inspect_payload", lambda: None)
    monkeypatch.setattr(resources, "resource_fingerprint", lambda: "mock sources")
    monkeypatch.setattr(resources, "materialize_resources", lambda path: "mock sources")
    monkeypatch.setattr(automatic, "sha256", lambda path: "mock source hash")
    monkeypatch.setattr(automatic, "fetch_model", download)
    monkeypatch.setattr(gpu, "fetch_model", download)
    monkeypatch.setattr(automatic, "_build", lambda *a: pytest.fail("Unexpected recipe build"))
    for device in ("cpu", "gpu"):
        with pytest.raises(DownloadReached):
            automatic.setup(device=device, cache_dir=tmp_path)
        assert not active
    assert downloads == [tmp_path / "models", tmp_path / "models"]
    assert not fake[1].calls


def test_stream_state_counts_empty_reset_flush_and_close(fake):
    torch, model = fake
    d = decoder(fake)
    a, b = d.stream(), d.stream()
    empty = np.empty((1, 64, 0), np.float32)
    assert a.decode_chunk(empty).shape == a.flush().shape == (1, 1, 0)
    assert torch.uploads == torch.syncs == 0
    x = np.ones((1, 64, 2), np.float32)
    y = a.decode_chunk(x)
    assert y.shape == (1, 1, 3840) and y.dtype == np.float32 and y.flags.owndata
    assert a.frames_decoded == 2 and b.frames_decoded == 0
    assert a._history["history"].device.type == "mps"
    assert a.decode_chunk(x)[0, 0, 0] == 3
    np.testing.assert_array_equal(b.decode_chunk(x), y)
    assert torch.syncs == 3
    a.reset(); assert a.frames_decoded == 0 and not a._history
    np.testing.assert_array_equal(a.decode_chunk(x), y)
    a.close(); a.close(); assert not a._history
    for fn in (lambda: a.decode_chunk(empty), a.flush, a.reset, a.__enter__):
        with pytest.raises(RuntimeError, match="closed"):
            fn()
    with d.stream() as stream:
        stream.decode_chunk(x)
    with pytest.raises(RuntimeError, match="closed"):
        stream.flush()


def test_batch_is_independent_and_noncontiguous_input_is_owned(fake):
    torch, model = fake
    d = decoder(fake, "batch")
    base = np.ones((1, 64, 4), np.float32)
    x = base[..., ::2]
    assert not x.flags.c_contiguous
    first = d.decode(x); second = d.decode(x)
    np.testing.assert_array_equal(first, second)
    assert all(not history for _, history in model.calls)
    assert np.all(base == 1) and torch.syncs == 2
    assert d.decode(x[..., :0]).shape == (1, 1, 0) and len(model.calls) == 2
    with pytest.raises(RuntimeError, match="streaming"):
        d.stream()
    with pytest.raises(RuntimeError, match="stream"):
        decoder(fake).decode(x)


def test_prepare_does_not_advance_existing_streams_or_publish_failed_state(fake):
    _, model = fake
    d = decoder(fake)
    stream = d.stream()
    x = np.ones((1, 64, 2), np.float32)
    stream.decode_chunk(x)
    state = stream._history
    before = state["history"].data.copy()
    assert d.prepare() is d
    assert stream._history is state and stream.frames_decoded == 2
    np.testing.assert_array_equal(state["history"].data, before)
    assert [z.shape[-1] for z, _ in model.calls[-2:]] == [1, 2]
    assert all(not h for _, h in model.calls[-2:])
    model.bad = "nan_state"
    with pytest.raises(RuntimeError, match="nonfinite"):
        d.prepare()
    assert stream._history is state and stream.frames_decoded == 2
    model.bad = None
    assert stream.decode_chunk(x)[0, 0, 0] == 3
    calls = len(model.calls)
    assert decoder(fake, "batch").prepare().mode == "batch"
    assert len(model.calls) == calls


@pytest.mark.parametrize("bad", [np.zeros((1, 64, 1), np.float64), np.zeros((2, 64, 1), np.float32),
                                  np.zeros((1, 63, 1), np.float32), np.full((1, 64, 1), np.nan, np.float32), []])
def test_bad_input_never_reaches_model(fake, bad):
    stream = decoder(fake).stream()
    with pytest.raises(ValueError):
        stream.decode_chunk(bad)
    assert stream.frames_decoded == 0 and not fake[1].calls


@pytest.mark.parametrize("bad", ["raise", "missing", "cpu_state", "dtype_state", "shape_state",
                                  "nan_state", "cpu_audio", "shape_audio", "nan_audio"])
def test_failure_does_not_publish_state_and_recovery_works(fake, bad):
    torch, model = fake
    stream = decoder(fake).stream(); x = np.ones((1, 64, 1), np.float32)
    stream.decode_chunk(x)
    history = stream._history; original = history["history"].data.copy()
    model.bad = bad
    with pytest.raises(RuntimeError):
        stream.decode_chunk(x)
    assert stream._history is history and stream.frames_decoded == 1
    np.testing.assert_array_equal(history["history"].data, original)
    model.bad = None
    assert stream.decode_chunk(x)[0, 0, 0] == 2 and stream.frames_decoded == 2


def test_shared_decoder_serializes_calls_and_keeps_histories_independent(fake):
    torch, model = fake
    model.entered, model.release = threading.Event(), threading.Event()
    a, b = decoder(fake).stream(), None
    b = a._decoder.stream()
    attempting = threading.Event()
    x = np.ones((1, 64, 1), np.float32)
    def second():
        attempting.set(); return b.decode_chunk(x)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(a.decode_chunk, x)
        assert model.entered.wait(3)
        other = pool.submit(second)
        assert attempting.wait(3)
        model.release.set()
        np.testing.assert_array_equal(first.result(3), other.result(3))
    assert model.maximum_active == 1 and a.frames_decoded == b.frames_decoded == 1
    assert a._history["history"] is not b._history["history"] and torch.syncs == 2


@pytest.mark.parametrize("setting", ["PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH",
                                      "TORCHINDUCTOR_USE_FAST_MATH", "PYTORCH_MPS_PREFER_METAL"])
def test_pre_enabled_backend_setting_rejected_before_import(monkeypatch, setting):
    monkeypatch.setattr(gpu.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(gpu.platform, "machine", lambda: "arm64")
    monkeypatch.setenv(setting, "1")
    monkeypatch.setattr(gpu.importlib, "import_module", lambda *a: pytest.fail("Torch imported with unsafe flag"))
    with pytest.raises(RuntimeError, match=setting):
        gpu._torch()


def test_zero_metal_preference_is_also_rejected_before_import(monkeypatch):
    monkeypatch.setattr(gpu.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(gpu.platform, "machine", lambda: "arm64")
    monkeypatch.setenv("PYTORCH_MPS_PREFER_METAL", "0")
    monkeypatch.setattr(gpu.importlib, "import_module", lambda *a: pytest.fail("Unqualified Metal route imported"))
    with pytest.raises(RuntimeError, match="PYTORCH_MPS_PREFER_METAL"):
        gpu._torch()


@pytest.mark.parametrize("failure", ["platform", "missing", "unavailable", "version"])
def test_explicit_gpu_has_no_cpu_fallback(monkeypatch, failure):
    for name in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH",
                 "TORCHINDUCTOR_USE_FAST_MATH", "PYTORCH_MPS_PREFER_METAL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(gpu.platform, "system", lambda: "Linux" if failure == "platform" else "Darwin")
    monkeypatch.setattr(gpu.platform, "machine", lambda: "arm64")
    torch = Torch()
    if failure == "unavailable":
        torch.backends.mps.is_available = lambda: False
    if failure == "version":
        torch.__version__ = "2.8.0"
    def imported(name):
        assert name == "torch"
        if failure == "missing":
            raise ImportError("not installed")
        return torch
    monkeypatch.setattr(gpu.importlib, "import_module", imported)
    with pytest.raises(RuntimeError):
        gpu._torch()
