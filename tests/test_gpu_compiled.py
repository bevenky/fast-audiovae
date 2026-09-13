"""Bounded compilation contracts with tiny CPU tensors and a fake compiler."""
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")

from fast_audiovae import mps_compiled, mps_decoder


class Stem(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.ones(1))

    def decode(self, x, history):
        return x, x[..., -1:].clone()


class Mean(torch.nn.Module):
    def forward(self, x):
        return x.mean(1, keepdim=True)


class Final(torch.nn.Module):
    def decode(self, x, history):
        value = x.cumsum(-1) + history
        return value.repeat_interleave(2, -1), value[..., -1:].clone()


class TinyModel(torch.nn.Module):
    sample_rate, hop_samples, latent_channels = 48000, 2, 64
    state_shapes = {"stem.history": (1, 64, 1), "final.history": (1, 1, 1)}
    state_bytes = 260
    _checked_input = mps_decoder.MPSModel._checked_input
    _checked_states = mps_decoder.MPSModel._checked_states

    def __init__(self):
        super().__init__()
        self.stem, self.pointwise = Stem(), Mean()
        self.stages = torch.nn.ModuleList()
        self.final_snake, self.final_conv = torch.nn.Identity(), Final()
        self.eager_lengths = []

    @property
    def device(self):
        return self.stem.weight.device

    def decode(self, z, state):
        self._checked_input(z)
        history = self._checked_states(z, state)
        self.eager_lengths.append(z.shape[-1])
        if not z.shape[-1]:
            return z.new_empty(1, 1, 0), {k: v.clone() for k, v in history.items()}
        cumulative = z.mean(1, keepdim=True).cumsum(-1) + history["final.history"]
        return torch.tanh(cumulative.repeat_interleave(2, -1)), {
            "stem.history": z[..., -1:].clone(), "final.history": cumulative[..., -1:].clone()}


def install_compiler(monkeypatch, transform=None):
    built, invoked = [], []

    def compile_step(step, **options):
        built.append(options)

        def call(z, *state):
            invoked.append(z.shape[-1])
            result = step(z, *state)
            return transform(result) if transform else result

        return call

    monkeypatch.setattr(mps_compiled.torch, "compile", compile_step)
    return built, invoked


def test_fixed_cache_reuses_graphs_and_other_lengths_stay_on_model(monkeypatch):
    built, invoked = install_compiler(monkeypatch)
    model = TinyModel()
    wrapped = mps_compiled.CompiledModel(model)
    state = {}
    for length in (1, 2, 1, 3, 0, 13, 2):
        x = torch.arange(64*length, dtype=torch.float32).reshape(1, 64, length) / 256
        saved = {k: v.clone() for k, v in state.items()}
        reference, reference_state = TinyModel().decode(x, state)
        audio, updated = wrapped.decode(x, state)
        assert torch.equal(audio, reference)
        assert set(updated) == set(model.state_shapes)
        assert all(torch.equal(updated[k], v) for k, v in reference_state.items())
        assert all(torch.equal(state[k], v) for k, v in saved.items())
        assert all(v.is_contiguous() and v.device.type == "cpu" for v in updated.values())
        state = updated
    assert set(wrapped.compiled) == {1, 2}
    assert built == [dict(backend="inductor", fullgraph=True, dynamic=False)] * 2
    assert invoked == [1, 2, 1, 2]
    assert model.eager_lengths == [3, 0, 13]


def test_compiled_result_histories_are_materialized_contiguously(monkeypatch):
    returned = []

    def strided(result):
        values = [result[0]]
        for history in result[1:]:
            backing = torch.empty(1, history.shape[1]*2, 1)
            view = backing[:, ::2, :]
            view.copy_(history)
            values.append(view)
        returned.extend(values)
        return tuple(values)

    install_compiler(monkeypatch, strided)
    wrapped = mps_compiled.CompiledModel(TinyModel())
    _, state = wrapped.decode(torch.ones(1, 64, 2), {})
    assert not returned[1].is_contiguous()
    assert state["stem.history"].is_contiguous()
    assert torch.equal(state["stem.history"], returned[1])
    assert state["stem.history"].data_ptr() != returned[1].data_ptr()


@pytest.mark.parametrize("failure", ["compile", "execute", "inventory"])
def test_failed_compile_or_call_is_not_cached_or_retried_eagerly(monkeypatch, failure):
    model = TinyModel()
    wrapped = mps_compiled.CompiledModel(model)
    x = torch.ones(1, 64, 1)
    state = model._checked_states(x, {})
    before = {k: v.clone() for k, v in state.items()}

    def broken(step, **options):
        if failure == "compile":
            raise RuntimeError("compile rejected")

        def run(*args):
            if failure == "execute":
                raise RuntimeError("execution rejected")
            return step(*args)[:-1]

        return run

    monkeypatch.setattr(mps_compiled.torch, "compile", broken)
    with pytest.raises((RuntimeError, ValueError)):
        wrapped.decode(x, state)
    assert not wrapped.compiled and not model.eager_lengths
    assert all(torch.equal(state[k], v) for k, v in before.items())
    install_compiler(monkeypatch)
    wrapped.decode(x, state)
    assert set(wrapped.compiled) == {1}


@pytest.mark.parametrize("invalid", ["input_type", "input_channels", "input_strides", "partial_state", "state_dtype"])
def test_validation_precedes_compiler_dispatch(monkeypatch, invalid):
    compiler = Mock(side_effect=AssertionError("Compiler reached invalid input"))
    monkeypatch.setattr(mps_compiled.torch, "compile", compiler)
    model = TinyModel()
    wrapped = mps_compiled.CompiledModel(model)
    x = torch.ones(1, 64, 1)
    state = {}
    if invalid == "input_type":
        x = x.double()
    elif invalid == "input_channels":
        x = x[:, :63]
    elif invalid == "input_strides":
        x = torch.ones(1, 64, 4)[..., ::2]
    elif invalid == "partial_state":
        state = {"stem.history": torch.zeros(1, 64, 1)}
    elif invalid == "state_dtype":
        state = model._checked_states(x, {})
        state["stem.history"] = state["stem.history"].double()
    with pytest.raises(ValueError):
        wrapped.decode(x, state)
    compiler.assert_not_called()


def test_cpu_package_import_does_not_import_gpu_or_torch():
    code = """
import importlib.abc
import sys
class RejectGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.') or fullname in (
                'fast_audiovae.gpu', 'fast_audiovae.mps_decoder', 'fast_audiovae.mps_compiled'):
            raise AssertionError('CPU import reached optional GPU module: ' + fullname)
sys.meta_path.insert(0, RejectGPU())
import fast_audiovae
import fast_audiovae.automatic
assert 'torch' not in sys.modules
"""
    source = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": source},
                               capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr
