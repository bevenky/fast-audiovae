"""Cold execution, exact quiet parity and no-update warmup preservation."""
import copy
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn.utils import weight_norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'convnext'))
import quiet_constraint_warmup as warm

q = warm.q


@pytest.fixture(autouse=True)
def cpu_policy():
    rng, threads = q.screen.rng_state(), torch.get_num_threads()
    torch.manual_seed(722)
    torch.set_num_threads(1)
    yield
    q.screen.restore_rng(rng)
    torch.set_num_threads(threads)


def entry(length=1920, marker=0., *, context=0, valid=None):
    target = torch.zeros(1, 1, length)
    crop = {'source_id': 'fixture', 'context_frames': context, 'start_frame': context,
            'context_start_frame': 0, 'valid_scored_samples': valid or length - context * 1920}
    result = q.window_layout(crop, target)
    result['group_input'] = torch.full_like(target, marker)
    return result


class ExecutionWave(nn.Module):
    def __init__(self, *, cold=False, persistent=False, cold_marker=None, mutate=False):
        super().__init__()
        self.level = nn.Parameter(torch.tensor(2e-5))
        self.suffix_scale = nn.Parameter(torch.tensor(1.), requires_grad=False)
        self.register_buffer('protected', torch.tensor(0.))
        self.cold, self.persistent, self.cold_marker, self.mutate = cold, persistent, cold_marker, mutate
        self.grad_calls = self.no_grad_calls = self.backward_calls = 0
        self.marker_calls = {}
        self.suffix_inputs = []

    def group_from_input(self, x):
        grad = torch.is_grad_enabled()
        self.grad_calls += int(grad)
        self.no_grad_calls += int(not grad)
        marker = float(x[..., 0])
        calls = self.marker_calls.get(marker, 0)
        if grad:
            self.marker_calls[marker] = calls + 1
        perturb = ((grad and self.cold and self.grad_calls == 1)
                   or (grad and marker == self.cold_marker and calls == 0)
                   or (self.persistent and not grad))
        # Advancing disposable RNG is allowed; the guard must restore all three.
        torch.rand(()); random.random(); np.random.random()
        if self.mutate:
            self.protected.add_(1)
        return x * 0 + self.level + (1e-7 if perturb else 0.)

    def suffix_from_group(self, x):
        self.suffix_inputs.append((torch.is_grad_enabled(), x.requires_grad))
        out = x * self.suffix_scale
        if out.requires_grad:
            out.register_hook(self._backward)
        return out

    def _backward(self, grad):
        self.backward_calls += 1
        return grad


def test_cold_first_grad_is_reported_then_exactly_qualified_without_backward():
    model, cache = ExecutionWave(cold=True), set()
    with torch.no_grad():
        report = warm.warm_quiet_entries(model, None, [entry()], cache)
    assert report['passed'] and report['preserved'] and len(cache) == 1
    cold = report['cold_comparisons'][0]
    assert not cold['first_vs_third']['waveform_exact']
    assert not cold['first_vs_third']['constraints_exact']
    assert cold['second_vs_third']['passed']
    assert all(v['passed'] for k, v in report['entries'][0].items() if k != 'entry_index')
    assert (model.grad_calls, model.no_grad_calls, model.backward_calls) == (5, 1, 0)
    assert all(requires for enabled, requires in model.suffix_inputs if enabled)
    assert model.level.grad is None


def test_second_input_of_same_shape_is_checked_before_shape_is_cached():
    model, cache = ExecutionWave(cold_marker=1.), set()
    with pytest.raises(RuntimeError, match='entry 1'):
        warm.warm_quiet_entries(model, None, [entry(), entry(marker=1.)], cache)
    assert not cache and model.backward_calls == 0


def test_persistent_mode_difference_fails_and_preserves_existing_cache_and_rng():
    model, cache = ExecutionWave(persistent=True), {('previous-qualified-key',)}
    rng = q.screen.rng_state()
    with pytest.raises(RuntimeError, match='parity failed'):
        warm.warm_quiet_entries(model, None, [entry()], cache)
    assert cache == {('previous-qualified-key',)}
    assert q.replay.compare_tree(q.screen.rng_state(), rng)['equal']
    assert model.backward_calls == 0 and model.level.grad is None


def test_constraint_only_difference_is_not_hidden_by_exact_waveform(monkeypatch):
    original = q.window_excesses
    def differing_values(prediction, e):
        rows = original(prediction, e)
        if torch.is_grad_enabled():
            for row in rows:
                row['values'] = tuple(value + 1e-15 for value in row['values'])
        return rows
    monkeypatch.setattr(q, 'window_excesses', differing_values)
    cache = set()
    with pytest.raises(RuntimeError, match="'waveform_exact': True, 'constraints_exact': False"):
        warm.warm_quiet_entries(ExecutionWave(), None, [entry()], cache)
    assert not cache


def test_optimizer_existing_gradient_objects_modes_teacher_rng_and_backend_preserved():
    model = ExecutionWave().train()
    teacher = nn.Sequential(nn.Linear(1, 1), nn.Identity()).eval().requires_grad_(False)
    teacher[1].train()
    optimizer = torch.optim.AdamW([model.level], lr=3e-5, betas=(.9, .99), weight_decay=0)
    model.level.grad = torch.ones_like(model.level)
    optimizer.step()
    for p in list(model.parameters()) + list(teacher.parameters()):
        p.grad = torch.full_like(p, 7.)
    parameters = list(model.parameters()) + list(teacher.parameters())
    slots = [p.grad for p in parameters]
    saved = [p.grad.clone() for p in parameters]
    state = copy.deepcopy(model.state_dict()), copy.deepcopy(teacher.state_dict()), copy.deepcopy(optimizer.state_dict())
    rng, backend = q.screen.rng_state(), q.replay.backend_state()
    modes = [m.training for parent in (model, teacher) for m in parent.modules()]
    report = warm.warm_quiet_entries(model, teacher, [entry(3857, context=1, valid=1937)], set(), optimizer=optimizer)
    assert report['passed'] and model.backward_calls == 0
    for current, expected in zip((model.state_dict(), teacher.state_dict(), optimizer.state_dict()), state):
        assert q.replay.compare_tree(current, expected)['equal']
    assert all(p.grad is slot and torch.equal(p.grad, value) for p, slot, value in zip(parameters, slots, saved))
    assert [m.training for parent in (model, teacher) for m in parent.modules()] == modes
    assert q.replay.compare_tree(q.screen.rng_state(), rng)['equal']
    assert q.replay.backend_state() == backend


def test_guard_rejects_mutation_and_never_caches_even_after_successful_parity():
    model, cache = ExecutionWave(mutate=True), set()
    with pytest.raises(RuntimeError, match='Diagnostic mutated protected state'):
        warm.warm_quiet_entries(model, None, [entry()], cache)
    assert model.protected.item() == 0 and not cache


def test_cached_shape_skips_only_warming_and_new_geometry_is_warmed():
    model, cache = ExecutionWave(), set()
    warm.warm_quiet_entries(model, None, [entry()], cache)
    report = warm.warm_quiet_entries(model, None, [entry(), entry(2880)], cache)
    assert report['new_shapes'] == 1 and report['warmup_grad_forwards'] == 3
    assert report['entries_checked'] == 2 and report['verification_forwards'] == 6
    assert len(cache) == 2
    # A new decoder cannot inherit another decoder's execution qualification.
    assert warm.warm_quiet_entries(ExecutionWave(), None, [entry()], cache)['new_shapes'] == 1


@torch.jit.script
def scripted_snake(x: torch.Tensor, alpha: torch.Tensor):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    return x.reshape(shape)


class NativeWave(nn.Module):
    def __init__(self):
        super().__init__()
        self.group = weight_norm(nn.Conv1d(1, 1, 1))
        self.suffix = weight_norm(nn.Conv1d(1, 1, 1)).requires_grad_(False)
        self.alpha = nn.Parameter(torch.ones(1, 1, 1), requires_grad=False)
        with torch.no_grad():
            self.group.bias.fill_(2e-6)
            self.suffix.bias.fill_(1e-6)
    def group_from_input(self, x):
        return self.group(x)
    def suffix_from_group(self, x):
        return self.suffix(scripted_snake(x, self.alpha))


def test_native_weight_norm_and_scripted_frozen_suffix_keep_gradient_graph():
    model = NativeWave().eval()
    state = copy.deepcopy(model.state_dict())
    report = warm.warm_quiet_entries(model, None, [entry()], set())
    assert report['passed'] and q.replay.compare_tree(model.state_dict(), state)['equal']
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize('invalid', ['inference', 'attached_input', 'wrong_dtype'])
def test_invalid_execution_does_not_publish_a_warm_shape(invalid):
    model, cache, e = ExecutionWave(), set(), entry()
    if invalid == 'attached_input':
        e['group_input'].requires_grad_()
    elif invalid == 'wrong_dtype':
        e['group_input'] = e['group_input'].double()
    with torch.inference_mode(invalid == 'inference'), pytest.raises((ValueError, RuntimeError)):
        warm.warm_quiet_entries(model, None, [e], cache)
    assert not cache and model.grad_calls == 0
