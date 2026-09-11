"""CPU checks for exact late-segment replay and diagnostic isolation."""
import copy
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "convnext"))
import replay_late_segment as replay


def ledgers():
    original = [f"original-{i}" for i in range(3000)]
    fresh = [f"fresh-{i}" for i in range(27000)]
    identity = {"learning_rate": 3e-5, "coefficients": {"waveform": 1., "mel": .002, "feature": .01}}
    def payload(step):
        cursor = (step-1000)*3
        return {"step": step, "fit_cursor": 3000+cursor, "fresh_cursor": cursor,
                "sources_seen": original+fresh[:cursor], "identity": identity,
                "resume_identity": {"effective_batch_size": 3, "execution_batch_size": 1,
                                    "accumulation_steps": 3}}
    return payload(4500), payload(5000), original, fresh


def test_replay_interval_consumes_exactly_the_original_last_1500_sources():
    start, stop, original, fresh = ledgers()
    assert replay.replay_window(start, stop, original, fresh) == (10500, 12000)
    triplets = [fresh[index:index+3] for index in range(10500, 12000, 3)]
    assert len(triplets) == 500
    assert triplets[0] == ["fresh-10500", "fresh-10501", "fresh-10502"]
    assert triplets[-1] == ["fresh-11997", "fresh-11998", "fresh-11999"]
    assert start["sources_seen"] + sum(triplets, []) == stop["sources_seen"]


@pytest.mark.parametrize("change", ["start_step", "end_step", "start_cursor", "end_order", "identity", "batch"])
def test_replay_rejects_changed_steps_source_order_recipe_or_batch(change):
    start, stop, original, fresh = ledgers()
    if change == "start_step": start["step"] -= 1
    if change == "end_step": stop["step"] += 1
    if change == "start_cursor": start["fresh_cursor"] -= 3
    if change == "end_order": stop["sources_seen"][-2:] = reversed(stop["sources_seen"][-2:])
    if change == "identity": stop["identity"] = {**stop["identity"], "learning_rate": 1e-4}
    if change == "batch": stop["resume_identity"] = {**stop["resume_identity"], "effective_batch_size": 4}
    with pytest.raises((ValueError, RuntimeError)):
        replay.replay_window(start, stop, original, fresh)


def test_tree_compares_values_and_nested_types_not_serialized_byte_order():
    left = {"state": {7: {"exp_avg": torch.tensor([.25, -.5]), "step": torch.tensor(4500.)}},
            "groups": [{"betas": (.9, .99), "weight_decay": 0}], "flag": True}
    right = {"flag": True, "groups": copy.deepcopy(left["groups"]), "state": copy.deepcopy(left["state"])}
    assert replay.compare_tree(left, right)["equal"] is True
    for altered in (
        {**right, "flag": 1},
        {**right, "groups": [{"betas": [.9, .99], "weight_decay": 0}]},
        {**right, "state": {7: {"exp_avg": torch.tensor([.25, -.5], dtype=torch.float64), "step": torch.tensor(4500.)}}},
    ):
        result = replay.compare_tree(left, altered)
        assert result["equal"] is False and result["mismatches"]
    changed = copy.deepcopy(right)
    changed["state"][7]["exp_avg"][0] = torch.nextafter(torch.tensor(.25), torch.tensor(1.))
    result = replay.compare_tree(left, changed)
    assert result["equal"] is False
    assert any("exp_avg" in row["path"] for row in result["mismatches"])
    assert replay.compare_tree(torch.tensor([float("nan")]), torch.tensor([float("nan")]))["equal"] is False


def test_training_record_requires_exact_branch_values_and_order_but_ignores_wall_clock():
    values = {"total": .31, "waveform": .3, "mel": .4, "feature": .2, "gradient_norm": .8}
    ids = ["a", "b", "c"]
    original = {"step": 4525, "source_ids": ids, **values, "step_seconds": .04, "gpu_memory_gib": 8.}
    assert replay.validate_training_record(4525, ids, {**values, "step_seconds": 4.}, original)["equal"] is True
    for actual_step, actual_ids, actual_values in (
        (4524, ids, values), (4525, ids[::-1], values),
        (4525, ids, {**values, "mel": float(np.nextafter(.4, 1.))}),
        (4525, ids, {**values, "gradient_norm": .800000001}),
    ):
        assert replay.validate_training_record(actual_step, actual_ids, actual_values, original)["equal"] is False


class TinyGroup(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.group = nn.Linear(2, 1)
        self.decoder.suffix = nn.Linear(1, 1)
        self.decoder.suffix.requires_grad_(False)
        self.decoder.register_buffer("normalization", torch.tensor([.75]))

    def forward(self, x):
        return self.decoder.suffix(self.decoder.group(x))

    def group_named_parameters(self):
        return [("group."+name, value) for name, value in self.decoder.group.named_parameters()]

    def group_state_dict(self):
        return {"group."+name: value for name, value in self.decoder.group.state_dict().items()}


def test_diagnostic_guard_restores_rng_gradients_modes_and_requires_grad_even_on_exception():
    model = TinyGroup()
    teacher = nn.Linear(2, 1).requires_grad_(False)
    model.train(); model.decoder.suffix.eval(); teacher.eval()
    params = list(model.parameters())
    for p in params: p.grad = torch.full_like(p, .125)
    before_grads = [p.grad.clone() for p in params]
    before_modes = [m.training for m in model.modules()]
    before_requires = [p.requires_grad for p in params]
    rng = replay.screen.rng_state()
    with pytest.raises(RuntimeError, match="intentional diagnostic failure"):
        with replay.diagnostic_state_guard(model, teacher):
            torch.rand(5); np.random.rand(5); random.random()
            model.eval(); teacher.train()
            for p in params: p.grad = None; p.requires_grad_(False)
            raise RuntimeError("intentional diagnostic failure")
    assert replay.compare_tree(replay.screen.rng_state(), rng)["equal"]
    assert [m.training for m in model.modules()] == before_modes
    assert [p.requires_grad for p in params] == before_requires
    for p, saved in zip(params, before_grads):
        torch.testing.assert_close(p.grad, saved, rtol=0, atol=0)
    assert teacher.training is False


@pytest.mark.parametrize("target", ["weight", "buffer", "optimizer"])
def test_diagnostic_guard_rejects_persistent_training_state_mutation(target):
    model = TinyGroup()
    teacher = nn.Linear(2, 1).requires_grad_(False)
    optimizer = torch.optim.AdamW(model.decoder.group.parameters(), lr=3e-5, betas=(.9, .99), weight_decay=0)
    model(torch.tensor([[.4, .7]])).square().mean().backward(); optimizer.step()
    with pytest.raises(RuntimeError):
        with replay.diagnostic_state_guard(model, teacher, optimizer):
            if target == "weight":
                with torch.no_grad(): model.decoder.group.weight.add_(1)
            elif target == "buffer": model.decoder.normalization.add_(1)
            else: optimizer.state[model.decoder.group.weight]["exp_avg"].add_(1)


def test_replay_step_calls_the_original_update_once_and_does_not_change_its_arithmetic():
    uninterrupted = TinyGroup().double()
    instrumented = copy.deepcopy(uninterrupted)
    opt_a = torch.optim.AdamW(uninterrupted.decoder.group.parameters(), lr=3e-5, betas=(.9, .99), weight_decay=0)
    opt_b = torch.optim.AdamW(instrumented.decoder.group.parameters(), lr=3e-5, betas=(.9, .99), weight_decay=0)
    calls = []
    def original_update(model, teacher, crops, definition, common, coefficients, optimizer, *, record_diagnostics):
        calls.append((definition, record_diagnostics))
        optimizer.zero_grad(set_to_none=True)
        x = torch.rand(3, 2, dtype=torch.float64)
        loss = (model(x)-.6).square().mean()
        loss.backward(); optimizer.step()
        return {"total": float(loss.detach()), "waveform": float(loss.detach()), "mel": 0., "feature": 0.}
    rng = replay.screen.rng_state()
    expected = original_update(uninterrupted, None, [], 'current', None, {}, opt_a, record_diagnostics=True)
    after_rng = replay.screen.rng_state()
    replay.screen.restore_rng(rng)
    before = {name: p.detach().clone() for name, p in instrumented.group_named_parameters()}
    actual, direction = replay.replay_step(instrumented, None, [], None, {}, opt_b, True, original_update)
    assert calls == [('current', True), ('current', True)]
    assert replay.compare_tree(actual, expected)['equal']
    assert replay.compare_tree(instrumented.state_dict(), uninterrupted.state_dict())['equal']
    assert replay.compare_tree(opt_b.state_dict(), opt_a.state_dict())['equal']
    assert replay.compare_tree(replay.screen.rng_state(), after_rng)['equal']
    expected_dot = sum(float((p.grad*(p.detach()-before[name])).sum()) for name, p in instrumented.group_named_parameters())
    assert direction['gradient_dot_actual_update'] == pytest.approx(expected_dot, rel=1e-13, abs=1e-18)


def test_cache_observer_uses_only_original_forward_and_ignores_unscored_context(monkeypatch):
    cached = torch.ones(1, 1, 5760)
    waveform = cached.clone(); waveform[..., :1920] = 99.; waveform[..., 3840:] = -99.
    crop = {"source_id": "fresh-case", "context_frames": 1, "valid_scored_samples": 1920, "teacher_audio": cached}
    trace = {"waveform": waveform}
    calls = []
    def original(teacher, z): calls.append((teacher, z)); return trace
    monkeypatch.setattr(replay.base, "teacher_forward", original)
    latent = torch.zeros(1)
    with replay.observe_teacher_cache([crop]) as records:
        assert replay.base.teacher_forward(None, latent) is trace
    assert len(calls) == 1 and len(records) == 1
    assert records[0]['bitwise_equal'] and records[0]['max_abs'] == 0.
    assert records[0]['samples'] == 1920
    assert replay.base.teacher_forward is original


def test_cache_observer_restores_original_forward_after_missing_call_error(monkeypatch):
    original = lambda teacher, z: {"waveform": torch.zeros(1, 1, 1920)}
    monkeypatch.setattr(replay.base, "teacher_forward", original)
    with pytest.raises(RuntimeError, match="Missing teacher/cache"):
        with replay.observe_teacher_cache([{"source_id": "never-called"}]):
            pass
    assert replay.base.teacher_forward is original
