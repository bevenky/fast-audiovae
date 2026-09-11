"""CPU contracts for the bounded joint stage 2–4 continuation."""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "convnext"))
import joint_recovery_v1 as joint
from test_projected_hints_v1 import fixture
from test_group_model import assert_snapshot, snapshot, gm, selections


@pytest.fixture(autouse=True)
def cpu_state():
    rng, threads = joint.screen.rng_state(), torch.get_num_threads()
    torch.manual_seed(1904)
    torch.set_num_threads(1)
    yield
    joint.screen.restore_rng(rng)
    torch.set_num_threads(threads)


def ledger():
    original = [f"original-{i}" for i in range(3000)]
    fresh = [f"fresh-{i}" for i in range(27000)]
    candidate = {
        "historical_sources_seen": original + fresh[:10500],
        "additional_sources_seen": fresh[10500:12000],
    }
    return candidate, original, fresh


@pytest.mark.parametrize("updates", [250, 500, 750, 1000])
def test_source_window_uses_new_reserve_and_independent_optimizer_counter(updates):
    candidate, original, fresh = ledger()
    before = copy.deepcopy(candidate)
    historical, selected = joint.validate_source_window(candidate, original, fresh, updates=updates)
    assert historical == original + fresh[:12000]
    assert len(historical) == 15000
    assert selected == fresh[12000:12000 + updates * 12]
    assert not set(historical) & set(selected)
    assert len(set(historical + selected)) == 15000 + updates * 12
    # 4625 is an AdamW update count, not the 15000-source history divided by 3 or 12.
    assert joint.START_STEP + updates == 4625 + updates
    assert candidate == before


@pytest.mark.parametrize("options", [
    {"updates": 0}, {"updates": 249}, {"updates": 1250}, {"updates": True},
    {"start": 10500}, {"start": 12000.}, {"accumulation": 3},
])
def test_source_window_rejects_budget_or_accumulation_changes(options):
    candidate, original, fresh = ledger()
    with pytest.raises(ValueError):
        joint.validate_source_window(candidate, original, fresh, **options)


@pytest.mark.parametrize("damage", ["history_order", "history_missing", "reuse", "duplicate", "short"])
def test_source_window_rejects_history_changes_and_new_source_reuse(damage):
    candidate, original, fresh = ledger()
    if damage == "history_order":
        candidate["additional_sources_seen"][:2] = reversed(candidate["additional_sources_seen"][:2])
    elif damage == "history_missing":
        candidate["historical_sources_seen"].pop()
    elif damage == "reuse":
        fresh[12000] = original[0]
    elif damage == "duplicate":
        fresh[12001] = fresh[12000]
    elif damage == "short":
        fresh = fresh[:23999]
    with pytest.raises(ValueError):
        joint.validate_source_window(candidate, original, fresh)


def populated_candidate(monkeypatch):
    model, teacher, crops, objective, _ = fixture(monkeypatch, count=12)
    coefficients = {"waveform": 1., "mel": .0006674012905982311, "feature": .009304078923434964}
    optimizer = torch.optim.AdamW(model.trainable_group_parameters(), lr=3e-5,
                                  betas=(.9, .99), eps=1e-8, weight_decay=0.)
    joint.base.training_update(model, teacher, crops[:1], objective, coefficients, optimizer)
    for state in optimizer.state.values():
        state["step"].fill_(joint.START_STEP)
    candidate, original, fresh = ledger()
    candidate.update({
        "format": joint.prior.previous.VERSION, "optimizer_step": joint.START_STEP,
        "accumulation": 12, "group": copy.deepcopy(model.group_state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()), "rng": joint.screen.rng_state(),
        "original_training_identity": {"learning_rate": 3e-5, "coefficients": coefficients},
    })
    return model, teacher, crops, objective, candidate, original, fresh


def test_joint_update_is_exact_original_update_with_restored_moments_and_rng(monkeypatch):
    model, teacher, crops, objective, candidate, _, _ = populated_candidate(monkeypatch)
    # Legacy WN materializes a nonleaf weight after a forward, so construct an
    # independent graph and let the real checkpoint loader restore its group.
    reference_model = gm.build_student(teacher.model.decoder, *selections())
    untouched = copy.deepcopy(candidate)
    reference_opt, _ = joint.prior.restart_candidate(reference_model, candidate)
    expected = joint.base.training_update(reference_model, teacher, crops, objective,
        candidate["original_training_identity"]["coefficients"], reference_opt, record_diagnostics=True)
    expected_rng = joint.screen.rng_state()
    optimizer, restored = joint.prior.restart_candidate(model, candidate)
    assert all(check["equal"] for check in restored.values())
    frozen_before = joint.screen.frozen_versions(model)
    teacher_before = snapshot(teacher)
    actual = joint.screen.training_update(model, teacher, crops, "current", objective,
        candidate["original_training_identity"]["coefficients"], optimizer, record_diagnostics=True)
    assert actual == expected
    assert_snapshot(model, snapshot(reference_model))
    assert joint.replay.compare_tree(optimizer.state_dict(), reference_opt.state_dict())["equal"]
    assert joint.replay.compare_tree(joint.screen.rng_state(), expected_rng)["equal"]
    assert joint.replay.compare_tree(candidate, untouched)["equal"]
    assert joint.screen.frozen_versions(model) == frozen_before
    assert_snapshot(teacher, teacher_before)
    assert all(float(state["step"]) == 4626 for state in optimizer.state.values())
    # All residuals, conditioning and upsampling in the three-stage box remain joint.
    trainable = {name for name, p in model.decoder.named_parameters() if p.requires_grad}
    assert trainable == {name for name, _ in model.group_named_parameters()}
    assert any(name.startswith("model.3.") for name in trainable)
    assert any(name.startswith("model.4.") for name in trainable)
    assert any(name.startswith("model.5.block.1.") for name in trainable)
    assert any(name.startswith("model.5.block.4.") for name in trainable)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert all(p.grad is None for p in teacher.parameters())


def test_restore_rejects_incomplete_moments_before_mutating_weights_or_rng(monkeypatch):
    model, _, _, _, candidate, _, _ = populated_candidate(monkeypatch)
    damaged = copy.deepcopy(candidate)
    damaged["optimizer"]["state"].pop(next(iter(damaged["optimizer"]["state"])))
    with torch.no_grad():
        next(iter(model.trainable_group_parameters())).add_(.2)
    before = snapshot(model)
    rng = joint.screen.rng_state()
    with pytest.raises(ValueError, match="Missing or additional"):
        joint.prior.restart_candidate(model, damaged)
    assert_snapshot(model, before)
    assert joint.replay.compare_tree(joint.screen.rng_state(), rng)["equal"]


def save_fixture(monkeypatch, updates=250):
    model, teacher, crops, objective, candidate, _, fresh = populated_candidate(monkeypatch)
    optimizer, _ = joint.prior.restart_candidate(model, candidate)
    for state in optimizer.state.values():
        state["step"].fill_(joint.START_STEP + updates)
    seen = fresh[12000:12000 + updates * 12]
    identity = {
        "version": joint.VERSION,
        "source_interval": [12000, 24000],
        "source_ids_sha256": joint.screen.digest(fresh[12000:24000]),
        "starting_optimizer_step": 4625, "target_optimizer_step": 5625,
        "gradient_accumulation": 12, "execution_batch_size": 1,
        "coefficients": candidate["original_training_identity"]["coefficients"],
        "learning_rate": 3e-5,
    }
    return model, optimizer, candidate, seen, identity


def test_checkpoint_preserves_group_moments_rng_and_separate_source_ledger(monkeypatch, tmp_path):
    model, optimizer, candidate, seen, identity = save_fixture(monkeypatch)
    before = snapshot(model)
    moments, rng = copy.deepcopy(optimizer.state_dict()), joint.screen.rng_state()
    path = tmp_path / "checkpoint-step4875.pt"
    receipt = joint.save_state(path, model, optimizer, candidate, seen, 1234567, identity)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert saved["optimizer_step"] == 4875 and saved["updates"] == 250
    assert len(saved["historical_sources_seen"]) == 15000
    assert saved["additional_sources_seen"] == seen and len(seen) == 3000
    assert saved["accumulation"] == 12 and saved["scored_samples"] == 1234567
    assert saved["automatic_promotion"] is False
    assert saved["original_training_identity"] == candidate["original_training_identity"]
    assert joint.replay.compare_tree(saved["group"], dict(model.group_state_dict()))["equal"]
    assert joint.replay.compare_tree(saved["optimizer"], moments)["equal"]
    assert joint.replay.compare_tree(saved["rng"], rng)["equal"]
    assert receipt["checkpoint_sha256"] == joint.base.sha(path)
    assert receipt["optimizer_step"] == 4875 and receipt["sources_consumed"] == 3000
    assert receipt["optimizer_and_rng_saved"] is True
    assert not path.with_suffix(".tmp").exists()
    assert_snapshot(model, before)
    assert joint.replay.compare_tree(optimizer.state_dict(), moments)["equal"]
    assert joint.replay.compare_tree(joint.screen.rng_state(), rng)["equal"]
    with pytest.raises(FileExistsError):
        joint.save_state(path, model, optimizer, candidate, seen, 1234567, identity)


@pytest.mark.parametrize("damage", ["partial_update", "wrong_counter", "empty_moments", "missing_moment", "duplicate", "reuse", "over_budget"])
def test_save_rejects_incomplete_or_reused_state_without_creating_checkpoint(monkeypatch, tmp_path, damage):
    model, optimizer, candidate, seen, identity = save_fixture(monkeypatch)
    if damage == "partial_update": seen.pop()
    elif damage == "wrong_counter": next(iter(optimizer.state.values()))["step"].add_(1)
    elif damage == "empty_moments": optimizer.state.clear()
    elif damage == "missing_moment": optimizer.state.pop(next(iter(optimizer.state)))
    elif damage == "duplicate": seen[-1] = seen[-2]
    elif damage == "reuse": seen[-1] = candidate["historical_sources_seen"][0]
    elif damage == "over_budget":
        seen = [f"too-many-{i}" for i in range(12012)]
        for state in optimizer.state.values(): state["step"].fill_(5626)
    path = tmp_path / "invalid.pt"
    with pytest.raises((ValueError, RuntimeError)):
        joint.save_state(path, model, optimizer, candidate, seen, 1234567, identity)
    assert not path.exists() and not path.with_suffix(".tmp").exists()


def test_final_checkpoint_rejects_changed_order_against_sealed_source_digest(monkeypatch, tmp_path):
    model, optimizer, candidate, seen, identity = save_fixture(monkeypatch, updates=1000)
    seen[:2] = reversed(seen[:2])
    with pytest.raises(ValueError, match="digest"):
        joint.save_state(tmp_path / "invalid-final.pt", model, optimizer, candidate, seen, 900, identity)
    assert not (tmp_path / "invalid-final.pt").exists()


def test_near_silence_observer_preserves_existing_values_and_pools_valid_samples(monkeypatch):
    original = joint.base.quiet_window_metrics
    teacher = torch.cat([torch.full((1, 1, 960), 5e-6),
                         torch.full((1, 1, 960), 5e-4)], -1)
    prediction = teacher + torch.cat([torch.full((1, 1, 960), 2e-6),
                                     torch.full((1, 1, 960), 1e-3)], -1)
    valid = torch.ones_like(teacher, dtype=torch.bool)
    target2 = torch.full((1, 1, 237), 2e-6)
    prediction2 = target2 + 6e-6
    valid2 = torch.ones_like(target2, dtype=torch.bool)
    expected = [original(prediction, teacher, valid),
                original(prediction2, target2, valid2)]
    calls = []
    class Monitor:
        def evaluate(self, model, teacher_model, crops, common):
            calls.extend([joint.base.quiet_window_metrics(prediction, teacher, valid),
                          joint.base.quiet_window_metrics(prediction2, target2, valid2)])
            return {"aggregate": {"sentinel": 17.}, "rows": [{"source_id": "a"}, {"source_id": "b"}]}
    report = joint.evaluate_review(Monitor(), None, None, ["a", "b"], None)
    assert calls == expected
    assert report["aggregate"] == {"sentinel": 17.}
    assert report["rows"] == [{"source_id": "a"}, {"source_id": "b"}]
    near = report["recovery_window_metrics"]
    assert near["near_samples"] == 1197
    assert near["near_error_sum"] == pytest.approx(960 * (2e-6)**2 + 237 * (6e-6)**2, rel=2e-7)
    assert near["near_residual_rms"] == pytest.approx((near["near_error_sum"] / 1197)**.5)
    assert near["by_source"]["b"]["near_samples"] == 237
    assert joint.base.quiet_window_metrics is original


def test_near_silence_observer_restores_original_helper_after_exception():
    original = joint.base.quiet_window_metrics
    class FailingMonitor:
        def evaluate(self, *args):
            raise RuntimeError("scoring failure")
    with pytest.raises(RuntimeError, match="scoring failure"):
        joint.evaluate_review(FailingMonitor(), None, None, [], None)
    assert joint.base.quiet_window_metrics is original
