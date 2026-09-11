from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from audiovae_student.objective_comparison import state_fingerprint
from run_update_experiment import (fork_generator_rate, validate_gate, validate_pool, save_checkpoint,
    canonical_required, load_canonical_panel, evaluate_domains, combined_domain_result, load_fresh_training)
from test_diagnose_objectives_updates import prepared, deterministic


def _entries(crops):
    return [{"window": {"source_id": c.source_id, "start_frame": c.start_frame,
                         "valid_output_samples48k": c.valid_scored_samples},
             "discriminator_start_sample48k": None} for c in crops]


def test_cache_gate_is_required_and_bound_to_exact_inputs():
    expected = dict(parent_sha256="parent", target_cache_sha256="cache", data_plan_sha256="plan")
    gate = {"format_version": 1, "training_ready": True, "parent_checkpoint_sha256": "parent",
            "target_cache_sha256": "cache", "data_plan_sha256": "plan",
            "resolution": "Synthetic test receipt", "evidence": ["fixture.json"]}
    validate_gate(gate, **expected)
    for broken in ({**gate, "training_ready": False}, {**gate, "target_cache_sha256": "other"},
                   {**gate, "resolution": ""}, {**gate, "evidence": []}):
        with pytest.raises(ValueError):
            validate_gate(broken, **expected)


def test_quarter_migration_preserves_all_other_initial_state_and_rate_is_used(prepared):
    engine, crops = prepared
    before = deepcopy(engine.state_dict())
    migration = fork_generator_rate(engine, .25)
    after = deepcopy(engine.state_dict())
    for key in ("model", "discriminators", "discriminator_optimizer", "balancer", "calibration", "step", "crop_rng"):
        assert state_fingerprint(before[key]) == state_fingerprint(after[key])
    for name in engine.optimizer.optimizers:
        assert state_fingerprint(before["optimizer"]["optimizers"][name]["state"]) == state_fingerprint(after["optimizer"]["optimizers"][name]["state"])
    assert migration["generator_learning_rate"] == 5e-5
    updates = engine.balancer.updates
    result = engine.train_step(crops)
    assert result["learning_rate"] == 5e-5
    assert engine.balancer.updates == updates + 1
    assert engine.step == before["step"] + 1
    assert state_fingerprint(after["optimizer"]) != state_fingerprint(engine.optimizer.state_dict())


def test_control_migration_is_identical(prepared):
    engine, _ = prepared
    before = state_fingerprint(engine.state_dict())
    receipt = fork_generator_rate(engine, 1.)
    assert state_fingerprint(engine.state_dict()) == before
    assert receipt["fork_engine_sha256"] == receipt["source_engine_sha256"]


def test_unique_scored_intervals_allow_context_overlap_but_not_data_replay(prepared):
    _, crops = prepared
    first = crops[0]
    second = replace(first, start_frame=7, context_start_frame=6)
    pair = [first, second]
    assert validate_pool(pair, _entries(pair), [], expected_count=2)["scored_intervals_nonoverlapping"]
    duplicate = [first, replace(first)]
    with pytest.raises(ValueError, match="overlapping"):
        validate_pool(duplicate, _entries(duplicate), [], expected_count=2)
    with pytest.raises(ValueError, match="heldout"):
        validate_pool(pair, _entries(pair), [first], expected_count=2)
    invalid = _entries(pair); invalid[0]["discriminator_start_sample48k"] = first.valid_scored_samples
    with pytest.raises(ValueError, match="discriminator"):
        validate_pool(pair, invalid, [], expected_count=2)


def test_aliasing_audio_hash_cannot_bypass_unique_window_check(prepared):
    _, crops = prepared
    first = crops[0]
    second = replace(first, source_id="alias")
    rows = {first.source_id: {"audio_sha256": "same-audio"}, second.source_id: {"audio_sha256": "same-audio"}}
    with pytest.raises(ValueError, match="audio_hash"):
        validate_pool([first, second], _entries([first, second]), [], rows, expected_count=2)


def test_atomic_checkpoint_retains_original_on_insufficient_storage(tmp_path, monkeypatch):
    path = tmp_path / "last.pt"
    path.write_bytes(b"prior-checkpoint")
    monkeypatch.setattr("run_update_experiment.shutil.disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(RuntimeError, match="space"):
        save_checkpoint(path, {"x": torch.ones(2)}, expected_bytes=1000)
    assert path.read_bytes() == b"prior-checkpoint"


def test_corrected_domain_requires_receipt_and_boolean_policy():
    assert canonical_required({"requires_canonical_evaluation": True})
    assert canonical_required({"canonical_target_cache_sha256": "bound-cache"})
    assert not canonical_required({})
    with pytest.raises(ValueError, match="boolean"):
        canonical_required({"requires_canonical_evaluation": "true"})
    with pytest.raises(ValueError, match="canonical-receipt"):
        load_canonical_panel(None, {"requires_canonical_evaluation": True}, None)
    assert load_canonical_panel(None, {}, None) is None


def test_canonical_loader_binds_receipt_and_preserves_historical_geometry(prepared, tmp_path, monkeypatch):
    import sys
    from audiovae_student.restart_data import file_sha
    engine, crops = prepared
    metadata = {"fixture": {"condition": "speech"}}
    ctx = SimpleNamespace(heldout=crops, metadata=metadata)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text('{"test": "sealed"}')
    receipt = {"path": str(tmp_path / "heldout.pt"), "sha256": "cache", "contract": {"identity_sha256": "contract"}}
    returned = [crops, metadata, receipt]
    monkeypatch.setitem(sys.modules, "canonical_evaluation", SimpleNamespace(load_canonical_cache=lambda path: returned))
    gate = {"requires_canonical_evaluation": True, "canonical_receipt_sha256": file_sha(receipt_path),
            "canonical_target_contract_sha256": "contract", "canonical_target_cache_sha256": "cache"}
    result = load_canonical_panel(ctx, gate, receipt_path)
    assert result["identity"]["receipt_sha256"] == file_sha(receipt_path)
    assert result["identity"]["required_by_resolution"]
    with pytest.raises(ValueError, match="different canonical"):
        load_canonical_panel(ctx, {**gate, "canonical_target_cache_sha256": "another-cache"}, receipt_path)
    returned[0] = [replace(crops[0], source_id="other-source")]
    with pytest.raises(ValueError, match="geometry"):
        load_canonical_panel(ctx, gate, receipt_path)
    returned[0] = crops
    returned[1] = {"fixture": {"condition": "changed"}}
    with pytest.raises(ValueError, match="metadata"):
        load_canonical_panel(ctx, gate, receipt_path)


def test_both_evaluation_domains_use_their_own_inputs_and_receipt(monkeypatch):
    import sys
    calls = []
    def historical(engine, crops, metadata):
        calls.append(("historical", crops, metadata))
        return {"domain": "old"}
    def canonical(engine, crops, metadata, receipt):
        calls.append(("canonical", crops, metadata, receipt))
        return {"domain": "corrected"}
    monkeypatch.setattr("run_update_experiment.build_evaluation", historical)
    monkeypatch.setitem(sys.modules, "canonical_evaluation", SimpleNamespace(build_canonical_evaluation=canonical))
    ctx = SimpleNamespace(heldout="historical tensors", metadata="historical metadata")
    panel = {"crops": "canonical tensors", "metadata": "canonical metadata", "receipt": "sealed receipt"}
    result = evaluate_domains(object(), ctx, panel)
    assert result == {"historical": {"domain": "old"}, "canonical": {"domain": "corrected"}}
    assert calls == [("historical", "historical tensors", "historical metadata"),
                     ("canonical", "canonical tensors", "canonical metadata", "sealed receipt")]


@pytest.mark.parametrize("historical,canonical", [(False, True), (True, False), (False, False)])
def test_either_domain_failure_blocks_candidate(historical, canonical):
    report = combined_domain_result({"screen_passed": historical}, {"screen_passed": canonical}, required=True)
    assert report["screen_passed"] is False and report["promoted"] is False


def test_both_domains_must_pass_and_missing_required_result_cannot_pass():
    passed = {"screen_passed": True}
    assert combined_domain_result(passed, passed, required=True)["screen_passed"] is True
    assert combined_domain_result(passed, None, required=False)["screen_passed"] is True
    with pytest.raises(ValueError, match="required canonical"):
        combined_domain_result(passed, None, required=True)
    with pytest.raises(ValueError, match="defined boolean"):
        combined_domain_result(passed, {"screen_passed": None}, required=True)


def test_fresh_training_requirement_is_explicit_and_binds_exact_full_pool(monkeypatch):
    import sys
    with pytest.raises(ValueError, match="training-receipt"):
        load_fresh_training(None, {"requires_fresh_training_pairs": True}, None)
    assert load_fresh_training(None, {}, None) is None
    seen = []
    def loader(ctx, path, *, required_counts):
        seen.append(required_counts)
        return {"identity": {"receipt_sha256": "receipt", "contract_sha256": "contract", "cache_sha256": "cache"}}
    monkeypatch.setitem(sys.modules, "training_overlay", SimpleNamespace(load_training_overlay=loader))
    gate = {"requires_fresh_training_pairs": True, "training_target_cache_sha256": "cache"}
    assert load_fresh_training(None, gate, "fresh.json")["identity"]["cache_sha256"] == "cache"
    assert seen == [{"targeted_generator": 12800}]
    with pytest.raises(ValueError, match="different fresh-training"):
        load_fresh_training(None, {**gate, "training_target_cache_sha256": "old-cache"}, "fresh.json")


@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="Native Muon unavailable locally")
def test_joint_rate_migration_changes_both_native_optimizer_rates(prepared):
    from audiovae_student.optimizers import build_optimizer_bundle
    engine, crops = prepared
    engine.optimizer = build_optimizer_bundle(engine.model, optimizer="muon_adamw")
    engine.train_step(crops)
    fork_generator_rate(engine, .25)
    assert set(engine.optimizer.optimizers) == {"muon", "adamw"}
    assert all(g["lr"] == 5e-5 for opt in engine.optimizer.optimizers.values() for g in opt.param_groups)
    assert engine.train_step(crops)["learning_rate"] == 5e-5
