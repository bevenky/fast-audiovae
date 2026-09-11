from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from audiovae_student.objective_comparison import state_fingerprint
from corrected_training_overlay import seal_training_overlay
from training_overlay import load_training_overlay, verify_overlay_files
from run_corrected_update import native_rate_once, canonical_probes
from test_diagnose_objectives_updates import prepared, deterministic


def context(crops):
    return SimpleNamespace(pools={"gradient_calibration": crops},
        receipt={"target_cache_sha256": "base-cache"}, identity={"data_plan_sha256": "base-plan"},
        parent={"identity": {"data": {"teacher_state_sha256": "teacher"}}})


def seal(path, original, fresh):
    return seal_training_overlay(path, {"gradient_calibration": fresh}, {"gradient_calibration": original},
        source_provenance={"test": "fresh synthetic pairs"}, base_target_cache_sha256="base-cache",
        base_data_plan_sha256="base-plan", teacher_state_sha256="teacher")


def test_seal_and_load_allow_new_cache_identity_but_bind_old_inputs(prepared, tmp_path):
    _, crops = prepared
    fresh = [replace(crops[0], latents=crops[0].latents + .01,
                     teacher_audio=crops[0].teacher_audio + .001, cache_key="b" * 64)]
    receipt = seal(tmp_path, crops, fresh)
    overlay = load_training_overlay(context(crops), receipt, required_counts={"gradient_calibration": 1})
    assert overlay["pools"]["gradient_calibration"][0].cache_key == "b" * 64
    contract = overlay["receipt"]["contract"]["pools"]["gradient_calibration"]
    assert contract["baseline_pool_prefix_identity_sha256"] != contract["fresh_pool_identity_sha256"]
    assert verify_overlay_files(overlay)["fresh_training_overlay_files_unchanged"]
    with pytest.raises(ValueError, match="exact required"):
        load_training_overlay(context(crops), receipt, required_counts={"gradient_calibration": 2})
    wrong = context(crops); wrong.receipt["target_cache_sha256"] = "different"
    with pytest.raises(ValueError, match="different baseline"):
        load_training_overlay(wrong, receipt, required_counts={"gradient_calibration": 1})
    with (tmp_path / "pairs.pt").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="bytes"):
        load_training_overlay(context(crops), receipt, required_counts={"gradient_calibration": 1})


def test_geometry_reference_and_source_order_cannot_change(prepared, tmp_path):
    _, crops = prepared
    with pytest.raises(ValueError, match="geometry"):
        seal(tmp_path / "geometry", crops, [replace(crops[0], source_id="other")])
    with pytest.raises(ValueError, match="reference"):
        seal(tmp_path / "reference", crops, [replace(crops[0], reference16k=torch.zeros(1, 1, 7 * 640))])


def test_actual_current_and_quarter_steps_restore_all_trained_states(prepared):
    engine, crops = prepared
    before = state_fingerprint(engine.state_dict())
    full = native_rate_once(engine, crops, crops, factor=1.)
    assert state_fingerprint(engine.state_dict()) == before
    quarter = native_rate_once(engine, crops, crops, factor=.25)
    assert state_fingerprint(engine.state_dict()) == before
    assert full["baseline"] == quarter["baseline"]
    assert full["training_metrics"]["learning_rate"] == .0002
    assert quarter["training_metrics"]["learning_rate"] == .00005
    assert full["migration"]["source_engine_sha256"] == quarter["migration"]["source_engine_sha256"]
    assert full["state_restored"] and quarter["state_restored"]


def test_canonical_probe_selection_uses_same_keys_not_first_rows(prepared):
    _, crops = prepared
    other = replace(crops[0], source_id="other")
    fresh = replace(crops[0], teacher_audio=crops[0].teacher_audio + .001)
    ctx = SimpleNamespace(probe_crops=crops)
    assert canonical_probes(ctx, {"crops": [other, fresh]})[0] is fresh
    with pytest.raises(ValueError, match="Duplicate"):
        canonical_probes(ctx, {"crops": [fresh, fresh]})
