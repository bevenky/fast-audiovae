import copy
from pathlib import Path

import pytest
import torch

from audiovae_student.monitoring import evaluation_scalars, training_scalars


def test_balancer_reporting_uses_current_component_norms_not_nominal_shares():
    row = {"teacher_waveform": 2., "teacher_mel": 3., "feature_matching": 5., "adversarial": 7.,
           "teacher_waveform/scaled_norm": .3, "teacher_mel/scaled_norm": 1.2,
           "teacher_waveform/target_share": .3, "teacher_mel/target_share": .4, "total": 999.}
    before = copy.deepcopy(row)
    scalars = training_scalars(row)
    assert scalars["loss/reconstruction_only"] == 5.
    assert scalars["balance/component_norm_fraction/teacher_waveform"] == pytest.approx(.2)
    assert scalars["balance/scheduled_share/teacher_waveform"] == .3
    assert not any(k.endswith("total") for k in scalars)
    assert row == before


def test_peak_overshoot_prevents_combined_acceptance_without_changing_old_gate():
    evaluation = {"gate": {"passed": True}, "groups": {}, "rows": [
        {"samples": 100, "student_peak_abs": 1.4, "student_clipped_samples": 4,
         "quiet_windows": {"windows": [{"is_quiet": True, "residual_rms": .02},
                                          {"is_quiet": False, "residual_rms": .9}]}}]}
    scalars = evaluation_scalars(evaluation)
    assert scalars["quality/all/reconstruction_and_peak_checks_pass"] == 0
    assert scalars["quality/all/quiet_residual_rms_mean"] == .02
    assert evaluation["gate"]["passed"] is True
    evaluation["rows"][0]["student_peak_abs"] = 1.
    assert evaluation_scalars(evaluation)["quality/all/reconstruction_and_peak_checks_pass"] == 1
    del evaluation["rows"][0]["student_peak_abs"]
    with pytest.raises(ValueError, match="complete"):
        evaluation_scalars(evaluation)


def test_retention_pins_atomic_checkpoint_generation(tmp_path, monkeypatch):
    import shutil
    from audiovae_student import checkpoint_retention as observer
    monkeypatch.setattr(observer.shutil, "disk_usage", lambda _: shutil._ntuple_diskusage(100*1024**3, 0, 100*1024**3))
    latest = tmp_path / "latest.pt"
    value = {"format_version": "recipe_v2_pilot", "engine": {"step": 3000},
             "identity": {"batch_size": 32}, "sampler": {"cursor": 96000}, "journal_sha256": "original"}
    torch.save(value, latest)
    result = observer.snapshot_latest(latest, tmp_path / "kept", 3000)
    replacement = tmp_path / "new.pt"
    torch.save({**value, "journal_sha256": "later"}, replacement)
    replacement.replace(latest)
    retained = torch.load(result["path"], weights_only=True)
    assert retained["journal_sha256"] == "original"
    assert torch.load(latest, weights_only=True)["journal_sha256"] == "later"
    assert result["cursor"] == 96000
    # Simulate publication completed but the observer state was never committed.
    stale_state = {"checkpoints": [], "completed_thresholds": []}
    observer.reconcile_retention(tmp_path / "kept", stale_state)
    assert stale_state["checkpoints"] == [result]
    assert stale_state["completed_thresholds"] == [3000]
    # Simulate a crash after receipt publication and before the checkpoint rename.
    Path(result["path"]).rename(tmp_path / "kept/capture.pending.pt")
    observer.reconcile_retention(tmp_path / "kept", stale_state)
    assert Path(result["path"]).exists()
    # Pruning interrupted before the state write must not strand stale ownership.
    Path(result["path"]).unlink()
    observer.reconcile_retention(tmp_path / "kept", stale_state)
    assert stale_state["checkpoints"] == []
