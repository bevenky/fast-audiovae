"""Synthetic logging tests without constructing or evaluating an audio model."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import progressive_monitor as monitoring
from joint_recovery_gates_v2 import summarize_regions


class Writer:
    def __init__(self, path):
        self.path = Path(path); self.scalars = []; self.text = []; self.layouts = []; self.closed = False
    def add_scalar(self, tag, value, step): self.scalars.append((tag, value, step))
    def add_text(self, tag, value, step): self.text.append((tag, value, step))
    def add_custom_scalars(self, layout): self.layouts.append(layout)
    def flush(self): pass
    def close(self): self.closed = True


def report():
    windows = []
    for index, (start, teacher, zero) in enumerate([
        (0, 9e-6, True), (960, 6e-4, True), (1920, 9e-6, True),
        (48000, 9e-6, False), (48960, 3e-4, False),
    ]):
        windows.append({"window_id": str(index), "source_id": "source",
            "source_start_sample": start, "source_stop_sample": start+960,
            "valid_samples": 960, "is_quiet": True, "teacher_rms": teacher,
            "student_rms": teacher, "residual_limit": max(.02**.5*teacher, 1e-5),
            "output_rms_limit": max(10**.05*teacher, 1e-5),
            "source_reference_exact_zero": zero, "failure_category": "passed",
            "residual_square_sum": 4e-6**2*960, "centered_residual_square_sum": 3e-6**2*960})
    return {"aggregate": {"sources": 1, "samples": 4800, "mae": .02, "mel": .5,
        "group_mse": .1, "quiet_residual_rms_mean": 4e-6,
        "nonquiet_cosine_mean": .875, "peak_abs_max": .75,
        "quiet_windows": 5, "quiet_failed_windows": 0, "overshoot_samples": 0},
        "rows": [{"source_id": "source"}], "quiet_regions": summarize_regions(windows)}


def make(tmp_path): return monitoring.ProgressiveMonitor(tmp_path/"new", writer_factory=Writer)


def metric(monitor, prefix):
    return next(w for name, w in monitor.writers.items() if name.startswith(monitor.cut_name+" "+prefix))


def test_fresh_cut_has_one_layout_and_safe_metric_runs_and_unchanged_input(tmp_path):
    monitor = make(tmp_path); source = report(); before = deepcopy(source)
    monitor.log_validation(source, 0)
    assert source == before
    assert sum(len(w.layouts) for w in monitor.writers.values()) == 1
    overview = [w for name, w in monitor.writers.items() if not name.endswith(" Details")]
    assert len(overview) == 13
    assert all({tag for tag, _, _ in w.scalars} == {monitoring.COMMON_TAG} for w in overview)
    assert all(not any(c in w.path.name for c in "()/\\") for w in overview)
    assert metric(monitor, "01").scalars == [(monitoring.COMMON_TAG, 87.5, 0)]
    assert metric(monitor, "02").scalars == [(monitoring.COMMON_TAG, 99., 0)]
    assert metric(monitor, "03").scalars == [(monitoring.COMMON_TAG, 0., 0)]
    assert json.loads((monitor.path/"pruned-step0-baseline.json").read_text())["widths"] == [384, 256]
    assert "not perceptual accuracy" in monitor.details.text[0][1]
    assert "never the exact-copy teacher control" in monitor.details.text[0][1]


def test_raw_training_updates_do_not_fabricate_validation_measurements(tmp_path):
    monitor = make(tmp_path); monitor.log_validation(report(), 0)
    for step in (1, 2):
        monitor.log_training({"total": .3, "waveform": .02, "mel": .5,
            "feature": .1, "unique_sources": step*12, "audio_hours": .1}, step)
    assert len(metric(monitor, "01").scalars) == 1
    assert [s for _, _, s in metric(monitor, "13").scalars] == [0, 1, 2]
    assert [(v,s) for tag,v,s in monitor.details.scalars if tag == "loss/teacher_waveform"] == [(.02,1),(.02,2)]
    assert "Latest measured quality is step **0**" in monitor.details.text[-1][1]


def test_negative_reduction_remains_visible_and_baseline_never_moves(tmp_path):
    monitor = make(tmp_path); monitor.log_validation(report(), 0)
    original = (monitor.path/"pruned-step0-baseline.json").read_bytes()
    later = report(); later["aggregate"]["mae"] *= 1.25
    monitor.log_validation(later, 500)
    assert metric(monitor, "03").scalars[-1] == (monitoring.COMMON_TAG, -25., 500)
    assert (monitor.path/"pruned-step0-baseline.json").read_bytes() == original
    assert json.loads((monitor.path/"latest-validation.json").read_text())["aggregate"]["mae"] == .025


def test_zero_baseline_component_is_raw_only_without_division_or_invented_percentage(tmp_path):
    monitor = make(tmp_path); source = report(); source["aggregate"]["quiet_residual_rms_mean"] = 0.
    monitor.log_validation(source, 0)
    assert not any("06 Quiet" in name for name in monitor.writers)
    assert ("quality/quiet_residual_rms_mean", 0., 0) in monitor.details.scalars
    assert json.loads((monitor.path/"latest-validation.json").read_text())["omitted_zero_baseline_reductions"] == ["quiet_residual_rms_mean"]


def test_exact_copy_control_cannot_become_pruned_baseline(tmp_path):
    monitor = make(tmp_path); source = report()
    for key in monitoring.ERRORS: source["aggregate"][key] = 0.
    with pytest.raises(ValueError, match="exact-copy teacher"): monitor.log_validation(source, 0)
    assert monitor.baseline is None and monitor.last_validation_step == -1


@pytest.mark.parametrize("damage", ["source", "teacher", "quiet_count", "nonfinite"])
def test_changed_panel_or_invalid_metrics_are_rejected_before_logging(tmp_path, damage):
    monitor = make(tmp_path); monitor.log_validation(report(), 0)
    source = report()
    if damage == "source": source["rows"][0]["source_id"] = "other"
    elif damage == "teacher": source["quiet_regions"]["regions"]["near_silence"]["teacher_rms"] *= 2
    elif damage == "quiet_count": source["aggregate"]["quiet_windows"] += 1
    else: source["aggregate"]["mae"] = float("nan")
    before = {n:len(w.scalars) for n,w in monitor.writers.items()}
    with pytest.raises(ValueError): monitor.log_validation(source, 500)
    assert before == {n:len(w.scalars) for n,w in monitor.writers.items()}


def test_all_seven_quiet_regions_keep_raw_levels_errors_and_categories(tmp_path):
    monitor = make(tmp_path); monitor.log_validation(report(), 0)
    tags = {tag for tag, _, _ in monitor.details.scalars}
    for name in monitoring.REGIONS:
        for key in ("residual_rms", "output_rms", "output_limit_excess_rms", "failed", "amplitude_failed", "both"):
            assert "quiet/"+name+"/"+key in tags


def test_old_directory_steps_and_closed_writer_cannot_mix_new_history(tmp_path):
    monitor = make(tmp_path)
    with pytest.raises(FileExistsError): make(tmp_path)
    with pytest.raises(ValueError): monitor.log_validation(report(), 500)
    with pytest.raises(ValueError): monitor.log_training({"total": 1.}, 1)
    monitor.log_validation(report(), 0)
    with pytest.raises(ValueError): monitor.log_validation(report(), 0)
    with pytest.raises(ValueError): monitor.log_validation(report(), 250)
    with pytest.raises(ValueError): monitor.log_training({"total": 1.}, 1001)
    monitor.close(); monitor.close()
    assert all(w.closed for w in monitor.writers.values())
    with pytest.raises(ValueError): monitor.log_training({"total": 1.}, 1)


def test_real_tensorboard_events_share_one_safe_tag_and_keep_raw_quality(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.plugins.custom_scalar import metadata
    monitor = monitoring.ProgressiveMonitor(tmp_path/"events")
    try:
        monitor.log_validation(report(), 0)
        monitor.log_training({"total": .125, "waveform": .0625, "mel": .5, "feature": .25}, 1)
        monitor.log_validation(report(), 500)
    finally:
        monitor.close()
    directories = [p for p in monitor.path.iterdir() if p.is_dir()]
    assert len(directories) == 14
    layouts = 0
    for directory in directories:
        events = EventAccumulator(str(directory)).Reload()
        layouts += metadata.CONFIG_SUMMARY_TAG in events.Tags()["tensors"]
        if directory.name.endswith(" Details"):
            assert [(v.step, v.value) for v in events.Scalars("loss/teacher_waveform")] == [(1, .0625)]
            assert [v.step for v in events.Scalars("quality/nonquiet_cosine_mean")] == [0, 500]
        else:
            assert events.Tags()["scalars"] == [monitoring.COMMON_TAG]
            assert [v.step for v in events.Scalars(monitoring.COMMON_TAG)] == (
                [0, 1, 500] if "13 Training progress" in directory.name else [0, 500])
    assert layouts == 1
