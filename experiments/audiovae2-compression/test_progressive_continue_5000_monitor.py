"""Isolated0..5000 TensorBoard history without additional training exposure."""
from copy import deepcopy
import json
import random

import numpy as np
import pytest
import torch

import progressive_continue_5000_monitor as monitoring
from test_progressive_continue_monitor import quality, row, state
from test_progressive_monitor import Writer, metric


@pytest.fixture
def histories(tmp_path):
    first, second = tmp_path/"original", tmp_path/"recovery2000"
    first.mkdir(); second.mkdir()
    for directory, steps in ((first, (0, 500, 1000)), (second, (1000, 1500, 2000))):
        for step in steps:
            report = quality(1-step/10000)
            if directory == second and step == 1000:
                # A small resumed observation difference must not replace the
                # original1000 chart point. Runner verifies numeric parity.
                report["aggregate"]["mae"] += 1e-9
            (directory/f"development-step{step}.json").write_text(json.dumps(report))
        start = 1 if directory == first else 1001
        (directory/"train.jsonl").write_text("".join(json.dumps(row(s))+"\n" for s in range(start, start+1000)))
        (directory/"old-events").write_bytes(b"preserve old events")
    (first/"full-width-copy.json").write_text(json.dumps({"passed": True}))
    return first, second


def make(tmp_path, histories, **kwargs):
    first, second = histories
    return monitoring.ContinuationMonitor(tmp_path/"new-events", parent_dir=second,
        original_dir=first, source_metadata={"source": {"verified_source_labels": ["human_whistling_source_description"]}},
        writer_factory=Writer, **kwargs)


def test_both_histories_replayed_once_with_fixed_original_baseline(tmp_path, histories):
    before = [state(p) for p in histories]
    monitor = make(tmp_path, histories)
    assert [state(p) for p in histories] == before
    assert monitor.baseline == quality()["aggregate"]
    assert [s for _, _, s in metric(monitor, "01").scalars] == [0, 500, 1000, 1500, 2000]
    reductions = [value for _, value, _ in metric(monitor, "03").scalars]
    assert reductions == pytest.approx([0, 5, 10, 15, 20])
    assert [step for tag, _, step in monitor.details.scalars if tag == "loss/teacher_waveform"] == list(range(1, 2001))
    replay = monitor.history_identity
    assert replay["replayed_training_records"] == 2000
    assert replay["new_training_updates_from_replay"] == replay["new_audio_samples_from_replay"] == 0
    assert replay["prior_unique_sources"] == 24000
    assert replay["prior_scored_audio_hours"] == 2.
    assert replay["prior_elapsed_seconds"] == 20.
    assert replay["duplicate_resumed1000_observation_replayed"] is False
    assert len(monitor.writers) == 16
    assert len(replay["files_sha256"]) == 9


def test_every_progress_point_uses5000_and_new_logs_reach_final_without_duplicate_quality(tmp_path, histories):
    monitor = make(tmp_path, histories)
    progress = metric(monitor, "13").scalars
    assert len(progress) == 2001
    assert progress[1000][1] == 20.
    assert progress[2000][1] == 40.
    before = {n: len(w.scalars) for n, w in monitor.writers.items()}
    monitor.log_validation(quality(.8), 2000)
    assert before == {n: len(w.scalars) for n, w in monitor.writers.items()}
    with pytest.raises(ValueError): monitor.log_validation(quality(.8), 2000)
    baseline = (monitor.path/"pruned-step0-baseline.json").read_bytes()
    for step in range(2001, 5001):
        monitor.log_training(row(step), step)
        if step % 500 == 0: monitor.log_validation(quality(1-step/10000), step)
    assert [s for _, _, s in progress] == list(range(5001))
    assert progress[-1][1:] == (100., 5000)
    assert [s for _, _, s in metric(monitor, "01").scalars] == list(range(0, 5001, 500))
    assert metric(monitor, "03").scalars[-1][1] == 50.
    assert (monitor.path/"pruned-step0-baseline.json").read_bytes() == baseline
    with pytest.raises(ValueError): monitor.log_training(row(5001), 5001)


def test_fifteen_curves_seven_quiet_groups_and_missing_rms_denominator(tmp_path, histories):
    monitor = make(tmp_path, histories)
    for prefix in ("14", "15"):
        assert [v for _, v, _ in metric(monitor, prefix).scalars] == [50.]*5
    assert [v for _, v, _ in metric(monitor, "02").scalars] == [99.]*5
    tags = {tag for tag, _, _ in monitor.details.scalars}
    for name in monitoring.REGIONS:
        for key in ("output_rms", "residual_rms", "failed", "amplitude_failed", "both"):
            assert "quiet/"+name+"/"+key in tags
    monitor.close()
    first, second = histories
    for directory in histories:
        for path in directory.glob("development-step*.json"):
            report = json.loads(path.read_text())
            report["overview_window_metrics"]["by_source"]["source"]["active_teacher_energy"] = 0.
            path.write_text(json.dumps(report))
    zero = monitoring.ContinuationMonitor(tmp_path/"zero", parent_dir=second, original_dir=first, writer_factory=Writer)
    assert not any("14 Active" in name or "15 Whistling" in name for name in zero.writers)


@pytest.mark.parametrize("damage", ["missing", "duplicate", "nonfinite", "panel", "counter_reset", "teacher"])
def test_invalid_combined_history_rejected_before_events(tmp_path, histories, damage):
    _, second = histories
    if damage in ("missing", "duplicate", "nonfinite", "counter_reset"):
        rows = [row(s) for s in range(1001, 2001)]
        if damage == "missing": rows.pop(50)
        elif damage == "duplicate": rows[50]["step"] = rows[49]["step"]
        elif damage == "nonfinite": rows[50]["total"] = float("inf")
        else: rows[0]["audio_hours"] = .01
        (second/"train.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    else:
        report = quality()
        if damage == "panel": report["rows"][0]["source_id"] = "changed"
        else: report["quiet_regions"]["regions"]["all_quiet"]["teacher_rms"] *= 2
        (second/"development-step1500.json").write_text(json.dumps(report))
    with pytest.raises(ValueError): make(tmp_path, histories)
    assert not (tmp_path/"new-events").exists()


def test_new_updates_cannot_reset_counters_skip_steps_or_invent_quality(tmp_path, histories):
    monitor = make(tmp_path, histories)
    for key in ("audio_hours", "elapsed_seconds", "unique_sources"):
        record = row(2001); record[key] = 0
        with pytest.raises(ValueError, match="counter reset"): monitor.log_training(record, 2001)
    with pytest.raises(ValueError): monitor.log_training(row(2002), 2002)
    with pytest.raises(ValueError): monitor.log_training(row(2000), 2000)
    with pytest.raises(ValueError): monitor.log_validation(quality(), 2500)
    with pytest.raises(ValueError): monitor.log_validation(quality(), 3000)
    assert monitor.last_training_step == 2000
    monitor.close()
    with pytest.raises(ValueError): monitor.log_training(row(2001), 2001)


def test_original_files_logs_and_rng_state_remain_untouched(tmp_path, histories):
    python_rng, np_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    model = torch.nn.Linear(1, 1)
    # Snapshot after model creation; the monitor never receives model handles.
    torch_rng = torch.get_rng_state()
    weights = deepcopy(model.state_dict())
    before = [state(p) for p in histories]
    monitor = make(tmp_path, histories)
    monitor.assert_history_unchanged(); monitor.close()
    assert [state(p) for p in histories] == before
    assert random.getstate() == python_rng
    assert np.array_equal(np.random.get_state()[1], np_rng[1])
    assert torch.equal(torch.get_rng_state(), torch_rng)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, weights[name])
    (histories[1]/"train.jsonl").write_text("changed")
    with pytest.raises(RuntimeError): monitor.assert_history_unchanged()


def test_old_directories_and_wrong_targets_cannot_receive_new_history(tmp_path, histories):
    first, second = histories
    for old in histories:
        with pytest.raises(ValueError):
            monitoring.ContinuationMonitor(old/"new", parent_dir=second, original_dir=first, writer_factory=Writer)
    with pytest.raises(ValueError): make(tmp_path, histories, total_updates=2000)
    make(tmp_path, histories)
    with pytest.raises(FileExistsError): make(tmp_path, histories)


def test_real_tensorboard_history_has_single5000_target_and2000_raw_losses(tmp_path, histories):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    first, second = histories
    monitor = monitoring.ContinuationMonitor(tmp_path/"real", parent_dir=second, original_dir=first)
    monitor.close()
    progress_path = next(p for p in monitor.path.iterdir() if p.is_dir() and "13 Training" in p.name)
    progress = EventAccumulator(str(progress_path), size_guidance={"scalars": 0}).Reload().Scalars(monitoring.COMMON_TAG)
    assert [s.step for s in progress] == list(range(2001))
    assert progress[1000].value == 20 and progress[-1].value == 40
    details = EventAccumulator(str(monitor.path/(monitor.cut_name+" Details")), size_guidance={"scalars": 0}).Reload()
    assert len(details.Scalars("loss/teacher_waveform")) == 2000
    assert [s.step for s in details.Scalars("quality/mae")] == [0, 500, 1000, 1500, 2000]
