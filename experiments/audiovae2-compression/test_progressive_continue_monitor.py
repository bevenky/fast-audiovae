"""Read-only history replay and continuous2000-step chart contracts."""
from copy import deepcopy
import hashlib
import json
import random

import numpy as np
import pytest
import torch

import progressive_continue_monitor as monitor_api
from test_progressive_monitor import Writer, report, metric


def row(step):
    return {"step": step, "total": .125, "waveform": .02, "mel": .5,
            "feature": .1, "unique_sources": 12*step, "audio_hours": .001*step,
            "elapsed_seconds": .01*step, "step_seconds": .01}


def quality(scale=1):
    result = report()
    for key in monitor_api.original.ERRORS:
        result["aggregate"][key] *= scale
    result["overview_window_metrics"] = {"by_source": {"source": {
        "active_teacher_energy": 4., "active_student_energy": 1.}}}
    return result


@pytest.fixture
def parent(tmp_path):
    directory = tmp_path/"old-cut"
    directory.mkdir()
    for step, scale in ((0, 1.), (500, .8), (1000, .5)):
        (directory/f"development-step{step}.json").write_text(json.dumps(quality(scale)))
    (directory/"train.jsonl").write_text("".join(json.dumps(row(step))+"\n" for step in range(1, 1001)))
    (directory/"full-width-copy.json").write_text(json.dumps({"passed": True, "waveform_error": 0.}))
    (directory/"old-events").write_bytes(b"do not overwrite")
    return directory


def make(tmp_path, parent, **kwargs):
    return monitor_api.ContinuationMonitor(tmp_path/"new-chart", parent_dir=parent,
        source_metadata={"source": {"verified_source_labels": ["human_whistling_source_description"]}},
        writer_factory=Writer, **kwargs)


def state(directory):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()}


def test_replay_keeps_pruned_baseline_old_files_and_all_original_quality_points(tmp_path, parent):
    before = state(parent)
    monitor = make(tmp_path, parent)
    assert state(parent) == before
    assert monitor.baseline == quality()["aggregate"]
    assert [(v, s) for _, v, s in metric(monitor, "03").scalars] == [(0., 0), (pytest.approx(20.), 500), (50., 1000)]
    assert [s for _, _, s in metric(monitor, "01").scalars] == [0, 500, 1000]
    assert len([x for x in monitor.details.scalars if x[0] == "loss/teacher_waveform"]) == 1000
    assert monitor.history_identity["new_training_updates_from_replay"] == 0
    assert monitor.history_identity["new_audio_samples_from_replay"] == 0
    assert monitor.history_identity["prior_unique_sources"] == 12000
    assert sum(len(w.layouts) for w in monitor.writers.values()) == 1
    assert len(monitor.writers) == 16  # Details plus all15 measured curves.
    assert "zero training updates" in monitor.details.text[-1][1]


def test_progress_recomputed_consistently_and_logged_once_per_step(tmp_path, parent):
    monitor = make(tmp_path, parent)
    values = metric(monitor, "13").scalars
    assert len(values) == 1001
    assert [s for _, _, s in values] == list(range(1001))
    assert values[0][1] == 0 and values[500][1] == 25 and values[1000][1] == 50
    monitor.log_training(row(1001), 1001)
    assert values[-1][1:] == (50.05, 1001)
    assert len(metric(monitor, "01").scalars) == 3


def test_new_quality_keeps_original_zero_and_duplicate_resume_has_no_chart_point(tmp_path, parent):
    monitor = make(tmp_path, parent)
    baseline_bytes = (monitor.path/"pruned-step0-baseline.json").read_bytes()
    count = {n: len(w.scalars) for n, w in monitor.writers.items()}
    monitor.log_validation(quality(.5), 1000)
    assert count == {n: len(w.scalars) for n, w in monitor.writers.items()}
    with pytest.raises(ValueError):
        monitor.log_validation(quality(.5), 1000)
    for step in range(1001, 2001):
        monitor.log_training(row(step), step)
        if step == 1500: monitor.log_validation(quality(.4), step)
        if step == 2000: monitor.log_validation(quality(.3), step)
    assert [s for _, _, s in metric(monitor, "03").scalars] == [0, 500, 1000, 1500, 2000]
    assert metric(monitor, "03").scalars[-1][1] == 70.
    assert metric(monitor, "13").scalars[-1][1:] == (100., 2000)
    assert (monitor.path/"pruned-step0-baseline.json").read_bytes() == baseline_bytes
    with pytest.raises(ValueError): monitor.log_training(row(2001), 2001)


def test_all_quiet_regions_and_measured_rms_ratios_remain_raw(tmp_path, parent):
    monitor = make(tmp_path, parent)
    for prefix in ("14", "15"):
        assert [(v, s) for _, v, s in metric(monitor, prefix).scalars] == [(50., 0), (50., 500), (50., 1000)]
    tags = {tag for tag, _, _ in monitor.details.scalars}
    for name in monitor_api.REGIONS:
        for key in ("residual_rms", "output_rms", "amplitude_failed", "residual_only", "both"):
            assert "quiet/"+name+"/"+key in tags
    assert "RMS ratios, not least-squares" in monitor.details.text[0][1]


def test_missing_whistle_and_zero_teacher_energy_are_unmeasured_not_invented(tmp_path, parent):
    for step in (0, 500, 1000):
        q = quality()
        q["overview_window_metrics"]["by_source"]["source"]["active_teacher_energy"] = 0.
        (parent/f"development-step{step}.json").write_text(json.dumps(q))
    monitor = monitor_api.ContinuationMonitor(tmp_path/"new", parent_dir=parent, writer_factory=Writer)
    assert not any("14 Active" in name or "15 Whistling" in name for name in monitor.writers)
    values = json.loads((monitor.path/"latest-validation.json").read_text())["amplitude_ratios"]
    assert values == {"active_rms_ratio": None, "whistle_rms_ratio": None}


@pytest.mark.parametrize("damage", ["missing_step", "duplicate_step", "nonfinite", "changed_panel", "exact_copy"])
def test_bad_history_fails_before_creating_any_events(tmp_path, parent, damage):
    if damage in ("missing_step", "duplicate_step", "nonfinite"):
        rows = [row(s) for s in range(1, 1001)]
        if damage == "missing_step": rows.pop(23)
        elif damage == "duplicate_step": rows[23]["step"] = 23
        else: rows[23]["total"] = float("nan")
        (parent/"train.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    else:
        q = quality()
        if damage == "changed_panel": q["rows"][0]["source_id"] = "different"
        else:
            for key in ("mae", "mel", "group_mse"): q["aggregate"][key] = 0.
        (parent/f"development-step{500 if damage == 'changed_panel' else 0}.json").write_text(json.dumps(q))
    with pytest.raises(ValueError): make(tmp_path, parent)
    assert not (tmp_path/"new-chart").exists()


def test_replay_does_not_touch_rng_parameters_gradients_or_optimizer(tmp_path, parent):
    model = torch.nn.Linear(2, 3)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).square().sum().backward()
    optimizer.step()
    params = {n: (p.detach().clone(), p.grad.clone(), p.requires_grad) for n, p in model.named_parameters()}
    optim = deepcopy(optimizer.state_dict())
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    monitor = make(tmp_path, parent)
    monitor.close()
    assert random.getstate() == python_rng
    assert np.array_equal(np.random.get_state()[1], numpy_rng[1])
    assert torch.equal(torch.get_rng_state(), torch_rng)
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, params[name][0], rtol=0, atol=0)
        torch.testing.assert_close(p.grad, params[name][1], rtol=0, atol=0)
        assert p.requires_grad == params[name][2]
    for key, entries in optimizer.state_dict()["state"].items():
        for name, value in entries.items():
            torch.testing.assert_close(value, optim["state"][key][name], rtol=0, atol=0)


def test_duplicate_history_target_alias_and_future_quality_are_rejected(tmp_path, parent):
    with pytest.raises(ValueError):
        monitor_api.ContinuationMonitor(parent/"events", parent_dir=parent, writer_factory=Writer)
    with pytest.raises(ValueError): make(tmp_path, parent, total_updates=1000)
    monitor = make(tmp_path, parent)
    with pytest.raises(FileExistsError): make(tmp_path, parent)
    with pytest.raises(ValueError): monitor.log_training(row(1000), 1000)
    with pytest.raises(ValueError): monitor.log_training(row(1002), 1002)
    with pytest.raises(ValueError): monitor.log_validation(quality(), 1500)
    (parent/"train.jsonl").write_text("changed")
    with pytest.raises(RuntimeError): monitor.assert_history_unchanged()
    monitor.close()
    with pytest.raises(ValueError): monitor.log_training(row(1001), 1001)


def test_real_events_have_one_target_and_original_quality_history(tmp_path, parent):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    monitor = monitor_api.ContinuationMonitor(tmp_path/"real-events", parent_dir=parent)
    monitor.log_training(row(1001), 1001)
    monitor.close()
    progress_path = next(p for p in monitor.path.iterdir() if p.is_dir() and "13 Training" in p.name)
    progress = EventAccumulator(str(progress_path), size_guidance={"scalars": 0}).Reload().Scalars(monitor_api.COMMON_TAG)
    assert [x.step for x in progress] == list(range(1002))
    assert progress[1000].value == 50.
    details = EventAccumulator(str(monitor.path/(monitor.cut_name+" Details")), size_guidance={"scalars": 0}).Reload()
    assert len(details.Scalars("loss/teacher_waveform")) == 1001
    assert [x.step for x in details.Scalars("quality/mae")] == [0, 500, 1000]
