"""CPU-only observer lineage, contiguous histories and finite-plan retention."""
import copy
import fcntl
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest
import torch

from audiovae_student import checkpoint_retention as retention


SCRIPT = Path(__file__).resolve().parents[4] / "convnext-preflight/run_recipe_v2_monitor.py"
spec = importlib.util.spec_from_file_location("recipe_v2_observer", SCRIPT)
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def fixture(tmp_path):
    run = tmp_path / "training-runs/decoder-recipe-v2"
    child = tmp_path / "training-runs/decoder-recipe-v2-expressive"
    run.mkdir(parents=True)
    child.mkdir()
    (run.parent / ("." + run.name + ".runner.lock")).touch()
    parent_identity = {"batch_size": 2, "actual_plan": {"metadata": {"windows": 3}},
        "window_identity": "parent-windows", "recipe": {"total_steps": 2, "waveform_weight": 1.},
        "model_config": {"channels": 64}, "context_frames": 29, "heldout_metadata": {"clip": "hi"},
        "data": {"teacher_checkpoint_sha256": "teacher", "teacher_state_sha256": "state",
                 "heldout": {"hash": "same-original-panel"}}}
    (run / "run.json").write_text(json.dumps(parent_identity))
    rows = [{"step": i + 1, "teacher_waveform": 1., "teacher_mel": .5,
             "seconds_in_training_loop": (i + 1) * 2.} for i in range(2)]
    write_rows(run / "metrics.jsonl", rows)
    records, chain = [], retention.identity_digest([])
    for index in range(2):
        body = {"step": index + 1, "cursor": min((index + 1) * 2, 3),
                "window_identity": "parent-windows", "previous_sha256": chain}
        chain = retention.identity_digest(body)
        records.append({**body, "sha256": chain})
    write_rows(run / "exposure.jsonl", records)
    payload = {"format_version": "recipe_v2_pilot", "identity": parent_identity,
        "engine": {"step": 2, "calibration": {"frozen": True}},
        "sampler": {"cursor": 3, "identity_sha256": "parent-windows"},
        "journal_sha256": chain, "metrics_sha256": retention.identity_digest(rows),
        "latest_metrics": rows[-1], "seconds": 4.}
    torch.save(payload, run / "latest.pt")
    parent = {"step": 2, "batch_size": 2, "checkpoint_sha256": observer.file_seal(run / "latest.pt")["sha256"],
              "run_identity_sha256": retention.identity_digest(parent_identity),
              "journal_sha256": chain, "metrics_sha256": retention.identity_digest(rows)}
    identity = {"kind": retention.CONTINUATION_FORMAT, "parent": parent, "global_start_step": 2,
        "global_total_steps": 4, "segment_window_count": 3, "batch_size": 2,
        "window_identity": "child-windows", "evaluation_steps": [4],
        "calibration_sha256": retention.identity_digest(payload["engine"]["calibration"]),
        "recipe": {**parent_identity["recipe"], "total_steps": 4},
        **{k: parent_identity[k] for k in ("model_config", "context_frames", "heldout_metadata", "data")}}
    (child / "run.json").write_text(json.dumps(identity))
    child_rows = [{"step": i + 3, "segment_step": i + 1, "consumed_windows": min((i + 1) * 2, 3),
                   "teacher_waveform": .75, "teacher_mel": .25, "unique_scored_hours": i + 3.,
                   "seconds_in_training_loop": i + 1.} for i in range(2)]
    write_rows(child / "metrics.jsonl", child_rows)
    child_payload = {"format_version": retention.CONTINUATION_FORMAT, "identity": identity,
        "engine": {"step": 4}, "sampler": {"cursor": 3, "identity_sha256": "child-windows"},
        "journal_sha256": "child-chain", "metrics_sha256": retention.identity_digest(child_rows)}
    torch.save(child_payload, child / "latest.pt")
    return run, child, identity, child_payload


def empty_state():
    return {"offset": 0, "step": 0, "evaluations": {}, "checkpoints": [], "completed_thresholds": []}


def bind(run, child):
    state = empty_state()
    observer.consume_metrics(run / "metrics.jsonl", state, lambda *_: None)
    identity, guard = observer.bind_continuation(run, child, state)
    return state, identity, guard


def test_joined_history_uses_global_steps_separate_offsets_and_cumulative_seconds(tmp_path):
    run, child, _, _ = fixture(tmp_path)
    state, emitted = empty_state(), []
    emit = lambda step, values: emitted.append((step, values))
    observer.consume_metrics(run / "metrics.jsonl", state, emit)
    parent_offset = state["offset"]
    identity, guard = observer.bind_continuation(run, child, state)
    try:
        observer.consume_metrics(child / "metrics.jsonl", state, emit, identity=identity)
        assert state["offset"] == parent_offset
        assert state["continuation"]["offset"] == (child / "metrics.jsonl").stat().st_size
        assert [step for step, _ in emitted] == [1, 2, 3, 4]
        assert [v["progress/seconds_in_training_loop"] for _, v in emitted] == [2., 4., 5., 6.]
        assert emitted[-1][1]["progress/unique_scored_hours"] == 4.
        observer.consume_metrics(child / "metrics.jsonl", state, emit, identity=identity)
        assert len(emitted) == 4
        _, resumed_guard = observer.bind_continuation(run, child, state)
        resumed_guard.close()
    finally:
        guard.close()


@pytest.mark.parametrize("damage", ["parent_hash", "teacher", "panel", "recipe", "count", "calendar"])
def test_child_identity_must_match_actual_parent_and_evaluation_contract(tmp_path, damage):
    run, child, identity, _ = fixture(tmp_path)
    if damage == "parent_hash":
        identity["parent"]["checkpoint_sha256"] = "wrong"
    elif damage == "teacher":
        identity["data"]["teacher_state_sha256"] = "new-teacher"
    elif damage == "panel":
        identity["heldout_metadata"] = {"new": "test-contamination"}
    elif damage == "recipe":
        identity["recipe"]["waveform_weight"] = 9.
    elif damage == "count":
        identity["segment_window_count"] = 5
    else:
        identity["evaluation_steps"] = [2]
    (child / "run.json").write_text(json.dumps(identity))
    with pytest.raises(ValueError):
        bind(run, child)


def test_running_or_inflight_parent_cannot_be_attached(tmp_path):
    run, child, _, _ = fixture(tmp_path)
    with (run.parent / ("." + run.name + ".runner.lock")).open("rb") as guard:
        fcntl.flock(guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            bind(run, child)
    (run / "inflight.json").write_text("{}")
    with pytest.raises(ValueError, match="in-flight"):
        bind(run, child)


@pytest.mark.parametrize("damage", ["metrics", "identity", "rebind"])
def test_bound_files_cannot_change(tmp_path, damage):
    run, child, identity, _ = fixture(tmp_path)
    state, _, guard = bind(run, child)
    try:
        if damage == "metrics":
            with (run / "metrics.jsonl").open("a") as handle:
                handle.write('{"step":3}\n')
        else:
            identity["data"]["source_corpus"] = "different"
            (child / "run.json").write_text(json.dumps(identity))
        if damage == "rebind":
            with pytest.raises(ValueError, match="Previously bound"):
                observer.bind_continuation(run, child, state)
        else:
            with pytest.raises(ValueError, match="changed"):
                observer.verify_sealed_parent(run, child, state["continuation"])
    finally:
        guard.close()


@pytest.mark.parametrize("damage", ["gap", "repeat", "cursor", "segment", "truncated"])
def test_child_metrics_reject_gaps_repeats_and_wrong_segment_exposure(tmp_path, damage):
    run, child, _, _ = fixture(tmp_path)
    state, identity, guard = bind(run, child)
    try:
        rows = observer.read_complete_rows(child / "metrics.jsonl")
        if damage == "gap": rows[0]["step"] = 4
        if damage == "repeat": rows[0]["step"] = 2
        if damage == "cursor": rows[-1]["consumed_windows"] = 4
        if damage == "segment": rows[0]["segment_step"] = 3
        if damage == "truncated": state["continuation"]["offset"] = 99999
        write_rows(child / "metrics.jsonl", rows)
        with pytest.raises(ValueError):
            observer.consume_metrics(child / "metrics.jsonl", state, lambda *_: None, identity=identity)
    finally:
        guard.close()


def test_partial_json_line_is_not_consumed_until_complete(tmp_path):
    run, child, _, _ = fixture(tmp_path)
    state, identity, guard = bind(run, child)
    try:
        raw = (child / "metrics.jsonl").read_bytes()
        (child / "metrics.jsonl").write_bytes(raw[:-1])
        observer.consume_metrics(child / "metrics.jsonl", state, lambda *_: None, identity=identity)
        assert state["step"] == 3
        (child / "metrics.jsonl").write_bytes(raw)
        observer.consume_metrics(child / "metrics.jsonl", state, lambda *_: None, identity=identity)
        assert state["step"] == 4
    finally:
        guard.close()


def test_partial_final_retention_and_custom_threshold_recovery(tmp_path, monkeypatch):
    _, child, identity, payload = fixture(tmp_path)
    monkeypatch.setattr(retention.shutil, "disk_usage", lambda _: shutil._ntuple_diskusage(100*1024**3, 0, 100*1024**3))
    receipt = retention.snapshot_latest(child / "latest.pt", tmp_path / "kept", 4, expected_identity=identity)
    assert (receipt["step"], receipt["segment_step"], receipt["cursor"]) == (4, 2, 3)
    assert receipt["parent"] == identity["parent"]
    assert receipt["run_identity_sha256"] == retention.identity_digest(identity)
    state = empty_state()
    retention.reconcile_retention(tmp_path / "kept", state)
    assert state["completed_thresholds"] == [4]
    assert state["checkpoints"] == [receipt]
    for step, cursor in ((2, 0), (3, 2), (4, 3)):
        current = copy.deepcopy(payload)
        current["engine"]["step"], current["sampler"]["cursor"] = step, cursor
        assert retention.checkpoint_position(current)["cursor"] == cursor
    payload["sampler"]["cursor"] = 4
    with pytest.raises(ValueError, match="cursor"):
        retention.checkpoint_position(payload)
    with pytest.raises(ValueError, match="identity"):
        retention.snapshot_latest(child / "latest.pt", tmp_path / "other", 4, expected_identity={})


def test_continuation_retention_keeps_only_two_owned_snapshots(tmp_path, monkeypatch):
    _, child, identity, payload = fixture(tmp_path)
    monkeypatch.setattr(retention.shutil, "disk_usage", lambda _: shutil._ntuple_diskusage(100*1024**3, 0, 100*1024**3))
    destination = tmp_path / "kept"
    for step, cursor in ((2, 0), (3, 2), (4, 3)):
        payload["engine"]["step"], payload["sampler"]["cursor"] = step, cursor
        # Atomic replacement must preserve already retained checkpoint inodes.
        replacement = child / "next.pt"
        torch.save(payload, replacement)
        replacement.replace(child / "latest.pt")
        retention.snapshot_latest(child / "latest.pt", destination, step, expected_identity=identity)
    state = empty_state()
    retention.reconcile_retention(destination, state)
    assert [r["step"] for r in state["checkpoints"]] == [3, 4]
    assert len(list(destination.glob("checkpoint-step*.pt"))) == 2
    assert len(list(destination.glob("checkpoint-step*.json"))) == 3
    assert (child / "latest.pt").is_file()


def test_main_resumes_same_tensorboard_run_and_includes_child_final_evaluation(tmp_path, monkeypatch):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    run, child, _, _ = fixture(tmp_path)
    evaluation = {"step": 4, "gate": {"passed": False}, "groups": {}, "rows": [
        {"samples": 100, "student_peak_abs": 1.1, "student_clipped_samples": 2}]}
    (child / "evaluation-step000004.json").write_text(json.dumps(evaluation))
    monkeypatch.setattr(retention.shutil, "disk_usage", lambda _: shutil._ntuple_diskusage(100*1024**3, 0, 100*1024**3))
    monkeypatch.setattr(observer.signal, "signal", lambda *_: None)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--base", str(tmp_path), "--once"])
    observer.main()  # Persist the preexisting parent history first.
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--base", str(tmp_path), "--once", "--continuation-run", str(child)])
    observer.main()
    observer.main()  # No duplicate steps after the observer restarts.
    output = tmp_path / "remediation/monitoring"
    state = json.loads((output / "state.json").read_text())
    assert state["step"] == 4
    assert state["evaluations"][str(child / "evaluation-step000004.json")] == 4
    accumulator = EventAccumulator(str(output / "tensorboard/decoder-recipe-v2")).Reload()
    assert [e.step for e in accumulator.Scalars("loss/teacher_waveform")] == [1, 2, 3, 4]
    assert [e.value for e in accumulator.Scalars("progress/seconds_in_training_loop")] == [2., 4., 5., 6.]
    assert accumulator.Scalars("quality/all/peak_abs_max")[-1].value == pytest.approx(1.1)
    assert len(list((output / "tensorboard").iterdir())) == 1
