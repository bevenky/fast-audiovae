"""Read-only training observer with an isolated dashboard and bounded retention."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import time

from audiovae_student.monitoring import training_scalars, evaluation_scalars


from audiovae_student.checkpoint_retention import (CONTINUATION_FORMAT, atomic_json,
    checkpoint_position, identity_digest, snapshot_latest, reconcile_retention)


def file_seal(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Sealed run files must be regular files")
    before = file_stamp(path)
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if file_stamp(path) != before:
        raise ValueError("Run file changed while binding the observer")
    return {"stamp": before, "sha256": digest}


def file_stamp(path):
    stat = path.stat()
    return {"device": stat.st_dev, "inode": stat.st_ino, "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns}


def read_complete_rows(path):
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("Stopped parent has an incomplete journal record")
    return [json.loads(line) for line in raw.splitlines()]


def bind_continuation(run, child, state):
    """Pin the stopped parent and a child identity before joining their charts.

    The returned shared lock must remain open for the observer's lifetime. It
    cooperates with the runners' locks and excludes original-parent resumption.
    """
    import torch
    torch.set_num_threads(1)
    run, child = run.resolve(strict=True), child.resolve(strict=True)
    if child == run or child in run.parents or run in child.parents:
        raise ValueError("Continuation must use a separate run directory")
    lock_path = run.parent / ("." + run.name + ".runner.lock")
    if lock_path.is_symlink() or not lock_path.is_file():
        raise ValueError("Stopped parent runner lock is missing or unsafe")
    guard = lock_path.open("rb")
    try:
        fcntl.flock(guard.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        if (run / "inflight.json").exists():
            raise ValueError("Parent still has uncertain in-flight exposure")
        seals = {name: file_seal(run / name) for name in
                 ("run.json", "latest.pt", "metrics.jsonl", "exposure.jsonl")}
        child_seal = file_seal(child / "run.json")
        identity = json.loads((child / "run.json").read_text())
        parent_identity = json.loads((run / "run.json").read_text())
        payload = torch.load(run / "latest.pt", map_location="cpu", weights_only=True)
        position = checkpoint_position(payload, expected_identity=parent_identity)
        if position["format_version"] != "recipe_v2_pilot":
            raise ValueError("Only the original pilot can be the observer's parent")
        rows = read_complete_rows(run / "metrics.jsonl")
        records = read_complete_rows(run / "exposure.jsonl")
        step, batch = position["step"], parent_identity["batch_size"]
        chain = identity_digest([])
        for index, record in enumerate(records):
            body = {k: v for k, v in record.items() if k != "sha256"}
            cursor = (index + 1) * batch
            if position["segment_window_count"] is not None:
                cursor = min(cursor, position["segment_window_count"])
            if (record.get("step") != index + 1 or record.get("cursor") != cursor
                    or record.get("previous_sha256") != chain
                    or record.get("sha256") != identity_digest(body)
                    or record.get("window_identity") != parent_identity.get("window_identity")):
                raise ValueError("Parent exposure journal changed")
            chain = record["sha256"]
        if (len(records) != step or len(rows) != step
                or any(row.get("step") != i + 1 for i, row in enumerate(rows))
                or chain != payload["journal_sha256"]
                or identity_digest(rows) != payload["metrics_sha256"]
                or (rows and rows[-1] != payload["latest_metrics"])):
            raise ValueError("Parent metrics or exposure extend past its committed checkpoint")
        parent = {"checkpoint_sha256": seals["latest.pt"]["sha256"],
                  "run_identity_sha256": identity_digest(parent_identity),
                  "journal_sha256": chain, "metrics_sha256": identity_digest(rows),
                  "step": step, "batch_size": batch}
        if identity.get("parent") != parent or identity.get("global_start_step") != step:
            raise ValueError("Continuation does not bind to this stopped parent")
        checkpoint_position({"format_version": CONTINUATION_FORMAT, "identity": identity,
            "engine": {"step": step}, "sampler": {"cursor": 0,
                "identity_sha256": identity.get("window_identity")}})
        expected_recipe = {**parent_identity["recipe"], "total_steps": identity["global_total_steps"]}
        if (identity["recipe"] != expected_recipe
                or any(identity.get(k) != parent_identity.get(k) for k in
                       ("model_config", "batch_size", "context_frames", "heldout_metadata"))
                or any(identity["data"].get(k) != parent_identity["data"].get(k) for k in
                       ("teacher_checkpoint_sha256", "teacher_state_sha256", "heldout"))
                or identity.get("calibration_sha256") != identity_digest(payload["engine"]["calibration"])):
            raise ValueError("Continuation changed the inherited model, teacher, recipe or heldout panel")
        calendar = identity["evaluation_steps"]
        if (calendar != sorted(set(calendar)) or any(type(s) is not int or not step < s <=
                identity["global_total_steps"] for s in calendar)):
            raise ValueError("Continuation evaluation calendar is invalid")
        binding = {"run": str(child), "identity_sha256": identity_digest(identity),
                   "parent_step": step, "global_total_steps": identity["global_total_steps"],
                   "parent_seconds": payload["seconds"], "parent_files": seals,
                   "identity_file": child_seal}
        previous = state.get("continuation")
        if previous is not None:
            if any(previous.get(k) != v for k, v in binding.items()):
                raise ValueError("Previously bound continuation or stopped parent changed")
        else:
            if state["step"] != step or state["offset"] != seals["metrics.jsonl"]["stamp"]["bytes"]:
                raise ValueError("Drain the complete parent metrics before attaching continuation")
            state["continuation"] = {**binding, "offset": 0}
        verify_sealed_parent(run, child, state["continuation"])
        return identity, guard
    except BaseException:
        guard.close()
        raise


def verify_sealed_parent(run, child, binding):
    if (run / "inflight.json").exists():
        raise ValueError("Stopped parent acquired new in-flight exposure")
    for name, seal in binding["parent_files"].items():
        path = run / name
        if path.is_symlink() or file_stamp(path) != seal["stamp"]:
            raise ValueError("Stopped parent changed after continuation attachment")
    if (child / "run.json").is_symlink() or file_stamp(child / "run.json") != binding["identity_file"]["stamp"]:
        raise ValueError("Continuation run identity changed after attachment")


def consume_metrics(path, state, emit, *, identity=None):
    """Maintain independent byte cursors and one contiguous global update axis."""
    segment = state["continuation"] if identity is not None else state
    if not path.exists() and identity is not None and segment["offset"] == 0:
        return
    if path.stat().st_size < segment["offset"]:
        raise ValueError("Metrics file was truncated behind the observer")
    with path.open("rb") as handle:
        handle.seek(segment["offset"])
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line or not line.endswith(b"\n"):
                segment["offset"] = start
                break
            row = json.loads(line)
            if type(row.get("step")) is not int or row["step"] != state["step"] + 1:
                raise ValueError("Metrics sequence changed or skipped an update")
            values = training_scalars(row)
            if identity is not None:
                step = row["step"] - identity["global_start_step"]
                cursor = min(step * identity["batch_size"], identity["segment_window_count"])
                if (row["step"] > identity["global_total_steps"] or step < 1
                        or row.get("segment_step") != step or row.get("consumed_windows") != cursor):
                    raise ValueError("Continuation metric global step or segment cursor changed")
                if "progress/seconds_in_training_loop" in values:
                    values["progress/seconds_in_training_loop"] += segment["parent_seconds"]
            emit(row["step"], values)
            state["step"], segment["offset"] = row["step"], handle.tell()


def evaluation_sources(base, run, child=None, identity=None):
    sources = [(p, None) for p in run.glob("evaluation-step*.json")]
    sources += [(p, None) for p in (base / "audit-step1000").glob("evaluation-step*.json")]
    sources += [(p, None) for p in (base / "remediation/evaluations").glob("evaluation-step*.json")]
    if identity is not None:
        sources += [(p, identity["evaluation_steps"]) for p in child.glob("evaluation-step*.json")]
    return sorted(sources, key=lambda item: str(item[0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--continuation-run", type=Path,
                        help="Optional continuation directory; original dashboard history stays visible")
    args = parser.parse_args()
    base = args.base.resolve(strict=True)
    run = base / "training-runs/decoder-recipe-v2"
    out = base / "remediation/monitoring"
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "observer.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = out / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "offset": 0, "step": 0, "evaluations": {}, "checkpoints": [], "completed_thresholds": []}
    child = args.continuation_run.resolve() if args.continuation_run else None
    if state.get("continuation") and (child is None or str(child) != state["continuation"]["run"]):
        raise ValueError("A bound observer requires its original --continuation-run")
    thresholds = {3000, 5000, 10000}
    if state.get("continuation"):
        thresholds.add(state["continuation"]["global_total_steps"])
    reconcile_retention(out / "checkpoints", state, thresholds=thresholds)
    atomic_json(state_path, state)
    stop_requested = False
    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    writer = EventFileWriter(str(out / "tensorboard/decoder-recipe-v2"), max_queue_size=10, flush_secs=2)

    def emit(step, values):
        writer.add_event(Event(wall_time=time.time(), step=step,
            summary=Summary(value=[Summary.Value(tag=key, simple_value=float(value)) for key, value in values.items()])))

    identity, parent_guard = None, None
    try:
        while True:
            if state.get("continuation") and not (child / "run.json").is_file():
                raise ValueError("Bound continuation run identity disappeared")
            if not state.get("continuation"):
                consume_metrics(run / "metrics.jsonl", state, emit)
            if child is not None and identity is None and (child / "run.json").exists():
                identity, parent_guard = bind_continuation(run, child, state)
                thresholds.add(identity["global_total_steps"])
            if identity is not None:
                verify_sealed_parent(run, child, state["continuation"])
                consume_metrics(child / "metrics.jsonl", state, emit, identity=identity)
            for path, calendar in evaluation_sources(base, run, child, identity):
                name = str(path)
                if name in state["evaluations"]:
                    continue
                evaluation = json.loads(path.read_text())
                if calendar is not None and (evaluation["step"] not in calendar or
                        path.name != f"evaluation-step{evaluation['step']:06d}.json"):
                    raise ValueError("Continuation evaluation is not in its bound calendar")
                if evaluation["step"] > state["step"]:
                    continue
                emit(evaluation["step"], evaluation_scalars(evaluation))
                state["evaluations"][name] = evaluation["step"]
            for threshold in sorted(thresholds):
                if state["step"] < threshold or threshold in state["completed_thresholds"]:
                    continue
                latest = (child if identity is not None else run) / "latest.pt"
                if not latest.exists():
                    continue
                result = snapshot_latest(latest, out / "checkpoints", threshold,
                                         expected_identity=identity)
                state["capture_status"] = result
                if result["state"] == "retained":
                    state["checkpoints"].append(result)
                    state["completed_thresholds"] = sorted(set(state["completed_thresholds"]) |
                        {target for target in thresholds if target <= result["step"]})
                    # Remove only older snapshots created and listed by this observer.
                    while len(state["checkpoints"]) > 2:
                        old = state["checkpoints"].pop(0)
                        path = Path(old["path"])
                        if path.parent != out / "checkpoints" or path.name == "latest.pt":
                            raise ValueError("Unsafe retention path")
                        path.unlink(missing_ok=True)
                        # Receipts remain as a record of the retention decision.
                break
            writer.flush()
            state["updated_unix"] = time.time()
            state["training_modified"] = False
            atomic_json(state_path, state)
            if args.once or stop_requested:
                break
            time.sleep(5)
    finally:
        writer.close()
        if parent_guard is not None:
            parent_guard.close()


if __name__ == "__main__":
    main()
