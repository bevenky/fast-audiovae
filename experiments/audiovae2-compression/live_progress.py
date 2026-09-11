"""Publish completed training steps and data readiness without running a model.

The shared-run dashboard must use --load_fast false --reload_multifile true
--reload_multifile_inactive_secs 86400 so both active event writers stay visible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time


def read_json(path):
    try:
        value = json.loads(Path(path).read_text())
    except (FileNotFoundError, OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def last_complete_record(path, tail_bytes=262144):
    """A trailing line is uncommitted until its newline has been written."""
    try:
        with Path(path).open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            start = max(0, size-tail_bytes)
            handle.seek(start)
            lines = handle.read().split(b"\n")
    except (FileNotFoundError, OSError):
        return None
    if start: lines = lines[1:]
    for line in reversed(lines[:-1]):
        try:
            value = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if (isinstance(value, dict) and type(value.get("step")) is int and value["step"] >= 0
                and type(value.get("unique_sources")) is int and value["unique_sources"] >= 0):
            return value
    return None


def latest_validation_step(run_dir, current_step):
    files = []
    for path in Path(run_dir).glob("development-step*.json"):
        match = re.fullmatch(r"development-step(\d+)\.json", path.name)
        if match and int(match[1]) <= current_step:
            files.append((int(match[1]), path))
    for step, path in sorted(files, reverse=True):
        report = read_json(path)
        if report is not None and isinstance(report.get("aggregate"), dict):
            return step
    return None


def progress_percent(step, target_step):
    if type(step) is not int or type(target_step) is not int or not 0 <= step <= target_step or target_step <= 0:
        raise ValueError("Progress needs an actual completed step within the target")
    return 100.0*step/target_step


def process_status(pid, run_dir):
    """Linux-only verification; an unavailable process view is not proof of exit."""
    if pid is None or not Path("/proc").is_dir(): return "unknown"
    directory = Path("/proc")/str(pid)
    try:
        stat = (directory/"stat").read_text()
        command = (directory/"cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return "ended"
    except (OSError, PermissionError):
        return "unknown"
    if stat.rsplit(")", 1)[-1].split()[0] in ("Z", "X"): return "ended"
    tokens = [part.decode(errors="replace") for part in command if part]
    if not any(Path(token).name == "resume_settings.py" for token in tokens): return "ended"
    try:
        output = Path(tokens[tokens.index("--out")+1]).resolve()
    except (ValueError, IndexError):
        return "ended"
    return "running" if output == Path(run_dir).resolve() else "ended"


def snapshot(run_dir, producer_progress, target_step, training_pid=None):
    run_dir = Path(run_dir)
    record = last_complete_record(run_dir/"train.jsonl")
    step = record["step"] if record is not None else None
    if step is not None: progress_percent(step, target_step)
    launch = read_json(run_dir/"launch.json") or {}
    config = launch.get("resume_identity", {})
    batch_size = config.get("effective_batch_size", 3)
    cadence = launch.get("validation_every_updates", 500)
    if type(batch_size) is not int or batch_size <= 0 or type(cadence) is not int or cadence <= 0:
        raise ValueError("Invalid run batch or validation metadata")
    producer = read_json(producer_progress) or {}
    prepared, approved = producer.get("sealed_source_prefix"), producer.get("approved_stop_index")
    if (type(prepared) is not int or type(approved) is not int or not 0 <= prepared <= approved):
        prepared = approved = None
    fresh = record.get("fresh_sources") if record is not None else None
    if type(fresh) is not int or fresh < 0: fresh = None
    latest = latest_validation_step(run_dir, step) if step is not None else None
    completed = read_json(run_dir/"completed.json")
    finished = (completed is not None and completed.get("step") == target_step
                and step == target_step and completed.get("frozen_state_preserved") is True)
    process = process_status(training_pid, run_dir)
    waiting = (step is not None and step < target_step and prepared is not None and fresh is not None
               and prepared < fresh+batch_size)
    if finished:
        status = "Completed the target; stopped for the quality decision."
    elif process == "ended":
        status = "Training process ended; no completed target receipt is available."
    elif waiting:
        status = "Waiting for the next sealed training-data shard."
    elif step is None:
        status = "Waiting for the first complete optimizer-update record."
    else:
        status = "Training or validation in progress; prepared data is available." if prepared is not None else "Training status available; producer readiness is unreported."
    if latest == target_step:
        next_validation = None
    elif latest is not None:
        next_validation = min(target_step, ((latest//cadence)+1)*cadence)
    elif step is not None:
        next_validation = min(target_step, ((step+cadence-1)//cadence)*cadence)
    else:
        next_validation = None
    return {"step":step, "target_step":target_step, "latest_validation_step":latest,
            "next_validation_step":next_validation, "prepared_sources":prepared,
            "remaining_to_prepare":None if approved is None else approved-prepared,
            "fresh_sources_consumed":fresh, "waiting":waiting, "status":status,
            "stop":finished or process == "ended"}


def publish(writer, state, previous=None):
    """Quality metrics belong to the trainer; this writer emits progress only."""
    if state == previous: return False
    step = state["step"]
    if step is not None and (previous is None or previous["step"] != step):
        writer.add_scalar(f"overview/Training progress to {state['target_step']} steps (%)",
                          progress_percent(step, state["target_step"]), step)
    def shown(value): return "not yet available" if value is None else str(value)
    text = (f"**{state['status']}**\n\n"
            f"Completed optimizer step: **{shown(step)} / {state['target_step']}**. "
            f"Most recent measured validation: **{shown(state['latest_validation_step'])}**. "
            f"Next validation milestone: **{shown(state['next_validation_step'])}**.\n\n"
            f"Fresh sources prepared: **{shown(state['prepared_sources'])}**; "
            f"remaining to prepare: **{shown(state['remaining_to_prepare'])}**; "
            f"already consumed: **{shown(state['fresh_sources_consumed'])}**. "
            "The quality curves change only when a validation measurement finishes.")
    writer.add_text("Monitor/Live status", text, step if step is not None else 0)
    writer.flush()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--producer-progress", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--target-step", type=int, default=5000)
    parser.add_argument("--training-pid", type=int)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.target_step <= 0 or (args.training_pid is not None and args.training_pid <= 0):
        parser.error("Target step and optional PID must be positive")
    # Lazy logging import: no project model, inference, data or training modules.
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(args.logdir), filename_suffix=".live-progress")
    previous = None
    try:
        while True:
            state = snapshot(args.run_dir, args.producer_progress, args.target_step, args.training_pid)
            publish(writer, state, previous)
            previous = state
            if args.once or state["stop"]: break
            time.sleep(10)
    finally:
        writer.flush(); writer.close()


if __name__ == "__main__": main()
