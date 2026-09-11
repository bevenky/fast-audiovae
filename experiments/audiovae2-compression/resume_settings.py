"""Resume the selected 1,000-step student with sealed, previously unseen data.

The current reconstruction losses, coefficients, teacher, model and AdamW state
are retained. This entry point stops at the requested decision point, at most
5,000 total updates. It neither ranks settings nor consumes the later reserve.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import torch

import continue_settings as continuation

screen = continuation.screen
base = screen.base
VERSION = "audiovae2_settings_resume_v1"
ANCHOR_STEP = 1000
ANCHOR_SOURCES = 3000
ANCHOR_SHA256 = "bfecd759eb58db1eeeca3dd693660b26be91bd69a93e3457b9d91523c5de0573"
MAX_TARGET_STEP = 5000
AUTHORIZED_FRESH_SOURCES = 12000


def validate_anchor(payload, receipt, completed, launch, original_arm, fitting, checksum):
    """Authenticate the selected completed state, not merely its reported score."""
    if checksum != ANCHOR_SHA256 or receipt.get("checkpoint_sha256") != checksum:
        raise ValueError("The selected checkpoint is not the approved step-1000 anchor")
    if (payload.get("format") != screen.VERSION or payload.get("step") != ANCHOR_STEP
            or payload.get("identity") != original_arm
            or original_arm.get("arm") != "current_lr3e-5"
            or original_arm.get("definition") != "current"):
        raise ValueError("Anchor model or selected settings changed")
    continuation.validate_source_cursor(payload, fitting)
    for key in ("step", "fit_cursor", "checkpoint_sha256", "arm", "common_quality",
                "selected_screen_checkpoint_sha256", "optimizer_and_rng_saved", "frozen_state_preserved"):
        if receipt.get(key) != completed.get(key):
            raise ValueError("Anchor receipt and completed run disagree: "+key)
    if (receipt.get("step") != ANCHOR_STEP or receipt.get("fit_cursor") != ANCHOR_SOURCES
            or receipt.get("arm") != original_arm["arm"]
            or receipt.get("optimizer_and_rng_saved") is not True
            or receipt.get("frozen_state_preserved") is not True):
        raise ValueError("Anchor completion is missing preservation or source accounting")
    if (launch.get("version") != continuation.VERSION
            or launch.get("continuation_sha256") != base.sha(continuation.__file__)
            or launch.get("target_step") != ANCHOR_STEP or launch.get("starting_step") != 256
            or launch.get("starting_source_cursor") != 768
            or launch.get("manual_arm") != original_arm["arm"]
            or launch.get("coefficients") != original_arm["coefficients"]
            or launch.get("learning_rate") != original_arm["learning_rate"]
            or launch.get("selected_checkpoint_sha256") != receipt["selected_screen_checkpoint_sha256"]
            or launch.get("screen_identity_sha256") != original_arm["screen_identity_sha256"]
            or launch.get("optimizer_restored") is not True or launch.get("rng_restored") is not True):
        raise ValueError("Anchor continuation does not preserve the original training recipe")


def validate_source_ledger(payload, original_ids, fresh_ids, effective_batch_size=3):
    """The complete history must be an exact prefix, with no validation reuse."""
    if type(effective_batch_size) is not int or effective_batch_size < 1:
        raise ValueError("Effective batch size must be a positive integer")
    if len(original_ids) != ANCHOR_SOURCES or len(set(original_ids)) != len(original_ids):
        raise ValueError("The original run must contain exactly 3000 distinct fitting sources")
    if len(fresh_ids) < AUTHORIZED_FRESH_SOURCES or len(set(fresh_ids)) != len(fresh_ids):
        raise ValueError("Fresh fitting sources are insufficient or repeated")
    if set(original_ids).intersection(fresh_ids):
        raise ValueError("Fresh source plan repeats an original fitting source")
    step = payload.get("step")
    if type(step) is not int or not ANCHOR_STEP <= step <= MAX_TARGET_STEP:
        raise ValueError("Checkpoint global step lies outside this approved continuation")
    fresh_cursor = (step-ANCHOR_STEP)*effective_batch_size
    if fresh_cursor > AUTHORIZED_FRESH_SOURCES:
        raise ValueError("The checkpoint consumed the conditional data reserve")
    expected = list(original_ids)+list(fresh_ids[:fresh_cursor])
    if payload.get("sources_seen") != expected or payload.get("fit_cursor") != len(expected):
        raise ValueError("Saved history is not the exact original plus fresh ordered source prefix")
    if payload.get("fresh_cursor", fresh_cursor) != fresh_cursor:
        raise ValueError("Saved fresh-source cursor disagrees with optimizer updates")
    return fresh_cursor


def validate_target(start_step, target_step, effective_batch_size):
    if (type(target_step) is not int or not start_step < target_step <= MAX_TARGET_STEP
            or target_step % 500):
        raise ValueError("Choose a later 500-step decision point, no later than step 5000")
    needed = (target_step-ANCHOR_STEP)*effective_batch_size
    if needed > AUTHORIZED_FRESH_SOURCES:
        raise ValueError("Requested updates need more than the 12000 authorized fresh sources")
    return needed


def read_training_metadata(directory, payload):
    last = json.loads((directory/"train.jsonl").read_text().splitlines()[-1])
    if (last["step"] != payload["step"] or last["unique_sources"] != payload["fit_cursor"]
            or any(not math.isfinite(last[key]) or last[key] < 0 for key in ("elapsed_seconds", "audio_hours"))):
        raise ValueError("Saved exposure and training time do not match the checkpoint")
    return last


def authenticate(args):
    """Follow the original preflight -> screen -> step-1000 provenance chain."""
    original_args = copy.copy(args)
    original_args.checkpoint = args.screen_out/"current_lr3e-5"/"final.pt"
    metadata, selection, screen_identity, manifest, pools, screen_payload, _, screen_sha, _ = continuation.authenticate(original_args)
    anchor_path = args.anchor_checkpoint or args.checkpoint
    anchor_sha = base.sha(anchor_path)
    payload = torch.load(anchor_path, map_location="cpu", weights_only=True, mmap=True)
    receipt = json.loads(anchor_path.with_suffix(".json").read_text())
    completed = json.loads((anchor_path.parent/"completed.json").read_text())
    launch = json.loads((anchor_path.parent/"launch.json").read_text())
    validate_anchor(payload, receipt, completed, launch, screen_payload["identity"], pools["fit"], anchor_sha)
    if receipt["selected_screen_checkpoint_sha256"] != screen_sha:
        raise ValueError("Step-1000 anchor descends from a different screen checkpoint")
    starting_report = json.loads((anchor_path.parent/"development-step1000.json").read_text())
    if starting_report["aggregate"] != receipt["common_quality"]:
        raise ValueError("Saved step-1000 quality differs from its checkpoint receipt")
    training_metadata = read_training_metadata(anchor_path.parent, payload)
    if args.checkpoint.resolve() != anchor_path.resolve():
        checksum = base.sha(args.checkpoint)
        candidate_receipt = json.loads(args.checkpoint.with_suffix(".json").read_text())
        candidate = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if (candidate.get("format") != VERSION or candidate_receipt.get("checkpoint_sha256") != checksum
                or candidate_receipt.get("anchor_checkpoint_sha256") != ANCHOR_SHA256
                or candidate.get("identity") != payload["identity"]
                or candidate_receipt.get("step") != candidate.get("step")
                or candidate_receipt.get("fit_cursor") != candidate.get("fit_cursor")
                or candidate_receipt.get("resume_identity_sha256") != screen.digest(candidate["resume_identity"])
                or candidate_receipt.get("optimizer_and_rng_saved") is not True
                or candidate_receipt.get("frozen_state_preserved") is not True):
            raise ValueError("Resume checkpoint does not authenticate to the selected step-1000 state")
        payload = candidate
        starting_report = json.loads((args.checkpoint.parent/f"development-step{payload['step']}.json").read_text())
        if starting_report["aggregate"] != candidate_receipt["common_quality"]:
            raise ValueError("Saved resume quality report differs from its receipt")
        # Each receipt records time/exposure, including when resuming an earlier milestone.
        training_metadata = candidate_receipt["training_metadata"]
        if (training_metadata["step"] != payload["step"]
                or training_metadata["unique_sources"] != payload["fit_cursor"]):
            raise ValueError("Resume checkpoint training metadata differs")
    else:
        checksum = anchor_sha
    return metadata, selection, screen_identity, manifest, pools, payload, starting_report, checksum, training_metadata


def validate_restored_quality(actual, expected):
    if set(actual) != set(expected): raise ValueError("Restored validation metric coverage changed")
    for key, reference in expected.items():
        value = actual[key]
        if reference is None or type(reference) in (bool, int):
            passed = type(value) is type(reference) and value == reference
        else:
            passed = isinstance(value, (int, float)) and math.isclose(value, reference, rel_tol=1e-5, abs_tol=1e-6)
        if not passed: raise ValueError("Restored checkpoint validation changed: "+key)


def take_when_ready(data, start, count, *, step, writer=None, monitor=None, poll_seconds=30, wait_seconds=3600):
    """Only sealed shards can be consumed; waiting never changes the data cursor."""
    began = time.monotonic()
    while True:
        try:
            chosen = data.take(start, count)
        except FileNotFoundError:
            elapsed = time.monotonic()-began
            base.event("waiting_for_sealed_training_data", step=step, start_index=start,
                       stop_index=start+count, waiting_seconds=elapsed)
            if writer is not None:
                writer.add_scalar("data/waiting_for_shard", 1, step)
                writer.add_scalar("data/shard_wait_seconds", elapsed, step)
                writer.flush()
            if monitor is not None:
                monitor.log_waiting(f"Waiting for sealed fresh sources {start} to {start+count}; no fitting update or source reuse occurs while waiting.", step)
            if elapsed >= wait_seconds:
                raise TimeoutError("Required training shard has not been sealed within the waiting limit")
            time.sleep(min(poll_seconds, wait_seconds-elapsed))
            continue
        if [crop["source_id"] for crop in chosen] != list(data.source_ids[start:start+count]):
            raise ValueError("Loaded fresh crops differ from the sealed ordered source plan")
        if writer is not None: writer.add_scalar("data/waiting_for_shard", 0, step)
        return chosen


def save_checkpoint(path, model, optimizer, step, arm, seen, resume_identity):
    if path.exists(): raise FileExistsError("Refusing to overwrite a saved training state")
    payload = {"format": VERSION, "group": {key:value.detach().cpu() for key,value in model.group_state_dict().items()},
               "optimizer": optimizer.state_dict(), "rng": screen.rng_state(), "step": step,
               "fit_cursor": len(seen), "fresh_cursor": len(seen)-ANCHOR_SOURCES,
               "sources_seen": list(seen), "identity": arm, "resume_identity": resume_identity}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    for name in ("checkpoint", "screen-out", "base-out", "manifest", "fresh-manifest", "shards", "assets", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, help="Required when checkpoint is a later saved resume milestone")
    parser.add_argument("--tensorboard", type=Path)
    parser.add_argument("--target-step", type=int, default=5000)
    parser.add_argument("--effective-batch-size", type=int, default=3)
    parser.add_argument("--shard-wait-seconds", type=float, default=3600)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()): raise FileExistsError("Use a new empty resume directory")
    if args.shard_wait_seconds <= 0: raise ValueError("Shard waiting limit must be positive")
    base.policy()
    metadata, selection, identity, manifest, pools, payload, starting_report, checksum, previous = authenticate(args)
    from fresh_training_data import FreshTrainingData
    from unified_monitor import UnifiedMonitor
    data = FreshTrainingData(args.fresh_manifest, args.manifest, pools, args.shards)
    original_ids = [crop["source_id"] for crop in pools["fit"]]
    cursor = validate_source_ledger(payload, original_ids, data.source_ids, args.effective_batch_size)
    validate_target(payload["step"], args.target_step, args.effective_batch_size)
    fresh_identity = data.plan["identity_sha256"]
    resume_identity = {"version": VERSION, "anchor_checkpoint_sha256": ANCHOR_SHA256,
        "resume_sha256": base.sha(__file__), "screen_identity_sha256": screen.digest(identity),
        "original_manifest_sha256": base.sha(args.manifest), "fresh_plan_identity_sha256": fresh_identity,
        "fresh_manifest_sha256": base.sha(args.fresh_manifest),
        "fresh_loader_sha256": base.sha(Path(__file__).with_name("fresh_training_data.py")),
        "monitor_sha256": base.sha(Path(__file__).with_name("unified_monitor.py")),
        "effective_batch_size": args.effective_batch_size, "execution_batch_size": 1,
        "accumulation_steps": args.effective_batch_size, "approved_source_range": [0, AUTHORIZED_FRESH_SOURCES],
        "precision": "float32", "tf32": False, "coefficients": payload["identity"]["coefficients"],
        "learning_rate": payload["identity"]["learning_rate"], "original_seen_sha256": screen.digest(original_ids)}
    if payload["step"] != ANCHOR_STEP and payload.get("resume_identity") != resume_identity:
        raise ValueError("Saved continuation source, data, execution mode or training configuration changed")
    teacher = base.FrozenAudioVAE2.from_files(args.assets/"audio_vae_v2.py", args.assets/"audiovae.pth", device="cuda")
    model = base.build_student(teacher.model.decoder, selection["stage2_indices"], selection["stage3_indices"])
    common = base.objective()
    if screen.digest(asdict(common.config)) != screen.digest(identity["current_mel"]):
        raise ValueError("Current mel objective differs from the selected checkpoint")
    source_metadata = {row["source_id"]:row for row in manifest["splits"]["development"]["rows"]}
    boundary_panel, boundary_selection = base.select_boundary_panel(pools["development"], source_metadata)
    if boundary_selection != identity["boundary_panel"]: raise ValueError("Boundary diagnostic inputs changed")
    frozen, frozen_teacher = screen.frozen_versions(model), continuation.teacher_versions(teacher)
    optimizer = continuation.restore_training_state(model, payload)
    if screen.frozen_versions(model) != frozen: raise RuntimeError("Restore changed frozen decoder state")
    args.out.mkdir(parents=True, exist_ok=True)
    arm = payload["identity"]
    launch = {"version": VERSION, "resume_identity": resume_identity, "selected_checkpoint_sha256": checksum,
        "starting_step": payload["step"], "target_step": args.target_step,
        "starting_source_cursor": payload["fit_cursor"], "starting_fresh_cursor": cursor,
        "optimizer_restored": True, "rng_restored": True, "automatic_selection": False,
        "stop_for_quality_decision": True, "validation_every_updates": 500,
        "starting_group_sha256": screen.group_digest(model), "previous_training_metadata": previous}
    base.write_json(args.out/"launch.json", launch)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(args.tensorboard or args.out/"tensorboard"/"active_training"))
    baseline = json.loads((args.screen_out/arm["arm"]/"development-step0.json").read_text())["aggregate"]
    monitor = UnifiedMonitor(writer, baseline, source_metadata)
    monitor.initialize_layout()
    screen.restore_rng(payload["rng"])
    seen, seen_set = list(payload["sources_seen"]), set(payload["sources_seen"])
    exposure = previous["audio_hours"]*3600
    started = time.monotonic()
    try:
        evaluation_rng = screen.rng_state()
        try:
            restored_report = monitor.evaluate(model, teacher, pools["development"], common)
            validate_restored_quality(restored_report["aggregate"], starting_report["aggregate"])
            starting_report = restored_report
        finally:
            screen.restore_rng(evaluation_rng)
        base.write_json(args.out/"restore-check.json", {"passed":True, "step":payload["step"],
            "selected_checkpoint_sha256":checksum, "relative_tolerance":1e-5, "absolute_tolerance":1e-6,
            "common_quality":starting_report["aggregate"]})
        base.write_json(args.out/f"development-step{payload['step']}.json", starting_report)
        base.log_validation(writer, starting_report, payload["step"], source_metadata)
        monitor.log_validation(starting_report, payload["step"])
        writer.flush()
        for step in range(payload["step"]+1, args.target_step+1):
            chosen = take_when_ready(data, cursor, args.effective_batch_size, step=step, writer=writer, monitor=monitor,
                                     wait_seconds=args.shard_wait_seconds)
            ids = [crop["source_id"] for crop in chosen]
            if (len(ids) != args.effective_batch_size or len(set(ids)) != len(ids)
                    or any(source in seen_set for source in ids)):
                raise RuntimeError("Training would repeat or omit fitting sources")
            step_start = time.monotonic()
            diagnostics = step == payload["step"]+1 or step % 25 == 0
            values = screen.training_update(model, teacher, chosen, "current", common, arm["coefficients"],
                                            optimizer, record_diagnostics=diagnostics)
            seen.extend(ids); seen_set.update(ids); cursor += len(ids)
            exposure += sum(crop["valid_scored_samples"] for crop in chosen)/48000
            elapsed = time.monotonic()-started
            record = {"step":step, **values, "source_ids":ids, "unique_sources":len(seen),
                "fresh_sources":cursor, "audio_hours":exposure/3600,
                "elapsed_seconds":previous["elapsed_seconds"]+elapsed,
                "continuation_elapsed_seconds":elapsed, "continuation_updates":step-payload["step"]}
            if diagnostics: record["step_seconds"] = time.monotonic()-step_start
            with (args.out/"train.jsonl").open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False)+"\n")
            base.log_training(writer, record, arm["coefficients"], arm["learning_rate"], step)
            monitor.log_training(record, step)
            if diagnostics: base.event("resume_training", **record); writer.flush()
            if step % 500 == 0:
                # Evaluation is read-only and must not perturb the continued RNG sequence.
                evaluation_rng = screen.rng_state()
                try:
                    report = monitor.evaluate(model, teacher, pools["development"], common)
                    base.write_json(args.out/f"development-step{step}.json", report)
                    groups = base.log_validation(writer, report, step, source_metadata)
                    base.write_json(args.out/f"validation-groups-step{step}.json", groups)
                    monitor.log_validation(report, step)
                    if step == args.target_step:
                        boundaries = base.evaluate_boundaries(teacher, model, boundary_panel, selection, base.batch, base.teacher_forward)
                        base.write_json(args.out/f"boundaries-step{step}.json", boundaries)
                        base.log_boundaries(writer, boundaries, step)
                finally:
                    screen.restore_rng(evaluation_rng)
                if screen.frozen_versions(model) != frozen or continuation.teacher_versions(teacher) != frozen_teacher:
                    raise RuntimeError("Training changed frozen model or teacher tensors")
                data.assert_unchanged()
                path = args.out/f"checkpoint-step{step}.pt"
                save_checkpoint(path, model, optimizer, step, arm, seen, resume_identity)
                receipt = {"step":step, "fit_cursor":len(seen), "fresh_cursor":cursor,
                    "checkpoint_sha256":base.sha(path), "anchor_checkpoint_sha256":ANCHOR_SHA256,
                    "resume_identity_sha256":screen.digest(resume_identity), "common_quality":report["aggregate"],
                    "arm":arm["arm"], "optimizer_and_rng_saved":True, "frozen_state_preserved":True,
                    "training_metadata":{key:record[key] for key in ("step","unique_sources","audio_hours","elapsed_seconds")}}
                base.write_json(path.with_suffix(".json"), receipt)
                base.event("resume_validation", step=step, **report["aggregate"])
                writer.flush()
        if base.sha(args.checkpoint) != checksum: raise RuntimeError("Selected input checkpoint changed")
        base.write_json(args.out/"completed.json", {**receipt, "automatic_promotion":False,
            "stopped_for_quality_decision":True, "target_step":args.target_step})
    finally:
        preserved = screen.frozen_versions(model) == frozen and continuation.teacher_versions(teacher) == frozen_teacher
        base.write_json(args.out/"preservation.json", {"frozen_model_and_teacher_preserved":preserved})
        writer.flush(); writer.close()
        if not preserved: raise RuntimeError("Frozen model/teacher preservation failed")


if __name__ == "__main__": main()
