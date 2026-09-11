"""Continue an explicitly selected screen checkpoint without repeating sources.

This entry point never ranks or selects arms. It retains the selected optimizer,
loss coefficients and RNG states and completes the approved1000-update pilot.
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

import settings_screen as screen

base = screen.base
VERSION = "audiovae2_settings_continuation_v1"
FINAL_STEP = 1000


def validate_source_cursor(payload, fitting):
    step, cursor, seen = payload["step"], payload["fit_cursor"], payload["sources_seen"]
    if type(step) is not int or not 0 < step <= FINAL_STEP or type(cursor) is not int or cursor != step*3:
        raise ValueError("Checkpoint step and source cursor disagree")
    available = [crop["source_id"] for crop in fitting]
    if len(available) < FINAL_STEP*3 or len(set(available[:FINAL_STEP*3])) != FINAL_STEP*3:
        raise ValueError("The continuation needs3000 distinct ordered fitting sources")
    if type(seen) is not list or seen != available[:cursor] or len(set(seen)) != cursor:
        raise ValueError("Checkpoint history is not the exact ordered fitting-source prefix")
    return cursor


def restore_training_state(model, payload):
    """Validate moments before changing weights; restore RNG after construction."""
    params = base.parameters(model)
    state = copy.deepcopy(payload["optimizer"])
    if not isinstance(state, dict) or len(state.get("param_groups", [])) != 1:
        raise ValueError("Expected one complete AdamW parameter group")
    group = state["param_groups"][0]
    rate = payload["identity"]["learning_rate"]
    if (group.get("lr") != rate or tuple(group.get("betas", ())) != screen.OPTIMIZER["betas"]
            or group.get("weight_decay") != 0 or group.get("eps") != 1e-8):
        raise ValueError("Optimizer settings differ from the selected arm")
    ids = group["params"]
    if len(ids) != len(params) or len(set(ids)) != len(params) or set(state.get("state", {})) != set(ids):
        raise ValueError("Missing or additional optimizer parameter states")
    for identity, param in zip(ids, params):
        item = state["state"][identity]
        if set(item) != {"step", "exp_avg", "exp_avg_sq"} or float(item["step"]) != payload["step"]:
            raise ValueError("AdamW moments do not match the checkpoint update")
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = item[name]
            if not isinstance(tensor, torch.Tensor) or tensor.shape != param.shape or not torch.isfinite(tensor).all():
                raise ValueError("Invalid optimizer moment: "+name)
    rng = payload["rng"]
    if any(param.is_cuda for param in params) and len(rng["cuda"]) != torch.cuda.device_count():
        raise ValueError("Visible CUDA devices differ from the saved RNG states")
    model.load_group_state_dict(payload["group"])
    optimizer = torch.optim.AdamW(params, lr=rate, **screen.OPTIMIZER)
    optimizer.load_state_dict(state)
    for param in params: param.grad = None
    screen.restore_rng(rng)
    return optimizer


def teacher_versions(teacher):
    return {name: (value.data_ptr(), value._version) for name, value in teacher.state_dict().items()}


def authenticate(args):
    metadata, selection, _, preflight, manifest, pools = screen.authenticate_inputs(args)
    identity = json.loads((args.screen_out/"screen-identity.json").read_text())
    if (identity["screen_sha256"] != base.sha(screen.__file__)
            or identity["author_mel_sha256"] != base.sha(Path(__file__).with_name("author_mel.py"))
            or identity["original_identity"] != preflight["identity"]
            or identity["original_preflight"] != metadata):
        raise ValueError("Screen source, losses or original model identity changed")
    if identity["torch"] != str(torch.__version__) or identity["cudnn"] != torch.backends.cudnn.version():
        raise ValueError("The saved training runtime differs from the continuation runtime")
    original_fit = [crop["source_id"] for crop in pools["fit"][:screen.UPDATES*3]]
    if identity["fit_source_ids"] != original_fit or identity["fit_source_ids_sha256"] != screen.digest(original_fit):
        raise ValueError("The screen's fitting-source order changed")
    checksum = base.sha(args.checkpoint)
    completed = json.loads((args.checkpoint.parent/"completed.json").read_text())
    if completed["checkpoint_sha256"] != checksum or completed["step"] != screen.UPDATES:
        raise ValueError("Checkpoint bytes do not match the completed screen arm")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload["format"] != screen.VERSION or payload["step"] != screen.UPDATES:
        raise ValueError("Select an original completed256-update settings-screen checkpoint")
    arm = payload["identity"]
    planned = {name: (definition, rate) for name, definition, rate in screen.ARMS}
    if (arm["arm"] not in planned or (arm["definition"], arm["learning_rate"]) != planned[arm["arm"]]
            or completed["arm"] != arm["arm"] or completed["unique_sources"] != payload["fit_cursor"]
            or arm["screen_identity_sha256"] != screen.digest(identity)
            or arm["original_identity"] != preflight["identity"] or arm["channel_selection"] != selection):
        raise ValueError("Selected arm does not match the authenticated screen")
    if args.checkpoint.parent.resolve() != (args.screen_out/arm["arm"]).resolve():
        raise ValueError("Selected checkpoint is outside its screen arm directory")
    if json.loads((args.checkpoint.parent/"launch.json").read_text()) != arm:
        raise ValueError("Selected arm launch identity changed")
    coefficients = arm["coefficients"]
    if set(coefficients) != {"waveform", "mel", "feature"} or any(not math.isfinite(x) or x <= 0 for x in coefficients.values()):
        raise ValueError("Invalid saved loss coefficients")
    validate_source_cursor(payload, pools["fit"])
    report = json.loads((args.checkpoint.parent/"development-step256.json").read_text())
    if report["aggregate"] != completed["common_quality"]:
        raise ValueError("Saved starting quality report differs from the completed arm")
    last_training = json.loads((args.checkpoint.parent/"train.jsonl").read_text().splitlines()[-1])
    if (last_training["step"] != payload["step"] or last_training["unique_sources"] != payload["fit_cursor"]
            or not math.isfinite(last_training["elapsed_seconds"]) or last_training["elapsed_seconds"] < 0):
        raise ValueError("Saved training-time metadata differs from the checkpoint")
    return metadata, selection, identity, manifest, pools, payload, report, checksum, last_training["elapsed_seconds"]


def main():
    parser = argparse.ArgumentParser()
    for name in ("checkpoint", "screen-out", "base-out", "manifest", "assets", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--tensorboard", type=Path)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()): raise FileExistsError("Use a new, empty continuation directory")
    base.policy()
    metadata, selection, identity, manifest, pools, payload, starting_report, checksum, previous_elapsed = authenticate(args)
    teacher = base.FrozenAudioVAE2.from_files(args.assets/"audio_vae_v2.py", args.assets/"audiovae.pth", device="cuda")
    model = base.build_student(teacher.model.decoder, selection["stage2_indices"], selection["stage3_indices"])
    common = base.objective()
    definition = payload["identity"]["definition"]
    objective = common if definition == "current" else screen.AuthorMelLoss(screen.AuthorMelConfig()).cuda()
    if screen.digest(asdict(objective.config)) != screen.digest(identity[definition+"_mel"]):
        raise ValueError("Loss configuration changed since the screen")
    source_metadata = {row["source_id"]: row for row in manifest["splits"]["development"]["rows"]}
    boundary_panel, boundary_selection = base.select_boundary_panel(pools["development"], source_metadata)
    if boundary_selection != identity["boundary_panel"]: raise ValueError("Diagnostic panel changed")
    frozen, frozen_teacher = screen.frozen_versions(model), teacher_versions(teacher)
    optimizer = restore_training_state(model, payload)
    if screen.frozen_versions(model) != frozen: raise RuntimeError("Restore changed frozen decoder state")
    args.out.mkdir(parents=True, exist_ok=True)
    arm = payload["identity"]
    launch = {"version": VERSION, "manual_arm": arm["arm"], "selected_checkpoint": str(args.checkpoint.resolve()),
              "selected_checkpoint_sha256": checksum, "starting_step": payload["step"], "target_step": FINAL_STEP,
              "starting_source_cursor": payload["fit_cursor"], "screen_identity_sha256": screen.digest(identity),
              "previous_training_elapsed_seconds": previous_elapsed,
              "continuation_sha256": base.sha(__file__), "optimizer_restored": True, "rng_restored": True,
              "coefficients": arm["coefficients"], "learning_rate": arm["learning_rate"],
              "automatic_selection": False, "source_order": "Continue the unchanged fitting manifest after its saved prefix"}
    base.write_json(args.out/"launch.json", launch)
    from torch.utils.tensorboard import SummaryWriter
    tag = f"{arm['arm']}_continue256to1000"
    writer = SummaryWriter(str((args.tensorboard or args.out/"tensorboard")/tag))
    # Constructing the writer must not advance the model's saved random state.
    screen.restore_rng(payload["rng"])
    seen = list(payload["sources_seen"])
    exposure = sum(crop["valid_scored_samples"] for crop in pools["fit"][:len(seen)])/48000
    started = time.monotonic()
    try:
        base.write_json(args.out/"development-step256.json", starting_report)
        base.log_validation(writer, starting_report, 256, source_metadata)
        writer.flush()
        for step in range(payload["step"]+1, FINAL_STEP+1):
            step_start = time.monotonic()
            chosen = pools["fit"][(step-1)*3:step*3]
            ids = [crop["source_id"] for crop in chosen]
            if len(chosen) != 3 or len(set(ids)) != 3 or any(source in seen for source in ids):
                raise RuntimeError("Continuation would repeat or omit fitting sources")
            diagnostics = step == payload["step"]+1 or step % 25 == 0
            values = screen.training_update(model, teacher, chosen, definition, objective, arm["coefficients"],
                                            optimizer, record_diagnostics=diagnostics)
            seen.extend(ids); exposure += sum(crop["valid_scored_samples"] for crop in chosen)/48000
            record = {"step": step, **values, "source_ids": ids, "unique_sources": len(seen),
                      "audio_hours": exposure/3600, "elapsed_seconds": previous_elapsed+time.monotonic()-started,
                      "continuation_elapsed_seconds": time.monotonic()-started,
                      "continuation_updates": step-payload["step"]}
            if diagnostics: record["step_seconds"] = time.monotonic()-step_start
            with (args.out/"train.jsonl").open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False)+"\n")
            base.log_training(writer, record, arm["coefficients"], arm["learning_rate"], step)
            writer.add_scalar("performance/seconds_per_continuation_update", record["continuation_elapsed_seconds"]/record["continuation_updates"], step)
            if diagnostics:
                base.event("continuation_training", arm=arm["arm"], **record); writer.flush()
            if step in (500, FINAL_STEP):
                report = base.evaluate(model, teacher, pools["development"], common)
                base.write_json(args.out/f"development-step{step}.json", report)
                groups = base.log_validation(writer, report, step, source_metadata)
                base.write_json(args.out/f"validation-groups-step{step}.json", groups)
                if step == FINAL_STEP:
                    boundaries = base.evaluate_boundaries(teacher, model, boundary_panel, selection, base.batch, base.teacher_forward)
                    base.write_json(args.out/f"boundaries-step{step}.json", boundaries)
                    base.log_boundaries(writer, boundaries, step)
                if screen.frozen_versions(model) != frozen or teacher_versions(teacher) != frozen_teacher:
                    raise RuntimeError("Continuation changed frozen model or teacher state")
                path = args.out/f"checkpoint-step{step}.pt"
                screen.save_checkpoint(path, model, optimizer, step, arm, seen)
                receipt = {"step": step, "fit_cursor": len(seen), "checkpoint_sha256": base.sha(path),
                           "selected_screen_checkpoint_sha256": checksum, "arm": arm["arm"],
                           "common_quality": report["aggregate"], "optimizer_and_rng_saved": True,
                           "frozen_state_preserved": True}
                base.write_json(path.with_suffix(".json"), receipt)
                base.event("continuation_validation", step=step, **report["aggregate"])
        if base.sha(args.checkpoint) != checksum: raise RuntimeError("Original selected checkpoint changed")
        base.write_json(args.out/"completed.json", {**receipt, "audio_hours": exposure/3600, "automatic_promotion": False})
    finally:
        writer.flush(); writer.close()


if __name__ == "__main__": main()
