"""Bounded, silence-first adaptation of an existing nonlinear decoder head.

Only the head convolution, PReLU and waveform projection are updated. Frozen
pre-head features are captured with the original singleton execution geometry.
Targets and quiet masks come from sealed post-tanh teacher pairs. This script
does not add an inference gate, change the architecture, or promote a model.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import hashlib
import importlib
import json
import math
from pathlib import Path
import time

import torch


HEAD_NAMES = {"head.conv.weight", "head.conv.bias", "activation.weight", "output.weight"}
ARMS = ("teacher_reconstruction", "quiet_preservation")


def head_forward(model, value):
    return model._waveform(model.output(model.activation(model.head(value))))


@torch.no_grad()
def capture_prehead(model, latents):
    captured = []
    handle = model.head.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    try:
        audio = model(latents)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("Expected exactly one head input capture")
    features = captured[0]
    replay = head_forward(model, features)
    if not torch.equal(replay, audio):
        raise RuntimeError("Singleton frozen-feature head replay is not exact")
    return features.detach(), audio.detach()


def head_parameters(model):
    named = dict(model.named_parameters())
    if not HEAD_NAMES.issubset(named):
        raise ValueError("Saved decoder lacks the four declared head parameters")
    for name, parameter in named.items():
        parameter.requires_grad_(name in HEAD_NAMES)
    return [(name, named[name]) for name in sorted(HEAD_NAMES)]


def head_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters() if name in HEAD_NAMES}


@torch.no_grad()
def restore_head(model, state):
    named = dict(model.named_parameters())
    if set(state) != HEAD_NAMES:
        raise ValueError("Head snapshot does not contain exactly the declared parameters")
    for name, value in state.items():
        if value.shape != named[name].shape or value.dtype != named[name].dtype:
            raise ValueError("Head snapshot shape/dtype changed")
        named[name].copy_(value.to(named[name].device))


def masked_sums(prediction, target, baseline, valid, quiet):
    if prediction.shape != target.shape or prediction.shape != baseline.shape:
        raise ValueError("Teacher, baseline and candidate waveform geometry must agree")
    if valid.shape != prediction.shape or quiet.shape != prediction.shape:
        raise ValueError("Waveform masks changed shape")
    if valid.dtype != torch.bool or quiet.dtype != torch.bool or bool((quiet & ~valid).any()):
        raise ValueError("Quiet mask must be a teacher-only subset of valid samples")
    outside = valid & ~quiet
    error = prediction - target
    drift = prediction - baseline
    return {"quiet_teacher_square": error[quiet].square().sum(),
            "outside_teacher_square": error[outside].square().sum(),
            "outside_preservation_square": drift[outside].square().sum(),
            "quiet_samples": int(quiet.sum()), "outside_samples": int(outside.sum())}


def normalized_loss(sums, totals, scales, arm, preservation_weight):
    """One microbatch contribution to sample-pooled batch losses.

    Scales are fixed training-pool baseline errors, never per-clip RMS or
    candidate-dependent values. A batch without quiet samples has a zero quiet
    contribution; its nonquiet preservation/reconstruction term still trains.
    """
    if arm not in ARMS or min(scales.values()) <= 0:
        raise ValueError("Invalid objective or global baseline error scales")
    quiet = sums["quiet_teacher_square"] / max(totals["quiet_samples"], 1) / scales["quiet_mse"]
    field = "outside_teacher_square" if arm == ARMS[0] else "outside_preservation_square"
    outside = sums[field] / max(totals["outside_samples"], 1) / scales["outside_mse"]
    factor = 1.0 if arm == ARMS[0] else preservation_weight
    return quiet + factor * outside


def new_optimizer(parameters, learning_rate):
    # These original head parameters belong to AdamW, not the matrix-only Muon
    # body route. Fresh matched states explicitly isolate this head-fit pilot.
    return torch.optim.AdamW(parameters, lr=learning_rate, betas=(.9, .999), eps=1e-8,
                            weight_decay=0., foreach=False, fused=False)


def train_batch(model, bank, indices, optimizer, scales, arm, preservation_weight, *, device):
    selected = [bank[i] for i in indices]
    totals = {key: sum(row[key] for row in selected) for key in ("quiet_samples", "outside_samples")}
    if totals["quiet_samples"] + totals["outside_samples"] <= 0:
        raise ValueError("Training batch contains no scored samples")
    optimizer.zero_grad(set_to_none=True)
    summary = {"loss": 0., **totals}
    for row in selected:
        h, target, baseline, valid, quiet = [row[k].to(device) for k in ("features", "target", "baseline", "valid", "quiet")]
        prediction = head_forward(model, h)
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("Nonfinite candidate waveform")
        sums = masked_sums(prediction, target, baseline, valid, quiet)
        loss = normalized_loss(sums, totals, scales, arm, preservation_weight)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite objective")
        loss.backward()
        summary["loss"] += float(loss.detach())
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if {n for n, _ in named} != HEAD_NAMES:
        raise RuntimeError("Unexpected trainable parameter route")
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for _, p in named):
        raise FloatingPointError("Missing or nonfinite head gradient")
    # A declared uniform global cap preserves gradient direction; no adaptive
    # historical loss balancer is reused across these different objectives.
    summary["gradient_norm_before_clip"] = float(torch.nn.utils.clip_grad_norm_([p for _, p in named], 1.0))
    before = [p.detach().clone() for _, p in named]
    optimizer.step()
    summary["parameter_displacement"] = math.sqrt(sum(float((p.detach().double()-v.double()).square().sum()) for (_, p), v in zip(named, before)))
    if any(not bool(torch.isfinite(p).all()) for _, p in named):
        raise FloatingPointError("Nonfinite head parameter")
    return summary


@torch.no_grad()
def bank_score(model, bank, device):
    rows = []
    for row in bank:
        h, target, baseline, valid, quiet = [row[k].to(device) for k in ("features", "target", "baseline", "valid", "quiet")]
        prediction = head_forward(model, h)
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("Nonfinite validation waveform")
        p, t, b = prediction.double(), target.double(), baseline.double()
        outside = valid & ~quiet
        e = p-t
        rows.append({"source_id": row["source_id"], "start_frame": row["start_frame"],
            "language": row["language"], "condition": row["condition"], "dataset": row["dataset"],
            "samples": int(valid.sum()), "quiet_samples": int(quiet.sum()), "outside_samples": int(outside.sum()),
            "square": float(e[valid].square().sum()), "absolute": float(e[valid].abs().sum()),
            "quiet_square": float(e[quiet].square().sum()), "outside_square": float(e[outside].square().sum()),
            "outside_drift_square": float((p-b)[outside].square().sum()),
            "peak": float(p[valid].abs().max()), "overshoot_samples": int((p[valid].abs()>1).sum())})
    def aggregate(selected):
        ns = sum(r["samples"] for r in selected)
        nq = sum(r["quiet_samples"] for r in selected)
        no = sum(r["outside_samples"] for r in selected)
        if not ns:
            return None
        return {"sources": len(selected), "samples": ns, "quiet_samples": nq, "outside_samples": no,
            "mse": sum(r["square"] for r in selected)/ns, "mae": sum(r["absolute"] for r in selected)/ns,
            "quiet_mse": sum(r["quiet_square"] for r in selected)/nq if nq else None,
            "outside_mse": sum(r["outside_square"] for r in selected)/no if no else None,
            "outside_drift_mse": sum(r["outside_drift_square"] for r in selected)/no if no else None,
            "peak": max(r["peak"] for r in selected), "overshoot_samples": sum(r["overshoot_samples"] for r in selected)}
    groups = {"all": aggregate(rows)}
    for field in ("language", "condition", "dataset"):
        for value in sorted({str(r[field]) for r in rows}):
            groups[field+"/"+value] = aggregate([r for r in rows if str(r[field]) == value])
    return {"aggregate": groups["all"], "groups": groups, "rows": rows}


def selection_checks(candidate, baseline):
    a, b = baseline["aggregate"], candidate["aggregate"]
    if any(a[k] != b[k] for k in ("samples", "quiet_samples", "outside_samples", "sources")):
        raise ValueError("Selection masks or source counts changed")
    checks = [{"name": "all/mae", "passed": b["mae"] <= a["mae"]*1.01},
              {"name": "all/mse", "passed": b["mse"] <= a["mse"]*1.01},
              {"name": "outside/preservation", "passed": b["outside_drift_mse"] <= a["outside_mse"]*.01},
              {"name": "all/peak", "passed": b["peak"] <= a["peak"]},
              {"name": "all/overshoot_samples", "passed": b["overshoot_samples"] <= a["overshoot_samples"]}]
    for key, ref in baseline["groups"].items():
        if key == "all" or ref["sources"] < 2:
            continue
        observed = candidate["groups"][key]
        checks.append({"name": key+"/mae", "passed": observed["mae"] <= ref["mae"]*1.05})
    quiet_ratio = math.sqrt(b["quiet_mse"]/a["quiet_mse"])
    checks.append({"name": "quiet/meaningful_reduction", "passed": quiet_ratio <= .90})
    return {"qualified": all(c["passed"] for c in checks), "quiet_rms_ratio": quiet_ratio,
            "checks": checks, "policy": "Selection split only; unchanged canonical quality screen remains required"}


def canonical_peak_screen(baseline, candidate):
    """Silence improvement may not introduce or worsen full-scale overshoot.

    Counts here are scored observations on the unchanged overlapping panel,
    not a count of distinct physical events. The per-crop check also catches a
    new affected crop hidden by a lower aggregate maximum elsewhere.
    """
    reference = {(r["source_id"], r["start_frame"]): r for r in baseline["rows"]}
    observed = {(r["source_id"], r["start_frame"]): r for r in candidate["rows"]}
    if (reference.keys() != observed.keys() or len(reference) != len(baseline["rows"])
            or len(observed) != len(candidate["rows"])):
        raise ValueError("Canonical peak-screen crop identities differ")
    regressions = []
    for key, row in reference.items():
        before, after = row["student_peak_abs"], observed[key]["student_peak_abs"]
        before_count, after_count = row["student_overshoot_samples"], observed[key]["student_overshoot_samples"]
        if after > max(1., before) or after_count > before_count:
            regressions.append({"source_id": key[0], "start_frame": key[1],
                                "baseline_peak": before, "candidate_peak": after,
                                "baseline_overshoot_observations": before_count,
                                "candidate_overshoot_observations": after_count})
    a, b = baseline["recovery_metrics"]["all"], candidate["recovery_metrics"]["all"]
    return {"passed": not regressions and b["maximum_peak"] <= a["maximum_peak"] and
            b["scored_overshoot_samples"] <= a["scored_overshoot_samples"],
            "regressing_crops": regressions, "baseline_maximum_peak": a["maximum_peak"],
            "candidate_maximum_peak": b["maximum_peak"],
            "baseline_overshoot_observations": a["scored_overshoot_samples"],
            "candidate_overshoot_observations": b["scored_overshoot_samples"]}


@torch.no_grad()
def build_bank(model, crops, indices, rows, config, device, status, split):
    from fit_support import validation_masks
    bank = []
    for position, index in enumerate(indices, 1):
        crop = crops[index]
        h, baseline = capture_prehead(model, crop.latents.to(device))
        target, valid, quiet = validation_masks(crop, device="cpu", quiet_config=config)
        if baseline.shape != target.shape:
            raise ValueError("Frozen-feature output and sealed teacher shape differ")
        meta = rows[crop.source_id]
        bank.append({"features": h.cpu(), "baseline": baseline.cpu(), "target": target,
            "valid": valid, "quiet": quiet, "quiet_samples": int(quiet.sum()),
            "outside_samples": int((valid & ~quiet).sum()), "source_id": crop.source_id,
            "start_frame": crop.start_frame, **{k: meta.get(k) for k in ("language", "condition", "dataset")}})
        if position % 64 == 0:
            status("joint_head_features", split=split, completed=position, total=len(indices))
    return bank


def baseline_scales(bank):
    quiet_square = outside_square = 0.
    quiet_count = outside_count = 0
    for row in bank:
        e = row["baseline"].double()-row["target"].double()
        q, o = row["quiet"], row["valid"] & ~row["quiet"]
        quiet_square += float(e[q].square().sum())
        outside_square += float(e[o].square().sum())
        quiet_count += int(q.sum()); outside_count += int(o.sum())
    if min(quiet_count, outside_count, quiet_square, outside_square) <= 0:
        raise ValueError("Fit corpus needs positive quiet and nonquiet baseline error")
    return {"quiet_mse": quiet_square/quiet_count, "outside_mse": outside_square/outside_count}, {
        "quiet_samples": quiet_count, "outside_samples": outside_count,
        "quiet_seconds": quiet_count/48000, "outside_seconds": outside_count/48000}


def calibrate_rate(model, bank, initial, parameters, scales, args, device):
    indices = list(range(args.batch_size))
    first = [bank[i] for i in indices]
    baseline = bank_score(model, first, device)
    if baseline["aggregate"]["quiet_samples"] <= 0 or baseline["aggregate"]["outside_samples"] <= 0:
        raise ValueError("First fixed batch must contain both quiet and nonquiet samples for calibration")
    trials = []
    try:
        for half in range(7):
            rate = args.learning_rate/(2**half)
            outcomes = []
            for arm in ARMS:
                restore_head(model, initial)
                optimizer = new_optimizer(parameters, rate)
                update = train_batch(model, bank, indices, optimizer, scales, arm,
                                     args.preservation_weight, device=device)
                score = bank_score(model, first, device)["aggregate"]
                ref = baseline["aggregate"]
                quiet_ratio = score["quiet_mse"]/ref["quiet_mse"]
                drift_ratio = score["outside_drift_mse"]/ref["outside_mse"]
                outcomes.append({"arm": arm, "quiet_mse_ratio": quiet_ratio,
                    "outside_drift_to_baseline_error_mse_ratio": drift_ratio,
                    "passed": quiet_ratio <= 1.01 and drift_ratio <= .01,
                    "update": update})
            trials.append({"learning_rate": rate, "arms": outcomes})
            if all(v["passed"] for v in outcomes):
                return {"chosen_learning_rate": rate, "trials": trials,
                    "policy": "One fixed training batch, at most six halvings; temporary updates discarded. Same LR for both arms.",
                    "retained_calibration_optimizer_updates": 0,
                    "calibration_batch_replayed_as_first_training_batch": True}
        return {"chosen_learning_rate": None, "trials": trials, "reason": "No common rate met fixed finite-update safeguards"}
    finally:
        restore_head(model, initial)
        for p in parameters:
            p.grad = None


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def run(args):
    from audiovae_student.objective_comparison import state_fingerprint
    from audiovae_student.restart_data import file_sha
    from audiovae_student.training import _rng_state, _restore_rng
    from bounded_head import transient_errors
    from canonical_evaluation import build_canonical_evaluation
    from diagnostic_common import load_context, atomic_json, status
    from fit_support import select_training_sources, validation_masks
    from native_chain import QUARTER_SHA, restore_quarter
    from run_comparison import heldout_hashes
    from run_heads import compare
    from run_update_experiment import load_canonical_panel
    from training_overlay import load_training_overlay, verify_overlay_files

    if not 0 < args.learning_rate <= 1e-6 or args.steps*args.batch_size > 2048:
        raise ValueError("Pilot is capped at LR1e-6 and 2,048 distinct training crops per arm")
    if min(args.steps, args.batch_size, args.validation_count, args.eval_every, args.preservation_weight) <= 0:
        raise ValueError("Positive pilot settings required")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    if file_sha(checkpoint) != QUARTER_SHA:
        raise ValueError("Wrong retained checkpoint")
    ctx = load_context()
    panel = load_canonical_panel(ctx, {"requires_canonical_evaluation": True}, args.canonical_receipt)
    overlay = load_training_overlay(ctx, args.training_receipt, required_counts={"targeted_generator": 12800})
    hashes, inventory = heldout_hashes(panel, args.canonical_inventory)
    pool = overlay["pools"]["targeted_generator"]
    selection = select_training_sources(pool, ctx.data["rows"], {c.source_id for c in panel["crops"]}, hashes,
        fit_count=args.steps*args.batch_size, validation_count=args.validation_count)
    atomic_json(out/"selection.json", selection)
    saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    engine = ctx.engine("targeted", device="cuda")
    original_engine = restore_quarter(engine, saved["engine"])
    model = engine.model
    original_modes = {m: m.training for m in model.modules()}
    original_requires = {name: p.requires_grad for name, p in model.named_parameters()}
    original_rng = _rng_state()
    fit_rng = deepcopy(saved["rng"])
    initial = head_state(model)
    started = time.monotonic()
    model.eval()
    named = head_parameters(model); parameters = [p for _, p in named]
    frozen_hash = state_fingerprint({k: v for k, v in model.state_dict().items() if k not in HEAD_NAMES})
    def verify_frozen():
        if state_fingerprint({k: v for k, v in model.state_dict().items() if k not in HEAD_NAMES}) != frozen_hash:
            raise RuntimeError("Frozen body or normalization changed")
    def compare_silence(reference, observed):
        result = compare(reference, observed)
        result["no_new_peak_regression"] = canonical_peak_screen(reference, observed)
        result["silence_target_passed"] = (result["quality_screen_passed"] and
            result["natural_quiet_ratio"] <= .90 and result["steady_silence_ratio"] < 1 and
            result["no_new_peak_regression"]["passed"])
        return result
    def canonical(name):
        transient = []
        def observe(module, inputs, prediction):
            crop = panel["crops"][len(transient)]
            target, valid, _ = validation_masks(crop, device=engine.device, quiet_config=engine.config.quiet_audio)
            transient.append({"source_id": crop.source_id, "start_frame": crop.start_frame,
                              **transient_errors(prediction, target, valid)})
        handle = model.register_forward_hook(observe)
        try:
            report = build_canonical_evaluation(engine, panel["crops"], panel["metadata"], panel["receipt"])
        finally:
            handle.remove()
        report["transient_diagnostics"] = transient
        report["joint_head_experiment"] = {"name": name, "starting_checkpoint_step": 8890,
            "pilot_updates": 0 if name == "baseline" else args.steps,
            "engine_training_clock_unchanged": True}
        atomic_json(out/(name+"-canonical.json"), report)
        return report
    try:
        sources = {Path(__file__).resolve()}
        for module in ("fit_support", "diagnostic_common", "native_chain", "canonical_evaluation", "run_heads", "training_overlay"):
            sources.add(Path(importlib.import_module(module).__file__).resolve())
        identity = {"version": "joint_head_silence_v1", "checkpoint_sha256": QUARTER_SHA,
            "restored_engine_sha256": original_engine, "frozen_tensor_sha256": frozen_hash,
            "canonical_panel": panel["identity"], "training_overlay": overlay["identity"], "inventory": inventory,
            "selection_sha256": selection["identity_sha256"], "torch": str(torch.__version__),
            "teacher_state_sha256": ctx.parent["identity"]["data"]["teacher_state_sha256"],
            "cudnn": torch.backends.cudnn.version(), "trainable": [{"name": n, "shape": list(p.shape), "elements": p.numel()} for n, p in named],
            "source_files": {str(p): file_sha(p) for p in sources}, "args": vars(args),
            "objectives": {ARMS[0]: "quiet teacher MSE/q0 + nonquiet teacher MSE/a0",
                           ARMS[1]: "quiet teacher MSE/q0 + preservation_weight * nonquiet frozen-baseline MSE/a0"},
            "optimizer": {"name": "AdamW", "fresh_identical_states": True, "betas": [.9, .999], "eps": 1e-8,
                          "weight_decay": 0., "gradient_norm_cap": 1., "mixed_precision": False},
            "scope": "Head-only objective experiment, not original full GAN/Muon continuation. No synthetic fit, tanh, gate, new layer or automatic promotion.",
            "metadata_policy": "Missing condition labels remain unknown; dataset names do not prove which expressive event occurs in a crop."}
        atomic_json(out/"identity.json", identity)
        banks = {split: build_bank(model, pool, selection["splits"][split]["indices"], ctx.data["rows"],
                                  engine.config.quiet_audio, engine.device, status, split) for split in ("fit", "validation")}
        scales, counts = baseline_scales(banks["fit"])
        atomic_json(out/"fit-accounting.json", {"scales": scales, "counts": counts,
            "sample_pooling": True, "distinct_source_crops_per_arm": len(banks["fit"]),
            "source_repeats_per_arm": 0, "matched_arms_share_exact_order": True,
            "quiet_mask": "Fixed teacher-only 20ms windows; training only, absent at inference"})
        reference_selection = bank_score(model, banks["validation"], engine.device)
        if min(reference_selection["aggregate"]["quiet_samples"], reference_selection["aggregate"]["outside_samples"]) <= 0:
            raise ValueError("Selection split lacks quiet/nonquiet coverage")
        atomic_json(out/"baseline-validation.json", reference_selection)
        calibration = calibrate_rate(model, banks["fit"], initial, parameters, scales, args, engine.device)
        atomic_json(out/"rate-calibration.json", calibration)
        if calibration["chosen_learning_rate"] is None:
            atomic_json(out/"complete.json", {"complete": False, "stopped_before_training": True,
                "reason": calibration["reason"], "promoted": False})
            return
        baseline_report = canonical("baseline")
        all_results = {}
        for arm in ARMS:
            restore_head(model, initial)
            _restore_rng(fit_rng)
            optimizer = new_optimizer(parameters, calibration["chosen_learning_rate"])
            validations = []; best = None; best_state = None
            arm_start = time.monotonic()
            with (out/(arm+"-training.jsonl")).open("w") as logfile:
                for step in range(1, args.steps+1):
                    indices = list(range((step-1)*args.batch_size, step*args.batch_size))
                    update = train_batch(model, banks["fit"], indices, optimizer, scales, arm,
                                         args.preservation_weight, device=engine.device)
                    log = {"step": step, "arm": arm, "sources_seen": step*args.batch_size,
                           "seconds": time.monotonic()-arm_start, **update}
                    logfile.write(json.dumps(log, allow_nan=False)+"\n"); logfile.flush()
                    if step % 16 == 0:
                        status("joint_head_training", **log)
                    if step % args.eval_every == 0 or step == args.steps:
                        verify_frozen()
                        score = bank_score(model, banks["validation"], engine.device)
                        checks = selection_checks(score, reference_selection)
                        entry = {"step": step, "score": score, "selection": checks}
                        validations.append(entry)
                        atomic_json(out/(arm+"-validation.json"), validations)
                        if checks["qualified"] and (best is None or score["aggregate"]["quiet_mse"] < best["score"]["aggregate"]["quiet_mse"]):
                            best, best_state = entry, head_state(model)
                        status("joint_head_validation", arm=arm, step=step, qualified=checks["qualified"], quiet_rms_ratio=checks["quiet_rms_ratio"])
            final_state = head_state(model)
            torch.save({"format_version": "joint_head_silence_v1", "base_checkpoint_sha256": QUARTER_SHA,
                "arm": arm, "mode": "raw", "updates": args.steps, "head_parameters": final_state,
                "selected_step": best["step"] if best else None, "selected_head_parameters": best_state,
                "qualification": "Experimental head parameters only; no optimizer state, not a resumable full engine"}, out/(arm+"-heads.pt"))
            final_report = canonical(arm+"-final")
            comparison = compare_silence(baseline_report, final_report)
            selected_comparison = None
            if best_state is not None:
                if best["step"] == args.steps:
                    selected_comparison = comparison
                else:
                    restore_head(model, best_state)
                    selected_report = canonical(arm+"-selected")
                    selected_report["joint_head_experiment"]["pilot_updates"] = best["step"]
                    atomic_json(out/(arm+"-selected-canonical.json"), selected_report)
                    selected_comparison = compare_silence(baseline_report, selected_report)
            all_results[arm] = {"final": comparison, "selected_step": best["step"] if best else None,
                "selected": selected_comparison, "seconds": time.monotonic()-arm_start,
                "head_sha256": file_sha(out/(arm+"-heads.pt")), "promoted": False}
            atomic_json(out/"comparisons.json", all_results)
            verify_frozen()
        restore_head(model, initial)
        for name, p in model.named_parameters():
            p.requires_grad_(original_requires[name]); p.grad = None
        for module, mode in original_modes.items():
            module.training = mode
        _restore_rng(original_rng)
        if state_fingerprint(engine.state_dict()) != original_engine:
            raise RuntimeError("Retained engine state changed after restoration")
        if file_sha(checkpoint) != QUARTER_SHA:
            raise RuntimeError("Retained checkpoint file changed")
        for key in ("receipt", "cache"):
            if file_sha(panel["identity"][key+"_path"]) != panel["identity"][key+"_sha256"]:
                raise RuntimeError("Canonical inputs changed")
        atomic_json(out/"complete.json", {"complete": True, "seconds": time.monotonic()-started,
            "engine_state_preserved": True, "original_files": ctx.verify_files(),
            "fresh_files": verify_overlay_files(overlay), "comparisons": all_results, "promoted": False})
        status("joint_head_experiments_complete", out=str(out))
    finally:
        restore_head(model, initial)
        for name, p in model.named_parameters():
            p.requires_grad_(original_requires[name]); p.grad = None
        for module, mode in original_modes.items():
            module.training = mode
        _restore_rng(original_rng)
        # These preservation checks also run if finite-update calibration finds
        # no acceptable rate, or any later candidate fails before completion.
        if state_fingerprint(engine.state_dict()) != original_engine:
            raise RuntimeError("Original engine was not preserved on experiment exit")
        if file_sha(checkpoint) != QUARTER_SHA:
            raise RuntimeError("Original checkpoint file changed on experiment exit")
        for key in ("receipt", "cache"):
            if file_sha(panel["identity"][key+"_path"]) != panel["identity"][key+"_sha256"]:
                raise RuntimeError("Canonical input bytes changed on experiment exit")
        atomic_json(out/"preservation.json", {"original_engine_preserved": True,
            "original_checkpoint_preserved": True, "canonical_inputs_preserved": True,
            "fresh_files": verify_overlay_files(overlay), "promoted": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("checkpoint", "training-receipt", "canonical-receipt", "canonical-inventory", "out"):
        parser.add_argument("--"+arg, required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-count", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--preservation-weight", type=float, default=100.)
    parser.add_argument("--lock", default="/workspace/fast-audiovae-convnext-20260909-r9/training-runs/.decoder-recipe-v2-expressive.runner.lock")
    args = parser.parse_args()
    with Path(args.lock).open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)
