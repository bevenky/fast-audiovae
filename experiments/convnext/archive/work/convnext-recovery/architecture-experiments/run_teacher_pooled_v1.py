"""One bounded head pilot with sample-pooled teacher waveform supervision.

Weights, canonical targets and source selection are pinned. Each requested arm
starts from the same retained candidate. These are isolated pilots without
new deployed layers, inference gates, automatic promotion or checkpoint edits.
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
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import weight_norm

from audiovae_student.reconstruction_v2 import ReconstructionV2Config
from run_joint_heads import (HEAD_NAMES, head_forward, capture_prehead, restore_head,
    masked_sums, normalized_loss, new_optimizer, build_bank, baseline_scales,
    canonical_peak_screen)
from run_spectral_heads import (SpectralObjective, spectral_counts, pooled_spectral_loss,
    contiguous_spans, aggregate_source_spectra, source_spectral_checks,
    legacy_source_spectral_checks, bank_score)

ARMS = ("pooled",)


def parse_arms(text):
    values = tuple(piece.strip() for piece in text.split(","))
    if not values or len(set(values)) != len(values) or any(value not in ARMS for value in values):
        raise ValueError("Need distinct supported comma-separated refinement arms")
    return values


def _weighted_modules(model):
    return (("head.conv", model.head.conv), ("output", model.output))


def effective_head_state(model):
    return {"head.conv.weight": model.head.conv.weight.detach().cpu().clone(),
            "head.conv.bias": model.head.conv.bias.detach().cpu().clone(),
            "activation.weight": model.activation.weight.detach().cpu().clone(),
            "output.weight": model.output.weight.detach().cpu().clone()}


@torch.no_grad()
def fold_weight_norm(model):
    receipts = []
    for name, module in _weighted_modules(model):
        if parametrize.is_parametrized(module, "weight"):
            before = module.weight.detach().clone()
            if not bool(torch.isfinite(before).all()):
                raise FloatingPointError("Nonfinite effective weight before folding")
            required = any(p.requires_grad for p in module.parametrizations.weight.parameters())
            # Multi-original parametrizations choose Parameter vs buffer from
            # the effective tensor's autograd flag; removal under no_grad
            # otherwise silently turns a trained weight into a buffer.
            with torch.enable_grad():
                parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
            if not isinstance(module.weight, torch.nn.Parameter):
                value = module.weight.detach()
                delattr(module, "weight")
                module.register_parameter("weight", torch.nn.Parameter(value, requires_grad=required))
            after = module.weight.detach()
            if not torch.equal(before, after):
                raise RuntimeError("Weight-normalization removal changed effective weight")
            receipts.append({"module": name, "effective_weights_exact": True,
                             "maximum_abs_difference": float((before-after).abs().max())})
    return {"folded": receipts, "plain_weight_export": True,
            "runtime_weight_normalization_operations": 0}


@torch.no_grad()
def restore_effective_head(model, state):
    fold_weight_norm(model)
    restore_head(model, state)


def configure_head(model, arm):
    if arm not in ARMS:
        raise ValueError("Unknown refinement arm")
    fold_weight_norm(model)
    if arm in ("wn", "wn_short"):
        for _, module in _weighted_modules(model):
            original = module.weight.detach().clone()
            weight_norm(module, name="weight", dim=0)
            # W=0 can be represented exactly by g=0 with an arbitrary unit v.
            # Leaving v=0 would cause 0/0 and invalidate otherwise valid rows.
            with torch.no_grad():
                zero_rows = original.flatten(1).square().sum(1) == 0
                if bool(zero_rows.any()):
                    magnitude = module.parametrizations.weight.original0
                    direction = module.parametrizations.weight.original1
                    flat = direction.flatten(1)
                    flat[zero_rows] = 0
                    flat[zero_rows, 0] = 1
                    magnitude[zero_rows] = 0
            if not bool(torch.isfinite(module.weight).all()):
                raise FloatingPointError("Weight-normalization migration produced nonfinite weights")
    wanted = {"head.conv.bias", "activation.weight"}
    for name, module in _weighted_modules(model):
        if parametrize.is_parametrized(module, "weight"):
            wanted.update({name+".parametrizations.weight.original0", name+".parametrizations.weight.original1"})
        else:
            wanted.add(name+".weight")
    named = dict(model.named_parameters())
    if not wanted.issubset(named):
        raise ValueError("Refinement head parameters are missing")
    for name, parameter in named.items():
        parameter.requires_grad_(name in wanted)
        parameter.grad = None
    model.eval()
    return [(name, named[name]) for name in sorted(wanted)]


def load_candidate_head(model, path, expected_sha256, base_sha256):
    path = Path(path).resolve(strict=True)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError("Retained candidate bytes differ from the approved SHA256")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved.get("format_version") != "spectral_head_factorial_v1" or saved.get("base_checkpoint_sha256") != base_sha256 or saved.get("arm") != "joint_spectral"
            or saved.get("mode") != "raw" or saved.get("updates") != 256):
        raise ValueError("Retained candidate identity or base checkpoint differs")
    state = saved.get("head_parameters")
    if not isinstance(state, dict) or set(state) != HEAD_NAMES:
        raise ValueError("Retained candidate does not contain the complete plain head")
    if any(not torch.is_tensor(value) or not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("Retained candidate contains invalid head tensors")
    current = effective_head_state(model)
    if any(state[name].shape != current[name].shape or state[name].dtype != current[name].dtype for name in HEAD_NAMES):
        raise ValueError("Retained candidate head shape or dtype differs")
    restore_effective_head(model, state)
    return {"path": str(path), "sha256": actual, "base_checkpoint_sha256": base_sha256,
            "retained_arm": saved["arm"], "retained_pilot_updates": saved["updates"]}


def pooled_target_scale(bank):
    """One fixed valid-sample-pooled teacher mean square for the entire fit set."""
    square = 0.; samples = 0; quiet_samples = 0; sources = []
    for row in bank:
        target, valid, quiet = row["target"], row["valid"], row["quiet"]
        if (target.shape != valid.shape or target.shape != quiet.shape or valid.dtype != torch.bool
                or quiet.dtype != torch.bool or bool((quiet & ~valid).any())):
            raise ValueError("Invalid fixed teacher calibration mask")
        selected = target.detach()[valid].double()
        if not bool(torch.isfinite(selected).all()):
            raise ValueError("Nonfinite valid teacher calibration samples")
        square += float(selected.square().sum())
        samples += selected.numel(); quiet_samples += int(quiet.sum())
        sources.append(str(row["source_id"]))
    if not samples or not math.isfinite(square) or square <= 0 or len(sources) != len(set(sources)):
        raise ValueError("Need distinct fit sources and positive finite teacher mean square")
    scale = square/samples
    return {"teacher_mean_square": scale}, {"teacher_square_sum": square,
        "valid_samples": samples, "quiet_samples": quiet_samples, "outside_samples": samples-quiet_samples,
        "teacher_mean_square": scale, "teacher_rms": math.sqrt(scale), "source_ids": sources,
        "sources": len(sources), "calibration_pool": "Entire fixed FIT set only",
        "policy": "One immutable teacher RMS-squared over all valid fit samples. No separate quiet/active normalization, per-source RMS normalization or baseline-error scale."}


def pooled_teacher_wave_loss(sums, totals, scales):
    """Sample-pooled MSE with one fixed teacher scale, independent of quiet mask."""
    count = totals["quiet_samples"]+totals["outside_samples"]
    scale = scales["teacher_mean_square"]
    if (min(totals["quiet_samples"], totals["outside_samples"]) < 0
            or not math.isfinite(scale) or scale <= 0):
        raise ValueError("Invalid pooled teacher count or scale")
    return (sums["quiet_teacher_square"]+sums["outside_teacher_square"]) / max(count, 1) / scale


def teacher_selection_checks(candidate, baseline):
    a, b = baseline["aggregate"], candidate["aggregate"]
    if any(a[k] != b[k] for k in ("samples", "quiet_samples", "outside_samples", "sources")):
        raise ValueError("Selection source or mask counts changed")
    checks = [{"name": "all/mae", "passed": b["mae"] <= a["mae"]*1.01},
              {"name": "all/mse", "passed": b["mse"] <= a["mse"]*1.01},
              {"name": "active/teacher_mse", "passed": b["outside_mse"] <= a["outside_mse"]*1.01},
              {"name": "all/peak", "passed": b["peak"] <= max(1., a["peak"])},
              {"name": "all/overshoot_samples", "passed": b["overshoot_samples"] <= a["overshoot_samples"]}]
    for key, ref in baseline["groups"].items():
        if key != "all" and ref["sources"] >= 2:
            checks.append({"name": key+"/mae", "passed": candidate["groups"][key]["mae"] <= ref["mae"]*1.05})
    if a["quiet_mse"] == 0:
        quiet_ratio = 1. if b["quiet_mse"] == 0 else None
        quiet_passed = b["quiet_mse"] == 0
    else:
        quiet_ratio = math.sqrt(b["quiet_mse"]/a["quiet_mse"])
        quiet_passed = quiet_ratio <= .90
    checks.append({"name": "quiet/meaningful_reduction_or_already_perfect", "passed": quiet_passed})
    spectra = source_spectral_checks(candidate["spectral_sources"], baseline["spectral_sources"])
    return {"qualified": all(c["passed"] for c in checks) and spectra["passed"],
            "quiet_rms_ratio": quiet_ratio, "checks": checks, "source_spectral": spectra,
            "policy": "Teacher-relative active error replaces baseline-preservation drift as a gate. Existing-source mel guard retained; no automatic promotion."}


def refinement_peak_screen(baseline, candidate):
    result = canonical_peak_screen(baseline, candidate)
    result["passed"] = (not result["regressing_crops"] and
        result["candidate_maximum_peak"] <= max(1., result["baseline_maximum_peak"]) and
        result["candidate_overshoot_observations"] <= result["baseline_overshoot_observations"])
    result["policy"] = "Restoring legitimate sub-full-scale teacher peaks is allowed; new or worse full-scale overshoot remains prohibited."
    return result


def batch_totals(selected, objectives):
    wave = {key: sum(row[key] for row in selected) for key in ("quiet_samples", "outside_samples")}
    spectra = {name: tuple(sum(spectral_counts(row["valid"], objective.config)[i] for row in selected)
              for i in range(len(objective.config.fft_sizes))) for name, objective in objectives.items()}
    return wave, spectra


def loss_branches(prediction, row, totals, spectral_totals, scales, objectives, arm, *, device,
                  preservation_weight, active_weight):
    target, baseline, valid, quiet = [row[k].to(device) for k in ("target", "baseline", "valid", "quiet")]
    sums = masked_sums(prediction, target, baseline, valid, quiet)
    if arm != "pooled":
        raise ValueError("This pilot supports only sample-pooled teacher supervision")
    wave = pooled_teacher_wave_loss(sums, totals, scales)
    branches = {"wave": wave}
    for name, objective in objectives.items():
        branches[name] = pooled_spectral_loss(objective.sums(prediction, target, valid), spectral_totals[name], objective.config)
    return branches


def backward_batch(model, selected, scales, objectives, arm, coefficients, *, device,
                   preservation_weight, active_weight):
    totals, mel_counts = batch_totals(selected, objectives)
    if totals["quiet_samples"]+totals["outside_samples"] <= 0:
        raise ValueError("No scored training samples")
    summary = {"loss": 0., "branch_losses": {}, **totals, "mel_element_counts": mel_counts}
    for row in selected:
        prediction = head_forward(model, row["features"].to(device))
        branches = loss_branches(prediction, row, totals, mel_counts, scales, objectives, arm,
            device=device, preservation_weight=preservation_weight, active_weight=active_weight)
        loss = sum(coefficients[name]*value for name, value in branches.items())
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite refinement objective")
        loss.backward()
        for name, value in branches.items():
            summary["branch_losses"][name] = summary["branch_losses"].get(name, 0.)+float(value.detach())
        summary["loss"] += float(loss.detach())
    return summary


def train_batch(model, bank, indices, optimizer, scales, arm, objectives, coefficients, *, device,
                preservation_weight, active_weight):
    optimizer.zero_grad(set_to_none=True)
    summary = backward_batch(model, [bank[i] for i in indices], scales, objectives, arm,
        coefficients, device=device, preservation_weight=preservation_weight, active_weight=active_weight)
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not named or any(not (name.startswith("head.conv.") or name.startswith("output.") or name == "activation.weight") for name, _ in named):
        raise RuntimeError("Unexpected refinement trainable parameters")
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for _, p in named):
        raise FloatingPointError("Missing or nonfinite head gradient")
    summary["gradient_norm_before_clip"] = float(torch.nn.utils.clip_grad_norm_([p for _, p in named], 1.))
    before = [p.detach().clone() for _, p in named]
    optimizer.step()
    summary["parameter_displacement"] = math.sqrt(sum(float((p.detach().double()-value.double()).square().sum()) for (_, p), value in zip(named, before)))
    if any(not bool(torch.isfinite(p).all()) for _, p in named):
        raise FloatingPointError("Nonfinite refinement parameters")
    return summary


def gradient_probe(model, bank, indices, scales, arm, objectives, coefficients, *, device,
                   preservation_weight, active_weight):
    """Diagnostic branch gradients on fixed training examples only; no update."""
    chosen = [bank[i] for i in indices]
    totals, mel_counts = batch_totals(chosen, objectives)
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named]
    parameter_sums = {}
    output_energy = {}; output_dot = {}
    wave_region_energy = {"quiet": 0., "active": 0.}
    for row in chosen:
        prediction = head_forward(model, row["features"].to(device))
        branches = loss_branches(prediction, row, totals, mel_counts, scales, objectives, arm,
            device=device, preservation_weight=preservation_weight, active_weight=active_weight)
        gradients = {}
        for name, value in branches.items():
            grads = torch.autograd.grad(value, [prediction, *parameters], retain_graph=True, allow_unused=True)
            gradients[name] = grads[0].detach().double()
            output_energy[name] = output_energy.get(name, 0.) + float(gradients[name].square().sum())
            if name == "wave":
                q = row["quiet"].to(device); a = row["valid"].to(device) & ~q
                wave_region_energy["quiet"] += float(gradients[name][q].square().sum())
                wave_region_energy["active"] += float(gradients[name][a].square().sum())
            if name not in parameter_sums:
                parameter_sums[name] = [torch.zeros_like(p) for p in parameters]
            for target, grad in zip(parameter_sums[name], grads[1:]):
                if grad is not None:
                    target.add_(grad.detach())
        for i, left in enumerate(gradients):
            for right in list(gradients)[i+1:]:
                key = left+"/"+right
                output_dot[key] = output_dot.get(key, 0.)+float((gradients[left]*gradients[right]).sum())
    param_energy = {name: sum(float(g.double().square().sum()) for g in grads) for name, grads in parameter_sums.items()}
    param_cosine = {}
    for i, left in enumerate(parameter_sums):
        for right in list(parameter_sums)[i+1:]:
            den = math.sqrt(param_energy[left]*param_energy[right])
            dot = sum(float((a.double()*b.double()).sum()) for a, b in zip(parameter_sums[left], parameter_sums[right]))
            param_cosine[left+"/"+right] = dot/den if den else None
    out_cosine = {}
    for key, dot in output_dot.items():
        left, right = key.split("/")
        den = math.sqrt(output_energy[left]*output_energy[right])
        out_cosine[key] = dot/den if den else None
    return {"source_ids": [row["source_id"] for row in chosen], "trainable": [name for name, _ in named],
        "output_gradient_energies": output_energy, "parameter_gradient_energies": param_energy,
        "wave_output_gradient_energy_by_region": wave_region_energy,
        "weighted_output_gradient_norms": {name: coefficients[name]*math.sqrt(value) for name, value in output_energy.items()},
        "weighted_parameter_gradient_norms": {name: coefficients[name]*math.sqrt(value) for name, value in param_energy.items()},
        "output_gradient_cosines": out_cosine, "parameter_gradient_cosines": param_cosine,
        "optimizer_updates": 0, "policy": "Fixed training-only examples; diagnostic autograd.grad does not populate optimizer .grad."}


def calibrate_coefficients(bank, scales, objectives, args, device):
    """Long-mel coefficient from the same first four training batches."""
    if set(objectives) != {"long"}:
        raise ValueError("Pooled pilot uses only the unchanged long-mel branch")
    rows = []
    total = {"wave": 0., "long": 0.}
    total_regions = {"quiet": 0., "active": 0.}
    for batch in range(4):
        chosen = bank[batch*args.batch_size:(batch+1)*args.batch_size]
        if len(chosen) != args.batch_size:
            raise ValueError("Need four complete fixed training batches")
        totals, mel_counts = batch_totals(chosen, objectives)
        energy = {name: 0. for name in total}; regions = {"quiet": 0., "active": 0.}
        dot = 0.
        for row in chosen:
            prediction = row["baseline"].to(device).detach().clone().requires_grad_(True)
            branches = loss_branches(prediction, row, totals, mel_counts, scales, objectives, "pooled",
                device=device, preservation_weight=args.preservation_weight, active_weight=args.active_weight)
            grads = {name: torch.autograd.grad(value, prediction, retain_graph=True)[0].detach().double()
                     for name, value in branches.items()}
            for name, grad in grads.items():
                energy[name] += float(grad.square().sum())
            quiet = row["quiet"].to(device); active = row["valid"].to(device) & ~quiet
            regions["quiet"] += float(grads["wave"][quiet].square().sum())
            regions["active"] += float(grads["wave"][active].square().sum())
            dot += float((grads["wave"]*grads["long"]).sum())
        if any(not math.isfinite(value) or value < 0 for value in energy.values()):
            raise ValueError("Invalid calibration output gradient energy")
        for key in total:
            total[key] += energy[key]
        for key in total_regions:
            total_regions[key] += regions[key]
        denominator = math.sqrt(energy["wave"]*energy["long"])
        rows.append({"batch": batch, "source_ids": [r["source_id"] for r in chosen],
            "output_gradient_energies": energy, "wave_output_gradient_energy_by_region": regions,
            "output_gradient_cosines": {"wave/long": dot/denominator if denominator else None}})
    if min(total.values()) <= 0:
        raise ValueError("Calibration needs positive combined wave/mel gradient energies")
    long_weight = math.sqrt(total["wave"]/total["long"])
    return {"coefficients": {"wave": 1., "long": long_weight},
        "summed_output_gradient_energies": total,
        "summed_wave_output_gradient_energy_by_region": total_regions,
        "batches": rows, "optimizer_updates": 0,
        "policy": "Same output-gradient-energy calibration as v2 on first four fixed training batches; long-mel initially equals sample-pooled teacher waveform gradient norm. No short spectral branch. Quiet/active energies are diagnostics only, not separate loss weights."}


def finite_update_checks(candidate, baseline):
    a, b = baseline["aggregate"], candidate["aggregate"]
    if any(a[key] != b[key] for key in ("samples", "quiet_samples", "outside_samples", "sources")):
        raise ValueError("Finite-update calibration mask geometry changed")
    checks = []
    for name in ("mae", "mse", "quiet_mse", "outside_mse"):
        if a[name] is not None:
            checks.append({"name": name, "baseline": a[name], "candidate": b[name],
                           "passed": b[name] <= a[name]*1.01})
    spectra = source_spectral_checks(candidate["spectral_sources"], baseline["spectral_sources"])
    original_peak_sources = {row["source_id"] for row in baseline["rows"] if row["overshoot_samples"] > 0}
    candidate_peak_sources = {row["source_id"] for row in candidate["rows"] if row["overshoot_samples"] > 0}
    checks.extend([
        {"name": "peak/aggregate", "baseline": a["peak"], "candidate": b["peak"],
         "limit": max(1., a["peak"])*1.01+2e-6, "passed": b["peak"] <= max(1., a["peak"])*1.01+2e-6},
        {"name": "peak/aggregate_count", "baseline": a["overshoot_samples"], "candidate": b["overshoot_samples"],
         "passed": b["overshoot_samples"] <= a["overshoot_samples"]},
        {"name": "peak/no_new_source", "baseline": sorted(original_peak_sources), "candidate": sorted(candidate_peak_sources),
         "passed": candidate_peak_sources <= original_peak_sources}])
    return {"passed": all(row["passed"] for row in checks) and spectra["passed"],
            "checks": checks, "source_spectral": spectra,
            "policy": "Disposable finite-update entry check: teacher errors and aggregate peak permit at most 1% movement (peak +2e-6 numerical tolerance); overshoot count/source guards retained. Final selection/canonical peak rules remain strict. No old-student displacement gate or required 10% quiet improvement per calibration step."}


def migration_parity(model, first, device):
    maximum = 0.
    rows = []
    with torch.no_grad():
        for row in first:
            actual = head_forward(model, row["features"].to(device))
            difference = float((actual-row["baseline"].to(device)).abs().max())
            maximum = max(maximum, difference)
            rows.append({"source_id": row["source_id"], "max_abs_difference": difference})
    if maximum > 2e-6:
        raise RuntimeError("Head parametrization migration exceeded waveform parity tolerance")
    return {"passed": True, "max_abs_difference": maximum, "tolerance": 2e-6, "rows": rows}


def calibrate_rate(model, bank, initial, scales, objectives, coefficients, args, device):
    panel = bank[:4*args.batch_size]
    if len(panel) != 4*args.batch_size:
        raise ValueError("Incomplete fixed calibration panel")
    restore_effective_head(model, initial)
    baseline = bank_score(model, panel, device, objectives["long"])
    if min(baseline["aggregate"]["quiet_samples"], baseline["aggregate"]["outside_samples"]) <= 0:
        raise ValueError("Calibration panel needs both quiet and active samples")
    trials = []
    try:
        for half in range(7):
            rate = args.learning_rate/(2**half)
            outcomes = []
            for arm in ARMS:
                for batch in range(4):
                    restore_effective_head(model, initial)
                    parameters = [p for _, p in configure_head(model, arm)]
                    parity = migration_parity(model, panel, device)
                    optimizer = new_optimizer(parameters, rate)
                    indices = list(range(batch*args.batch_size, (batch+1)*args.batch_size))
                    update = train_batch(model, bank, indices, optimizer, scales, arm, objectives, coefficients,
                        device=device, preservation_weight=args.preservation_weight, active_weight=args.active_weight)
                    score = bank_score(model, panel, device, objectives["long"])
                    checks = finite_update_checks(score, baseline)
                    outcomes.append({"arm": arm, "batch": batch, "passed": checks["passed"],
                                     "checks": checks, "migration_parity": parity, "update": update})
            trials.append({"learning_rate": rate, "outcomes": outcomes})
            if all(row["passed"] for row in outcomes):
                return {"chosen_learning_rate": rate, "trials": trials, "retained_optimizer_updates": 0,
                    "calibrated_arms": list(ARMS), "policy": "Sole pooled arm; four isolated single-batch updates restored from retained candidate and scored on the same 32-source training-only panel. At most six halvings; temporary updates discarded. Same finite-update and strict endpoint checks as v2."}
        return {"chosen_learning_rate": None, "trials": trials,
                "reason": "No common learning rate passed the finite-update teacher safeguards"}
    finally:
        restore_effective_head(model, initial)
        for parameter in model.parameters():
            parameter.grad = None


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

    arms = parse_arms(args.arms)
    if not 0 < args.learning_rate <= 1e-6 or not 4 <= args.steps or args.steps*args.batch_size > 2048:
        raise ValueError("Pilot capped at 2,048 distinct training crops and LR1e-6, minimum four batches")
    if min(args.batch_size, args.validation_count, args.eval_every) <= 0 or args.active_weight != 1. or args.preservation_weight != 100.:
        raise ValueError("Legacy compatibility flags must retain defaults; pooled loss does not use separate quiet/active weights")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    if file_sha(checkpoint) != QUARTER_SHA:
        raise ValueError("Wrong retained base checkpoint")
    if args.candidate_sha256 != "b612baf57c5a3ae113ed67ea1a48f6f9cb2d740001aa5b2b5dc3a7a5ba9097e2":
        raise ValueError("Run requires the approved retained joint_spectral candidate")
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
    original_head = effective_head_state(model)
    original_modes = {m: m.training for m in model.modules()}
    original_requires = {name: p.requires_grad for name, p in model.named_parameters()}
    original_rng = _rng_state(); fit_rng = deepcopy(saved["rng"])
    started = time.monotonic()
    candidate_identity = load_candidate_head(model, args.candidate_heads, args.candidate_sha256, QUARTER_SHA)
    initial = effective_head_state(model)
    model.eval(); configure_head(model, "pooled")
    objectives = {"long": SpectralObjective().to(engine.device)}
    def frozen_state():
        return {k: v for k, v in model.state_dict().items()
                if not (k.startswith("head.conv.") or k.startswith("output.") or k == "activation.weight")}
    frozen_hash = state_fingerprint(frozen_state())
    def verify_frozen():
        if state_fingerprint(frozen_state()) != frozen_hash:
            raise RuntimeError("Frozen body or normalization changed")
    def canonical(name, updates=0):
        transient = []; spectral_rows = []
        def observe(module, inputs, prediction):
            crop = panel["crops"][len(transient)]
            target, valid, quiet = validation_masks(crop, device=engine.device, quiet_config=engine.config.quiet_audio)
            transient.append({"source_id": crop.source_id, "start_frame": crop.start_frame,
                              **transient_errors(prediction, target, valid)})
            spectral_rows.append({"source_id": crop.source_id, "start_frame": crop.start_frame,
                "regions": objectives["long"].regions(prediction, target, valid, quiet)})
        handle = model.register_forward_hook(observe)
        try:
            report = build_canonical_evaluation(engine, panel["crops"], panel["metadata"], panel["receipt"])
        finally:
            handle.remove()
        report["transient_diagnostics"] = transient
        report["spectral_head_rows"] = spectral_rows
        report["spectral_head_sources"] = aggregate_source_spectra(spectral_rows, objectives["long"].config)
        report["teacher_refinement"] = {"name": name, "base_checkpoint_step": 8890,
            "retained_head_updates": 256, "additional_pilot_updates": updates,
            "engine_training_clock_unchanged": True}
        atomic_json(out/(name+"-canonical.json"), report)
        return report
    def compare_refinement(reference, observed):
        result = compare(reference, observed)
        result["source_spectral"] = source_spectral_checks(observed["spectral_head_sources"], reference["spectral_head_sources"])
        result["legacy_source_spectral"] = legacy_source_spectral_checks(observed, reference)
        result["no_new_peak_regression"] = refinement_peak_screen(reference, observed)
        result["silence_target_passed"] = (result["quality_screen_passed"] and result["natural_quiet_ratio"] <= .90
            and result["steady_silence_ratio"] < 1 and result["source_spectral"]["passed"]
            and result["legacy_source_spectral"]["passed"] and result["no_new_peak_regression"]["passed"])
        return result
    try:
        sources = {Path(__file__).resolve()}
        for module in ("run_spectral_heads", "run_joint_heads", "fit_support", "diagnostic_common", "native_chain",
                       "canonical_evaluation", "run_heads", "training_overlay", "audiovae_student.reconstruction_v2"):
            sources.add(Path(importlib.import_module(module).__file__).resolve())
        atomic_json(out/"identity.json", {"version": "teacher_pooled_v1", "args": vars(args),
            "arms": arms, "candidate": candidate_identity, "checkpoint_sha256": QUARTER_SHA,
            "restored_engine_sha256": original_engine, "frozen_body_sha256": frozen_hash,
            "canonical_panel": panel["identity"], "training_overlay": overlay["identity"], "inventory": inventory,
            "selection_sha256": selection["identity_sha256"], "torch": str(torch.__version__),
            "cudnn": torch.backends.cudnn.version(), "teacher_state_sha256": ctx.parent["identity"]["data"]["teacher_state_sha256"],
            "source_files": {str(p): file_sha(p) for p in sources},
            "spectral_configs": {name: value.config.__dict__ for name, value in objectives.items()},
            "optimizer": {"name": "AdamW", "fresh_identical_states": True, "weight_decay": 0., "gradient_norm_cap": 1.,
                          "betas": [.9, .999], "eps": 1e-8, "mixed_precision": False},
            "scope": "Existing plain head only with pooled teacher waveform and unchanged long-mel supervision. No separate quiet weighting, WN, short mel, GAN, new layer or automatic promotion.",
            "derived_from": "run_teacher_refinements_v2.py",
            "derived_source_sha256": file_sha(Path(__file__).with_name("run_teacher_refinements_v2.py"))})
        banks = {split: build_bank(model, pool, selection["splits"][split]["indices"], ctx.data["rows"],
            engine.config.quiet_audio, engine.device, status, split) for split in ("fit", "validation")}
        scales, counts = pooled_target_scale(banks["fit"])
        atomic_json(out/"fit-accounting.json", {"scales": scales, "counts": counts,
            "baseline": "Same retained candidate; waveform normalization comes from fixed teacher targets, not baseline errors", "distinct_sources_per_arm": len(banks["fit"]),
            "matched_order": True, "source_repeats_per_arm": 0,
            "spectral_eligibility": {split: {name: {
                "short_spans": sum(stop-start < max(obj.config.fft_sizes) for row in bank for start, stop in contiguous_spans(row["valid"])),
                "skipped_samples": sum(stop-start for row in bank for start, stop in contiguous_spans(row["valid"]) if stop-start < max(obj.config.fft_sizes))}
                for name, obj in objectives.items()} for split, bank in banks.items()}})
        reference_selection = bank_score(model, banks["validation"], engine.device, objectives["long"])
        atomic_json(out/"baseline-validation.json", reference_selection)
        calibration = calibrate_coefficients(banks["fit"], scales, objectives, args, engine.device)
        atomic_json(out/"coefficient-calibration.json", calibration)
        coefficients = calibration["coefficients"]
        rate = calibrate_rate(model, banks["fit"], initial, scales, objectives, coefficients, args, engine.device)
        atomic_json(out/"rate-calibration.json", rate)
        if rate["chosen_learning_rate"] is None:
            atomic_json(out/"complete.json", {"complete": False, "stopped_before_training": True,
                "reason": rate["reason"], "promoted": False})
            return
        baseline_report = canonical("baseline")
        train_panel = banks["fit"][:4*args.batch_size]
        baseline_train_panel = bank_score(model, train_panel, engine.device, objectives["long"])
        atomic_json(out/"baseline-training-panel.json", baseline_train_panel)
        all_results = {}
        for arm in arms:
            restore_effective_head(model, initial); _restore_rng(fit_rng)
            named = configure_head(model, arm); parameters = [p for _, p in named]
            migration = migration_parity(model, train_panel, engine.device)
            optimizer = new_optimizer(parameters, rate["chosen_learning_rate"])
            probe_indices = list(range(4*args.batch_size))
            probes = {"begin": gradient_probe(model, banks["fit"], probe_indices, scales, arm, objectives, coefficients,
                device=engine.device, preservation_weight=args.preservation_weight, active_weight=args.active_weight)}
            atomic_json(out/(arm+"-gradient-probes.json"), probes)
            validations = []; best = None; best_state = None
            arm_start = time.monotonic()
            with (out/(arm+"-training.jsonl")).open("w") as logfile:
                for step in range(1, args.steps+1):
                    indices = list(range((step-1)*args.batch_size, step*args.batch_size))
                    update = train_batch(model, banks["fit"], indices, optimizer, scales, arm, objectives, coefficients,
                        device=engine.device, preservation_weight=args.preservation_weight, active_weight=args.active_weight)
                    log = {"step": step, "arm": arm, "sources_seen": step*args.batch_size,
                           "seconds": time.monotonic()-arm_start, **update}
                    logfile.write(json.dumps(log, allow_nan=False)+"\n"); logfile.flush()
                    if step % 16 == 0:
                        status("teacher_pooled_training", **log)
                    if step % args.eval_every == 0 or step == args.steps:
                        verify_frozen()
                        score = bank_score(model, banks["validation"], engine.device, objectives["long"])
                        checks = teacher_selection_checks(score, reference_selection)
                        entry = {"step": step, "score": score, "selection": checks}
                        validations.append(entry); atomic_json(out/(arm+"-validation.json"), validations)
                        if checks["qualified"] and (best is None or score["aggregate"]["quiet_mse"] < best["score"]["aggregate"]["quiet_mse"]):
                            best, best_state = entry, effective_head_state(model)
                        status("teacher_pooled_validation", arm=arm, step=step, qualified=checks["qualified"], quiet_rms_ratio=checks["quiet_rms_ratio"])
            probes["end"] = gradient_probe(model, banks["fit"], probe_indices, scales, arm, objectives, coefficients,
                device=engine.device, preservation_weight=args.preservation_weight, active_weight=args.active_weight)
            atomic_json(out/(arm+"-gradient-probes.json"), probes)
            pre_fold = []
            with torch.no_grad():
                for row in train_panel:
                    pre_fold.append(head_forward(model, row["features"].to(engine.device)).detach().cpu())
            folding = fold_weight_norm(model)
            with torch.no_grad():
                differences = [float((head_forward(model, row["features"].to(engine.device)).cpu()-before).abs().max())
                               for row, before in zip(train_panel, pre_fold)]
            if max(differences) > 2e-6:
                raise RuntimeError("Folding changed waveform beyond parity tolerance")
            folding["waveform_max_abs_difference"] = max(differences)
            folding["waveform_tolerance"] = 2e-6
            final_state = effective_head_state(model)
            torch.save({"format_version": "teacher_pooled_v1", "base_checkpoint_sha256": QUARTER_SHA,
                "retained_candidate_sha256": args.candidate_sha256, "arm": arm, "mode": "raw", "updates": args.steps,
                "head_parameters": final_state, "selected_step": best["step"] if best else None,
                "selected_head_parameters": best_state, "qualification": "Experimental plain head weights; no automatic promotion or resume state"},
                out/(arm+"-heads.pt"))
            final_report = canonical(arm+"-final", args.steps)
            comparison = compare_refinement(baseline_report, final_report)
            final_train = bank_score(model, train_panel, engine.device, objectives["long"])
            atomic_json(out/(arm+"-training-panel.json"), final_train)
            selected_comparison = None
            if best_state is not None:
                if best["step"] == args.steps:
                    selected_comparison = comparison
                else:
                    restore_effective_head(model, best_state)
                    selected_comparison = compare_refinement(baseline_report, canonical(arm+"-selected", best["step"]))
            all_results[arm] = {"final": comparison, "selected_step": best["step"] if best else None,
                "selected": selected_comparison, "seconds": time.monotonic()-arm_start, "migration": migration,
                "folding": folding, "head_sha256": file_sha(out/(arm+"-heads.pt")), "promoted": False}
            atomic_json(out/"comparisons.json", all_results)
            verify_frozen()
        completion = {"complete": True, "seconds": time.monotonic()-started,
            "comparisons": all_results, "promoted": False}
    finally:
        restore_effective_head(model, original_head)
        for name, p in model.named_parameters():
            p.requires_grad_(original_requires[name]); p.grad = None
        for module, mode in original_modes.items():
            module.training = mode
        _restore_rng(original_rng)
        if state_fingerprint(engine.state_dict()) != original_engine:
            raise RuntimeError("Original engine was not preserved")
        if file_sha(checkpoint) != QUARTER_SHA or file_sha(args.candidate_heads) != args.candidate_sha256:
            raise RuntimeError("Retained checkpoint or candidate file changed")
        for key in ("receipt", "cache"):
            if file_sha(panel["identity"][key+"_path"]) != panel["identity"][key+"_sha256"]:
                raise RuntimeError("Canonical input bytes changed")
        atomic_json(out/"preservation.json", {"original_engine_preserved": True, "base_checkpoint_preserved": True,
            "retained_candidate_preserved": True, "canonical_inputs_preserved": True,
            "original_files": ctx.verify_files(), "fresh_files": verify_overlay_files(overlay), "promoted": False})
    completion["original_engine_preserved"] = True
    completion["retained_candidate_preserved"] = True
    atomic_json(out/"complete.json", completion)
    status("teacher_pooled_complete", out=str(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("checkpoint", "candidate-heads", "training-receipt", "canonical-receipt", "canonical-inventory", "out"):
        parser.add_argument("--"+arg, required=True)
    parser.add_argument("--candidate-sha256", default="b612baf57c5a3ae113ed67ea1a48f6f9cb2d740001aa5b2b5dc3a7a5ba9097e2")
    parser.add_argument("--arms", default="pooled")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-count", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--preservation-weight", type=float, default=100.)
    parser.add_argument("--active-weight", type=float, default=1.)
    parser.add_argument("--lock", default="/workspace/fast-audiovae-convnext-20260909-r9/training-runs/.decoder-recipe-v2-expressive.runner.lock")
    args = parser.parse_args()
    with Path(args.lock).open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)
