"""Matched final-existing-block adaptation with optional auxiliary teacher loss.

The fixed prefix, outer normalization and encoder remain unchanged. Teacher
features are authenticated in the original full-source execution geometry.
The fitted auxiliary readout exists during training only and is not exported.
Version 3 restores authenticated reference-teacher startup warmup before capture.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import fcntl
import importlib
import json
import math
from pathlib import Path
import time

import torch

from run_joint_heads import HEAD_NAMES, baseline_scales, new_optimizer
from run_spectral_heads import (SpectralObjective, aggregate_source_spectra, source_spectral_checks,
    legacy_source_spectral_checks, contiguous_spans, bank_score as head_bank_score)
from run_teacher_refinements_v2 import (load_candidate_head, effective_head_state, restore_effective_head,
    teacher_selection_checks, finite_update_checks, refinement_peak_screen, batch_totals,
    loss_branches, calibrate_coefficients)
from audiovae_student.reconstruction_v2 import ReconstructionV2Config
from refinement_features import (capture_last_block_input, last_block_forward,
    last_block_parameter_names, select_last_block_parameters, capture_full_source_teacher_features,
    complete_teacher_feature_mask, fit_feature_normalization, AuxiliaryPhaseReadout,
    fit_auxiliary_ridge, auxiliary_feature_loss)

from teacher_reference_warmup import warmup_authenticated_teacher

ARMS = ("late", "aux")


def parse_arms(text):
    values = tuple(value.strip() for value in text.split(","))
    if not values or len(set(values)) != len(values) or any(value not in ARMS for value in values):
        raise ValueError("Need distinct late,aux arms")
    return values


def late_state(model):
    names = last_block_parameter_names(model)
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if name in names}


@torch.no_grad()
def restore_late_state(model, state):
    names = last_block_parameter_names(model)
    named = dict(model.named_parameters())
    if set(state) != names or any(state[name].shape != named[name].shape or state[name].dtype != named[name].dtype
                                 or not bool(torch.isfinite(state[name]).all()) for name in names):
        raise ValueError("Invalid complete final-block/head state")
    for name, value in state.items():
        named[name].copy_(value.to(named[name].device))


class _ScoreView:
    """Reuse unchanged waveform/spectral scoring with explicit prefix replay."""
    def __init__(self, model):
        self.head = lambda value: model.head(model.affine(model.blocks[9](value)))
        self.activation = model.activation
        self.output = model.output
        self._waveform = model._waveform


def bank_score(model, bank, device, spectral):
    return head_bank_score(_ScoreView(model), bank, device, spectral)


@torch.no_grad()
def build_bank(model, crops, indices, rows, config, device, status, split):
    from fit_support import validation_masks
    bank = []
    for position, index in enumerate(indices, 1):
        crop = crops[index]
        features, baseline = capture_last_block_input(model, crop.latents.to(device))
        target, valid, quiet = validation_masks(crop, device="cpu", quiet_config=config)
        if baseline.shape != target.shape:
            raise ValueError("Frozen-prefix waveform and sealed teacher geometry differ")
        meta = rows[crop.source_id]
        bank.append({"features": features.cpu(), "baseline": baseline.cpu(), "target": target,
            "valid": valid, "quiet": quiet, "quiet_samples": int(quiet.sum()),
            "outside_samples": int((valid & ~quiet).sum()), "source_id": crop.source_id,
            "start_frame": crop.start_frame, "pool_index": index,
            **{key: meta.get(key) for key in ("language", "condition", "dataset")}})
        if position % 64 == 0:
            status("late_prefix_features", split=split, completed=position, total=len(indices))
    return bank


def attach_teacher_features(bank, indices, teacher, crops, data, teacher_sha, *, status):
    receipts = []
    for position, index in enumerate(indices, 1):
        row = bank[index]
        if "teacher_features" in row:
            continue
        capture = capture_full_source_teacher_features(teacher, crops[row["pool_index"]], data,
            expected_teacher_state_sha256=teacher_sha, verify_teacher_state_hash=False)
        if not torch.equal(capture["valid"].cpu(), row["valid"]) or not torch.equal(
                capture["feature_valid"].cpu(), complete_teacher_feature_mask(row["valid"])):
            raise RuntimeError("Auxiliary capture changed the sealed score mask")
        row["teacher_features"] = capture["teacher_features"].detach().cpu()
        row["feature_valid"] = capture["feature_valid"].detach().cpu()
        row["feature_receipt"] = capture["receipt"]
        receipts.append(capture["receipt"])
        if position % 16 == 0:
            status("auxiliary_teacher_capture", completed=position, requested=len(indices))
    return receipts


def sample_branches(model, row, totals, mel_counts, total_cells, scales, objectives, arm,
                    auxiliary, *, device, args):
    prediction, features = last_block_forward(model, row["features"].to(device), return_features=True)
    branches = loss_branches(prediction, row, totals, mel_counts, scales, objectives, "teacher",
        device=device, preservation_weight=args.preservation_weight, active_weight=args.active_weight)
    if arm == "aux":
        branches["feature"] = auxiliary_feature_loss(auxiliary["adapter"], features,
            row["teacher_features"].to(device), row["feature_valid"].to(device), auxiliary["normalization"],
            total_valid_cells=total_cells)
    return prediction, branches


def train_batch(model, bank, indices, optimizer, scales, arm, objectives, coefficients,
                auxiliary, *, device, args):
    chosen = [bank[index] for index in indices]
    totals, mel_counts = batch_totals(chosen, objectives)
    total_cells = sum(int(row["feature_valid"].sum()) for row in chosen) if arm == "aux" else 0
    optimizer.zero_grad(set_to_none=True)
    summary = {"loss": 0., "branch_losses": {}, **totals, "feature_cells": total_cells}
    for row in chosen:
        _, branches = sample_branches(model, row, totals, mel_counts, total_cells, scales,
            objectives, arm, auxiliary, device=device, args=args)
        loss = sum(coefficients[name]*value for name, value in branches.items())
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite final-block objective")
        loss.backward()
        summary["loss"] += float(loss.detach())
        for name, value in branches.items():
            summary["branch_losses"][name] = summary["branch_losses"].get(name, 0.)+float(value.detach())
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if {name for name, _ in named} != last_block_parameter_names(model):
        raise RuntimeError("Unexpected final-block trainable scope")
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for _, p in named):
        raise FloatingPointError("Missing/nonfinite final-block gradient")
    summary["gradient_norm_before_clip"] = float(torch.nn.utils.clip_grad_norm_([p for _, p in named], 1.))
    before = [p.detach().clone() for _, p in named]
    optimizer.step()
    summary["parameter_displacement"] = math.sqrt(sum(float((p.detach().double()-old.double()).square().sum())
        for (_, p), old in zip(named, before)))
    if any(not bool(torch.isfinite(p).all()) for _, p in named):
        raise FloatingPointError("Nonfinite final-block parameters")
    return summary


def gradient_probe(model, bank, indices, scales, arm, objectives, coefficients, auxiliary,
                   *, device, args):
    chosen = [bank[index] for index in indices]
    totals, mel_counts = batch_totals(chosen, objectives)
    cells = sum(int(row["feature_valid"].sum()) for row in chosen) if arm == "aux" else 0
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    parameter_sums = {}; output_energy = {}; output_dot = {}
    for row in chosen:
        prediction, branches = sample_branches(model, row, totals, mel_counts, cells, scales,
            objectives, arm, auxiliary, device=device, args=args)
        output_grads = {}
        for name, value in branches.items():
            gradients = torch.autograd.grad(value, [prediction, *params], retain_graph=True, allow_unused=True)
            grad = gradients[0]
            if grad is not None:
                output_grads[name] = grad.detach().double()
                output_energy[name] = output_energy.get(name, 0.)+float(output_grads[name].square().sum())
            if name not in parameter_sums:
                parameter_sums[name] = [torch.zeros_like(p) for p in params]
            for total, g in zip(parameter_sums[name], gradients[1:]):
                if g is not None:
                    total.add_(g.detach())
        for i, left in enumerate(output_grads):
            for right in list(output_grads)[i+1:]:
                key = left+"/"+right
                output_dot[key] = output_dot.get(key, 0.)+float((output_grads[left]*output_grads[right]).sum())
    energies = {name: sum(float(g.double().square().sum()) for g in values) for name, values in parameter_sums.items()}
    cosines = {}
    for i, left in enumerate(parameter_sums):
        for right in list(parameter_sums)[i+1:]:
            denominator = math.sqrt(energies[left]*energies[right])
            value = sum(float((a.double()*b.double()).sum()) for a, b in zip(parameter_sums[left], parameter_sums[right]))
            cosines[left+"/"+right] = value/denominator if denominator else None
    shared_indices = [i for i, (name, _) in enumerate(named) if name.startswith("blocks.9.")]
    shared_energy = {name: sum(float(values[i].double().square().sum()) for i in shared_indices)
                     for name, values in parameter_sums.items()}
    shared_cosines = {}
    for i, left in enumerate(parameter_sums):
        for right in list(parameter_sums)[i+1:]:
            denominator = math.sqrt(shared_energy[left]*shared_energy[right])
            value = sum(float((parameter_sums[left][j].double()*parameter_sums[right][j].double()).sum()) for j in shared_indices)
            shared_cosines[left+"/"+right] = value/denominator if denominator else None
    return {"source_ids": [row["source_id"] for row in chosen],
        "parameter_gradient_energies": energies, "output_gradient_energies": output_energy,
        "weighted_parameter_gradient_norms": {name: coefficients[name]*math.sqrt(value) for name, value in energies.items()},
        "parameter_gradient_cosines": cosines,
        "shared_block_gradient_energies": shared_energy, "shared_block_gradient_cosines": shared_cosines,
        "weighted_shared_block_gradient_norms": {name: coefficients[name]*math.sqrt(value) for name, value in shared_energy.items()},
        "output_gradient_cosines": {key: value/math.sqrt(output_energy[key.split('/')[0]]*output_energy[key.split('/')[1]])
            if output_energy[key.split('/')[0]]*output_energy[key.split('/')[1]] > 0 else None for key, value in output_dot.items()},
        "policy": "Auxiliary branch has no gradient through final waveform. Global head+block and shared-block gradients are reported separately; coefficient calibration uses shared block only. No optimizer updates."}


def calibrate_auxiliary_weight(model, bank, scales, objectives, coefficients, auxiliary, args, device):
    if args.batch_size <= 0 or len(bank) < 4*args.batch_size:
        raise ValueError("Need four complete fixed calibration batches")
    params = [p for name, p in model.named_parameters() if name.startswith("blocks.9.") and p.requires_grad]
    energies = {"reconstruction": 0., "feature": 0.}; rows = []
    for batch in range(4):
        chosen = bank[batch*args.batch_size:(batch+1)*args.batch_size]
        totals, mel_counts = batch_totals(chosen, objectives)
        cells = sum(int(row["feature_valid"].sum()) for row in chosen)
        summed = {name: [torch.zeros_like(p) for p in params] for name in energies}
        for row in chosen:
            _, branches = sample_branches(model, row, totals, mel_counts, cells, scales,
                objectives, "aux", auxiliary, device=device, args=args)
            values = {"reconstruction": coefficients["wave"]*branches["wave"]+coefficients["long"]*branches["long"],
                      "feature": branches["feature"]}
            for name, value in values.items():
                gradients = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
                for total, grad in zip(summed[name], gradients):
                    if grad is not None:
                        total.add_(grad.detach())
        energy = {name: sum(float(g.double().square().sum()) for g in values) for name, values in summed.items()}
        if any(not math.isfinite(v) or v < 0 for v in energy.values()):
            raise ValueError("Nonfinite auxiliary calibration gradient energy")
        for name in energies:
            energies[name] += energy[name]
        dot = sum(float((a.double()*b.double()).sum()) for a, b in zip(summed["reconstruction"], summed["feature"]))
        denominator = math.sqrt(energy["reconstruction"]*energy["feature"])
        rows.append({"batch": batch, "source_ids": [r["source_id"] for r in chosen], "shared_block_gradient_energies": energy,
                     "cosine": dot/denominator if denominator else None})
    if min(energies.values()) <= 0:
        raise ValueError("Auxiliary calibration needs positive shared-block branch gradients")
    return {"feature_weight": .1*math.sqrt(energies["reconstruction"]/energies["feature"]),
        "summed_shared_block_energies": energies, "batches": rows,
        "policy": "Auxiliary branch starts at 10% of combined weighted reconstruction norm on shared final-block parameters; first four training batches after fixed ridge calibration."}


def calibrate_rate(model, bank, initial, scales, objectives, coefficients, auxiliary, args, device):
    if args.batch_size <= 0 or len(bank) < 4*args.batch_size:
        raise ValueError("Need four complete fixed calibration batches")
    selected = bank[:4*args.batch_size]
    baseline = bank_score(model, selected, device, objectives["long"])
    trials = []
    try:
        for half in range(7):
            rate = args.learning_rate/(2**half)
            outcomes = []
            for arm in ARMS:
                for batch in range(4):
                    restore_late_state(model, initial)
                    parameters = [p for _, p in select_last_block_parameters(model)]
                    optimizer = new_optimizer(parameters, rate)
                    update = train_batch(model, bank, list(range(batch*args.batch_size, (batch+1)*args.batch_size)),
                        optimizer, scales, arm, objectives, coefficients, auxiliary, device=device, args=args)
                    score = bank_score(model, selected, device, objectives["long"])
                    checks = finite_update_checks(score, baseline)
                    outcomes.append({"arm": arm, "batch": batch, "update": update, "checks": checks, "passed": checks["passed"]})
            trials.append({"learning_rate": rate, "outcomes": outcomes})
            if all(value["passed"] for value in outcomes):
                return {"chosen_learning_rate": rate, "trials": trials, "retained_optimizer_updates": 0,
                    "policy": "Common late/aux rate; four isolated updates per arm from same candidate; fixed training-only 32-source panel; at most six halvings."}
        return {"chosen_learning_rate": None, "trials": trials, "reason": "No common final-block rate passed fixed safeguards"}
    finally:
        restore_late_state(model, initial)
        for p in model.parameters():
            p.grad = None


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
        raise ValueError("Fixed alpha1 teacher and alpha100 control protocol required")
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
    original_late = late_state(model)
    original_modes = {m: m.training for m in model.modules()}
    original_requires = {name: p.requires_grad for name, p in model.named_parameters()}
    original_rng = _rng_state(); fit_rng = deepcopy(saved["rng"])
    started = time.monotonic()
    candidate_identity = load_candidate_head(model, args.candidate_heads, args.candidate_sha256, QUARTER_SHA)
    initial = late_state(model)
    model.eval(); select_last_block_parameters(model)
    objectives = {"long": SpectralObjective().to(engine.device), "short": SpectralObjective(ReconstructionV2Config(
        fft_sizes=(256, 512), mel_bands=(32, 64))).to(engine.device)}
    selected_names = last_block_parameter_names(model)
    def frozen_state():
        return {k: v for k, v in model.state_dict().items() if k not in selected_names}
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
        report["late_refinement"] = {"name": name, "base_checkpoint_step": 8890,
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
        for module in ("run_spectral_heads", "run_joint_heads", "run_teacher_refinements_v2", "refinement_features", "full_source_teacher_head", "teacher_reference_warmup", "fit_support", "diagnostic_common", "native_chain",
                       "canonical_evaluation", "run_heads", "training_overlay", "audiovae_student.reconstruction_v2"):
            sources.add(Path(importlib.import_module(module).__file__).resolve())
        atomic_json(out/"identity.json", {"version": "late_refinements_v3", "args": vars(args),
            "arms": arms, "candidate": candidate_identity, "checkpoint_sha256": QUARTER_SHA,
            "restored_engine_sha256": original_engine, "frozen_body_sha256": frozen_hash,
            "canonical_panel": panel["identity"], "training_overlay": overlay["identity"], "inventory": inventory,
            "selection_sha256": selection["identity_sha256"], "torch": str(torch.__version__),
            "cudnn": torch.backends.cudnn.version(), "teacher_state_sha256": ctx.parent["identity"]["data"]["teacher_state_sha256"],
            "source_files": {str(p): file_sha(p) for p in sources},
            "spectral_configs": {name: value.config.__dict__ for name, value in objectives.items()},
            "optimizer": {"name": "AdamW", "fresh_identical_states": True, "weight_decay": 0., "gradient_norm_cap": 1.,
                          "betas": [.9, .999], "eps": 1e-8, "mixed_precision": False},
            "trainable": sorted(selected_names),
            "scope": "Existing block9 and head, including block9 LayerNorm learned parameters. First nine blocks and outer normalization parameters/statistics frozen. Auxiliary adapter is training-only and excluded from exported student state. No new deployed layer or automatic promotion."})
        banks = {split: build_bank(model, pool, selection["splits"][split]["indices"], ctx.data["rows"],
            engine.config.quiet_audio, engine.device, status, split) for split in ("fit", "validation")}
        teacher = ctx.teacher(); teacher.model.eval()
        teacher_sha = ctx.parent["identity"]["data"]["teacher_state_sha256"]
        if state_fingerprint(teacher.model.state_dict()) != teacher_sha:
            raise RuntimeError("Frozen teacher state differs before auxiliary calibration")
        first_fit_crop = pool[banks["fit"][0]["pool_index"]]
        warmup_receipt = warmup_authenticated_teacher(teacher, first_fit_crop, ctx.data,
            expected_teacher_state_sha256=teacher_sha)
        atomic_json(out/"teacher-reference-warmup.json", warmup_receipt)
        status("teacher_reference_warmup", source_id=first_fit_crop.source_id,
            encoder_calls=3, decoder_calls=3, exact_cached_parity=True)
        calibration_indices = list(range(4*args.batch_size))
        feature_receipts = attach_teacher_features(banks["fit"], calibration_indices, teacher, pool,
            ctx.data, teacher_sha, status=status)
        for row in banks["fit"][:4*args.batch_size]:
            with torch.no_grad():
                _, features = last_block_forward(model, row["features"].to(engine.device), return_features=True)
            row["student_features"] = features.detach().cpu()
        calibration_rows = banks["fit"][:4*args.batch_size]
        normalization = fit_feature_normalization(calibration_rows)
        adapter = AuxiliaryPhaseReadout().to(engine.device)
        ridge = fit_auxiliary_ridge(adapter, calibration_rows, normalization, device=engine.device)
        auxiliary = {"adapter": adapter, "normalization": normalization}
        adapter_sha = state_fingerprint(adapter.state_dict())
        atomic_json(out/"auxiliary-ridge.json", ridge)
        atomic_json(out/"auxiliary-identity.json", {"adapter_sha256": adapter_sha,
            "normalization_sha256": state_fingerprint({"mean": normalization.mean, "scale": normalization.scale}),
            "calibration_source_ids": normalization.source_ids, "calibration_pool": "First four fixed FIT batches only",
            "baseline": "Both late and aux start retained joint_spectral, not a selected WN/short candidate",
            "inference_adapter": False})
        atomic_json(out/"teacher-feature-receipts.json", feature_receipts)
        atomic_json(out/"teacher-feature-accounting.json", {
            "captured_sources": len(feature_receipts),
            "valid_teacher_cells": sum(int(row["feature_valid"].sum()) for row in banks["fit"] if "feature_valid" in row),
            "partial_cell_excluded_samples": sum(int(row["valid"].sum())-240*int(row["feature_valid"].sum())
                for row in banks["fit"] if "feature_valid" in row),
            "teacher_cell_samples": 240, "teacher_state_sha256": teacher_sha})
        torch.save({"adapter_state": adapter.state_dict(), "normalization_mean": normalization.mean,
            "normalization_scale": normalization.scale, "normalization_valid_cells": normalization.valid_cells,
            "normalization_source_ids": normalization.source_ids, "normalization_std_floor": normalization.std_floor,
            "deployment": "Auxiliary sidecar only; never part of exported student"}, out/"training-only-auxiliary.pt")
        scales, counts = baseline_scales(banks["fit"])
        atomic_json(out/"fit-accounting.json", {"scales": scales, "counts": counts,
            "baseline": "Retained candidate, not earlier student", "distinct_sources_per_arm": len(banks["fit"]),
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
        aux_coefficient = calibrate_auxiliary_weight(model, banks["fit"], scales, objectives, coefficients,
            auxiliary, args, engine.device)
        coefficients = {**coefficients, "feature": aux_coefficient["feature_weight"]}
        atomic_json(out/"auxiliary-coefficient-calibration.json", aux_coefficient)
        rate = calibrate_rate(model, banks["fit"], initial, scales, objectives, coefficients, auxiliary, args, engine.device)
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
            restore_late_state(model, initial); _restore_rng(fit_rng)
            named = select_last_block_parameters(model); parameters = [p for _, p in named]
            if arm == "aux":
                feature_receipts.extend(attach_teacher_features(banks["fit"], list(range(len(banks["fit"]))),
                    teacher, pool, ctx.data, teacher_sha, status=status))
                atomic_json(out/"teacher-feature-receipts.json", feature_receipts)
                atomic_json(out/"teacher-feature-accounting.json", {
                    "captured_sources": len(feature_receipts),
                    "valid_teacher_cells": sum(int(row["feature_valid"].sum()) for row in banks["fit"]),
                    "partial_cell_excluded_samples": sum(int(row["valid"].sum())-240*int(row["feature_valid"].sum()) for row in banks["fit"]),
                    "teacher_cell_samples": 240, "teacher_state_sha256": teacher_sha})
            if state_fingerprint(teacher.model.state_dict()) != teacher_sha:
                raise RuntimeError("Frozen teacher changed during authenticated feature capture")
            optimizer = new_optimizer(parameters, rate["chosen_learning_rate"])
            probe_indices = list(range(4*args.batch_size))
            probes = {"begin": gradient_probe(model, banks["fit"], probe_indices, scales, arm, objectives, coefficients,
                auxiliary, device=engine.device, args=args)}
            atomic_json(out/(arm+"-gradient-probes.json"), probes)
            validations = []; best = None; best_state = None
            arm_start = time.monotonic()
            with (out/(arm+"-training.jsonl")).open("w") as logfile:
                for step in range(1, args.steps+1):
                    indices = list(range((step-1)*args.batch_size, step*args.batch_size))
                    update = train_batch(model, banks["fit"], indices, optimizer, scales, arm, objectives, coefficients,
                        auxiliary, device=engine.device, args=args)
                    log = {"step": step, "arm": arm, "sources_seen": step*args.batch_size,
                           "seconds": time.monotonic()-arm_start, **update}
                    logfile.write(json.dumps(log, allow_nan=False)+"\n"); logfile.flush()
                    if step % 16 == 0:
                        status("late_refinement_training", **log)
                    if step % args.eval_every == 0 or step == args.steps:
                        verify_frozen()
                        score = bank_score(model, banks["validation"], engine.device, objectives["long"])
                        checks = teacher_selection_checks(score, reference_selection)
                        entry = {"step": step, "score": score, "selection": checks}
                        validations.append(entry); atomic_json(out/(arm+"-validation.json"), validations)
                        if checks["qualified"] and (best is None or score["aggregate"]["quiet_mse"] < best["score"]["aggregate"]["quiet_mse"]):
                            best, best_state = entry, late_state(model)
                        status("late_refinement_validation", arm=arm, step=step, qualified=checks["qualified"], quiet_rms_ratio=checks["quiet_rms_ratio"])
            probes["end"] = gradient_probe(model, banks["fit"], probe_indices, scales, arm, objectives, coefficients,
                auxiliary, device=engine.device, args=args)
            atomic_json(out/(arm+"-gradient-probes.json"), probes)
            if state_fingerprint(adapter.state_dict()) != adapter_sha:
                raise RuntimeError("Training-only auxiliary adapter changed during student fit")
            final_state = late_state(model)
            torch.save({"format_version": "late_refinements_v3", "base_checkpoint_sha256": QUARTER_SHA,
                "retained_candidate_sha256": args.candidate_sha256, "arm": arm, "mode": "raw", "updates": args.steps,
                "student_parameters": final_state, "selected_step": best["step"] if best else None,
                "selected_student_parameters": best_state, "qualification": "Experimental final-block/head overlay; auxiliary adapter excluded; no automatic promotion or resume state"},
                out/(arm+"-student.pt"))
            final_report = canonical(arm+"-final", args.steps)
            comparison = compare_refinement(baseline_report, final_report)
            final_train = bank_score(model, train_panel, engine.device, objectives["long"])
            atomic_json(out/(arm+"-training-panel.json"), final_train)
            selected_comparison = None
            if best_state is not None:
                if best["step"] == args.steps:
                    selected_comparison = comparison
                else:
                    restore_late_state(model, best_state)
                    selected_comparison = compare_refinement(baseline_report, canonical(arm+"-selected", best["step"]))
            all_results[arm] = {"final": comparison, "selected_step": best["step"] if best else None,
                "selected": selected_comparison, "seconds": time.monotonic()-arm_start, "student_sha256": file_sha(out/(arm+"-student.pt")),
                "auxiliary_adapter_preserved": True, "auxiliary_exported": False, "promoted": False}
            atomic_json(out/"comparisons.json", all_results)
            verify_frozen()
        completion = {"complete": True, "seconds": time.monotonic()-started,
            "comparisons": all_results, "promoted": False}
    finally:
        restore_late_state(model, original_late)
        for name, p in model.named_parameters():
            p.requires_grad_(original_requires[name]); p.grad = None
        for module, mode in original_modes.items():
            module.training = mode
        _restore_rng(original_rng)
        if "teacher" in locals() and state_fingerprint(teacher.model.state_dict()) != teacher_sha:
            raise RuntimeError("Teacher state changed during late refinement")
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
    status("late_refinements_complete", out=str(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("checkpoint", "candidate-heads", "training-receipt", "canonical-receipt", "canonical-inventory", "out"):
        parser.add_argument("--"+arg, required=True)
    parser.add_argument("--candidate-sha256", default="b612baf57c5a3ae113ed67ea1a48f6f9cb2d740001aa5b2b5dc3a7a5ba9097e2")
    parser.add_argument("--arms", default="late")
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
