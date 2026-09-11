"""Matched head-layer and spectral-supervision pilot on sealed AudioVAE2 pairs.

Four arms keep the deployed graph unchanged. STFT/mel operations occur only in
training and evaluation, on contiguous scored audio without mask-to-zero edits.
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
from torch import nn

from run_joint_heads import (HEAD_NAMES, head_forward, capture_prehead, head_state,
    restore_head, masked_sums, normalized_loss, new_optimizer, build_bank,
    baseline_scales, canonical_peak_screen, bank_score as waveform_bank_score,
    selection_checks as waveform_selection_checks)
from audiovae_student.reconstruction_v2 import ReconstructionV2, ReconstructionV2Config

ARMS = ("joint_wave", "joint_spectral", "projection_wave", "projection_spectral")
REGIONS = ("all", "quiet", "active", "transition")


def trainable_names(arm):
    if arm not in ARMS:
        raise ValueError("Unknown factorial arm")
    return set(HEAD_NAMES) if arm.startswith("joint_") else {"output.weight"}


def select_parameters(model, arm):
    wanted = trainable_names(arm)
    named = dict(model.named_parameters())
    if not wanted.issubset(named):
        raise ValueError("Saved decoder lacks declared head parameters")
    for name, parameter in named.items():
        parameter.requires_grad_(name in wanted)
        parameter.grad = None
    return [(name, named[name]) for name in sorted(wanted)]


def contiguous_spans(valid):
    if valid.dtype != torch.bool or valid.ndim != 3 or valid.shape[:2] != (1, 1):
        raise ValueError("Expected singleton boolean scored mask [1,1,T]")
    value = valid[0, 0].detach().cpu()
    padded = torch.nn.functional.pad(value.to(torch.int8), (1, 1))
    boundaries = (padded[1:] - padded[:-1]).nonzero().flatten().tolist()
    return list(zip(boundaries[::2], boundaries[1::2]))


def spectral_counts(valid, config):
    spans = contiguous_spans(valid)
    return tuple(sum(max(0, 1 + (stop-start-size)//(size//4)) * bands
                     for start, stop in spans if stop-start >= max(config.fft_sizes))
                 for size, bands in zip(config.fft_sizes, config.mel_bands))


def pooled_spectral_loss(stats, totals, config):
    if len(totals) != len(config.fft_sizes) or any(n < 0 for n in totals):
        raise ValueError("Invalid pooled spectral element counts")
    active = [i for i, count in enumerate(totals) if count]
    zero = stats["linear_sums"][0] * 0
    if not active:
        return zero
    return sum((config.mel_linear_weight * stats["linear_sums"][i]
                + config.mel_log_weight * stats["log_sums"][i]) / totals[i]
               for i in active) / len(active)


class SpectralObjective(nn.Module):
    """Existing ReconstructionV2 sums, with explicit short-span accounting.

    Full scored spans use the unchanged multi-resolution implementation. Short
    spans below the largest FFT contribute waveform loss only, with exclusions
    reported explicitly. No padding or synthetic boundaries are introduced.
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = ReconstructionV2Config() if config is None else config
        self.full = ReconstructionV2(self.config)

    def _validate(self, prediction, target, valid):
        if (prediction.shape != target.shape or prediction.shape != valid.shape
                or prediction.device != target.device or prediction.device != valid.device):
            raise ValueError("Prediction, teacher and scored-mask geometry/device differ")
        if (not prediction.is_floating_point() or not target.is_floating_point()
                or not bool(torch.isfinite(prediction).all() and torch.isfinite(target).all())):
            raise ValueError("Finite floating audio required")
        return contiguous_spans(valid)

    def sums(self, prediction, target, valid):
        spans = self._validate(prediction, target, valid)
        size_count = len(self.config.fft_sizes)
        # A graph-connected zero keeps wave-only/empty contributions valid.
        zero = prediction.sum() * 0
        linear, logarithmic = [zero for _ in range(size_count)], [zero for _ in range(size_count)]
        counts = [0] * size_count
        skipped = [0] * size_count
        for start, stop in spans:
            p, t = prediction[..., start:stop], target[..., start:stop].detach()
            if stop-start >= max(self.config.fft_sizes):
                terms = self.full.group_terms(p, t)
                for i in range(size_count):
                    linear[i] = linear[i] + terms.mel_linear_sums[i]
                    logarithmic[i] = logarithmic[i] + terms.mel_log_sums[i]
                    counts[i] += terms.mel_element_counts[i]
            else:
                for i in range(size_count):
                    skipped[i] += stop-start
        return {"linear_sums": tuple(linear), "log_sums": tuple(logarithmic),
                "element_counts": tuple(counts), "skipped_samples_by_resolution": tuple(skipped),
                "scored_samples": sum(stop-start for start, stop in spans), "spans": spans}

    @torch.no_grad()
    def regions(self, prediction, target, valid, quiet):
        spans = self._validate(prediction, target, valid)
        if quiet.dtype != torch.bool or quiet.shape != valid.shape or quiet.device != valid.device or bool((quiet & ~valid).any()):
            raise ValueError("Quiet mask must be a teacher-only subset of valid samples")
        result = {name: {"linear_sums": [0.] * len(self.config.fft_sizes),
                         "log_sums": [0.] * len(self.config.fft_sizes),
                         "element_counts": [0] * len(self.config.fft_sizes)} for name in REGIONS}
        for start, stop in spans:
            if stop-start < max(self.config.fft_sizes):
                continue
            p, t = prediction[..., start:stop].float(), target[..., start:stop].float()
            q = quiet[0, 0, start:stop]
            for i, (size, bands) in enumerate(zip(self.config.fft_sizes, self.config.mel_bands)):
                if stop-start < size:
                    continue
                window = getattr(self.full, f"window_{i}").to(p)
                bank = getattr(self.full, f"mel_{i}").to(p)
                def mel(audio):
                    spectrum = torch.stft(audio[:, 0], n_fft=size, hop_length=size//4,
                        win_length=size, window=window, center=False, normalized=False,
                        onesided=True, return_complex=True)
                    return torch.matmul(bank, spectrum.abs())
                a, b = mel(p), mel(t)
                linear = (a-b).abs()
                logarithmic = (a.clamp_min(self.config.log_epsilon).log()-b.clamp_min(self.config.log_epsilon).log()).abs()
                coverage = q.unfold(0, size, size//4).sum(-1)
                masks = {"all": torch.ones_like(coverage, dtype=torch.bool),
                         "quiet": coverage == size, "active": coverage == 0,
                         "transition": (coverage > 0) & (coverage < size)}
                for name, frame_mask in masks.items():
                    result[name]["linear_sums"][i] += float(linear[..., frame_mask].double().sum())
                    result[name]["log_sums"][i] += float(logarithmic[..., frame_mask].double().sum())
                    result[name]["element_counts"][i] += int(frame_mask.sum()) * bands
        for value in result.values():
            value.update(summarize_spectral(value, self.config))
        return result


def summarize_spectral(stats, config):
    active = [i for i, count in enumerate(stats["element_counts"]) if count]
    if not active:
        return {"mel": None, "linear": None, "log": None, "active_resolutions": 0}
    linear = sum(stats["linear_sums"][i]/stats["element_counts"][i] for i in active)/len(active)
    logarithmic = sum(stats["log_sums"][i]/stats["element_counts"][i] for i in active)/len(active)
    return {"mel": config.mel_linear_weight*linear + config.mel_log_weight*logarithmic,
            "linear": linear, "log": logarithmic, "active_resolutions": len(active)}


def aggregate_source_spectra(rows, config):
    grouped = {}
    for row in rows:
        sid = row["source_id"]
        if sid not in grouped:
            grouped[sid] = {"source_id": sid, "crops": 0, "regions": {name: {
                "linear_sums": [0.] * len(config.fft_sizes), "log_sums": [0.] * len(config.fft_sizes),
                "element_counts": [0] * len(config.fft_sizes)} for name in REGIONS}}
        grouped[sid]["crops"] += 1
        for region in REGIONS:
            for key in ("linear_sums", "log_sums", "element_counts"):
                grouped[sid]["regions"][region][key] = [a+b for a, b in zip(
                    grouped[sid]["regions"][region][key], row["regions"][region][key])]
    for source in grouped.values():
        for region in source["regions"].values():
            region.update(summarize_spectral(region, config))
    return [grouped[key] for key in sorted(grouped)]


def source_spectral_checks(candidate, baseline, max_ratio=1.01, absolute_tolerance=1e-6):
    before = {r["source_id"]: r for r in baseline}
    after = {r["source_id"]: r for r in candidate}
    if (before.keys() != after.keys() or len(before) != len(baseline) or len(after) != len(candidate)):
        raise ValueError("Spectral source identities differ or repeat")
    checks = []
    for source in sorted(before):
        for region in REGIONS:
            a, b = before[source]["regions"][region], after[source]["regions"][region]
            if a["element_counts"] != b["element_counts"]:
                raise ValueError("Source spectral window counts changed")
            if a["mel"] is None or b["mel"] is None:
                if a["mel"] is not None or b["mel"] is not None:
                    raise ValueError("Missing source spectral coverage changed")
                continue
            ratio = b["mel"]/a["mel"] if a["mel"] > 0 else (1. if b["mel"] == 0 else None)
            checks.append({"source_id": source, "region": region, "baseline_mel": a["mel"],
                "candidate_mel": b["mel"], "ratio": ratio, "passed": b["mel"] <= a["mel"]*max_ratio + absolute_tolerance,
                "gate": region == "all"})
    gates = [row for row in checks if row["gate"]]
    return {"passed": bool(gates) and all(row["passed"] for row in gates), "max_ratio": max_ratio,
            "absolute_tolerance": absolute_tolerance, "source_gates": len(gates), "all_region_failed_sources": [r["source_id"] for r in gates if not r["passed"]],
            "checks": checks, "policy": "Each source overall mel is a gate; quiet/active/transition windows are additionally reported without synthetic joins."}


def legacy_source_spectral_checks(candidate, baseline):
    """Retain the original canonical equal-crop mel criterion per source too."""
    a = {k: v for k, v in baseline["recovery_metrics"].items() if k.startswith("source/")}
    b = {k: v for k, v in candidate["recovery_metrics"].items() if k.startswith("source/")}
    if not a or a.keys() != b.keys():
        raise ValueError("Canonical source metrics changed identity")
    checks = [{"source_id": key.removeprefix("source/"), "baseline_mel": a[key]["mel"],
        "candidate_mel": b[key]["mel"],
        "ratio": b[key]["mel"]/a[key]["mel"] if a[key]["mel"] > 0 else None,
        "passed": b[key]["mel"] <= a[key]["mel"]*1.01 + 1e-6} for key in sorted(a)]
    return {"passed": all(row["passed"] for row in checks), "checks": checks,
            "policy": "Unchanged canonical equal-crop mel definition for every source; max 1% + 1e-6 absolute regression."}


def batch_totals(selected, spectral):
    wave = {key: sum(row[key] for row in selected) for key in ("quiet_samples", "outside_samples")}
    counts = [spectral_counts(row["valid"], spectral.config) for row in selected]
    return wave, tuple(sum(row[i] for row in counts) for i in range(len(spectral.config.fft_sizes)))


def backward_batch(model, selected, scales, spectral, *, device, preservation_weight,
                   spectral_weight, branch="combined"):
    totals, mel_counts = batch_totals(selected, spectral)
    if totals["quiet_samples"] + totals["outside_samples"] <= 0:
        raise ValueError("Training batch has no scored samples")
    summary = {"wave_loss": 0., "spectral_loss": 0., "loss": 0., **totals,
               "mel_element_counts": mel_counts}
    for row in selected:
        h, target, baseline, valid, quiet = [row[k].to(device) for k in ("features", "target", "baseline", "valid", "quiet")]
        prediction = head_forward(model, h)
        sums = masked_sums(prediction, target, baseline, valid, quiet)
        wave = normalized_loss(sums, totals, scales, "quiet_preservation", preservation_weight)
        mel = (pooled_spectral_loss(spectral.sums(prediction, target, valid), mel_counts, spectral.config)
               if branch == "spectral" or spectral_weight else prediction.sum()*0)
        loss = wave if branch == "wave" else mel if branch == "spectral" else wave+spectral_weight*mel
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite pilot objective")
        loss.backward()
        summary["wave_loss"] += float(wave.detach())
        summary["spectral_loss"] += float(mel.detach())
        summary["loss"] += float(loss.detach())
    return summary


def train_batch(model, bank, indices, optimizer, scales, arm, preservation_weight, spectral,
                spectral_weight, *, device):
    optimizer.zero_grad(set_to_none=True)
    summary = backward_batch(model, [bank[i] for i in indices], scales, spectral, device=device,
        preservation_weight=preservation_weight, spectral_weight=spectral_weight if arm.endswith("spectral") else 0.)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if {n for n, _ in named} != trainable_names(arm):
        raise RuntimeError("Unexpected trainable parameter route")
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for _, p in named):
        raise FloatingPointError("Missing or nonfinite head gradient")
    summary["gradient_norm_before_clip"] = float(torch.nn.utils.clip_grad_norm_([p for _, p in named], 1.))
    before = [p.detach().clone() for _, p in named]
    optimizer.step()
    summary["parameter_displacement"] = math.sqrt(sum(float((p.detach().double()-v.double()).square().sum()) for (_, p), v in zip(named, before)))
    if any(not bool(torch.isfinite(p).all()) for _, p in named):
        raise FloatingPointError("Nonfinite head parameter")
    return summary


def calibrate_spectral_weight(model, bank, initial, scales, args, spectral, device):
    """Scope-invariant coefficient from acoustic-output gradient energies.

    Frozen baseline predictions are independent differentiable leaves. Neither
    a head parameter nor an optimizer state is changed during this calibration.
    """
    rows = []
    wave_energy = mel_energy = 0.
    for batch in range(4):
        chosen = bank[batch*args.batch_size:(batch+1)*args.batch_size]
        if len(chosen) != args.batch_size:
            raise ValueError("Need four complete fixed training batches for coefficient calibration")
        totals, mel_counts = batch_totals(chosen, spectral)
        energies = {"wave": 0., "spectral": 0.}
        dot = 0.
        for row in chosen:
            target, baseline, valid, quiet = [row[k].to(device) for k in ("target", "baseline", "valid", "quiet")]
            prediction = baseline.detach().clone().requires_grad_(True)
            sums = masked_sums(prediction, target, baseline, valid, quiet)
            wave = normalized_loss(sums, totals, scales, "quiet_preservation", args.preservation_weight)
            mel = pooled_spectral_loss(spectral.sums(prediction, target, valid), mel_counts, spectral.config)
            gw = torch.autograd.grad(wave, prediction, retain_graph=True)[0].detach().double()
            gm = torch.autograd.grad(mel, prediction)[0].detach().double()
            energies["wave"] += float(gw.square().sum())
            energies["spectral"] += float(gm.square().sum())
            dot += float((gw*gm).sum())
        if any(not math.isfinite(value) or value < 0 for value in energies.values()):
            raise ValueError("Coefficient calibration requires finite nonnegative branch gradient energies")
        wave_energy += energies["wave"]
        mel_energy += energies["spectral"]
        rows.append({"batch": batch, "source_ids": [r["source_id"] for r in chosen],
            "output_gradient_energies": energies,
            "output_gradient_norms": {key: math.sqrt(value) for key, value in energies.items()},
            "gradient_cosine": dot/math.sqrt(energies["wave"]*energies["spectral"]) if min(energies.values()) > 0 else None,
            "spectral_weight_to_equalize": math.sqrt(energies["wave"]/energies["spectral"]) if min(energies.values()) > 0 else None})
    if min(wave_energy, mel_energy) <= 0:
        raise ValueError("Combined calibration batches need positive wave and mel gradient energies")
    weight = math.sqrt(wave_energy/mel_energy)
    return {"spectral_weight": weight, "batches": rows, "optimizer_updates": 0,
        "summed_wave_output_gradient_energy": wave_energy, "summed_mel_output_gradient_energy": mel_energy,
        "policy": "sqrt(sum wave-output-gradient energy / sum mel-output-gradient energy) on first four fixed training batches. Shared by both spectral arms; no parameter-gradient scope advantage. Preservation derivative is zero at baseline; initial balance is quiet teacher versus all-valid teacher mel and will change during fitting."}


@torch.no_grad()
def bank_score(model, bank, device, spectral):
    result = waveform_bank_score(model, bank, device)
    regions = []
    for row in bank:
        h, target, valid, quiet = [row[k].to(device) for k in ("features", "target", "valid", "quiet")]
        prediction = head_forward(model, h)
        regions.append({"source_id": row["source_id"], "start_frame": row["start_frame"],
                        "regions": spectral.regions(prediction, target, valid, quiet)})
    result["spectral_rows"] = regions
    result["spectral_sources"] = aggregate_source_spectra(regions, spectral.config)
    result["spectral_aggregate"] = aggregate_source_spectra(
        [{**row, "source_id": "__all__"} for row in regions], spectral.config)[0]["regions"]
    return result


def selection_checks(candidate, baseline):
    result = waveform_selection_checks(candidate, baseline)
    spectral = source_spectral_checks(candidate["spectral_sources"], baseline["spectral_sources"])
    result["source_spectral"] = spectral
    result["qualified"] = result["qualified"] and spectral["passed"]
    return result


def calibrate_rate(model, bank, initial, scales, args, spectral, spectral_weight, device):
    indices = list(range(args.batch_size))
    first = [bank[i] for i in indices]
    baseline = bank_score(model, first, device, spectral)
    if baseline["aggregate"]["quiet_samples"] <= 0 or baseline["aggregate"]["outside_samples"] <= 0:
        raise ValueError("First fixed calibration batch needs quiet and active samples")
    trials = []
    try:
        for half in range(7):
            rate = args.learning_rate/(2**half)
            outcomes = []
            for arm in ARMS:
                restore_head(model, initial)
                parameters = [p for _, p in select_parameters(model, arm)]
                optimizer = new_optimizer(parameters, rate)
                update = train_batch(model, bank, indices, optimizer, scales, arm,
                    args.preservation_weight, spectral, spectral_weight, device=device)
                scored = bank_score(model, first, device, spectral)
                score, ref = scored["aggregate"], baseline["aggregate"]
                quiet_ratio = score["quiet_mse"]/ref["quiet_mse"]
                drift_ratio = score["outside_drift_mse"]/ref["outside_mse"]
                mel_check = source_spectral_checks(scored["spectral_sources"], baseline["spectral_sources"])
                outcomes.append({"arm": arm, "quiet_mse_ratio": quiet_ratio,
                    "outside_drift_to_baseline_error_mse_ratio": drift_ratio,
                    "source_spectral": mel_check,
                    "passed": quiet_ratio <= 1.01 and drift_ratio <= .01 and mel_check["passed"], "update": update})
            trials.append({"learning_rate": rate, "arms": outcomes})
            if all(v["passed"] for v in outcomes):
                return {"chosen_learning_rate": rate, "trials": trials, "retained_calibration_optimizer_updates": 0,
                    "policy": "First training batch; at most six halvings; identical rate for all four arms. Temporary candidate updates discarded; calibration batches replayed in one-pass fit."}
        return {"chosen_learning_rate": None, "trials": trials, "reason": "No common rate met predeclared finite-update safeguards"}
    finally:
        restore_head(model, initial)
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

    if not 0 < args.learning_rate <= 1e-6 or args.steps*args.batch_size > 2048:
        raise ValueError("Pilot is capped at LR1e-6 and 2,048 distinct training crops per arm")
    if args.steps < 4:
        raise ValueError("Need at least four complete batches for fixed calibration")
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
    named = select_parameters(model, "joint_wave")
    spectral = SpectralObjective().to(engine.device)
    frozen_hash = state_fingerprint({k: v for k, v in model.state_dict().items() if k not in HEAD_NAMES})
    def verify_frozen():
        if state_fingerprint({k: v for k, v in model.state_dict().items() if k not in HEAD_NAMES}) != frozen_hash:
            raise RuntimeError("Frozen body or normalization changed")
    def compare_silence(reference, observed):
        result = compare(reference, observed)
        result["source_spectral"] = source_spectral_checks(observed["spectral_head_sources"], reference["spectral_head_sources"])
        result["legacy_source_spectral"] = legacy_source_spectral_checks(observed, reference)
        result["no_new_peak_regression"] = canonical_peak_screen(reference, observed)
        result["silence_target_passed"] = (result["quality_screen_passed"] and
            result["natural_quiet_ratio"] <= .90 and result["steady_silence_ratio"] < 1 and
            result["no_new_peak_regression"]["passed"] and result["source_spectral"]["passed"] and result["legacy_source_spectral"]["passed"])
        return result
    def canonical(name):
        transient = []
        spectral_rows = []
        def observe(module, inputs, prediction):
            crop = panel["crops"][len(transient)]
            target, valid, quiet = validation_masks(crop, device=engine.device, quiet_config=engine.config.quiet_audio)
            transient.append({"source_id": crop.source_id, "start_frame": crop.start_frame,
                              **transient_errors(prediction, target, valid)})
            spectral_rows.append({"source_id": crop.source_id, "start_frame": crop.start_frame,
                "regions": spectral.regions(prediction, target, valid, quiet)})
        handle = model.register_forward_hook(observe)
        try:
            report = build_canonical_evaluation(engine, panel["crops"], panel["metadata"], panel["receipt"])
        finally:
            handle.remove()
        report["transient_diagnostics"] = transient
        report["spectral_head_rows"] = spectral_rows
        report["spectral_head_sources"] = aggregate_source_spectra(spectral_rows, spectral.config)
        report["spectral_head_experiment"] = {"name": name, "starting_checkpoint_step": 8890,
            "pilot_updates": 0 if name == "baseline" else args.steps,
            "engine_training_clock_unchanged": True}
        atomic_json(out/(name+"-canonical.json"), report)
        return report
    try:
        sources = {Path(__file__).resolve()}
        for module in ("fit_support", "diagnostic_common", "native_chain", "canonical_evaluation", "run_heads", "run_joint_heads", "training_overlay", "audiovae_student.reconstruction_v2", "audiovae_student.losses_distillation"):
            sources.add(Path(importlib.import_module(module).__file__).resolve())
        identity = {"version": "spectral_head_factorial_v1", "checkpoint_sha256": QUARTER_SHA,
            "restored_engine_sha256": original_engine, "frozen_tensor_sha256": frozen_hash,
            "canonical_panel": panel["identity"], "training_overlay": overlay["identity"], "inventory": inventory,
            "selection_sha256": selection["identity_sha256"], "torch": str(torch.__version__),
            "teacher_state_sha256": ctx.parent["identity"]["data"]["teacher_state_sha256"],
            "cudnn": torch.backends.cudnn.version(), "trainable_by_arm": {arm: [{"name": n, "shape": list(p.shape), "elements": p.numel()} for n, p in named if n in trainable_names(arm)] for arm in ARMS},
            "source_files": {str(p): file_sha(p) for p in sources}, "args": vars(args),
            "objectives": {arm: "quiet teacher MSE/q0 + preservation_weight * nonquiet frozen-baseline MSE/a0" +
                (" + calibrated_weight * pooled teacher linear/log mel" if arm.endswith("spectral") else "") for arm in ARMS},
            "spectral_config": spectral.config.__dict__,
            "spectral_region_policy": "Intact valid STFT windows classified by fixed teacher quiet mask into entirely quiet, entirely active, or mixed. Scored spans shorter than largest FFT omitted from spectral loss only and accounted.",
            "optimizer": {"name": "AdamW", "fresh_identical_states": True, "betas": [.9, .999], "eps": 1e-8,
                          "weight_decay": 0., "gradient_norm_cap": 1., "mixed_precision": False},
            "scope": "Matched existing-layer/spectral-supervision factorial, not a full GAN/Muon continuation. No synthetic fit, tanh, gate, new layer or automatic promotion.",
            "metadata_policy": "Missing condition labels remain unknown; dataset names do not prove which expressive event occurs in a crop."}
        atomic_json(out/"identity.json", identity)
        banks = {split: build_bank(model, pool, selection["splits"][split]["indices"], ctx.data["rows"],
                                  engine.config.quiet_audio, engine.device, status, split) for split in ("fit", "validation")}
        scales, counts = baseline_scales(banks["fit"])
        atomic_json(out/"fit-accounting.json", {"scales": scales, "counts": counts,
            "sample_pooling": True, "distinct_source_crops_per_arm": len(banks["fit"]),
            "source_repeats_per_arm": 0, "matched_arms_share_exact_order": True,
            "quiet_mask": "Fixed teacher-only 20ms windows; training only, absent at inference",
            "spectral_counts_by_split": {split: {
                "eligible_elements_by_resolution": [sum(spectral_counts(row["valid"], spectral.config)[i] for row in bank) for i in range(len(spectral.config.fft_sizes))],
                "short_spans": sum(stop-start < max(spectral.config.fft_sizes) for row in bank for start, stop in contiguous_spans(row["valid"])),
                "short_span_excluded_samples": sum(stop-start for row in bank for start, stop in contiguous_spans(row["valid"]) if stop-start < max(spectral.config.fft_sizes))
            } for split, bank in banks.items()}})
        reference_selection = bank_score(model, banks["validation"], engine.device, spectral)
        if min(reference_selection["aggregate"]["quiet_samples"], reference_selection["aggregate"]["outside_samples"]) <= 0:
            raise ValueError("Selection split lacks quiet/nonquiet coverage")
        atomic_json(out/"baseline-validation.json", reference_selection)
        coefficient = calibrate_spectral_weight(model, banks["fit"], initial, scales, args, spectral, engine.device)
        atomic_json(out/"spectral-calibration.json", coefficient)
        calibration = calibrate_rate(model, banks["fit"], initial, scales, args, spectral, coefficient["spectral_weight"], engine.device)
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
            parameters = [p for _, p in select_parameters(model, arm)]
            scope_frozen_hash = state_fingerprint({k: v for k, v in model.state_dict().items() if k not in trainable_names(arm)})
            optimizer = new_optimizer(parameters, calibration["chosen_learning_rate"])
            validations = []; best = None; best_state = None
            arm_start = time.monotonic()
            with (out/(arm+"-training.jsonl")).open("w") as logfile:
                for step in range(1, args.steps+1):
                    indices = list(range((step-1)*args.batch_size, step*args.batch_size))
                    update = train_batch(model, banks["fit"], indices, optimizer, scales, arm,
                                         args.preservation_weight, spectral, coefficient["spectral_weight"], device=engine.device)
                    log = {"step": step, "arm": arm, "sources_seen": step*args.batch_size,
                           "seconds": time.monotonic()-arm_start, **update}
                    logfile.write(json.dumps(log, allow_nan=False)+"\n"); logfile.flush()
                    if step % 16 == 0:
                        status("spectral_head_training", **log)
                    if step % args.eval_every == 0 or step == args.steps:
                        verify_frozen()
                        if state_fingerprint({k: v for k, v in model.state_dict().items() if k not in trainable_names(arm)}) != scope_frozen_hash:
                            raise RuntimeError("Arm-frozen head parameters changed")
                        score = bank_score(model, banks["validation"], engine.device, spectral)
                        checks = selection_checks(score, reference_selection)
                        entry = {"step": step, "score": score, "selection": checks}
                        validations.append(entry)
                        atomic_json(out/(arm+"-validation.json"), validations)
                        if checks["qualified"] and (best is None or score["aggregate"]["quiet_mse"] < best["score"]["aggregate"]["quiet_mse"]):
                            best, best_state = entry, head_state(model)
                        status("spectral_head_validation", arm=arm, step=step, qualified=checks["qualified"], quiet_rms_ratio=checks["quiet_rms_ratio"])
            final_state = head_state(model)
            torch.save({"format_version": "spectral_head_factorial_v1", "base_checkpoint_sha256": QUARTER_SHA,
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
                    selected_report["spectral_head_experiment"]["pilot_updates"] = best["step"]
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
        status("spectral_head_experiments_complete", out=str(out))
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
