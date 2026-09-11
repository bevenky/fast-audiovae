"""Paired, bounded reconstruction-settings screen from one authenticated model.

Only the mel definition and learning rate differ. This is decoder distillation,
not a reproduction of full VAE pretraining: GAN, KL and its schedule are absent.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

import run_pilot as base
from author_mel import AuthorMelConfig, AuthorMelLoss

VERSION = "audiovae2_settings_screen_v1"
ARMS = (("current_lr1e-4", "current", 1e-4), ("current_lr3e-5", "current", 3e-5),
        ("author_lr1e-4", "author", 1e-4), ("author_lr3e-5", "author", 3e-5))
UPDATES = 256
SOURCES_PER_UPDATE = 3
OPTIMIZER = {"betas": (.9, .99), "weight_decay": 0., "eps": 1e-8}


def authenticate_preflight(directory):
    """Read receipts without importing CPU-only export/streaming entry points."""
    pre = json.loads((directory/"preflight.json").read_text())
    if pre.get("passed") is not True: raise ValueError("Numerical preflight must pass")
    identity = pre["identity"]
    for field, filename in (("initial_sha256", "initial.pt"), ("selection_sha256", "channel-selection.json")):
        if base.sha(directory/filename) != pre[field]: raise ValueError("Authenticated input changed: "+filename)
    model_sha = base.sha(Path(__file__).with_name("group_model.py"))
    if identity["model_sha256"] != model_sha: raise ValueError("Group model source changed")
    if (identity["teacher_source_sha256"] != base.SOURCE_SHA256
            or identity["teacher_checkpoint_sha256"] != base.CHECKPOINT_SHA256):
        raise ValueError("The original teacher identity changed")
    selection = json.loads((directory/"channel-selection.json").read_text())
    if len(selection["stage2_indices"]) != 256 or len(selection["stage3_indices"]) != 128:
        raise ValueError("Expected the approved256/128/128 group")
    initial = torch.load(directory/"initial.pt", map_location="cpu", weights_only=True, mmap=True)
    if (initial["format"] != "audiovae2_group_width_v1" or initial["step"] != 0
            or initial["identity"] != identity or initial["optimizer"] is not None):
        raise ValueError("Expected the authenticated unfitted initial checkpoint")
    metadata = {"preflight_sha256": base.sha(directory/"preflight.json"),
                "initial_sha256": pre["initial_sha256"], "selection_sha256": pre["selection_sha256"],
                "model_sha256": model_sha, "teacher_source_sha256": base.SOURCE_SHA256,
                "teacher_checkpoint_sha256": base.CHECKPOINT_SHA256}
    return metadata, selection, initial


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def group_digest(model):
    result = hashlib.sha256()
    for name, value in sorted(model.group_state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        result.update(name.encode()); result.update(str(tuple(tensor.shape)).encode())
        result.update(str(tensor.dtype).encode()); result.update(tensor.numpy().tobytes())
    return result.hexdigest()


def rng_state():
    n = np.random.get_state()
    return {"torch": torch.get_rng_state().clone(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python": random.getstate(),
            "numpy": [n[0], n[1].tolist(), n[2], n[3], n[4]]}


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    if state["cuda"]: torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))


def frozen_versions(model):
    """No copies or GPU transfers: frozen tensors must retain identity/version."""
    group = set(model.group_state_dict())
    return {name: (value.data_ptr(), value._version, tuple(value.shape))
            for name, value in model.decoder.state_dict().items() if name not in group}


def counts(crops, definition, objective):
    if definition == "current": return base.reconstruction_denominators(crops, objective)
    if definition != "author": raise ValueError("Unknown mel definition")
    individual = [objective.element_counts(crop["valid_scored_samples"]) for crop in crops]
    if not individual: raise ValueError("Empty reconstruction panel")
    return {"samples": sum(crop["valid_scored_samples"] for crop in crops),
            "mel_elements": tuple(sum(row[i] for row in individual) for i in range(len(individual[0])))}


def losses(p, target, h, ht, valid, spans, definition, objective, denominators=None):
    if definition == "current":
        return base.losses(p, target, h, ht, valid, spans, objective, denominators)
    if definition != "author": raise ValueError("Unknown mel definition")
    if (p.shape != target.shape or h.shape != ht.shape or p.shape[-1] != h.shape[-1]*4
            or valid.shape != p.shape or valid.dtype != torch.bool or len(spans) != p.shape[0]):
        raise ValueError("Invalid waveform, feature or valid-mask geometry")
    expected = torch.zeros_like(valid)
    terms, waveform_sums = [], []
    for i, (a, b) in enumerate(spans):
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= p.shape[-1]:
            raise ValueError("Invalid scored span")
        expected[i, :, a:b] = True
        prediction, reference = p[i:i+1, :, a:b], target[i:i+1, :, a:b].detach()
        waveform_sums.append((prediction-reference).abs().sum())
        terms.append(objective.group_terms(prediction, reference))
    if not torch.equal(valid, expected): raise ValueError("Scored spans and mask disagree")
    sample_count = denominators["samples"] if denominators else int(valid.sum())
    if sample_count < int(valid.sum()): raise ValueError("Global sample count is too small")
    weights = valid.reshape(valid.shape[0], 1, h.shape[-1], 4).sum(-1).to(h)
    selected = weights > 0
    error = h.masked_fill(~selected, 0) - ht.detach().masked_fill(~selected, 0)
    return {"waveform": sum(waveform_sums)/sample_count,
            "mel": objective.loss_from_terms(terms, denominators["mel_elements"] if denominators else None),
            "feature": (error.square()*weights).sum()/(sample_count*h.shape[1])}


def forward_losses(model, teacher, crops, definition, objective, denominators):
    z, target, valid, spans = base.batch(crops)
    with torch.no_grad(): trace = base.teacher_forward(teacher, z)
    h = model.group_from_input(trace["group_input"])
    return losses(model.suffix_from_group(h), target, h, trace["group_output"], valid,
                  spans, definition, objective, denominators)


def training_update(model, teacher, crops, definition, objective, coefficients, optimizer, *, record_diagnostics=False):
    if definition == "current":
        return base.training_update(model, teacher, crops, objective, coefficients, optimizer,
                                    record_diagnostics=record_diagnostics)
    denominators = counts(crops, definition, objective)
    accumulated = {key: 0. for key in coefficients}
    optimizer.zero_grad(set_to_none=True)
    for crop in crops:
        values = forward_losses(model, teacher, [crop], definition, objective, denominators)
        total = sum(coefficients[key]*value for key, value in values.items())
        if not torch.isfinite(total): raise RuntimeError("Nonfinite training objective")
        total.backward()
        for key, value in values.items(): accumulated[key] += float(value.detach())
    params = base.parameters(model)
    if any(p.grad is None for p in params): raise RuntimeError("Missing group gradient")
    if not torch.stack([torch.isfinite(p.grad).all() for p in params]).all():
        raise RuntimeError("Nonfinite group gradient")
    diagnostics = {}
    if record_diagnostics:
        diagnostics["gradient_norm"] = float(torch.stack([p.grad.detach().norm() for p in params]).norm())
        if params[0].is_cuda: diagnostics["gpu_memory_gib"] = torch.cuda.memory_allocated(params[0].device)/1024**3
    optimizer.step()
    if record_diagnostics and params[0].is_cuda: torch.cuda.synchronize(params[0].device)
    return {"total": sum(coefficients[key]*value for key, value in accumulated.items()), **accumulated, **diagnostics}


def calibrate_author(model, teacher, crops, objective, current_calibration, rates=(1e-4, 3e-5)):
    """Change only the mel coefficient; restore every disposable update."""
    params = base.parameters(model)
    original = {key: value.detach().cpu().clone() for key, value in model.group_state_dict().items()}
    original_grads = [None if p.grad is None else p.grad.detach().clone() for p in params]
    rng = rng_state()
    initial_digest = group_digest(model)
    denominators = counts(crops, "author", objective)
    sums = {key: [torch.zeros_like(p) for p in params] for key in ("waveform", "mel", "feature")}
    before = {key: 0. for key in sums}
    probes = []
    try:
        for crop in crops:
            values = forward_losses(model, teacher, [crop], "author", objective, denominators)
            for key, value in values.items():
                before[key] += float(value.detach())
                grads = torch.autograd.grad(value, params, retain_graph=key != "feature")
                for total, grad in zip(sums[key], grads): total.add_(grad.detach())
        norms = {key: base.grad_norm(values) for key, values in sums.items()}
        if any(not math.isfinite(value) or value <= 0 for value in norms.values()):
            raise RuntimeError("Nondegenerate calibration gradients are required")
        # The current coefficients are authenticated by the original preflight.
        # Cross-check the two shared objectives before calibrating author mel.
        for key in ("waveform", "feature"):
            if not math.isclose(norms[key], current_calibration["pooled_parameter_gradient_norms"][key], rel_tol=1e-4, abs_tol=1e-10):
                raise RuntimeError("Shared calibration objective changed: "+key)
        coefficients = dict(current_calibration["coefficients"])
        coefficients["mel"] = .5*norms["waveform"]/norms["mel"]
        for rate in rates:
            model.load_group_state_dict(original)
            optimizer = torch.optim.AdamW(params, lr=rate, **OPTIMIZER)
            for i, param in enumerate(params):
                param.grad = sum(coefficients[key]*sums[key][i] for key in coefficients)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if not torch.stack([torch.isfinite(p).all() for p in params]).all():
                raise RuntimeError("Nonfinite disposable optimizer update")
            predicted = {key: sum(float((grad.double()*(param.detach()-original[name].to(param)).double()).sum())
                                 for grad, (name, param) in zip(sums[key], model.group_named_parameters())) for key in sums}
            after = {key: 0. for key in sums}
            with torch.no_grad():
                for crop in crops:
                    for key, value in forward_losses(model, teacher, [crop], "author", objective, denominators).items():
                        after[key] += float(value)
            if any(not math.isfinite(value) for value in after.values()):
                raise RuntimeError("Nonfinite disposable post-update loss")
            probes.append({"learning_rate": rate, "before": before, "after": after,
                           "predicted_changes": predicted, "actual_changes": {key: after[key]-before[key] for key in before}})
    finally:
        model.load_group_state_dict(original)
        for param, grad in zip(params, original_grads): param.grad = grad
        restore_rng(rng)
    if group_digest(model) != initial_digest: raise RuntimeError("Calibration did not restore initial weights")
    return {"definition": "author", "coefficients": coefficients, "pooled_parameter_gradient_norms": norms,
            "before": before, "denominators": denominators, "optimizer_probes": probes,
            "source_ids": [crop["source_id"] for crop in crops], "initial_group_sha256": initial_digest,
            "state_restored": True, "selection_policy": "Both preregistered rates are tested; no automatic acceptance or promotion"}


def start_arm(model, initial_group, initial_rng, learning_rate):
    model.load_group_state_dict(initial_group)
    restore_rng(initial_rng)
    optimizer = torch.optim.AdamW(base.parameters(model), lr=learning_rate, **OPTIMIZER)
    if optimizer.state: raise RuntimeError("A paired arm must start with fresh optimizer state")
    return optimizer


def save_checkpoint(path, model, optimizer, step, identity, seen):
    payload = {"format": VERSION, "group": {key: value.detach().cpu() for key, value in model.group_state_dict().items()},
               "optimizer": optimizer.state_dict(), "rng": rng_state(), "step": step,
               "fit_cursor": len(seen), "sources_seen": list(seen), "identity": identity}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def authenticate_inputs(args):
    metadata, selection, initial = authenticate_preflight(args.base_out)
    preflight = json.loads((args.base_out/"preflight.json").read_text())
    identity = preflight["identity"]
    if base.sha(args.manifest) != identity["manifest_sha256"] or base.sha(base.__file__) != identity["runner_sha256"]:
        raise ValueError("Original pilot source or data manifest changed")
    for name in ("monitoring", "boundary_diagnostics"):
        if base.sha(Path(__file__).with_name(name+".py")) != identity[name+"_sha256"]:
            raise ValueError("Original diagnostic source changed: "+name)
    for name, checksum in identity["shared_helpers_sha256"].items():
        if base.sha(base.sys.modules["audiovae_student."+name].__file__) != checksum:
            raise ValueError("Original shared helper changed: "+name)
    manifest, pools, receipt = base.load_data(args.manifest)
    if receipt["sha256"] != identity["cache_sha256"]: raise ValueError("Latent cache identity changed")
    expected = {key: metadata[key] for key in ("preflight_sha256", "initial_sha256", "selection_sha256",
                "model_sha256", "teacher_source_sha256", "teacher_checkpoint_sha256")}
    export = json.loads((args.base_out/"export-check.json").read_text())
    if export.get("passed") is not True or any(export.get(key) != value for key, value in expected.items()):
        raise ValueError("Authenticated export check is required")
    if base.sha(export["onnx_path"]) != export["onnx_sha256"]: raise ValueError("Export graph changed")
    streaming = json.loads((args.base_out/"streaming-check.json").read_text())
    base.authenticate_streaming_check(streaming, {**expected,
        "streaming_checker_sha256": base.sha(Path(__file__).with_name("streaming_preflight.py")),
        "manifest_sha256": identity["manifest_sha256"], "cache_sha256": identity["cache_sha256"],
        "runner_sha256": identity["runner_sha256"]})
    if len(pools["calibration"]) != 72 or len(pools["development"]) != 96 or len(pools["fit"]) < UPDATES*SOURCES_PER_UPDATE:
        raise ValueError("The preregistered 72/96/768-source screen panel is unavailable")
    if not preflight["calibration"]["passed"]: raise ValueError("Original loss calibration did not pass")
    return metadata, selection, initial, preflight, manifest, pools


def main():
    parser = argparse.ArgumentParser()
    for name in ("assets", "base-out", "manifest", "out"): parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--tensorboard", type=Path)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()): raise FileExistsError("Use a new, empty screen directory")
    args.out.mkdir(parents=True, exist_ok=True)
    base.policy()
    metadata, selection, initial, preflight, manifest, pools = authenticate_inputs(args)
    teacher = base.FrozenAudioVAE2.from_files(args.assets/"audio_vae_v2.py", args.assets/"audiovae.pth", device="cuda")
    model = base.build_student(teacher.model.decoder, selection["stage2_indices"], selection["stage3_indices"])
    model.load_group_state_dict(initial["group"])
    original = {key: value.detach().cpu().clone() for key, value in model.group_state_dict().items()}
    initial_digest = group_digest(model)
    frozen = frozen_versions(model)
    fitting = pools["fit"][:UPDATES*SOURCES_PER_UPDATE]
    fitting_ids = [crop["source_id"] for crop in fitting]
    if len(set(fitting_ids)) != len(fitting_ids): raise ValueError("Fitting sources must be unique within each arm")
    objectives = {"current": base.objective(), "author": AuthorMelLoss(AuthorMelConfig()).cuda()}
    source_metadata = {row["source_id"]: row for row in manifest["splits"]["development"]["rows"]}
    boundary_panel, boundary_selection = base.select_boundary_panel(pools["development"], source_metadata)
    identity = {"version": VERSION, "original_preflight": metadata, "original_identity": preflight["identity"],
        "screen_sha256": base.sha(__file__), "author_mel_sha256": base.sha(Path(__file__).with_name("author_mel.py")),
        "initial_group_sha256": initial_digest, "channel_selection": selection,
        "fit_source_ids": fitting_ids, "fit_source_ids_sha256": digest(fitting_ids),
        "calibration_source_ids": [crop["source_id"] for crop in pools["calibration"]],
        "development_source_ids": [crop["source_id"] for crop in pools["development"]],
        "boundary_panel": boundary_selection, "updates_per_arm": UPDATES, "effective_batch_sources": 3,
        "forward_batch_sources": 1, "gradient_accumulation": 3, "optimizer": OPTIMIZER,
        "current_mel": asdict(objectives["current"].config), "author_mel": asdict(objectives["author"].config),
        "shared_objectives": "Raw teacher-waveform L1 and full stage4 raw MSE; identical coefficients across all arms",
        "excluded": "No GAN, KL, LR schedule, gradient clipping or new stage2/3 matching losses",
        "reuse": "Paired reuse of the same sources across arms; no source repeats within an arm",
        "selection_policy": "No automatic winner; compare common96-source quality and quiet/peak regressions",
        "torch": str(torch.__version__), "cudnn": torch.backends.cudnn.version(), "precision": "FP32, TF32 disabled"}
    base.write_json(args.out/"screen-identity.json", identity)
    current = copy.deepcopy(preflight["calibration"])
    current.update({"definition": "current", "source_ids": identity["calibration_source_ids"],
                    "reused_from_authenticated_preflight": True, "initial_group_sha256": initial_digest})
    base.event("settings_calibration_started", sources=len(pools["calibration"]))
    author = calibrate_author(model, teacher, pools["calibration"], objectives["author"], current)
    calibrations = {"current": current, "author": author}
    base.write_json(args.out/"calibration.json", calibrations)
    if frozen_versions(model) != frozen: raise RuntimeError("Calibration changed frozen decoder state")
    arm_rng = rng_state()
    baseline = json.loads((args.base_out/"development-step0.json").read_text())
    if baseline["aggregate"] != preflight["development"]: raise ValueError("Original baseline report changed")
    base.write_json(args.out/"common-development-step0.json", baseline)
    baseline_boundaries = json.loads((args.base_out/"boundaries-step0.json").read_text())
    if baseline_boundaries["source_ids"] != [crop["source_id"] for crop in boundary_panel]:
        raise ValueError("Original boundary panel changed")
    from torch.utils.tensorboard import SummaryWriter
    results = []
    for name, definition, rate in ARMS:
        directory = args.out/name
        directory.mkdir(exist_ok=False)
        optimizer = start_arm(model, original, arm_rng, rate)
        if group_digest(model) != initial_digest: raise RuntimeError("Paired arm did not restore identical weights")
        coefficients = calibrations[definition]["coefficients"]
        arm_identity = {"screen_identity_sha256": digest(identity), "arm": name, "definition": definition,
                        "learning_rate": rate, "coefficients": coefficients, "initial_group_sha256": initial_digest,
                        "fit_source_ids_sha256": identity["fit_source_ids_sha256"], "optimizer_initial_state_entries": len(optimizer.state),
                        "channel_selection": selection, "original_identity": preflight["identity"]}
        base.write_json(directory/"launch.json", arm_identity)
        writer = SummaryWriter(str((args.tensorboard or args.out/"tensorboard")/name))
        seen, exposure = [], 0.
        started = time.monotonic()
        try:
            base.write_json(directory/"development-step0.json", baseline)
            base.log_validation(writer, baseline, 0, source_metadata)
            base.log_boundaries(writer, baseline_boundaries, 0)
            for step in range(1, UPDATES+1):
                step_start = time.monotonic()
                chosen = fitting[(step-1)*3:step*3]
                ids = [crop["source_id"] for crop in chosen]
                if any(source in seen for source in ids): raise RuntimeError("Repeated fitting source within arm")
                diagnostics = step == 1 or step % 25 == 0 or step == UPDATES
                values = training_update(model, teacher, chosen, definition, objectives[definition], coefficients,
                                         optimizer, record_diagnostics=diagnostics)
                seen.extend(ids); exposure += sum(crop["valid_scored_samples"] for crop in chosen)/48000
                record = {"step": step, **values, "source_ids": ids, "unique_sources": len(seen),
                          "audio_hours": exposure/3600, "elapsed_seconds": time.monotonic()-started}
                if diagnostics: record["step_seconds"] = time.monotonic()-step_start
                with (directory/"train.jsonl").open("a") as handle:
                    handle.write(json.dumps(record, allow_nan=False)+"\n")
                base.log_training(writer, record, coefficients, rate, step)
                if diagnostics:
                    base.event("settings_training", arm=name, **record); writer.flush()
                if step in (128, UPDATES):
                    common = base.evaluate(model, teacher, pools["development"], objectives["current"])
                    base.write_json(directory/f"development-step{step}.json", common)
                    summary = base.log_validation(writer, common, step, source_metadata)
                    base.write_json(directory/f"validation-groups-step{step}.json", summary)
                    base.event("settings_development", arm=name, step=step, **common["aggregate"])
                if step == UPDATES:
                    boundaries = base.evaluate_boundaries(teacher, model, boundary_panel, selection, base.batch, base.teacher_forward)
                    base.write_json(directory/f"boundaries-step{step}.json", boundaries)
                    base.log_boundaries(writer, boundaries, step)
                    save_checkpoint(directory/"final.pt", model, optimizer, step, arm_identity, seen)
            if frozen_versions(model) != frozen: raise RuntimeError("A fitting arm changed frozen decoder state")
            completion = {"arm": name, "step": UPDATES, "unique_sources": len(seen), "audio_hours": exposure/3600,
                          "checkpoint_sha256": base.sha(directory/"final.pt"), "common_quality": common["aggregate"],
                          "frozen_decoder_state_preserved": True, "automatic_promotion": False}
            base.write_json(directory/"completed.json", completion)
            results.append(completion)
            base.write_json(args.out/"results.json", {"identity": identity, "completed_arms": results,
                                                      "baseline": baseline["aggregate"], "automatic_promotion": False})
        finally:
            writer.flush(); writer.close()
    if base.sha(args.base_out/"initial.pt") != metadata["initial_sha256"]:
        raise RuntimeError("Original checkpoint changed during the screen")
    base.write_json(args.out/"completed.json", {"arms": [result["arm"] for result in results],
                    "updates_per_arm": UPDATES, "sources_per_arm": len(fitting_ids), "automatic_promotion": False})


if __name__ == "__main__": main()
