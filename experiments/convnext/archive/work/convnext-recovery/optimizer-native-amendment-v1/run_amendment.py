"""Transparent two-native-arm continuation of an unresolved calibration.

Uses the previously sealed radius and four unused comparison batches. It does
not repeat calibration, enlarge its grid, or evaluate the unqualified SGD arm.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import math
from pathlib import Path
import time

import torch

import run_comparison as original
from audiovae_student.corrected_calibration import _fixed_engine
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.restart_data import digest, file_sha
from diagnose_objectives_updates import _preserved_attributes
from native_chain import QUARTER_SHA, probe_snapshot, restore_quarter, select_probes
from run_corrected_screen import bind_views
from run_update_experiment import load_canonical_panel
from training_overlay import load_training_overlay, verify_overlay_files

PRIOR_REPORT_SHA = "d1f05cbb8b6159e5bc0f13e238f64419e4b60a4ec091482d0d03e60a021296ef"
ORIGINAL_SOURCE_SHA = "d615ebfd1b65644070a25e8191f46d2594a3880ec55f8a5324daf0bda24b2748"
NATIVE_METHODS = ("retained", "fresh_generator_state")
OUTCOMES = (*NATIVE_METHODS, "retained_native_unscaled")


def validate_prior(prior, report_sha):
    if report_sha != PRIOR_REPORT_SHA:
        raise ValueError("The original unresolved report bytes differ")
    if (prior.get("resolved") is not False or prior.get("comparisons") != []
            or prior.get("summary") is not None or prior.get("state_restored") is not True
            or prior.get("promoted") is not False or prior.get("retained_updates") != 0
            or prior.get("checkpoint_files_written") != 0):
        raise ValueError("Amendment requires an unresolved, unpromoted, state-preserved report with zero comparison batches")
    if (prior.get("original_files", {}).get("retained_checkpoint_hashes_unchanged") is not True
            or prior.get("fresh_files", {}).get("fresh_training_overlay_files_unchanged") is not True):
        raise ValueError("Original inputs were not proved immutable")
    identity = prior["identity"]
    if (identity["calibration_batches"] != 3 or identity["comparison_batches"] != 4 or identity["batch_size"] != 32
            or identity["grid"] != list(original.FRACTIONS) or identity["checkpoint_sha256"] != QUARTER_SHA):
        raise ValueError("The declared original experiment differs")
    source_hashes = [sha for p, sha in identity["source_file_hashes"].items() if Path(p).name == "run_comparison.py"]
    if source_hashes != [ORIGINAL_SOURCE_SHA]:
        raise ValueError("Original report did not use the pinned comparison source")
    selection = prior["selection"]
    if digest(selection) != identity["selection_sha256"]:
        raise ValueError("The original source split changed")
    batches = selection["batches"]
    expected_ids = ["calibration_" + str(i) for i in range(3)] + ["comparison_" + str(i) for i in range(4)]
    if [b["id"] for b in batches] != expected_ids or any(len(b["indices"]) != 32 or len(b["sources"]) != 32 for b in batches):
        raise ValueError("Three calibration and four untouched B32 batches are required")
    if any(b["split"] != ("calibration" if i < 3 else "comparison") for i, b in enumerate(batches)):
        raise ValueError("Batch split labels changed")
    sources = [r for b in batches for r in b["sources"]]
    if len({r["source_id"] for r in sources}) != 224 or len({r["audio_sha256"] for r in sources}) != 224:
        raise ValueError("Original source/hash disjointness failed")
    attempts = prior["calibration"]["attempts"]
    if [a["fraction"] for a in attempts] != list(original.FRACTIONS):
        raise ValueError("The original bounded calibration grid differs")
    common = []
    for attempt in attempts:
        cases = attempt["cases"]
        keys = [(c["batch"], c["method"]) for c in cases]
        if set(keys) != {(b, m) for b in expected_ids[:3] for m in original.METHODS} or len(keys) != 9:
            raise ValueError("Incomplete calibration batch/method coverage")
        for case in cases:
            derived = case["passed"] and case["rounding"]["rounding_qualified"] and case["half_rounding"]["rounding_qualified"]
            if case["qualified"] != derived:
                raise ValueError("Calibration qualification flags are inconsistent")
        if all(c["qualified"] for c in cases if c["method"] == "plain_sgd"):
            raise ValueError("SGD qualified; this stated amendment rationale would be false")
        if all(c["qualified"] for c in cases if c["method"] in NATIVE_METHODS):
            common.append(attempt)
    if not common:
        raise ValueError("Neither common native radius was qualified")
    chosen = max(common, key=lambda a: a["radius"])
    expected_radius = prior["calibration"]["initial_radius"] / 16
    if chosen["fraction"] != 1 / 16 or chosen["radius"] != expected_radius:
        raise ValueError("Largest already-qualified common native radius is not the pinned1/16")
    if not all(c["qualified"] for c in chosen["cases"] if c["method"] in NATIVE_METHODS):
        raise ValueError("The retained/fresh radius is not qualified on every calibration batch")
    return {"radius": chosen["radius"], "fraction": chosen["fraction"],
        "comparison_batches": batches[3:], "calibration_batches_repeated": 0,
        "calibration_vjps_repeated": 0, "calibration_grid_repeated": False}


def compare_two(engine, base_model, batch_id, crops, generated, probes, radius):
    pack = original.training_pack(engine, crops)
    original.copy_parameters(engine.model, base_model)
    baseline_audio = original.forward_pack(engine, pack)
    baseline_train = original.training_metrics(engine, pack, baseline_audio)
    baseline_panel, _ = probe_snapshot(engine, probes)
    outcomes = {}
    try:
        for method in OUTCOMES:
            if method == "retained_native_unscaled":
                original.copy_parameters(engine.model, generated["retained_after"])
                radius_info = {"requested_parameter_l2": generated["receipt"]["direction_norms"]["retained"],
                    "realized_parameter_l2": generated["receipt"]["direction_norms"]["retained"],
                    "scope": "Exact retained native parameter endpoint; unnormalized reference"}
            else:
                radius_info = original.apply_radius(engine.model, base_model, generated["directions"][method], radius)
                if not radius_info["rounding_qualified"]:
                    raise ValueError("Qualified common radius was lost to FP32 parameter rounding")
            prediction = original.forward_pack(engine, pack)
            train = original.training_metrics(engine, pack, prediction)
            panel, _ = probe_snapshot(engine, probes)
            outcomes[method] = {"radius": radius_info, "training_after": train,
                "training_changes": original.changes(baseline_train, train),
                "training_waveform_displacement_rms": float((prediction[pack["valid"]].double() - baseline_audio[pack["valid"]].double()).square().mean().sqrt()),
                "training_quiet_displacement_rms": float((prediction[pack["quiet"]].double() - baseline_audio[pack["quiet"]].double()).square().mean().sqrt()),
                "panel_after": panel, "panel_changes": original.changes(baseline_panel["metrics"], panel["metrics"])}
    finally:
        original.copy_parameters(engine.model, base_model)
    return {"batch": batch_id, "direction_generation": generated["receipt"],
        "training_before": baseline_train, "panel_before": baseline_panel, "outcomes": outcomes,
        "SGD_evaluated": False,
        "SGD_note": "The shared direction-generation utility also computes a negative-gradient tensor/norm. It is not applied, scored or considered qualified here."}


def summarize(results):
    methods = {}
    for method in OUTCOMES:
        methods[method] = {}
        for metric in results[0]["panel_before"]["metrics"]:
            values = [r["outcomes"][method]["panel_changes"][metric]["relative_change_percent"] for r in results]
            if any(v is None for v in values):
                raise ValueError("A required panel baseline metric is undefined")
            methods[method][metric] = {"batch_changes_percent": values, "mean_percent": math.fsum(values) / len(values),
                "minimum_percent": min(values), "maximum_percent": max(values), "improving_batches": sum(v < 0 for v in values), "batches": len(values)}
    return {"methods": methods, "promoted": False,
        "scope": "Two qualified generator-history conditions, four independent single-batch directions from the same checkpoint. Neither an AdamW-only causal intervention nor a long-run quality comparison.",
        "decision": "No automatic optimizer replacement, full-panel expansion, checkpoint promotion or continued training."}


def run(ctx, args):
    from diagnostic_common import atomic_json, status
    started = time.monotonic()
    prior_path = Path(args.prior_report).resolve(strict=True)
    prior_sha = file_sha(prior_path)
    prior = json.loads(prior_path.read_text())
    amendment = validate_prior(prior, prior_sha)
    if file_sha(original.__file__) != ORIGINAL_SOURCE_SHA:
        raise ValueError("Original immutable comparison implementation changed")
    # Validate all original source dependencies at their recorded locations before any model work.
    for path, sha in prior["identity"]["source_file_hashes"].items():
        if file_sha(path) != sha:
            raise ValueError("An original source dependency changed: " + path)
    if (str(torch.__version__) != prior["identity"]["torch"]
            or torch.backends.cudnn.version() != prior["identity"]["cudnn"]):
        raise ValueError("The comparison runtime differs from calibration")
    for path, sha in prior["identity"]["input_file_hashes"].items():
        if file_sha(path) != sha:
            raise ValueError("A pinned original input changed: " + path)
    def require_original_path(path):
        resolved = str(Path(path).resolve(strict=True))
        if resolved not in prior["identity"]["input_file_hashes"]:
            raise ValueError("Amendment changed a checkpoint, inventory or probe-selection path")
    for path in (args.checkpoint, args.canonical_inventory, args.selection):
        require_original_path(path)
    overlay = load_training_overlay(ctx, args.training_receipt, required_counts={"targeted_generator": 12800})
    panel = load_canonical_panel(ctx, {"requires_canonical_evaluation": True}, args.canonical_receipt)
    if overlay["identity"] != prior["identity"]["training_overlay"] or panel["identity"] != prior["identity"]["canonical_panel"]:
        raise ValueError("Training cache or canonical panel differs from calibration")
    hashes, inventory_id = original.heldout_hashes(panel, args.canonical_inventory)
    if inventory_id != prior["identity"]["canonical_inventory_identity"]:
        raise ValueError("Canonical recording identities changed")
    pool = overlay["pools"]["targeted_generator"]
    heldout_ids = {c.source_id for c in panel["crops"]}
    for batch in prior["selection"]["batches"]:
        for i, row in zip(batch["indices"], batch["sources"], strict=True):
            crop = pool[i]
            if (row["pool_index"] != i or row["source_id"] != crop.source_id or row["start_frame"] != crop.start_frame
                    or row["valid_scored_samples"] != crop.valid_scored_samples
                    or original._audio_sha(ctx.data["rows"][crop.source_id]) != row["audio_sha256"]
                    or crop.source_id in heldout_ids or row["audio_sha256"] in hashes):
                raise ValueError("A sealed calibration/comparison crop identity changed")
    probes = select_probes(panel["crops"], json.loads(Path(args.selection).read_text()))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    engine = ctx.engine("targeted", device="cuda")
    original.require_quiet_contract(payload["engine"]["config"]["quiet_audio"])
    restored = restore_quarter(engine, payload["engine"])
    if restored != prior["identity"]["restored_engine_sha256"]:
        raise ValueError("Restored8890 engine differs from calibration")
    bind_views(engine, ctx.data["pools"]["targeted_generator"])
    baseline = {n: p.detach().cpu().clone() for n, p in engine.model.named_parameters()}
    protocol = {"version": "optimizer-native-amendment-v1", "prior_report_path": str(prior_path),
        "prior_report_sha256": prior_sha, "original_source_sha256": ORIGINAL_SOURCE_SHA,
        "amendment_source_sha256": file_sha(__file__), "original_identity": prior["identity"],
        "reason": "Original three-arm calibration stopped as declared because plain SGD did not qualify anywhere in the fixed grid. Before any comparison-batch scoring, retain only the two native directions already qualified on all three calibration batches at their largest common radius.",
        "original_three_arm_result": "unresolved; zero comparison batches evaluated", "SGD_status": "unqualified and not evaluated",
        "normalized_methods": list(NATIVE_METHODS), "unnormalized_reference": "retained_native_unscaled",
        **amendment, "heldout_used_to_choose_radius_or_methods": False,
        "retained_updates": 0, "checkpoint_files_written": 0}
    atomic_json(out / "amendment-protocol.json", protocol)
    results = []
    for batch in amendment["comparison_batches"]:
        status("native_amendment_batch", batch=batch["id"])
        crops = [pool[i] for i in batch["indices"]]
        generated = original.generate_directions(engine, crops, global_rng=payload["rng"])
        with _fixed_engine(engine, 1861, _preserved_attributes(engine)):
            result = compare_two(engine, baseline, batch["id"], crops, generated, probes, amendment["radius"])
        if results and result["panel_before"] != results[0]["panel_before"]:
            raise RuntimeError("Identical baseline panel changed across comparison batches")
        if state_fingerprint(engine.state_dict()) != restored:
            raise RuntimeError("Comparison did not restore the saved engine")
        results.append(result)
        atomic_json(out / (batch["id"] + ".json"), result)
    result = {"protocol": protocol, "comparisons": results, "summary": summarize(results),
        "state_restored": state_fingerprint(engine.state_dict()) == restored, "retained_updates": 0,
        "checkpoint_files_written": 0, "promoted": False,
        "original_files": ctx.verify_files(), "fresh_files": verify_overlay_files(overlay), "seconds": time.monotonic() - started}
    if file_sha(prior_path) != PRIOR_REPORT_SHA or file_sha(original.__file__) != ORIGINAL_SOURCE_SHA:
        raise RuntimeError("Original report/source changed")
    for path, sha in prior["identity"]["input_file_hashes"].items():
        if file_sha(path) != sha:
            raise RuntimeError("An original input changed")
    for path, sha in ((panel["identity"]["receipt_path"], panel["identity"]["receipt_sha256"]),
                      (panel["identity"]["cache_path"], panel["identity"]["cache_sha256"])):
        if file_sha(path) != sha:
            raise RuntimeError("Canonical target cache changed")
    atomic_json(out / "two-native-comparison.json", result)
    status("native_amendment_complete", path=str(out / "two-native-comparison.json"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("prior-report", "checkpoint", "training-receipt", "canonical-receipt", "canonical-inventory", "selection", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--context-module", default="diagnostic_common")
    args = p.parse_args()
    lock = Path("/workspace/fast-audiovae-convnext-20260909-r9/training-runs/.decoder-recipe-v2-expressive.runner.lock")
    with lock.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(importlib.import_module(args.context_module).load_context(), args)
