"""Separate canonical-input evaluation using the unchanged recovery pilot gates.

Expected quiet counts come from the sealed teacher-only renderer receipt, never
from student predictions. Historical evaluation_audit.py is deliberately intact.
"""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import json

import evaluation_audit as historical
from render_canonical_targets import VERSION, digest, sha


def load_canonical_cache(receipt_path):
    import torch
    from audiovae_student.cache import TrainingCrop
    receipt_path = Path(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    path = Path(receipt["path"])
    if path.parent.resolve() != receipt_path.parent.resolve() or sha(path) != receipt["sha256"]:
        raise ValueError("Canonical heldout cache path or bytes changed")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["contract"] != receipt["contract"]:
        raise ValueError("Canonical heldout receipt and payload disagree")
    crops = [TrainingCrop(**c) for c in saved["heldout"]]
    verify_inputs(crops, saved["metadata"], receipt)
    return crops, saved["metadata"], receipt


def verify_inputs(crops, metadata, receipt):
    from audiovae_student.preflight_distillation import _crop_identity
    contract = receipt["contract"]
    if contract["version"] != VERSION:
        raise ValueError("Unsupported canonical teacher execution contract")
    expected = contract["identity_sha256"]
    if digest({k: v for k, v in contract.items() if k != "identity_sha256"}) != expected:
        raise ValueError("Canonical contract checksum differs")
    if contract["heldout_identity"] != _crop_identity(crops) or contract["metadata_sha256"] != digest(metadata):
        raise ValueError("Canonical source geometry, tensors or metadata changed")


def decorate_canonical(report, receipt):
    result = deepcopy(report); rows = result["rows"]; contract = receipt["contract"]
    keys = [(r["source_id"], r["start_frame"]) for r in rows]
    if len(keys) != len(set(keys)) or len(rows) != 285 or len({r["source_id"] for r in rows}) != 147:
        raise ValueError("Canonical evaluation requires every sealed crop and source")
    fixtures = [r for r in rows if r["source_id"] in historical.FIXTURES]
    if len(fixtures) != 3 or {r["source_id"] for r in fixtures} != historical.FIXTURES:
        raise ValueError("Canonical fixture identity changed")
    if any((r["metadata"].get("dataset") == "synthetic_fixture") != (r["source_id"] in historical.FIXTURES) for r in rows):
        raise ValueError("Canonical fixture classification changed")
    if result["quiet_config"] != contract["quiet_config"]:
        raise ValueError("Evaluator quiet configuration differs from sealed teacher renderer")
    groups = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        natural = row["source_id"] not in historical.FIXTURES
        groups["natural" if natural else "synthetic"].append(row)
        groups["source/" + row["source_id"]].append(row)
        if natural:
            groups["cohort/" + row["cohort"]].append(row)
            for field in ("language", "condition"):
                value = str(row["metadata"].get(field) or "unverified").strip().casefold().replace(" ", "_")
                groups[field + "/" + value].append(row)
    metrics = {key: historical._aggregate(value) for key, value in groups.items()}
    for name, field in (("raw_mae", "source_equal_mean_raw_mae"), ("mel", "source_equal_mean_mel")):
        values = [v[name] for k, v in metrics.items()
                  if k.startswith("source/") and k.removeprefix("source/") not in historical.FIXTURES]
        if len(values) != 144:
            raise ValueError("Natural source count changed")
        metrics["natural"][field] = sum(values) / 144
    expected_quiet = contract["canonical_quiet_windows"]
    if (metrics["all"]["samples"] != contract["scored_samples"]
            or metrics["all"]["samples"] != 26206830 or metrics["natural"]["samples"] != 25342830
            or metrics["natural"]["crops"] != 282 or metrics["natural"]["sources"] != 144
            or metrics["all"]["quiet_windows"] != expected_quiet["all"]
            or metrics["natural"]["quiet_windows"] != expected_quiet["natural"]
            or metrics["synthetic"]["quiet_windows"] != expected_quiet["fixtures"]
            or sum(r["context_excluded_samples"] for r in rows) != 618):
        raise ValueError("Canonical score geometry or sealed teacher quiet counts changed")
    zero = [r for r in fixtures if r["source_id"] == "encoded_zero"]
    if zero[0]["context_start_frame"] != 0 or zero[0]["samples"] != 288000:
        raise ValueError("Canonical encoded-zero fixture is not the full six seconds")
    steady = historical._quiet(zero, steady_start=96000)
    if any(steady[k] != contract["canonical_zero_steady"][k] for k in ("quiet_windows", "quiet_samples")):
        raise ValueError("Canonical steady encoded-zero mask differs from teacher receipt")
    result["recovery_metrics"] = metrics
    result["encoded_zero_steady_2_to_6_seconds"] = steady
    result["canonical_target_contract_sha256"] = contract["identity_sha256"]
    result["canonical_target_cache_sha256"] = receipt["sha256"]
    result["evaluation_domain"] = VERSION
    result["evaluation_contract_sha256"] = digest({"domain": VERSION,
        "canonical_target_contract_sha256": contract["identity_sha256"],
        "evaluation": historical._contract(report)})
    result["recovery_interpretation"] = {
        "role": "Canonical deployment-encoder generalization appendix; historical paired panel retained separately",
        "quiet": "Same20ms teacher-defined windows and sample-pooled residual RMS; masks sealed from canonical teacher",
        "criterion": "Unchanged common diagnostic criterion; waveform sample-weighted, mel equal-crop mean",
        "gates": "Same pilot screen thresholds; both historical and canonical domains must pass",
        "meaning": "No final quality-equivalence or deployment approval"}
    json.dumps(result, allow_nan=False)
    return result


def build_canonical_evaluation(engine, heldout, metadata, receipt):
    import torch
    from audiovae_student.fusion_evaluation import evaluate_fusion
    heldout = tuple(heldout)
    verify_inputs(heldout, metadata, receipt)
    peaks = []

    def capture(module, inputs, output):
        if len(peaks) >= len(heldout):
            raise ValueError("Unexpected extra canonical model forward")
        crop = heldout[len(peaks)]
        start = crop.scored_slice.start + (6 if crop.context_start_frame > 0 else 0)
        selected = output[..., start:crop.scored_slice.stop].detach().double()
        target = crop.teacher_audio[..., start:crop.scored_slice.stop]
        if not bool(torch.isfinite(selected).all()) or float(target.abs().max()) > 1:
            raise ValueError("Nonfinite canonical prediction or changed teacher peak contract")
        peaks.append((crop.source_id, crop.start_frame,
                      float((selected.abs() - 1).clamp_min(0).square().mean())))

    hook = engine.model.register_forward_hook(capture)
    try:
        report = evaluate_fusion(engine, heldout, metadata)
    finally:
        hook.remove()
    if len(peaks) != len(report["rows"]):
        raise ValueError("Canonical evaluator missed a crop")
    for row, (sid, start, energy) in zip(report["rows"], peaks):
        if (row["source_id"], row["start_frame"]) != (sid, start):
            raise ValueError("Canonical peak capture ordering differs")
        row["peak_excess_energy"] = energy
    return decorate_canonical(report, receipt)


def compare_canonical_reports(before, control, candidate):
    reports = (before, control, candidate)
    if any(r.get("evaluation_domain") != VERSION for r in reports):
        raise ValueError("Canonical comparison requires three decorated canonical reports")
    if len({r["canonical_target_cache_sha256"] for r in reports}) != 1:
        raise ValueError("Canonical comparison used different frozen caches")
    result = historical.compare_reports(before, control, candidate)
    result["evaluation_domain"] = VERSION
    result["canonical_target_cache_sha256"] = before["canonical_target_cache_sha256"]
    result["promotion_requires_historical_panel_also_pass"] = True
    return result
