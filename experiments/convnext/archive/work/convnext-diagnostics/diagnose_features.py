"""Frozen-checkpoint feature, output-head and source-disjoint readout diagnostics.

External least-squares coefficients are diagnostic probes. They never replace a
model parameter, alter a checkpoint, or become a deployed layer.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from audiovae_student.fusion_evaluation import (
    _MASK_POLICY, _cohort, _high_frequency, _phase_480,
    _preserved_evaluation, _summarize,
)
from audiovae_student.quiet_audio import QuietAudioConfig, quiet_window_metrics
from audiovae_student.batching import _validate_crop
from corrected_component_diagnostics import phase_decomposition


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def crop_key(crop):
    return (crop.source_id, crop.start_frame, crop.context_start_frame,
            crop.context_frames, crop.valid_scored_samples)


def stable_int(value):
    return int(hashlib.sha256(str(value).encode()).hexdigest()[:16], 16)


def identity_partition(crops, rows):
    """Union every supplied source/hash/parent/speaker/session identity first."""
    sources = sorted({c.source_id for c in crops})
    parent = {s: s for s in sources}

    def root(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    seen = {}
    fields = ("audio_sha256", "parent_recording_id", "speaker_id", "session_id")
    missing = Counter()
    for source in sources:
        row = rows.get(source, {})
        if hasattr(row, "to_dict"):
            row = row.to_dict()
        for field in fields:
            value = row.get(field)
            if not value:
                missing[field] += 1
                continue
            key = (field, value)
            if key in seen:
                a, b = sorted((root(source), root(seen[key])))
                parent[b] = a
            else:
                seen[key] = source
    groups = defaultdict(list)
    for source in sources:
        groups[root(source)].append(source)
    mapping = {}
    for members in groups.values():
        key = "|".join(sorted(members))
        split = "tune" if stable_int(key) % 5 == 0 else "fit"
        mapping.update({s: split for s in members})
    # No silent fallback that would break speaker/session disjointness.
    if set(mapping.values()) != {"fit", "tune"}:
        raise ValueError("Identity grouping did not yield both fit and tuning sources")
    return mapping, {
        "assignment": "SHA256 of full connected source group modulo five; 20 percent tune",
        "connected_groups": len(groups), "source_counts": dict(Counter(mapping.values())),
        "group_sizes_max": max(map(len, groups.values())),
        "shared_identities_across_fit_and_tune": 0,
        "identity_fields": list(fields), "missing_identity_source_counts": dict(missing),
        "limitations": "Only supplied identities are checked, not unknown speakers or acoustic duplicates",
    }


def select_probe_crops(crops, rows, heldout, *, maximum=1024):
    """One fixed selection for all checkpoints; never choose from held-out data."""
    heldout_ids = {c.source_id for c in heldout}
    if any(c.source_id in heldout_ids for c in crops):
        raise ValueError("Training pool overlaps held-out source IDs")
    unique = {crop_key(c): c for c in crops}
    selected = sorted(unique.values(), key=lambda c: stable_int(crop_key(c)))[:maximum]
    partition, report = identity_partition(selected, rows)
    report["window_language_counts"] = dict(Counter(rows.get(c.source_id, {}).get("language", "unknown") for c in selected))
    report["window_dataset_counts"] = dict(Counter(rows.get(c.source_id, {}).get("dataset", "unknown") for c in selected))
    report.update(selected_windows=len(selected), selected_sources=len(partition),
        selection="Deterministic hash ordering of targeted-generator training windows",
        heldout_source_id_overlap=0,
        heldout_identity_limit="The sealed corpus validates original split identities; this probe independently rechecks source IDs")
    return selected, partition, report


def complete_block_indices(crop, *, block=480, maximum=None):
    """Do not fit on six excluded history samples or a padded partial tail."""
    start = crop.scored_slice.start + (6 if crop.context_start_frame > 0 else 0)
    stop = crop.scored_slice.stop
    first, last = (start + block - 1) // block, stop // block
    indices = torch.arange(first, last, dtype=torch.long)
    if maximum is not None and len(indices) > maximum:
        generator = torch.Generator().manual_seed(stable_int(crop_key(crop)) % (2**63 - 1))
        selected = torch.randperm(len(indices), generator=generator)[:maximum].sort().values
        indices = indices[selected]
    return indices


def collect_features(engine, crops, *, full=False, max_blocks=64, batch_size=16):
    """Capture actual existing projection inputs with source/window accounting."""
    outputs = []
    crops = list(crops)
    with _preserved_evaluation(engine.model):
        for offset in range(0, len(crops), batch_size):
            batch = crops[offset:offset + batch_size]
            longest = max(c.latents.shape[-1] for c in batch)
            z = torch.cat([F.pad(c.latents, (0, longest - c.latents.shape[-1])) for c in batch]).to(engine.device)
            recorded = []
            hook = engine.model.output.register_forward_pre_hook(lambda module, args: recorded.append(args[0].detach()))
            try:
                prediction = engine.model(z)
            finally:
                hook.remove()
            if len(recorded) != 1 or prediction.shape != (len(batch), 1, longest * 1920):
                raise ValueError("Unexpected head execution or output length")
            features = recorded[0]
            if not torch.isfinite(features).all() or not torch.isfinite(prediction).all():
                raise ValueError("Nonfinite checkpoint features")
            for i, crop in enumerate(batch):
                _validate_crop(crop)
                n = crop.latents.shape[-1] * 4
                h = features[i, :, :n].transpose(0, 1).contiguous().cpu()
                p = prediction[i:i + 1, :, :n * 480].cpu()
                target = crop.teacher_audio.detach().cpu()
                if full:
                    outputs.append({"crop": crop, "features": h, "prediction": p, "target": target})
                else:
                    ids = complete_block_indices(crop, maximum=max_blocks)
                    if not len(ids):
                        raise ValueError("Selected fit window has no complete waveform block")
                    targets = target.reshape(-1, 480)[ids].contiguous()
                    outputs.append({"crop": crop, "features": h[ids].contiguous(), "target_blocks": targets,
                                    "block_indices": ids})
            print(json.dumps({"feature_collection": "heldout" if full else "training_probe",
                "step": int(engine.step), "windows": min(offset + batch_size, len(crops)),
                "total": len(crops)}), flush=True)
    return outputs


def matrix_spectrum(weight):
    value = weight.detach().reshape(weight.shape[0], -1).double()
    singular = torch.linalg.svdvals(value)
    maximum = float(singular.max())
    minimum = float(singular.min())
    eps64 = max(value.shape) * torch.finfo(torch.float64).eps * maximum
    eps32 = max(value.shape) * torch.finfo(torch.float32).eps * maximum
    return {"shape": list(value.shape), "singular_values": singular.cpu().tolist(),
        "rank_float64_threshold": int((singular > eps64).sum()),
        "rank_float32_threshold": int((singular > eps32).sum()),
        "singular_min": minimum, "singular_max": maximum,
        "condition_number": maximum / minimum if minimum > 0 else None,
        "threshold_float64": eps64, "threshold_float32": eps32,
        "full_row_rank_does_not_prove_upstream_features_can_reach_arbitrary_coefficients": True}


def mean_feature_terms(features, prediction, target, weight, *, quiet_rms=.001):
    """Exact quiet phase mean/deviation decomposition, no activation intervention."""
    cycles = min(len(features) // 4, prediction.numel() // 1920)
    if not cycles:
        return {"quiet_cycles": 0}
    h = features[:cycles * 4].double().reshape(cycles, 4, -1)
    p = prediction.reshape(-1)[:cycles * 1920].double().reshape(cycles, 4, 480)
    t = target.reshape(-1)[:cycles * 1920].double().reshape(cycles, 4, 480)
    selected = t.square().mean((1, 2)).sqrt() <= quiet_rms
    h, p, t = h[selected], p[selected], t[selected]
    if not len(h):
        return {"quiet_cycles": 0}
    w = weight.double().reshape(480, -1).cpu()
    mean_h, mean_t = h.mean(0), t.mean(0)
    mean_output = mean_h @ w.T
    mean_residual = mean_output - mean_t
    varying_residual = (h - mean_h) @ w.T - (t - mean_t)
    actual = p - t
    reconstructed = mean_residual[None] + varying_residual
    return {"quiet_cycles": len(h), "head_feature_phase_rms": h.square().mean((0, 2)).sqrt().tolist(),
        "mean_feature_phase_rms": mean_h.square().mean(1).sqrt().tolist(),
        "mean_output_phase_rms": mean_output.square().mean(1).sqrt().tolist(),
        "teacher_mean_phase_rms": mean_t.square().mean(1).sqrt().tolist(),
        "fixed_mean_residual_power": float(mean_residual.square().mean()),
        "varying_residual_power": float(varying_residual.square().mean()),
        "cross_term": float(2 * (mean_residual[None] * varying_residual).mean()),
        "observed_residual_power": float(actual.square().mean()),
        "linear_reconstruction_max_error": float((actual - reconstructed).abs().max()),
        "interpretation": "Exact linear-output attribution on observed features, not proof of a causal defect"}


def unique_peak_events(caches):
    """Merge overlapping crop observations on the original source sample grid."""
    per_source = defaultdict(dict)
    for item in caches:
        crop, prediction, target = item["crop"], item["prediction"], item["target"]
        start = crop.scored_slice.start + (6 if crop.context_start_frame > 0 else 0)
        stop = crop.scored_slice.stop
        p, t = prediction.reshape(-1)[start:stop], target.reshape(-1)[start:stop]
        for index in torch.nonzero(p.abs() > 1).reshape(-1).tolist():
            absolute = crop.context_start_frame * 1920 + start + index
            candidate = {"student": float(p[index]), "teacher": float(t[index])}
            old = per_source[crop.source_id].get(absolute)
            if old is None or abs(candidate["student"]) > abs(old["student"]):
                per_source[crop.source_id][absolute] = candidate
    events = []
    for source, samples in sorted(per_source.items()):
        ordered = sorted(samples)
        groups = []
        for sample in ordered:
            if not groups or sample > groups[-1][-1] + 1:
                groups.append([])
            groups[-1].append(sample)
        for group in groups:
            peak = max(group, key=lambda j: abs(samples[j]["student"]))
            events.append({"source_id": source, "start_sample": group[0], "stop_sample_exclusive": group[-1] + 1,
                "samples": len(group), "peak_sample": peak, "phase480": peak % 480, "phase1920": peak % 1920,
                "student_peak_signed": samples[peak]["student"], "teacher_at_student_peak": samples[peak]["teacher"]})
    return {"sources": len(per_source), "unique_samples": sum(len(v) for v in per_source.values()),
        "events": events, "event_definition": "Contiguous overshoot samples on source absolute timeline",
        "overlap_policy": "One source/sample entry, retain maximum absolute student value across crop observations"}


def component_panel(caches, weight, metadata):
    rows = []
    for item in caches:
        crop = item["crop"]
        skip = 1920 if crop.context_start_frame > 0 else 0
        start, stop = crop.scored_slice.start + skip, crop.scored_slice.stop
        if stop <= start:
            raise ValueError("Component exclusion removed the entire valid crop")
        p, t = item["prediction"][..., start:stop], item["target"][..., start:stop]
        h = item["features"][start // 480:stop // 480]
        rows.append({"source_id": crop.source_id, "start_frame": crop.start_frame,
            "metadata": metadata.get(crop.source_id, {}), "valid_samples": stop - start,
            "context_exclusion_samples": skip, "quiet": phase_decomposition(p, t),
            "features": mean_feature_terms(h, p, t, weight)})
    powers = defaultdict(float)
    cohort_powers = defaultdict(lambda: defaultdict(float))
    cohort_cycles = Counter()
    cycles = 0
    valid_rows = 0
    for row in rows:
        q = row["quiet"]
        if "power_components" not in q:
            continue
        valid_rows += 1
        cycles += q["quiet_cycles_40ms"]
        cohort = "synthetic" if row["source_id"].startswith("encoded_") else "natural"
        cohort_cycles[cohort] += q["quiet_cycles_40ms"]
        for name, power in q["power_components"].items():
            powers[name] += power * q["quiet_cycles_40ms"]
            cohort_powers[cohort][name] += power * q["quiet_cycles_40ms"]
    return {"rows": rows, "crops": len(rows), "sources": len({r["source_id"] for r in rows}),
        "cycles_with_at_least_eight_per_crop": cycles, "decomposable_crops": valid_rows,
        "pooled_component_power": {k: v / cycles for k, v in powers.items()} if cycles else {},
        "component_power_by_cohort": {cohort: {"quiet_cycles": cohort_cycles[cohort],
            "power": {k: v / cohort_cycles[cohort] for k,v in values.items()}} for cohort, values in cohort_powers.items()},
        "cohorts": dict(Counter(str(r["metadata"].get("condition", "unknown")) for r in rows)),
        "population_selection": "All held-out crops, no condition-name filtering",
        "component_mask": "Complete 40ms cycles; exclude first40ms only on interior archived crops",
        "quiet_threshold": .001, "minimum_quiet_cycles_per_crop": 8,
        "overlapping_crops_are_not_independent_sources": True,
        "unique_peaks": unique_peak_events(caches)}


def exact_output_change(old_h, new_h, old_w, new_w):
    old_h, new_h, old_w, new_w = [v.double() for v in (old_h, new_h, old_w, new_w)]
    head = ((old_h + new_h) / 2) @ (new_w - old_w).T
    upstream = (new_h - old_h) @ ((old_w + new_w) / 2).T
    change = new_h @ new_w.T - old_h @ old_w.T
    return head, upstream, change


def term_statistics(head, upstream, actual):
    if not head.numel():
        return None
    return {"samples": head.numel(), "head_change_power": float(head.square().mean()),
        "upstream_change_power": float(upstream.square().mean()),
        "cross_term": float(2 * (head * upstream).mean()), "total_change_power": float(actual.square().mean()),
        "identity_max_error": float((head + upstream - actual).abs().max()),
        "head_mean": float(head.mean()), "upstream_mean": float(upstream.mean())}


def checkpoint_change_panel(old, new, old_w, new_w, metadata):
    if [crop_key(v["crop"]) for v in old] != [crop_key(v["crop"]) for v in new]:
        raise ValueError("Checkpoint comparison did not retain identical crop order")
    rows = []
    for a, b in zip(old, new):
        c = a["crop"]
        head, upstream, delta = exact_output_change(a["features"], b["features"], old_w, new_w)
        head, upstream, delta = [v.reshape(-1) for v in (head, upstream, delta)]
        target = a["target"].reshape(-1).double()
        start = c.scored_slice.start + (6 if c.context_start_frame > 0 else 0)
        stop = c.scored_slice.stop
        hp, up, dp, tp = [v[start:stop] for v in (head, upstream, delta, target)]
        # Quiet samples use genuine complete20ms windows on original crop grid.
        quiet = torch.zeros_like(target, dtype=torch.bool)
        complete = target.numel() // 960
        valid = torch.zeros_like(target, dtype=torch.bool); valid[start:stop] = True
        windows = target[:complete * 960].reshape(-1, 960)
        selected = (windows.square().mean(1).sqrt() <= .001) & valid[:complete * 960].reshape(-1, 960).all(1)
        quiet[:complete * 960] = selected[:, None].expand(-1, 960).reshape(-1)
        peak = ((a["prediction"].reshape(-1).abs() > 1) | (b["prediction"].reshape(-1).abs() > 1)) & valid
        observed_change = b["prediction"].reshape(-1).double() - a["prediction"].reshape(-1).double()
        rows.append({"source_id": c.source_id, "start_frame": c.start_frame,
            "metadata": metadata.get(c.source_id, {}), "all": term_statistics(hp, up, dp),
            "observed_model_delta_vs_feature_matmul_max_error": float((observed_change[valid]-delta[valid]).abs().max()),
            "quiet": term_statistics(head[quiet], upstream[quiet], delta[quiet]),
            "overshoot_samples": term_statistics(head[peak], upstream[peak], delta[peak])})
    return {"rows": rows, "equation": "deltaY=meanH deltaW^T + deltaH meanW^T",
        "interpretation": "Exact symmetric attribution of checkpoint changes, not causal intervention",
        "parameter_updates": 0}


def probe_statistics(items, partition, split, device):
    width, outputs = items[0]["features"].shape[1], items[0]["target_blocks"].shape[1]
    xx = torch.zeros(width, width, dtype=torch.float64, device=device)
    xy = torch.zeros(width, outputs, dtype=torch.float64, device=device)
    sx = torch.zeros(width, dtype=torch.float64, device=device)
    sy = torch.zeros(outputs, dtype=torch.float64, device=device)
    count, yy = 0, 0.
    kept = [item for item in items if partition[item["crop"].source_id] == split]
    for start in range(0, len(kept), 32):
        batch = kept[start:start + 32]
        x = torch.cat([v["features"] for v in batch]).to(device=device, dtype=torch.float64)
        y = torch.cat([v["target_blocks"] for v in batch]).to(device=device, dtype=torch.float64)
        xx.add_(x.T @ x); xy.add_(x.T @ y)
        sx.add_(x.sum(0)); sy.add_(y.sum(0))
        yy += float(y.square().sum()); count += len(x)
    if count <= width:
        raise ValueError("Too few source-disjoint feature blocks for the diagnostic readout")
    return {"xx": xx / count, "xy": xy / count, "sx": sx / count,
        "sy": sy / count, "yy": yy / count, "count": count}


def fit_readout(stats, *, ridge, intercept):
    """RMS-scaled ridge solves with source-disjoint tuning; outputs [D,480]."""
    xx, xy, sx, sy = (stats[k] for k in ("xx", "xy", "sx", "sy"))
    scale = xx.diag().clamp_min(0).sqrt()
    scale = scale.clamp_min(scale.max().clamp_min(1e-20) * 1e-8)
    gram = xx - torch.outer(sx, sx) if intercept else xx
    cross = xy - torch.outer(sx, sy) if intercept else xy
    gram = gram / scale[:, None] / scale[None, :]
    cross = cross / scale[:, None]
    penalty = ridge * gram.diag().mean().clamp_min(1e-30)
    regularized = (gram + gram.T) / 2 + torch.eye(len(scale), device=gram.device, dtype=gram.dtype) * penalty
    cholesky, info = torch.linalg.cholesky_ex(regularized)
    if int(info) != 0:
        raise ValueError("Diagnostic ridge matrix is not positive definite")
    coefficient = torch.cholesky_solve(cross, cholesky) / scale[:, None]
    bias = sy - sx @ coefficient if intercept else sy.new_zeros(sy.shape)
    if not torch.isfinite(coefficient).all() or not torch.isfinite(bias).all():
        raise ValueError("Nonfinite diagnostic coefficients")
    return coefficient, bias, {"relative_ridge": ridge, "scaled_diagonal_penalty": float(penalty),
        "feature_rms_min": float(scale.min()), "feature_rms_max": float(scale.max()),
        "intercept": intercept, "fit_blocks": stats["count"], "external_probe_only": True}


def block_probe_metrics(items, partition, split, coefficient, bias):
    device = coefficient.device
    count = 0; absolute = squared = teacher_squared = 0.; peaks = 0; max_peak = 0.
    quiet_squared = quiet_count = 0
    for item in items:
        if partition[item["crop"].source_id] != split:
            continue
        x = item["features"].to(device=device, dtype=torch.float64)
        y = item["target_blocks"].to(device=device, dtype=torch.float64)
        p = x @ coefficient + bias
        e = p - y
        absolute += float(e.abs().sum()); squared += float(e.square().sum()); teacher_squared += float(y.square().sum())
        count += e.numel(); peaks += int((p.abs() > 1).sum()); max_peak = max(max_peak, float(p.abs().max()))
        quiet = y.square().mean(1).sqrt() <= .001
        quiet_squared += float(e[quiet].square().sum()); quiet_count += int(quiet.sum()) * y.shape[1]
    return {"samples": count, "raw_mae": absolute / count, "mse": squared / count,
        "teacher_mean_square": teacher_squared / count, "overshoot_samples": peaks, "peak_abs": max_peak,
        "quiet_10ms_samples": quiet_count, "quiet_10ms_residual_rms": (quiet_squared / quiet_count) ** .5 if quiet_count else None,
        "quiet_scope": "Selected complete10ms blocks for probe tuning, distinct from common20ms final metrics"}


def quality_from_cache(engine, caches, metadata, *, coefficient=None, bias=None):
    """Original48k quality criterion/mask applied to external readout waveforms."""
    quiet_config = getattr(engine.config, "quiet_audio", QuietAudioConfig())
    rows = []
    with _preserved_evaluation(engine.criterion):
        for item in caches:
            crop = item["crop"]
            if coefficient is None:
                prediction = item["prediction"].to(engine.device)
            else:
                h = item["features"].to(device=engine.device, dtype=torch.float64)
                prediction = (h @ coefficient + bias).reshape(1, 1, -1).float()
            target = item["target"].to(engine.device)
            excluded = 6 if crop.context_start_frame > 0 else 0
            start, stop = crop.scored_slice.start + excluded, crop.scored_slice.stop
            mask = torch.zeros_like(prediction, dtype=torch.bool); mask[..., start:stop] = True
            p, t = prediction[..., start:stop], target[..., start:stop]
            losses = engine.criterion(p, t)
            meta = metadata.get(crop.source_id, {})
            rows.append({"source_id": crop.source_id, "start_frame": crop.start_frame, "metadata": meta,
                "cohort": _cohort(meta), "samples": stop-start, "original_scored_samples": crop.valid_scored_samples,
                "context_excluded_samples": excluded, **{k: float(v) for k,v in losses.items()},
                "waveform_raw_mae": float((p-t).abs().mean()), "teacher_rms": float(t.square().mean().sqrt()),
                "waveform_cosine": float(F.cosine_similarity(p.flatten(), t.flatten(), dim=0, eps=1e-12)),
                "student_peak_abs": float(p.abs().max()), "teacher_peak_abs": float(t.abs().max()),
                "student_overshoot_samples": int((p.abs()>1).sum()), "teacher_overshoot_samples": int((t.abs()>1).sum()),
                "student_near_saturation_samples": int((p.abs()>=.999).sum()), "teacher_near_saturation_samples": int((t.abs()>=.999).sum()),
                "quiet_windows": quiet_window_metrics(prediction,target,mask,config=quiet_config),
                "quiet_phase480": _phase_480(prediction,target,mask,quiet_config), "high_frequency": _high_frequency(p,t)})
    groups = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        groups["synthetic" if row["source_id"].startswith("encoded_") else "natural"].append(row)
        groups["cohort/" + row["cohort"]].append(row)
    return {"groups": {k:_summarize(v,quiet_config) for k,v in groups.items()}, "rows": rows,
        "mask_policy": dict(_MASK_POLICY), "criterion_config": asdict(engine.criterion.config),
        "probe_coefficients_used_only_for_diagnostic_predictions": coefficient is not None,
        "probe_matmul_precision": "Float64, finalwaveformFloat32" if coefficient is not None else "originalcheckpointFP32"}


def run(ctx):
    out = Path(ctx.outPath) / "features"
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    pools = ctx.pools
    selected, partition, split_report = select_probe_crops(pools["targeted_generator"], ctx.data["rows"], ctx.heldout)
    atomic_json(out / "probe-split.json", {**split_report, "source_partition": partition,
        "windows": [list(crop_key(c)) for c in selected], "max_blocks_per_window": 64,
        "role": "External diagnostic readout fitting only, no student or optimizer updates"})
    baseline_cache = baseline_weight = None
    summary = {"format_version": 1, "parameter_updates": 0, "source_split": split_report,
        "readout_warning": "A better external readout localizes a representational opportunity; a poor readout does not prove architecture capacity insufficient",
        "arms": {}}
    for name in ("parent", "targeted", "complex"):
        print(json.dumps({"diagnostic": "features", "arm": name, "phase": "load"}), flush=True)
        engine = ctx.engine(name, device="cuda")
        weight = engine.model.output.weight.detach().reshape(480, -1).cpu()
        if engine.model.output.bias is not None:
            raise ValueError("Expected original bias-free output projection")
        report = {"model_config": engine.model.config.to_dict(), "step": int(engine.step),
            "head_weight_spectrum": matrix_spectrum(weight.to(engine.device))}
        cache = collect_features(engine, ctx.heldout, full=True)
        report["components"] = component_panel(cache, weight, ctx.metadata)
        report["original_quality"] = quality_from_cache(engine, cache, ctx.metadata)
        if name == "parent":
            baseline_cache, baseline_weight = cache, weight
        else:
            report["change_from_parent"] = checkpoint_change_panel(baseline_cache, cache, baseline_weight, weight, ctx.metadata)
        train_cache = collect_features(engine, selected, full=False)
        stats = probe_statistics(train_cache, partition, "fit", engine.device)
        standardized = stats["xx"] / stats["xx"].diag().clamp_min(1e-30).sqrt()[:,None] / stats["xx"].diag().clamp_min(1e-30).sqrt()[None,:]
        eig = torch.linalg.eigvalsh((standardized + standardized.T) / 2)
        report["fit_features"] = {"blocks": stats["count"], "rms_normalized_gram_eigen_min": float(eig.min()),
            "eigen_max": float(eig.max()), "eigenvalues": eig.cpu().tolist(),
            "eigenvalues_above_1e_minus6_of_max": int((eig > eig.max() * 1e-6).sum()),
            "empirical_gram_rank_is_not_structural_model_rank": True}
        report["readouts"] = {}
        original_coefficient = weight.T.to(device=engine.device, dtype=torch.float64)
        zeros = original_coefficient.new_zeros(480)
        report["original_selected_blocks"] = {s: block_probe_metrics(train_cache,partition,s,original_coefficient,zeros) for s in ("fit","tune")}
        for intercept in (False, True):
            label = "affine" if intercept else "same_dimension_bias_free"
            candidates = []
            best = None
            for ridge in (1e-8, 1e-6, 1e-4, 1e-2):
                coefficient, bias, details = fit_readout(stats, ridge=ridge, intercept=intercept)
                tuning = block_probe_metrics(train_cache, partition, "tune", coefficient, bias)
                candidates.append({**details, "tune": tuning})
                if best is None or tuning["mse"] < best[0]:
                    best = (tuning["mse"], coefficient, bias, details)
            _, coefficient, bias, details = best
            report["readouts"][label] = {"grid": candidates, "selection": "Lowest source-disjoint tuning MSE, heldout never used",
                "selected": details, "fit": block_probe_metrics(train_cache,partition,"fit",coefficient,bias),
                "tune": block_probe_metrics(train_cache,partition,"tune",coefficient,bias),
                "heldout": quality_from_cache(engine,cache,ctx.metadata,coefficient=coefficient,bias=bias)}
        atomic_json(out / (name + ".json"), report)
        summary["arms"][name] = {"model_config": report["model_config"], "step": report["step"],
            "head_weight_spectrum": {k:v for k,v in report["head_weight_spectrum"].items() if k!="singular_values"},
            "component_crops": report["components"]["crops"], "component_sources": report["components"]["sources"],
            "decomposable_crops": report["components"]["decomposable_crops"],
            "pooled_component_power": report["components"]["pooled_component_power"],
            "component_power_by_cohort": report["components"]["component_power_by_cohort"],
            "unique_peaks": report["components"]["unique_peaks"],
            "quality": report["original_quality"]["groups"],
            "readouts": {k:{key:value for key,value in v.items() if key!="heldout"} | {"heldout_groups":v["heldout"]["groups"]} for k,v in report["readouts"].items()}}
        del engine, train_cache, stats, standardized, eig, report, coefficient, bias, best, original_coefficient, zeros
        if name != "parent":
            del cache
        gc.collect(); torch.cuda.empty_cache()
    summary["seconds"] = time.monotonic() - started
    atomic_json(out / "summary.json", summary)
    return summary


if __name__ == "__main__":
    from diagnostic_common import load_context
    run(load_context())
