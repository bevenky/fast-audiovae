"""Fresh-student finite data plan with fixed, capacity-capped mixture quotas.

This module only plans metadata. Older models do not consume this student's
data. Calibration replays separate training windows without gradient updates.
"""
from collections import Counter, defaultdict
from dataclasses import replace
from fractions import Fraction
import heapq
import math
from pathlib import Path

from .comparison_data import (INDIC22, WEIGHTS, _primary, known_identities,
    unknown_session, verify_comparison_windows, write_comparison_plan, load_comparison_plan)
from .data import validate_manifest
from .restart_data import (ConsumedLedger, RATE, HOP, MINIMUM, _counts, _order,
    canonical, digest, windows_for)
from .sampling import SamplerExhausted


def _allocate(capacity, total, weights):
    """Integer weighted water filling; capped groups redistribute before use."""
    if total < 0 or total > sum(capacity.values()):
        raise SamplerExhausted("Unique window capacity is below the declared budget")
    result = {key: 0 for key in capacity}
    pending = {key for key in capacity if capacity[key]}
    remaining = total
    while remaining:
        weight_sum = math.fsum(weights[key] for key in sorted(pending))
        if weight_sum <= 0:
            raise ValueError("Positive weights are required for available groups")
        saturated = [key for key in pending if capacity[key] <= remaining * weights[key] / weight_sum]
        if saturated:
            for key in saturated:
                result[key] = capacity[key]
                remaining -= capacity[key]
                pending.remove(key)
            continue
        desired = {key: remaining * weights[key] / weight_sum for key in pending}
        for key in pending:
            result[key] = int(desired[key])
        remainder = remaining - sum(result[key] for key in pending)
        for key in sorted(pending, key=lambda k: (-(desired[k] - int(desired[k])), k))[:remainder]:
            result[key] += 1
        remaining = 0
    return result


def _group_quotas(capacity, total):
    buckets = {bucket: sum(n for (b, _), n in capacity.items() if b == bucket) for bucket in WEIGHTS}
    bucket_quota = _allocate(buckets, total, WEIGHTS)
    groups = {}
    for bucket in WEIGHTS:
        available = {key: n for key, n in capacity.items() if key[0] == bucket}
        groups.update(_allocate(available, bucket_quota[bucket], {key: 1 for key in available}))
    return groups


def _interleave_fixed(selected):
    """Evenly spread each finite group's windows instead of draining rare ones."""
    heap = [(Fraction(1, 2 * len(queue)), key, 0) for key, queue in selected.items() if queue]
    heapq.heapify(heap)
    result = []
    while heap:
        _, key, index = heapq.heappop(heap)
        result.append(selected[key][index])
        index += 1
        if index < len(selected[key]):
            heapq.heappush(heap, (Fraction(2 * index + 1, 2 * len(selected[key])), key, index))
    return result


def plan_recipe_v2_data(rows, counts, *, event_labels=None, reserved_rows=(), excluded_sources=(),
                        optimization_windows=320000, calibration_windows=512,
                        minimum_input_samples=3040, seed=43,
                        required_indic=INDIC22,
                        required_other=("cmn", "yue", "ar", "es", "pt", "fr", "ja", "de")):
    for name, value in (("optimization_windows", optimization_windows), ("calibration_windows", calibration_windows)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(minimum_input_samples) is not int or not MINIMUM <= minimum_input_samples <= 64 * HOP:
        raise ValueError("Invalid minimum scored window length")
    required_indic, required_other = tuple(required_indic), tuple(required_other)
    rows = sorted(rows, key=lambda row: row.source_id)
    _counts(rows, counts)
    reserved_rows, excluded_sources = tuple(reserved_rows), tuple(excluded_sources)
    reserved = set().union(*(known_identities(row) for row in reserved_rows))
    excluded = set().union(*(known_identities(row, people=False) for row in excluded_sources))
    eligible = [row for row in rows if row.split == "train" and not known_identities(row) & reserved
                and not known_identities(row, people=False) & excluded]
    validate_manifest(eligible, training_only=True)
    labels = event_labels or {}
    grouped = defaultdict(list)
    omitted = Counter()
    for row in eligible:
        category = _primary(row, labels)
        if category is None:
            omitted["unknown_language_without_verified_event"] += 1
        else:
            grouped[category].append(row)
    queues, discarded = {}, {}
    for category, sources in grouped.items():
        queue, skipped = [], 0
        for row in _order(sources, seed):
            used = 0
            for window in windows_for(row, counts[row.source_id], minimum=minimum_input_samples):
                condition = category[1] if category[0] == "events" else {
                    "jnv": "jnv_nonverbal", "jvnv": "jvnv_verbal_and_nonverbal"}.get(row.dataset, "speech")
                queue.append(replace(window, condition=condition))
                used += window.valid_input_samples16k
            skipped += counts[row.source_id] - used
        if queue:
            queues[category] = queue
            discarded[category] = skipped
    for category in [*(("indic", language) for language in required_indic),
                     *(("other", language) for language in required_other)]:
        if not queues.get(category):
            raise ValueError(f"Missing required speech group: {category}")
    capacity = {key: len(queue) for key, queue in queues.items()}
    preliminary = _group_quotas(capacity, optimization_windows + calibration_windows)
    # One calibration window per available group, while keeping at least one
    # optimization window. Then approximate the feasible optimization mixture.
    mandatory = {key: int(count >= 2 and preliminary[key] > 0) for key, count in capacity.items()}
    if sum(mandatory.values()) > calibration_windows:
        raise ValueError("Calibration budget is too small to represent every eligible subgroup")
    calibration_capacity = {key: max(0, count - 1 - mandatory[key]) for key, count in capacity.items()}
    extra = _allocate(calibration_capacity, calibration_windows - sum(mandatory.values()),
                      {key: max(1, preliminary[key]) for key in capacity})
    calibration_quota = {key: mandatory[key] + extra[key] for key in capacity}
    remaining_capacity = {key: capacity[key] - calibration_quota[key] for key in capacity}
    optimization_quota = _group_quotas(remaining_capacity, optimization_windows)
    calibration_groups = {key: queue[:calibration_quota[key]] for key, queue in queues.items()}
    optimization_groups = {key: queue[calibration_quota[key]:calibration_quota[key] + optimization_quota[key]]
                           for key, queue in queues.items()}
    optimization = _interleave_fixed(optimization_groups)
    calibration = _interleave_fixed(calibration_groups)
    ledger_body = {"format_version": 1, "input_sample_rate": RATE, "sources": [],
        "provenance": {"fresh_student": True, "older_independent_models_data_reusable": True},
        "exposure_policy": "No scored optimization repeats; calibration windows separate and replayed twice without gradients"}
    ledger = ConsumedLedger({**ledger_body, "identity_sha256": digest(ledger_body)})
    by_id = {row.source_id: row for row in eligible}
    all_selected = optimization + calibration
    selected_ids = {w.source_id for w in all_selected}
    combined_counts = {key: counts[key] for key in selected_ids}
    verify_comparison_windows(all_selected, {key: by_id[key] for key in selected_ids}, combined_counts, ledger,
                              reserved_rows=reserved_rows, excluded_sources=excluded_sources)

    def make_plan(windows, purpose):
        ids = {w.source_id for w in windows}
        chosen = [by_id[key] for key in sorted(ids)]
        samples = sum(w.valid_input_samples16k for w in windows)
        by_group, by_bucket = defaultdict(int), defaultdict(int)
        count_group = Counter()
        for window in windows:
            group = _primary(by_id[window.source_id], labels)
            by_group[":".join(group)] += window.valid_input_samples16k
            by_bucket[group[0]] += window.valid_input_samples16k
            count_group[":".join(group)] += 1
        return {"rows": chosen, "windows": windows, "counts": {key: counts[key] for key in ids},
                "ledger": ledger, "reserved": reserved_rows, "excluded": excluded_sources, "seed": seed,
                "metadata": {"purpose": purpose, "split": "train", "windows": len(windows), "rows": len(ids),
                    "input_samples": samples, "scored_hours": samples / RATE / 3600, "scored_frames": 64,
                    "context_frames": 29, "minimum_input_samples": minimum_input_samples,
                    "minimum_output_samples": minimum_input_samples * 3,
                    "by_subgroup_windows": dict(sorted(count_group.items())),
                    "by_subgroup_input_samples": dict(sorted(by_group.items())),
                    "by_bucket": {key: {"input_samples": by_bucket[key], "fraction": by_bucket[key] / samples}
                                  for key in WEIGHTS},
                    "speech_languages": sorted({w.language for w in windows if w.condition == "speech"}),
                    "requested_bucket_window_fractions": WEIGHTS,
                    "allocation_policy": "Capacity-capped fixed window quotas, midpoint-spaced throughout the run; actual duration shares reported",
                    "source_audio_validation": "Metadata-only: hash and exact decoded lengths must be checked before use",
                    "statistics_only_passes": 2 if purpose == "normalization_calibration" else 0,
                    "gradient_passes": 0 if purpose == "normalization_calibration" else 1,
                    "unknown_shared_session_placeholders": sorted({r.session_id for r in chosen if unknown_session(r)} &
                                                                  {r.session_id for r in reserved_rows if unknown_session(r)}),
                    "event_scope": "Whole windows from labeled recordings, not densely annotated event duration"}}

    optimization_plan = make_plan(optimization, "optimization")
    calibration_plan = make_plan(calibration, "normalization_calibration")
    if not set(required_indic + required_other) <= set(optimization_plan["metadata"]["speech_languages"]):
        raise ValueError("Budget did not reach every required speech language")
    report = {"format_version": 1, "fresh_student": True, "prior_model_retirements_applied": False,
        "optimization": optimization_plan["metadata"], "calibration": calibration_plan["metadata"],
        "capacity": {":".join(key): {"available_windows": capacity[key],
            "available_input_samples": sum(w.valid_input_samples16k for w in queues[key]),
            "optimization_windows": optimization_quota[key], "calibration_windows": calibration_quota[key],
            "unused_windows": capacity[key] - optimization_quota[key] - calibration_quota[key],
            "discarded_short_tail_input_samples": discarded[key]} for key in sorted(capacity)},
        "calibration_groups_unavailable_without_draining_optimization": [":".join(key) for key in sorted(capacity)
                                                                         if not calibration_quota[key]],
        "omitted_sources": dict(omitted), "optimization_calibration_scored_overlap": False,
        "shared_source_files": len({w.source_id for w in optimization} & {w.source_id for w in calibration}),
        "heldout_reservations": len(reserved_rows), "source_exclusions": len(excluded_sources),
        "effective_mixture_is_window_weighted": True,
        "scarce_event_policy": "Use each available selected window once and distribute its class across the full run; no oversampling"}
    return {"optimization": optimization_plan, "calibration": calibration_plan, "report": report}


def write_recipe_v2_plan(plan, directory, *, provenance):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    optimization = write_comparison_plan(plan["optimization"], directory / "optimization", provenance=provenance)
    calibration = write_comparison_plan(plan["calibration"], directory / "calibration", provenance=provenance)
    for name in ("optimization", "calibration"):
        checked = load_comparison_plan(directory / name)
        if checked["metadata"] != plan[name]["metadata"]:
            raise ValueError("Published plan metadata differs from its finite selection")
    (directory / "data-plan.json").write_bytes(canonical(plan["report"]))
    body = {"format_version": 1, "state": "metadata_ready_audio_validation_pending",
            "optimization_plan_identity": optimization["identity_sha256"],
            "calibration_plan_identity": calibration["identity_sha256"],
            "report_sha256": digest(plan["report"]), "provenance": provenance}
    ready = {**body, "identity_sha256": digest(body)}
    (directory / "ready.json").write_bytes(canonical(ready))
    return ready
