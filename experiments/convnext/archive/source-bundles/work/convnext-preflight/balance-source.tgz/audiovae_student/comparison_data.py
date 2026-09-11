"""Finite matched diagnostic plans with per-student exposure and honest identities.

Only FLEURS' explicitly unknown language-wide session placeholders are omitted
from people checks. Source metadata remains unchanged. No audio is read here.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import replace
import json
from pathlib import Path
import re

from .data import ManifestRow, load_manifest, validate_manifest
from .restart_data import (ConsumedLedger, FixedWindowSampler, PilotWindow, RATE, HOP,
    MINIMUM, canonical, digest, file_sha, identities, identity, verify_windows,
    windows_for, _counts, _order)
from .sampling import SamplerExhausted, normalize_language

INDIC22 = ("as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai",
           "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur")
RARE_EVENTS = ("Crying_and_sobbing", "Giggle", "Shout",
               "human_whistling_source_description", "Yell")
EVENTS = RARE_EVENTS + ("Chuckle_and_chortle", "Laughter", "Screaming", "Whispering",
                       "explicit_whisper_style", "Breathing")
WEIGHTS = {"english": .30, "indic": .30, "other": .35, "events": .05}
_UNKNOWN = re.compile(r"fleurs:unknown-session-group:[A-Za-z0-9_-]+\Z")


def unknown_session(row):
    return row.dataset == "fleurs" and bool(_UNKNOWN.fullmatch(row.session_id or ""))


def known_identities(row, *, people=True):
    result = identities(row, people=people)
    if unknown_session(row):
        result.discard(("session", row.session_id))
    return result


def assert_comparison_disjoint(rows, reserved_rows=(), excluded_sources=()):
    """Reject real source, parent, hash and known people leakage.

    Fitted diagnostic exclusions can be file-only; held-out exclusions include
    known people. A shared unknown placeholder is disclosed, not called a person.
    """
    reserved = set().union(*(known_identities(r) for r in reserved_rows))
    excluded = set().union(*(known_identities(r, people=False) for r in excluded_sources))
    for row in rows:
        if known_identities(row) & reserved or known_identities(row, people=False) & excluded:
            raise ValueError(f"Comparison source overlaps a reserved identity: {row.source_id}")


def verify_comparison_windows(windows, rows, counts, ledger, *, reserved_rows=(), excluded_sources=()):
    assert_comparison_disjoint(rows.values(), reserved_rows, excluded_sources)
    validate_manifest(rows.values(), training_only=True)
    verify_windows(windows, rows, counts, ledger)


def parent_student_ledger(rows, counts, *, checkpoint_sha256, provenance=None):
    """Conservatively retire the current parent's complete used source files."""
    rows = sorted(rows, key=lambda r: r.source_id)
    _counts(rows, counts)
    if not re.fullmatch(r"[a-f0-9]{64}", checkpoint_sha256):
        raise ValueError("Parent checkpoint identity must be a SHA256")
    body = {"format_version": 1, "input_sample_rate": RATE,
            "provenance": {"parent_checkpoint_sha256": checkpoint_sha256, **(provenance or {})},
            "exposure_policy": "Current student used files excluded conservatively; older independent models do not exclude data",
            "sources": [{"row": r.to_dict(), "input_samples": counts[r.source_id],
                         "retired_whole": True, "intervals": [[0, counts[r.source_id]]]}
                        for r in rows]}
    return ConsumedLedger({**body, "identity_sha256": digest(body)})


def _primary(row, event_labels):
    labels = event_labels.get(row.source_id, ())
    event = next((name for name in EVENTS if name in labels), None)
    if event:
        return "events", event
    language = normalize_language(row.language)
    if language == "en":
        return "english", row.dataset
    if language in INDIC22:
        return "indic", language
    if language != "und":
        return "other", language
    return None


def plan_comparison(rows, counts, ledger, *, event_labels=None, reserved_rows=(),
                    excluded_sources=(), windows_count=16000, seed=31,
                    required_indic=INDIC22, required_other=("cmn", "yue", "ar", "es", "pt", "fr", "ja", "de")):
    if type(windows_count) is not int or windows_count < 1:
        raise ValueError("windows_count must be positive")
    rows = sorted(rows, key=lambda r: r.source_id)
    _counts(rows, counts)
    event_labels = event_labels or {}
    reserved_rows, excluded_sources = tuple(reserved_rows), tuple(excluded_sources)
    reserved = set().union(*(known_identities(r) for r in reserved_rows))
    excluded = set().union(*(known_identities(r, people=False) for r in excluded_sources))
    candidates = [r for r in rows if r.split == "train" and ledger.whole_untouched(r)
                  and not known_identities(r) & reserved
                  and not known_identities(r, people=False) & excluded]
    validate_manifest(candidates, training_only=True)
    grouped = defaultdict(list)
    omitted = defaultdict(int)
    for row in candidates:
        category = _primary(row, event_labels)
        if category is None:
            omitted["unknown_language_without_verified_event"] += 1
        else:
            grouped[category].append(row)
    queues = {}
    capacity = defaultdict(lambda: {"windows": 0, "input_samples": 0, "discarded_tail_samples": 0})
    for (bucket, subgroup), values in grouped.items():
        queue = deque()
        for row in _order(values, seed):
            scored = 0
            for w in windows_for(row, counts[row.source_id]):
                condition = (subgroup if bucket == "events" else
                             {"jnv": "jnv_nonverbal", "jvnv": "jvnv_verbal_and_nonverbal"}
                             .get(row.dataset, "speech"))
                queue.append(replace(w, condition=condition))
                scored += w.valid_input_samples16k
            capacity[bucket]["discarded_tail_samples"] += counts[row.source_id] - scored
        queues[(bucket, subgroup)] = queue
        capacity[bucket]["windows"] += len(queue)
        capacity[bucket]["input_samples"] += sum(w.valid_input_samples16k for w in queue)
    for group in required_indic:
        if not queues.get(("indic", group)):
            raise ValueError(f"Missing required Indic coverage: {group}")
    for group in required_other:
        if not queues.get(("other", group)):
            raise ValueError(f"Missing required other-language coverage: {group}")
    if sum(len(q) for q in queues.values()) < windows_count:
        raise SamplerExhausted("Unique eligible window capacity is below comparison budget")
    weights = dict(WEIGHTS)
    used = defaultdict(int)
    subgroup_used = defaultdict(int)
    selected = []
    drained = []
    # Select in valid-sample deficit order. A finite queue is never reset.
    while len(selected) < windows_count:
        available = {bucket for (bucket, subgroup), queue in queues.items() if queue}
        for bucket in sorted(set(weights) - available):
            remaining_budget = windows_count - len(selected)
            drained.append({"bucket": bucket, "selected_windows": len(selected),
                            "consumed_input_samples": used[bucket],
                            "remaining_window_budget": remaining_budget,
                            "reallocated_to": "english" if "english" in available else sorted(available)[0]})
            target = "english" if "english" in available else sorted(available)[0]
            weights[target] = weights.get(target, 0) + weights.pop(bucket)
        bucket = min(available, key=lambda key: (used[key] / weights[key], key))
        options = [key for key, queue in queues.items() if key[0] == bucket and queue]
        if bucket == "events":
            rare = [key for key in options if key[1] in RARE_EVENTS]
            if rare:
                options = rare
        subgroup = min(options, key=lambda key: (subgroup_used[key], key))
        w = queues[subgroup].popleft()
        selected.append(w)
        used[bucket] += w.valid_input_samples16k
        subgroup_used[subgroup] += w.valid_input_samples16k
    selected_ids = {w.source_id for w in selected}
    selected_rows = [r for r in candidates if r.source_id in selected_ids]
    selected_counts = {r.source_id: counts[r.source_id] for r in selected_rows}
    verify_comparison_windows(selected, {r.source_id: r for r in selected_rows}, selected_counts,
                              ledger, reserved_rows=reserved_rows, excluded_sources=excluded_sources)
    total = sum(used.values())
    shared_unknown = sorted({r.session_id for r in selected_rows if unknown_session(r)} &
                            {r.session_id for r in reserved_rows if unknown_session(r)})
    languages = sorted({w.language for w in selected if w.condition == "speech"})
    if not set(required_indic) <= set(languages) or not set(required_other) <= set(languages):
        raise ValueError("Budget failed to reach required language coverage")
    stats = {"windows": len(selected), "rows": len(selected_rows), "input_samples": total,
             "scored_hours": total / RATE / 3600, "scored_frames": 64,
             "minimum_output_samples": MINIMUM * 3, "context_frames": 29,
             "by_bucket": {key: {"input_samples": used[key], "fraction": used[key] / total}
                           for key in WEIGHTS},
             "by_subgroup_input_samples": {f"{b}:{s}": n for (b, s), n in sorted(subgroup_used.items())},
             "capacity": dict(capacity), "drained_capacity": drained,
             "omitted_sources": dict(omitted), "speech_languages": languages,
             "unknown_shared_session_placeholders": shared_unknown,
             "event_duration_scope": "Whole scored windows from labeled files, not dense per-sample event annotations",
             "other_language_scope": "Identified-language recordings may include JNV nonverbal or JVNV mixed material; their condition labels remain explicit",
             "source_audio_validation": "Metadata sample counts; file hash and decoded length verification required before teacher preparation",
             "split_limit": "Known source/hash/parent/speaker/session exclusions; no acoustic fingerprints or unknown-speaker guarantee"}
    return {"rows": selected_rows, "windows": selected, "counts": selected_counts,
            "ledger": ledger, "reserved": reserved_rows, "excluded": excluded_sources,
            "metadata": stats, "seed": seed}


def write_comparison_plan(plan, directory, *, provenance=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    files = {
        "train-manifest.jsonl": b"".join(canonical(r.to_dict()) for r in plan["rows"]),
        "windows.jsonl": b"".join(canonical(w.to_dict()) for w in plan["windows"]),
        "input-sample-counts.json": canonical(plan["counts"]),
        "ledger.json": canonical(plan["ledger"].document),
        "reserved.jsonl": b"".join(canonical(r.to_dict()) for r in plan["reserved"]),
        "excluded-sources.jsonl": b"".join(canonical(r.to_dict()) for r in plan["excluded"]),
        "metadata.json": canonical(plan["metadata"]),
    }
    for name, payload in files.items():
        (directory / name).write_bytes(payload)
    body = {"format_version": 1, "state": "ready", "seed": plan["seed"],
            "scope": "Metadata-complete immutable finite plan; audio revalidation remains mandatory",
            "fixed_sampler_identity": FixedWindowSampler(plan["windows"]).identity_sha256,
            "ledger_identity": plan["ledger"].identity_sha256,
            "provenance": provenance or {},
            "files_sha256": {name: file_sha(directory / name) for name in files}}
    ready = {**body, "identity_sha256": digest(body)}
    (directory / "ready.json").write_bytes(canonical(ready))
    return ready


def load_comparison_plan(directory):
    directory = Path(directory).resolve(strict=True)
    ready = json.loads((directory / "ready.json").read_text())
    body = {key: value for key, value in ready.items() if key != "identity_sha256"}
    required = {"train-manifest.jsonl", "windows.jsonl", "input-sample-counts.json", "ledger.json",
                "reserved.jsonl", "excluded-sources.jsonl", "metadata.json"}
    if (ready.get("state") != "ready" or ready.get("identity_sha256") != digest(body)
            or set(ready.get("files_sha256", {})) != required):
        raise ValueError("Comparison plan identity or complete file set differs")
    for name, expected in ready["files_sha256"].items():
        path = directory / name
        if Path(name).name != name or path.is_symlink() or file_sha(path) != expected:
            raise ValueError("Comparison plan metadata changed")
    rows = load_manifest(directory / "train-manifest.jsonl")
    counts = json.loads((directory / "input-sample-counts.json").read_text())
    _counts(rows, counts)
    windows = []
    for line in (directory / "windows.jsonl").read_text().splitlines():
        value = json.loads(line)
        window = PilotWindow(**{key: value[key] for key in PilotWindow.__dataclass_fields__})
        if value != window.to_dict():
            raise ValueError("Comparison window sample accounting differs")
        windows.append(window)
    reserved = load_manifest(directory / "reserved.jsonl")
    excluded = load_manifest(directory / "excluded-sources.jsonl")
    ledger = ConsumedLedger(json.loads((directory / "ledger.json").read_text()))
    verify_comparison_windows(windows, {r.source_id: r for r in rows}, counts, ledger,
                              reserved_rows=reserved, excluded_sources=excluded)
    if (FixedWindowSampler(windows).identity_sha256 != ready["fixed_sampler_identity"]
            or ledger.identity_sha256 != ready["ledger_identity"]):
        raise ValueError("Comparison window or lineage identity differs")
    return dict(rows=rows, windows=windows, counts=counts, ledger=ledger, reserved=reserved,
                excluded=excluded, metadata=json.loads((directory / "metadata.json").read_text()),
                identity=ready)
