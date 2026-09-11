"""Select immutable, source-disjoint cached pilot metadata without model/audio work."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

VERSION = "audiovae2-group-compression-data-v1"
SEED = "20260910-group-width-pilot-v1"
INDIC = ("as", "bn", "brx", "doi", "gu", "hi", "kn", "ks", "kok", "mai", "ml",
         "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur")
INTERNATIONAL = ("en", "es", "pt", "cmn", "ja", "fr", "ar")
LABELS = ("Breathing", "Chuckle_and_chortle", "Crying_and_sobbing", "Giggle", "Laughter",
          "Screaming", "Shout", "Whispering", "Yell", "explicit_whisper_style",
          "human_whistling_source_description")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def rank(domain, value):
    return hashlib.sha256((SEED + "|" + domain + "|" + value).encode()).hexdigest()


def language(row):
    return row["language"].split("_")[0]


def load_inputs(args):
    inventory = json.loads(args.inventory.read_text())
    cache = json.loads(args.cache_metadata.read_text())
    canonical = json.loads(args.canonical_metadata.read_text())
    votes = json.loads(args.fsd_votes.read_text())
    mid = {}
    with args.fsd_labels.open() as handle:
        for row in csv.DictReader(handle):
            for label, identity in zip(row["labels"].split(","), row["mids"].split(","), strict=True):
                if label in LABELS:
                    if label in mid and mid[label] != identity:
                        raise ValueError("Inconsistent FSD label identity")
                    mid[label] = identity
    whistle = {"freesound:" + str(row["freesound_id"]): row
               for row in json.loads(args.whistle_evidence.read_text())}
    return inventory, cache, canonical, votes, mid, whistle


def build(args):
    inventory, cache, canonical, votes, mid, whistle = load_inputs(args)
    data = inventory["data"]
    receipt = inventory["overlay_receipt"]
    if (receipt["contract"]["base_data_plan_sha256"] != inventory["data_sha256"]
            or cache["cache_sha256_expected"] != receipt["sha256"]
            or not cache["contract_matched_receipt"]):
        raise ValueError("Cache/data receipt mismatch")
    source_rows = data["rows"]
    canonical_rows = [r["manifest_row"] for r in canonical["canonical_inventory"]["sources"]
                      if r.get("manifest_row")]
    forbidden = {key: {r.get(key) for r in canonical_rows if r.get(key)}
                 for key in ("source_id", "audio_sha256", "parent_recording_id")}
    pool = data["pools"]["targeted_generator"]
    crops = cache["crop_metadata"]
    if len(pool) != 12800 or len(crops) != 12800:
        raise ValueError("Expected sealed 12,800-crop pool")
    candidates = defaultdict(list)
    excluded = set()
    for index, (planned, crop) in enumerate(zip(pool, crops, strict=True)):
        window = planned["window"]
        sid = crop["source_id"]
        row = source_rows[sid]
        if any(row.get(k) in forbidden[k] for k in forbidden):
            excluded.add(sid)
            continue
        if (crop["pool_index"] != index or sid != window["source_id"]
                or crop["start_frame"] != window["start_frame"]
                or crop["valid_scored_samples"] != window["valid_output_samples48k"]):
            raise ValueError("Crop metadata differs from original plan")
        context, scored = crop["context_frames"], crop["scored_frames"]
        n = context + scored
        expected = {"latents": [1, 64, n], "teacher_audio": [1, 1, n * 1920],
                    "reference16k": [1, 1, n * 640]}
        if crop["tensor_shapes"] != expected or set(crop["tensor_dtypes"].values()) != {"torch.float32"}:
            raise ValueError("Unexpected cached tensor geometry or precision")
        if (crop["context_start_frame"] + context != crop["start_frame"]
                or not (crop["start_frame"] == 0 and context == 0 or context >= 20)
                or crop["start_frame"] * 1920 + crop["valid_scored_samples"] > data["counts"][sid] * 3):
            raise ValueError("Insufficient teacher history or invalid source extent")
        labels, evidence = [], {}
        for label in planned["source_labels"]:
            if label not in LABELS:
                continue
            if label == "explicit_whisper_style":
                valid = sid.startswith("thorsten_emotional:whisper/")
                evidence[label] = {"source_id_style_verified": valid}
            elif label == "human_whistling_source_description":
                entry = whistle.get(sid, {})
                valid = entry.get("state") == "metadata_verified"
                evidence[label] = {"source_page_sha256": entry.get("source_page_sha256"),
                                   "source_url": entry.get("source_url")}
            else:
                observed = votes.get(sid.removeprefix("freesound:"), {}).get(mid[label], [])
                valid = observed.count(1.0) >= 2 and not any(v < 0 for v in observed)
                evidence[label] = {"fsd_votes": observed, "class_id": mid[label]}
            if valid:
                labels.append(label)
        candidate = {**crop, "audio_sha256": row["audio_sha256"],
                     "parent_recording_id": row["parent_recording_id"],
                     "dataset": row["dataset"], "language": row["language"],
                     "normalized_language": language(row), "manifest_row": row,
                     "verified_source_labels": labels, "label_evidence": evidence,
                     "selection_kind": planned["selection_kind"],
                     "input_energy_qualification": planned["qualification"],
                     "has_semantic_event_timestamps": False,
                     "full_prepared_source_scored": crop["start_frame"] == 0 and
                     crop["valid_scored_samples"] == data["counts"][sid] * 3,
                     "score_start_sample48k": context * 1920,
                     "score_stop_sample48k": context * 1920 + crop["valid_scored_samples"],
                     "legacy_six_sample_omission_applied": False}
        candidates[sid].append(candidate)
    used = {key: set() for key in forbidden}
    splits = {}

    def available(c):
        return not any(c[key] in used[key] for key in used)

    def choose(name, reason, predicate, count=1, required=True):
        taken = []
        for sid, options in candidates.items():
            if not available(options[0]):
                continue
            eligible = [c for c in options if predicate(c)]
            if not eligible:
                continue
            # Use the whole prepared recording for source-level expressive tags
            # when it fits; otherwise prefer full-length existing scored crops.
            best = min(eligible, key=lambda c: (
                not (bool(c["verified_source_labels"]) and c["full_prepared_source_scored"]),
                -c["valid_scored_samples"],
                c["selection_kind"] != "explicit_event" if c["verified_source_labels"] else False,
                rank(name + "/crop", sid + "/" + str(c["pool_index"]))))
            taken.append(best)
        taken.sort(key=lambda c: (not c["full_prepared_source_scored"] if reason.startswith("expressive:") else False,
                                  rank(name + "/" + reason, c["source_id"]), c["source_id"]))
        if required and len(taken) < count:
            raise ValueError(f"Insufficient distinct sources for {name}/{reason}: {len(taken)} < {count}")
        for c in taken[:count]:
            item = dict(c, selection_reason=reason)
            splits[name].append(item)
            for key in used:
                used[key].add(c[key])
        return min(count, len(taken))

    for name, size in (("calibration", args.calibration), ("development", args.development),
                       ("fit", args.fit)):
        splits[name] = []
        for lang in INDIC + INTERNATIONAL:
            choose(name, "required_language:" + lang,
                   lambda c, lang=lang: c["normalized_language"] == lang)
        for label in LABELS:
            choose(name, "expressive:" + label,
                   lambda c, label=label: label in c["verified_source_labels"])
        for kind in ("quiet", "transition"):
            choose(name, "input_energy:" + kind,
                   lambda c, kind=kind: c["selection_kind"] == kind,
                   count=64 if name == "fit" else 8)
        # Broaden language coverage before deterministic identity-only filling.
        for lang in sorted({language(v) for v in source_rows.values()}):
            if len(splits[name]) >= size:
                break
            if not any(c["normalized_language"] == lang for c in splits[name]):
                choose(name, "additional_language:" + lang,
                       lambda c, lang=lang: c["normalized_language"] == lang, required=False)
        if len(splits[name]) > size:
            raise ValueError("Requested split too small for fixed coverage quotas")
        choose(name, "deterministic_source_fill", lambda c: True, size - len(splits[name]))
        splits[name].sort(key=lambda c: (rank(name + "/order", c["source_id"]), c["source_id"]))
    all_rows = [c for split in splits.values() for c in split]
    for key in used:
        if len({c[key] for c in all_rows}) != len(all_rows):
            raise ValueError("Cross-split repeated identity: " + key)
    summaries = {}
    for name, rows in splits.items():
        total = sum(c["valid_scored_samples"] for c in rows)
        expressive = sum(c["valid_scored_samples"] for c in rows if c["verified_source_labels"])
        summaries[name] = {"sources": len(rows), "scored_seconds": total / 48000,
            "full_source_seconds": sum(c["manifest_row"]["duration_seconds"] for c in rows),
            "normalized_language_counts": dict(sorted(Counter(c["normalized_language"] for c in rows).items())),
            "dataset_counts": dict(sorted(Counter(c["dataset"] for c in rows).items())),
            "verified_source_label_counts": {lab: sum(lab in c["verified_source_labels"] for c in rows) for lab in LABELS},
            "source_labeled_expressive_scored_percent": 100 * expressive / total,
            "input_qualified_selection_counts": dict(Counter(c["selection_kind"] for c in rows)),
            "all_22_indic_present": all(any(c["normalized_language"] == lang for c in rows) for lang in INDIC),
            "all_requested_international_present": all(any(c["normalized_language"] == lang for c in rows) for lang in INTERNATIONAL),
            "whole_prepared_source_crops": sum(c["full_prepared_source_scored"] for c in rows),
            "context_frames_counts": dict(Counter(c["context_frames"] for c in rows)),
            "minimum_nonstartup_context_frames": min(c["context_frames"] for c in rows if c["start_frame"]),
            "speaker_disjointness_established": False}
    if summaries["fit"]["source_labeled_expressive_scored_percent"] < 5:
        raise ValueError("Fitting panel has less than 5% source-labeled expressive duration")
    result = {"format_version": VERSION, "seed": SEED,
        "metadata_only": True, "source_audio_or_models_executed": False,
        "data_path": inventory["data_path"], "data_sha256": inventory["data_sha256"],
        "overlay_receipt_path": inventory["overlay_receipt_path"],
        "overlay_receipt_sha256": inventory["overlay_receipt_sha256"],
        "cache_path": cache["cache_path"], "cache_sha256": receipt["sha256"],
        "cache_contract_sha256": receipt["contract"]["identity_sha256"],
        "pool": "targeted_generator", "available_unique_sources": len(candidates),
        "canonical_excluded_sources": sorted(excluded),
        "source_hash_parent_disjoint": True, "canonical_source_hash_parent_disjoint": True,
        "updates": args.fit // args.batch_size, "batch_size": args.batch_size,
        "fitting_sources_repeated": False, "calibration_fit_development_disjoint": True,
        "inputs": {str(p): sha(p) for p in (args.inventory, args.cache_metadata,
            args.canonical_metadata, args.fsd_votes, args.fsd_labels, args.whistle_evidence)},
        "generator_sha256": sha(__file__), "summaries": summaries,
        "splits": {name: {"indices": [c["pool_index"] for c in rows], "rows": rows} for name, rows in splits.items()},
        "pending_before_model_use": ["Verify the full 11.4 GB cache byte hash against its receipt on Runpod.",
            "Authenticate source PCM and same full-source teacher/group targets under the preserved backend.",
            "Check complete copied decoder numerical/streaming parity with all declared valid samples scored.",
            "This panel is for debugging; reserve a fresh source-disjoint final panel later."],
        "limitations": ["Expressive labels are recording-level metadata, not timestamped semantic event occupancy.",
            "Quiet/transition flags are previously measured input-energy qualification, not new teacher-output qualification.",
            "Speaker/session disjointness is not claimed; actual shared speakers and unknown IDs are allowed only in this diagnostic panel.",
            "Calibration/development sources are repeatedly evaluated; fitting sources occur once in the 1,000-update schedule.",
            "The cache is volatile shared memory; durable source/generation manifests must remain available."]}
    result["identity_sha256"] = digest(result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--cache-metadata", type=Path, required=True)
    p.add_argument("--canonical-metadata", type=Path, required=True)
    p.add_argument("--fsd-votes", type=Path, required=True)
    p.add_argument("--fsd-labels", type=Path, required=True)
    p.add_argument("--whistle-evidence", type=Path, required=True)
    p.add_argument("--fit", type=int, default=3000)
    p.add_argument("--calibration", type=int, default=72)
    p.add_argument("--development", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.fit % args.batch_size:
        raise ValueError("No dropped partial fitting batch permitted")
    if args.output.exists():
        raise FileExistsError("Refusing to replace a sealed selection")
    result = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"identity_sha256": result["identity_sha256"], "updates": result["updates"],
                      "summaries": result["summaries"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
