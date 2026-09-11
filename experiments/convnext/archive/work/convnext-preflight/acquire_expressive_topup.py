"""Bounded supplemental FSD whisper/breath data, entirely on the existing pod."""
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys

BASE = Path("/workspace/fast-audiovae-convnext-20260908-r1")
ROOT = Path("/workspace/fast-audiovae-convnext-20260909-r5/data-expressive-topup-v1")
sys.path.insert(0, str(BASE))
from audiovae_student import acquire_fsd_vocal as fsd
from audiovae_student.acquire import _json_bytes
from audiovae_student.data import load_manifest

PLAN = Path("/workspace/fast-audiovae-convnext-20260909-r2/data/selection-v1")
PANEL = Path("/workspace/fast-audiovae-convnext-20260909-r4/data/dev-panel-v1/manifest.jsonl")
FULL = BASE / "expanded-pilot/corpus/source-manifest.jsonl"
RESERVED_WHISPER = {"freesound:uploader:bigfriendlyjiant", "freesound:uploader:carmsie"}


def main():
    if ROOT.exists():
        raise ValueError("Supplement root already exists; inspect instead of silently repeating acquisition")
    fsd.TARGET_LABELS = fsd.TARGET_LABELS | {"Whispering", "Breathing"}
    full_selection = fsd.read_selection(BASE / "assets/fsd50k-metadata")
    original_selection = json.loads((BASE / "data-fsd-vocal/prepared/provenance/selection.json").read_text())
    old_ids = {x["id"] for x in original_selection["selected"]}
    prior_manifests = [BASE / stem / "prepared" / (split + ".jsonl")
                       for stem in ("data-fsd-vocal", "data-human-whistling") for split in ("train", "dev")]
    exclusions = fsd.Exclusions(reserved=prior_manifests)
    ledger = json.loads((PLAN / "ledger.json").read_text())
    for row in (entry["row"] for entry in ledger["sources"]):
        exclusions.source_ids.add(row["source_id"])
        exclusions.parents.add(row["parent_recording_id"])
        exclusions.hashes.add(row["audio_sha256"])
        if row.get("session_id"):
            exclusions.sessions.add(row["session_id"])
    for path in (PANEL, PLAN / "train-manifest.jsonl"):
        for row in load_manifest(path):
            exclusions.source_ids.add(row.source_id)
            exclusions.parents.add(row.parent_recording_id)
            exclusions.hashes.add(row.audio_sha256)
            if row.session_id:
                exclusions.sessions.add(row.session_id)
    # Entire already prepared source ID/hash inventory is excluded without
    # reading every audio file or per-utterance trace during the pilot.
    with FULL.open() as handle:
        for line in handle:
            row = json.loads(line)
            exclusions.source_ids.add(row["source_id"])
            exclusions.parents.add(row["parent_recording_id"])
            exclusions.hashes.add(row["audio_sha256"])
    available = [x for x in full_selection["selected"] if x["id"] not in old_ids and
                 set(x["target_labels"]) & {"Whispering", "Breathing"} and not exclusions.metadata_matches(x)]
    whisper_dev = []
    for session in sorted(RESERVED_WHISPER):
        candidates = [x for x in available if x["session_id"] == session and "Whispering" in x["target_labels"]]
        if len(candidates) < 2:
            raise ValueError("Both fresh whisper contributors must supply at least two candidate recordings")
        whisper_dev.extend(sorted(candidates, key=lambda x: int(x["id"]))[:2])
    natural_dev = [x for x in available if x["split"] == "dev" and x["session_id"] not in RESERVED_WHISPER]
    if len(natural_dev) > 12:
        raise ValueError("Unexpected development candidate growth requires a new bounded selection")
    training = []
    for event in ("Whispering", "Breathing"):
        groups = defaultdict(list)
        for item in available:
            if item["split"] == "train" and item["session_id"] not in RESERVED_WHISPER and event in item["target_labels"]:
                groups[item["session_id"]].append(item)
        queues = [deque(sorted(groups[key], key=lambda x: int(x["id"]))) for key in sorted(groups)]
        chosen = []
        while queues and len(chosen) < 12:
            for queue in queues:
                if len(chosen) == 12:
                    break
                chosen.append(queue.popleft())
            queues = [queue for queue in queues if queue]
        training.extend(chosen)
    selected = [dict(x) for x in natural_dev + whisper_dev + training]
    if len(selected) > 36 or len(training) > 24 or len({x["id"] for x in selected}) != len(selected):
        raise ValueError("Selected clip count exceeds authorization or duplicates an event target")
    # This explicitly documented supplemental assignment supersedes only the
    # old hash split for two never-trained contributors. It never changes the
    # official FSD source partition, old manifests or the active pilot panel.
    original_split = fsd.uploader_split
    def supplemental_split(username):
        return "dev" if "freesound:uploader:" + fsd.canonical_uploader(username) in RESERVED_WHISPER else original_split(username)
    fsd.uploader_split = supplemental_split
    for item in selected:
        item["split"] = supplemental_split(item["uploader"])
    policy = dict(full_selection["selection_policy"])
    policy.update({"target_labels": ["Whispering", "Breathing"],
        "uploader_split": "Original uploader hash split, except complete never-trained bigfriendlyjiant and carmsie contributor groups reserved for supplemental development before any new training selection",
        "reserved_supplemental_dev_sessions": sorted(RESERVED_WHISPER),
        "training_selection": "Up to twelve recordings per target; deterministic contributor round robin",
        "purpose": "Separate future-stage supplement. Not inserted into active pilot or immutable development panel.",
        "source_split_preservation": "All source files remain official FSD50K train; internal heldout records say train-uploader-heldout-dev"})
    selection = {**full_selection, "selected": selected, "selection_policy": policy,
        "selected_clip_count": len(selected), "counts_by_split": dict(Counter(x["split"] for x in selected)),
        "counts_by_target": dict(Counter(label for x in selected for label in x["target_labels"])),
        "counts_by_split_and_target": {s: dict(Counter(label for x in selected if x["split"] == s for label in x["target_labels"])) for s in ("train", "dev")},
        "uploader_groups_by_split": {s: len({x["session_id"] for x in selected if x["split"] == s}) for s in ("train", "dev")}}
    selection_hash = hashlib.sha256(_json_bytes(selection)).hexdigest()
    ROOT.mkdir(parents=True)
    audit = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "metadata_only_available_after_source_and_uploader_exclusions": {
            s: {label: sum(x["split"] == s and label in x["target_labels"] for x in available)
                for label in ("Whispering", "Breathing")} for s in ("train", "dev")},
        "selection_sha256": selection_hash, "selected": [{k:x[k] for k in ("id", "split", "target_labels", "license", "session_id")} for x in selected],
        "reserved_whole_contributor_candidate_ids": {session: [x["id"] for x in available if x["session_id"] == session] for session in sorted(RESERVED_WHISPER)},
        "exclusion_input_sha256": {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (PLAN/"ledger.json", PANEL, PLAN/"train-manifest.jsonl", FULL)},
        "download_policy": "At most 36 individually hash-verified pinned WAVs, at most four concurrent requests, no complete archives",
        "cpu_only": True, "audio_downloaded_to_local_mac": False}
    (ROOT / "selection-audit.json").write_bytes(_json_bytes(audit))
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {x["id"]: pool.submit(fsd.fetch_pinned_clip, x["id"]) for x in selected}
        result = fsd.acquire(selection, ROOT, expected_selection_sha256=selection_hash, exclusions=exclusions,
            max_bytes=256 * 1024**2, reserve_bytes=4 * 1024**3,
            fetch=lambda identifier: futures[identifier].result())
    summary = {key: result[key] for key in ("state", "rows", "hours", "hours_by", "counts_by_target", "quarantined", "source_revision")}
    summary["selection_audit"] = audit
    (ROOT / "result-summary.json").write_bytes(_json_bytes(summary))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
