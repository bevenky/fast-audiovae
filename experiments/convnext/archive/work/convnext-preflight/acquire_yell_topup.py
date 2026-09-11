"""Add three separately authorized Yell clips without changing supplement v1."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

BASE = Path("/workspace/fast-audiovae-convnext-20260908-r1")
ROOT = Path("/workspace/fast-audiovae-convnext-20260909-r5/data-expressive-topup-v1")
ADDITION = ROOT / "version2-yell"
sys.path.insert(0, str(BASE))
from audiovae_student import acquire_fsd_vocal as fsd
from audiovae_student.acquire import _json_bytes
from audiovae_student.data import load_manifest, validate_manifest


def main():
    if ADDITION.exists():
        raise ValueError("Yell addition already exists; inspect before another acquisition")
    fsd.TARGET_LABELS = fsd.TARGET_LABELS | {"Yell"}
    full = fsd.read_selection(BASE / "assets/fsd50k-metadata")
    selected = [dict(x) for x in full["selected"] if x["id"] in {"17643", "47829", "266716"}]
    if len(selected) != 3 or any(x["target_labels"] != ["Yell"] for x in selected):
        raise ValueError("Expected three metadata-qualified Yell sources")
    current = [BASE / stem / "prepared" / (split + ".jsonl")
               for stem in ("data-fsd-vocal", "data-human-whistling") for split in ("train", "dev")]
    current.extend(ROOT / "prepared" / (split + ".jsonl") for split in ("train", "dev"))
    current.append(Path("/workspace/fast-audiovae-convnext-20260909-r4/data/dev-panel-v1/manifest.jsonl"))
    exclusions = fsd.Exclusions(reserved=current)
    ledger = json.loads(Path("/workspace/fast-audiovae-convnext-20260909-r2/data/selection-v1/ledger.json").read_text())
    for row in (x["row"] for x in ledger["sources"]):
        exclusions.source_ids.add(row["source_id"])
        exclusions.parents.add(row["parent_recording_id"])
        exclusions.hashes.add(row["audio_sha256"])
        if row.get("session_id"):
            exclusions.sessions.add(row["session_id"])
    for line in Path("/workspace/fast-audiovae-convnext-20260909-r2/data/selection-v1/train-manifest.jsonl").read_text().splitlines():
        row = json.loads(line)
        exclusions.source_ids.add(row["source_id"])
        exclusions.parents.add(row["parent_recording_id"])
        exclusions.hashes.add(row["audio_sha256"])
        if row.get("session_id"):
            exclusions.sessions.add(row["session_id"])
    with (BASE / "expanded-pilot/corpus/source-manifest.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            exclusions.source_ids.add(row["source_id"])
            exclusions.parents.add(row["parent_recording_id"])
            exclusions.hashes.add(row["audio_sha256"])
    if any(exclusions.metadata_matches(x) for x in selected):
        raise ValueError("New Yell identity overlaps existing data or a reserved contributor")
    reserved = min((x["session_id"] for x in selected),
                   key=lambda value: hashlib.sha256(("expressive-yell-v2:" + value).encode()).hexdigest())
    original_split = fsd.uploader_split
    def split(username):
        return "dev" if "freesound:uploader:" + fsd.canonical_uploader(username) == reserved else original_split(username)
    fsd.uploader_split = split
    for item in selected:
        item["split"] = split(item["uploader"])
    policy = {**full["selection_policy"], "target_labels": ["Yell"],
        "uploader_split": "Reserve the minimum sha256(expressive-yell-v2: + canonical session) contributor for supplemental development before any training use; retain original hash split for others",
        "reserved_supplemental_dev_session": reserved, "purpose": "Supplement version2 only; keep Yell distinct from Shout and leave active training/panel unchanged"}
    selection = {**full, "selected": selected, "selected_clip_count": 3, "selection_policy": policy,
        "counts_by_split": dict(Counter(x["split"] for x in selected)), "counts_by_target": {"Yell": 3},
        "counts_by_split_and_target": {s: {"Yell": sum(x["split"] == s for x in selected)} for s in ("train", "dev")},
        "uploader_groups_by_split": {s: len({x["session_id"] for x in selected if x["split"] == s}) for s in ("train", "dev")}}
    digest = hashlib.sha256(_json_bytes(selection)).hexdigest()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {x["id"]: pool.submit(fsd.fetch_pinned_clip, x["id"]) for x in selected}
        result = fsd.acquire(selection, ADDITION, expected_selection_sha256=digest, exclusions=exclusions,
            max_bytes=128 * 1024**2, reserve_bytes=4 * 1024**3, fetch=lambda key: futures[key].result())
    combined = [row for parent in (ROOT, ADDITION) for s in ("train", "dev")
                for row in load_manifest(parent / "prepared" / (s + ".jsonl"))]
    validate_manifest(combined)
    version = ROOT / "versions/v2"
    version.mkdir(parents=True)
    for s in ("train", "dev"):
        (version / (s + ".jsonl")).write_bytes(b"".join(_json_bytes(row.to_dict()) for row in combined if row.split == s))
    report = {"state": "prepared_validated_not_trained", "version": 2, "clips": len(combined),
        "clips_by_split": dict(Counter(row.split for row in combined)),
        "seconds_by_split": {s: sum(row.duration_seconds for row in combined if row.split == s) for s in ("train", "dev")},
        "contributors_by_split": {s: len({row.session_id for row in combined if row.split == s}) for s in ("train", "dev")},
        "new_yell_records": [{key:x[key] for key in ("id", "split", "session_id", "license", "target_labels")} for x in selected],
        "reserved_yell_contributor": reserved, "new_yell_quarantined": result["quarantined"],
        "original_v1_report_sha256": hashlib.sha256((ROOT / "prepared-audit.json").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "files_sha256": {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in version.iterdir()},
        "active_training_or_panel_modified": False, "all_audio_on_runpod": True,
        "label_policy": "Yell is independently labeled and not silently merged into Shout"}
    (version / "ready.json").write_bytes(_json_bytes(report))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
