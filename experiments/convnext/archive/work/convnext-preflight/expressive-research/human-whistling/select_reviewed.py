from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
REJECT = {474356: "Postprocessed unusual bird effect", 474357: "Postprocessed realistic bird effect",
          122272: "Entirely synthetic", 122268: "Entirely synthetic", 122269: "Entirely synthetic",
          74388: "Explicit 60 Hz background hum"}
rows = json.loads((ROOT / "source-evidence.json").read_text())
selected = []
for row in rows:
    if row["freesound_id"] in REJECT:
        continue
    uploader = unquote(urlparse(row["resolved_source_url"]).path.split("/")[2])
    uploader_hash = sha256(("freesound-uploader-split-v1:" + uploader.strip().casefold()).encode()).hexdigest()
    split = "dev" if int(uploader_hash[:8], 16) % 20 == 0 else "train"
    identifier, license = str(row["freesound_id"]), row["verified_license"]
    dataset = {"CC0-1.0": "freesound_human_whistle_cc0", "CC-BY-3.0": "freesound_human_whistle_cc_by_3",
               "CC-BY-4.0": "freesound_human_whistle_cc_by_4"}[license]
    evidence = row["source_evidence"]
    selected.append({"id": identifier, "source_id": "freesound:" + identifier,
        "parent_recording_id": "freesound:" + identifier, "speaker_id": None,
        "session_id": "freesound:uploader:" + uploader.strip().casefold(), "uploader": uploader,
        "split": split, "dataset": dataset, "license": license, "license_url": evidence["license_url"],
        "source_url": row["resolved_source_url"], "title": evidence["title"],
        "duration_seconds": evidence["duration_seconds"], "sample_rate_hz": evidence["sample_rate_hz"],
        "source_description": evidence["description"], "source_page_path": row["source_page_path"],
        "source_page_sha256": row["source_page_sha256"], "source_page_verified_at": row["verified_at_utc"],
        "mirror_metadata": {key: row[key] for key in ("title", "description", "tags", "username", "freesound_id",
            "license", "attribution_required", "commercial_use", "shard", "row", "row_group", "row_within_group")}})
fsd_path = ROOT.parents[1] / "fsd-vocal-review/prepared/provenance/selection.json"
fsd = json.loads(fsd_path.read_bytes())
forbidden = sorted({str(row["id"]) for row in fsd["selected"]} | set(fsd["forbidden_official_clip_ids"]), key=int)
assert not {row["id"] for row in selected} & set(forbidden)
value = {"format_version": 1, "preparation_version": 1, "repo": "MoamenElSayed/freesound-commercial-50k",
    "revision": "ac5aed8cf1aeb97b26a375f24794d42e15f97163", "selected": selected,
    "excluded_after_original_page_review": {str(key): val for key, val in REJECT.items()},
    "fsd_selection_sha256": sha256(fsd_path.read_bytes()).hexdigest(), "forbidden_fsd_clip_ids": forbidden,
    "selection_policy": {
        "scope": "Human lip/finger whistling inferred from creator descriptions and titles; not listening-confirmed purity",
        "exclude": "No synthetic whistles, instruments, actual birds, crowd/music mixes, strong processing, documented hum, FSD selected or official held-out clip IDs",
        "uploader_split": "sha256(freesound-uploader-split-v1: + normalized source uploader) first8hex modulo20 equals0 -> dev",
        "identity_limit": "Uploader is not an identified physical speaker; separate files can repeat gestures; no fuzzy acoustic deduplication claim",
        "channel_policy": "Mono unchanged; stereo fixed channel0 without mixing or gain normalization; original channels retained",
        "recordings": "Retain encoded bytes embedded in the pinned mirror; original Freesound file-byte equality is unverified"}}
payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
(ROOT / "selection.json").write_bytes(payload)
summary = {"clips": len(selected), "uploaders": len({row["uploader"] for row in selected}),
           "seconds": sum(row["duration_seconds"] for row in selected),
           "licenses": dict(Counter(row["license"] for row in selected)),
           "splits": dict(Counter(row["split"] for row in selected)), "selection_sha256": sha256(payload).hexdigest()}
(ROOT / "shortlist-summary.json").write_text(json.dumps(summary, indent=2))
lines = ["Human-whistling shortlist", "", json.dumps(summary, indent=2), "",
         "| ID | Uploader | Seconds | License | Split | Title |", "| --- | --- | ---: | --- | --- | --- |"]
for row in selected:
    lines.append(f"| [{row['id']}]({row['source_url']}) | {row['uploader']} | {row['duration_seconds']:.2f} | {row['license']} | {row['split']} | {row['title']} |")
(ROOT / "shortlist.md").write_text("\n".join(lines) + "\n")
print(json.dumps(summary, indent=2))
