"""Acquire a pinned, metadata-selected human vocal subset of FSD50K.

Only official training clips with reviewed per-file licenses and positive human
annotations enter this subset. Original WAV bytes remain available. Labels are
not a guarantee of background-free or single-speaker recordings: FSD50K labels
are weak and may be incomplete. No model or accelerator is used here.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import signal
import time

import soundfile as sf

from .acquire import _atomic_bytes, _json_bytes, _open_url, load_evaluation_exclusions
from .acquire_expressive import GAIN_POLICY, StorageBudget, deduplicate_prepared, prepare_audio
from .data import DATASET_LICENSES, ManifestRow, load_manifest, summarize_manifest, validate_manifest


REPO = "Fhrozen/FSD50k"
REVISION = "ccf1acaff12f3f4a4c10052dddafbb6d22152c9b"
OFFICIAL_SOURCE = "https://zenodo.org/records/4060432"
SPLIT_PREFIX = "freesound-uploader-split-v1:"
MAX_CLIP_BYTES = 16 * 1024**2
TARGET_LABELS = frozenset({"Laughter", "Giggle", "Chuckle_and_chortle", "Shout", "Screaming", "Crying_and_sobbing"})
# Permit only human vocal co-labels, so a target plus music, instruments,
# animal, crowd, movement or other background-event labels is excluded.
ALLOWED_LABELS = TARGET_LABELS | {
    "Yell", "Human_voice", "Speech", "Male_speech_and_man_speaking",
    "Female_speech_and_woman_speaking", "Child_speech_and_kid_speaking",
    "Whispering", "Conversation", "Chatter", "Breathing", "Respiratory_sounds",
    "Gasp", "Sigh", "Cough", "Sneeze", "Burping_and_eructation",
}
LICENSES = {
    "creativecommons.org/publicdomain/zero/1.0": ("fsd50k_vocal_cc0", "CC0-1.0"),
    "creativecommons.org/licenses/by/3.0": ("fsd50k_vocal_cc_by_3", "CC-BY-3.0"),
}
METADATA_SHA256 = {
    "dev.csv": "ce4ab5f01baa5a52b0ec892edaf137a7560743fce3a70155f8bf975c197ded1b",
    "eval.csv": "ef861a4619691783bfceb7f43e1a395dcf969150ba4ad34dcb36b93abd26b3d6",
    "dev_clips_info_FSD50K.json": "b31a2eb130aea6d545d4b48511b24dfd0120420567a8d1066c52d54968bb3735",
    "pp_pnp_ratings_FSD50K.json": "c6ad677deac1d68ebbab48d51d2ccf511204ddd0111bd5c4820051180ba2d35b",
    "vocabulary.csv": "992b6e11422101ee575ab3022d37b79b98e19407b211331c72f5d9fdd16e7174",
    "LICENSE-DATASET.md": "ebee1f320405cc09abfcc3c77e14fbc3adb9734d53f2ef5274d35eddcbb476f1",
    "README.md": "6af09867075823ad3ea74467234b23209bb9db9a2e38626377fce2f6a1327273",
}


def canonical_uploader(username: str) -> str:
    if not isinstance(username, str) or not username.strip() or len(username) > 512:
        raise ValueError("Missing or invalid Freesound uploader identity")
    return username.strip().casefold()


def uploader_split(username: str) -> str:
    """Shared with other Freesound subsets; uploader is not a speaker claim."""
    digest = hashlib.sha256((SPLIT_PREFIX + canonical_uploader(username)).encode()).hexdigest()
    return "dev" if int(digest[:8], 16) % 20 == 0 else "train"


def clip_identity(identifier: str, username: str) -> dict:
    if not isinstance(identifier, str) or not re.fullmatch(r"[1-9][0-9]*", identifier):
        raise ValueError("Freesound IDs must be canonical positive integers")
    return {"source_id": f"freesound:{identifier}", "parent_recording_id": f"freesound:{identifier}",
            "speaker_id": None, "session_id": "freesound:uploader:" + canonical_uploader(username),
            "split": uploader_split(username)}


def reviewed_license(url: str) -> tuple[str, str] | None:
    if not isinstance(url, str) or not re.fullmatch(r"https?://[^?#]+", url):
        return None
    return LICENSES.get(re.sub(r"^https?://", "", url).rstrip("/"))


def select_candidates(catalog: list[dict], evaluation: list[dict], info: dict,
                      ratings: dict, vocabulary: dict) -> dict:
    """Pure metadata selection, separately testable without network or audio."""
    if TARGET_LABELS - vocabulary.keys():
        raise ValueError("Missing target labels in pinned FSD vocabulary")
    seen, excluded, selected = set(), Counter(), []
    forbidden = {str(row["fname"]) for row in evaluation}
    forbidden.update(str(row["fname"]) for row in catalog if row["split"] != "train")
    for row in sorted(catalog, key=lambda value: int(value["fname"])):
        identifier = str(row["fname"])
        if identifier in seen:
            raise ValueError("Duplicate FSD catalog clip ID")
        seen.add(identifier)
        labels, mids = row["labels"].split(","), row["mids"].split(",")
        if len(labels) != len(mids) or any(vocabulary.get(label) != mid for label, mid in zip(labels, mids)):
            raise ValueError("FSD label/MID mapping disagrees with the vocabulary")
        if not set(labels) & TARGET_LABELS:
            excluded["no_target_label"] += 1
            continue
        if identifier in forbidden or row["split"] != "train":
            excluded["official_heldout_clip"] += 1
            continue
        clip_info = info[identifier]
        license_pair = reviewed_license(clip_info.get("license"))
        if license_pair is None:
            excluded["unapproved_audio_license"] += 1
            continue
        if set(labels) - ALLOWED_LABELS:
            excluded["non_vocal_or_background_colabel"] += 1
            continue
        targets = []
        for label in sorted(set(labels) & TARGET_LABELS):
            votes = ratings.get(identifier, {}).get(vocabulary[label], [])
            if any(type(vote) not in {int, float} or vote not in {-1, 0, 0.5, 1} for vote in votes):
                raise ValueError("Unknown FSD rating value")
            # Official release legend: PP=1, PNP=.5, U=0, NP=-1. Require
            # two present-and-predominant votes and no negative presence vote.
            if votes.count(1) >= 2 and -1 not in votes:
                targets.append(label)
        if not targets:
            excluded["no_two_PP_votes_without_NP"] += 1
            continue
        ids = clip_identity(identifier, clip_info["uploader"])
        selected.append({"id": identifier, **ids, "labels": sorted(labels), "target_labels": targets,
                         "ratings": ratings[identifier], "uploader": clip_info["uploader"],
                         "license_url": clip_info["license"], "dataset": license_pair[0],
                         "license": license_pair[1], "title": clip_info.get("title", ""),
                         "clip_info_sha256": hashlib.sha256(_json_bytes(clip_info)).hexdigest()})
    counts, per_split = Counter(), {"train": Counter(), "dev": Counter()}
    for item in selected:
        counts.update(item["target_labels"])
        per_split[item["split"]].update(item["target_labels"])
    return {"format_version": 1, "selection_version": 1, "repo": REPO, "revision": REVISION,
            "official_source": OFFICIAL_SOURCE, "collection_license": "CC-BY-4.0",
            "metadata_sha256": METADATA_SHA256, "selected": selected,
            "selected_clip_count": len(selected), "counts_by_target": dict(counts),
            "counts_by_split": dict(Counter(item["split"] for item in selected)),
            "counts_by_split_and_target": {key: dict(value) for key, value in per_split.items()},
            "uploader_groups_by_split": {split: len({item["session_id"] for item in selected if item["split"] == split})
                                         for split in ("train", "dev")},
            "excluded_counts": dict(excluded), "forbidden_official_clip_ids": sorted(forbidden, key=int),
            "selection_policy": {"target_labels": sorted(TARGET_LABELS), "allowed_colabels": sorted(ALLOWED_LABELS),
                "source_split": "official train only; all official val/eval clip IDs excluded",
                "votes": "at least two PP=1 votes and zero NP=-1 votes for a requested target label",
                "uploader_split": "sha256('freesound-uploader-split-v1:' + username.strip().casefold()) first8hex modulo20 equals0 -> dev",
                "identity_limit": "Uploader is a conservative source group, not an identified physical speaker",
                "quality_limit": "Weak labels may be incomplete; no claim of background-free or single-speaker recordings"}}


def read_selection(metadata_dir: Path) -> dict:
    raw = {}
    for name, digest in METADATA_SHA256.items():
        raw[name] = (metadata_dir / name).read_bytes()
        if hashlib.sha256(raw[name]).hexdigest() != digest:
            raise ValueError(f"Pinned FSD metadata SHA-256 mismatch: {name}")
    rows = lambda name: list(csv.DictReader(io.StringIO(raw[name].decode())))
    vocabulary = {label: mid for _, label, mid in csv.reader(io.StringIO(raw["vocabulary.csv"].decode()))}
    return select_candidates(rows("dev.csv"), rows("eval.csv"), json.loads(raw["dev_clips_info_FSD50K.json"]),
                             json.loads(raw["pp_pnp_ratings_FSD50K.json"]), vocabulary)


def review(metadata_dir: Path, root: Path) -> dict:
    selection = read_selection(metadata_dir)
    path = root / "prepared/provenance/selection.json"
    payload = _json_bytes(selection)
    if path.exists() and path.read_bytes() != payload:
        raise ValueError("Existing selection differs; use a separate acquisition directory")
    _atomic_bytes(path, payload)
    return {"selection_path": str(path.resolve()), "selection_sha256": hashlib.sha256(payload).hexdigest(),
            **{key: value for key, value in selection.items() if key not in {"selected", "forbidden_official_clip_ids"}}}


def _fetch_pinned_once(identifier: str) -> tuple[bytes, dict]:
    """Fetch only one reviewed clip, without creating a Hub audio cache."""
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    clip_identity(identifier, "validation")
    url = hf_hub_url(REPO, f"clips/dev/{identifier}.wav", repo_type="dataset", revision=REVISION)
    meta = get_hf_file_metadata(url, token=False)
    if meta.commit_hash != REVISION or not isinstance(meta.size, int) or not 1 <= meta.size <= MAX_CLIP_BYTES:
        raise ValueError("Pinned clip revision/size mismatch")
    if not isinstance(meta.etag, str) or not re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", meta.etag):
        raise ValueError("No usable upstream clip content hash")
    with _open_url(url) as response:
        payload = response.read(MAX_CLIP_BYTES + 1)
    if len(payload) != meta.size or len(payload) > MAX_CLIP_BYTES:
        raise ValueError("Pinned clip byte length mismatch")
    digest = hashlib.sha256(payload).hexdigest() if len(meta.etag) == 64 else hashlib.sha1(
        f"blob {len(payload)}\0".encode() + payload).hexdigest()
    if digest != meta.etag:
        raise ValueError("Pinned upstream clip content hash mismatch")
    return payload, {"url": url, "repo": REPO, "revision": REVISION, "bytes": len(payload),
                     "upstream_content_hash": digest, "upstream_hash_algorithm": "sha256" if len(digest) == 64 else "git-blob-sha1",
                     "upstream_content_hash_verified": True}


def fetch_pinned_clip(identifier: str) -> tuple[bytes, dict]:
    for attempt in range(3):
        try:
            return _fetch_pinned_once(identifier)
        except Exception as error:
            # Retry transport failures only. Source revision/hash/format errors
            # must remain hard failures, never trigger another source choice.
            module = type(error).__module__.split(".")[0]
            if attempt == 2 or (not isinstance(error, (OSError, TimeoutError)) and
                                module not in {"httpx", "httpcore", "requests", "urllib", "huggingface_hub"}):
                raise
            time.sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


def prepare_clip(item: dict, payload: bytes, root: Path, budget: StorageBudget,
                 source_receipt: dict, selection_sha256: str) -> tuple[ManifestRow, dict]:
    ids = clip_identity(item["id"], item["uploader"])
    if any(item[key] != value for key, value in ids.items()):
        raise ValueError("Selected clip identity changed")
    if reviewed_license(item["license_url"]) != (item["dataset"], item["license"]):
        raise ValueError("Selected clip license changed")
    info = sf.info(io.BytesIO(payload))
    if (info.channels, info.samplerate, info.subtype) != (1, 44100, "PCM_16"):
        raise ValueError("FSD source differs from the published mono44.1kPCM16 format")
    if info.frames > 180 * 44100:
        raise ValueError("FSD utterance exceeds the teacher duration guard")
    prepared, trace = prepare_audio(payload)
    original_path = root / "originals" / (item["id"] + ".wav")
    path = root / "prepared/audio" / (item["id"] + ".wav")
    provenance_path = root / "prepared/provenance/utterances" / (item["id"] + ".json")
    digest = hashlib.sha256(prepared).hexdigest()
    trace.update(preparation_version=2, dataset=item["dataset"], source_revision=REVISION,
                 selected_metadata=item, selection_sha256=selection_sha256,
                 source_receipt=source_receipt, original_path=str(original_path.resolve()),
                 prepared_audio_sha256=digest, prepared_path=str(path.resolve()),
                 collection_license="CC-BY-4.0", official_source=OFFICIAL_SOURCE,
                 source_processing="FSD release converted files to mono44.1kPCM16; prior Freesound processing is not independently verified")
    budget.write(original_path, payload)
    budget.write(path, prepared)
    row = ManifestRow(
        dataset=item["dataset"], source_revision=REVISION, source_id=ids["source_id"],
        source_url=f"https://freesound.org/s/{item['id']}/", audio_path=str(path.resolve()),
        audio_sha256=digest, parent_recording_id=ids["parent_recording_id"], parent_start_seconds=0,
        speaker_id=None, session_id=ids["session_id"], language="und", sample_rate_hz=16000,
        original_sample_rate_hz=44100, bandwidth_hz=8000, bandwidth_class="speech_band",
        native_recording=False, enhanced=False,
        bandwidth_evidence="Prepared 16 kHz reference has an 8 kHz Nyquist upper bound, not measured recording bandwidth. FSD-delivered mono44.1kPCM16 original retained; microphone bandwidth and prior Freesound processing are unverified.",
        duration_seconds=trace["prepared_frames"] / 16000, split=ids["split"],
        source_split="train-uploader-heldout-dev" if ids["split"] == "dev" else "train-uploader-selected",
        license=DATASET_LICENSES[item["dataset"]], license_url=item["license_url"],
        attribution=f"{item['uploader']}: {item['title']} (Freesound {item['id']}); FSD50K, Fonseca et al.",
        access_record=str(provenance_path.resolve()), gain_policy=GAIN_POLICY,
        resampler_policy=trace["resampler_policy"], teacher_cache_key=None)
    trace["manifest"] = row.to_dict()
    budget.write(provenance_path, _json_bytes(trace))
    return row, trace


class Exclusions:
    def __init__(self, prior=(), reserved=(), legacy=()):
        self.source_ids, self.parents, self.sessions = set(), set(), set()
        self.hashes = load_evaluation_exclusions(legacy)["sha256"]
        self.input_sha256 = {}
        for paths, heldout in ((prior, False), (reserved, True)):
            for path in paths:
                path = Path(path).resolve()
                self.input_sha256[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                for row in load_manifest(path):
                    self.source_ids.add(row.source_id)
                    self.parents.add(row.parent_recording_id)
                    self.hashes.add(row.audio_sha256)
                    if heldout and row.session_id:
                        self.sessions.add(row.session_id)
                    access = Path(row.access_record)
                    if not access.is_absolute():
                        access = path.parent / access
                    if access.is_file():
                        original = json.loads(access.read_bytes()).get("original_audio_sha256")
                        if original:
                            self.hashes.add(original)
        for path in legacy:
            path = Path(path).resolve()
            self.input_sha256[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    def metadata_matches(self, item):
        return (item["source_id"] in self.source_ids or item["parent_recording_id"] in self.parents
                or item["session_id"] in self.sessions)

    def identity(self):
        return {"sources": sorted(self.source_ids), "parents": sorted(self.parents),
                "sessions": sorted(self.sessions), "hashes": sorted(self.hashes),
                "input_sha256": self.input_sha256}


def acquire(selection: dict, root: Path, *, expected_selection_sha256: str,
            exclusions: Exclusions | None = None, max_bytes=2 * 1024**3,
            reserve_bytes=8 * 1024**3, fetch=fetch_pinned_clip) -> dict:
    root = root.resolve()
    selection_bytes = _json_bytes(selection)
    selection_hash = hashlib.sha256(selection_bytes).hexdigest()
    if selection_hash != expected_selection_sha256:
        raise ValueError("Reviewed selection SHA-256 changed")
    exclusions = exclusions or Exclusions()
    budget = StorageBudget(root, max_bytes, reserve_bytes)
    budget.write(root / "prepared/provenance/selection.json", selection_bytes)
    policy = {"preparation_version": 2, "selection_sha256": selection_hash, "exclusions": exclusions.identity()}
    budget.write(root / "prepared/provenance/acquisition-policy.json", _json_bytes(policy))
    rows, quarantined = [], []
    progress_path, marker_path = root / "prepared/progress.json", root / "prepared/provenance/complete.json"
    existing_complete = marker_path.read_bytes() if marker_path.exists() else None

    def progress(state, error=None):
        record = {"state": state, "preparation_version": 2, "processed": len(rows) + len(quarantined),
                  "selected": len(selection["selected"]), "rows": len(rows),
                  "hours": sum(row.duration_seconds for row in rows) / 3600,
                  "quarantined": len(quarantined), "stored_bytes": budget.used, "updated_at": time.time()}
        if error is not None:
            record["error"] = str(error)
        _atomic_bytes(progress_path, _json_bytes(record))

    try:
        progress("starting")
        forbidden = set(selection["forbidden_official_clip_ids"])
        for index, item in enumerate(selection["selected"], 1):
            if item["id"] in forbidden:
                raise ValueError("Official held-out clip appeared in the reviewed selection")
            if exclusions.metadata_matches(item):
                quarantined.append({"source_id": item["source_id"], "reason": "prior/reserved source, parent or reserved uploader group"})
                continue
            receipt_path = root / "prepared/provenance/utterances" / (item["id"] + ".json")
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_bytes())
                if receipt["selection_sha256"] != selection_hash or receipt["selected_metadata"] != item:
                    raise ValueError("Resume clip receipt differs from reviewed selection")
                row = ManifestRow.from_dict(receipt["manifest"])
                for key, expected in (("original_path", receipt["original_audio_sha256"]),
                                      ("prepared_path", row.audio_sha256)):
                    if hashlib.sha256(Path(receipt[key]).read_bytes()).hexdigest() != expected:
                        raise ValueError("Resume original/prepared audio hash mismatch")
            else:
                payload, source_receipt = fetch(item["id"])
                if hashlib.sha256(payload).hexdigest() in exclusions.hashes:
                    quarantined.append({"source_id": item["source_id"], "reason": "prior/reserved original audio hash"})
                    continue
                row, receipt = prepare_clip(item, payload, root, budget, source_receipt, selection_hash)
            if row.audio_sha256 in exclusions.hashes or receipt["original_audio_sha256"] in exclusions.hashes:
                quarantined.append({"source_id": row.source_id, "reason": "prior/reserved original or prepared audio hash"})
                continue
            rows.append(row)
            if index == 1 or index % 20 == 0:
                progress("preparing")
        rows = deduplicate_prepared(rows, quarantined)
        validate_manifest(rows)
        summary = {"state": "complete", "preparation_version": 2, **summarize_manifest(rows),
                   "selection_sha256": selection_hash, "exclusion_policy": policy["exclusions"],
                   "quarantined": quarantined, "source_revision": REVISION, "official_source": OFFICIAL_SOURCE,
                   "counts_by_target": dict(Counter(label for row in rows for label in json.loads(Path(row.access_record).read_bytes())["selected_metadata"]["target_labels"])),
                   "split_policy": selection["selection_policy"]["uploader_split"],
                   "quality_limit": selection["selection_policy"]["quality_limit"]}
        for split in ("train", "dev"):
            payload = b"".join((json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n").encode()
                               for row in rows if row.split == split)
            budget.write(root / "prepared" / (split + ".jsonl"), payload)
        encoded = _json_bytes(summary)
        if existing_complete is not None and existing_complete != encoded:
            raise ValueError("Existing complete FSD corpus audit changed")
        budget.write(marker_path, encoded)
        progress("complete")
        return summary
    except BaseException as error:
        progress("interrupted" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed", error)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("review", "acquire"))
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-sha256")
    parser.add_argument("--exclude-training-manifest", type=Path, action="append", default=[])
    parser.add_argument("--reserved-manifest", type=Path, action="append", default=[])
    parser.add_argument("--reserved-evaluation-manifest", type=Path, action="append", default=[])
    parser.add_argument("--max-gib", type=float, default=2)
    parser.add_argument("--min-free-gib", type=float, default=8)
    args = parser.parse_args()
    reviewed = review(args.metadata_dir, args.output_dir)
    if args.command == "review":
        print(json.dumps(reviewed, indent=2))
        return
    if args.selection_sha256 != reviewed["selection_sha256"]:
        parser.error("acquire requires --selection-sha256 from the metadata review")
    def interrupted(signum, _frame):
        raise InterruptedError(f"Acquisition interrupted by signal {signum}; verified receipts retained")
    signal.signal(signal.SIGTERM, interrupted)
    selection = json.loads(Path(reviewed["selection_path"]).read_bytes())
    summary = acquire(selection, args.output_dir, expected_selection_sha256=args.selection_sha256,
                      exclusions=Exclusions(args.exclude_training_manifest, args.reserved_manifest,
                                            args.reserved_evaluation_manifest),
                      max_bytes=int(args.max_gib * 1024**3), reserve_bytes=int(args.min_free_gib * 1024**3))
    print(json.dumps({key: summary[key] for key in ("state", "rows", "hours", "hours_by", "counts_by_target")}, indent=2))


if __name__ == "__main__":
    main()
