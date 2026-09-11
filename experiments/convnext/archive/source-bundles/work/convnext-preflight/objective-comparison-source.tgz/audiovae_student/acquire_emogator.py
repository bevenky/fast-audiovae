"""Acquire the pinned, Apache-2.0 EmoGator vocal-burst release independently.

Original MP3 bytes remain in the verified official archive. Prepared files are
whole utterances in mono16k IEEE FLOAT with explicit SoXR VHQ provenance, no gain
normalization and participant-disjoint dev. Emotion categories are not relabeled
as verified laughter, giggling, crying, shouting or whistling action classes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import time

import numpy as np
import soundfile as sf

from .acquire import _atomic_bytes, _json_bytes, _open_url
from .acquire_expressive import StorageBudget, prepare_audio, sha256_file, deduplicate_prepared, GAIN_POLICY
from .data import ManifestRow, summarize_manifest, validate_manifest


REVISION = "51eeca515a95f168ab73391cd9e7c975ab964429"
URL = f"https://codeload.github.com/fredbuhl/EmoGator/tar.gz/{REVISION}"
README_URL = f"https://github.com/fredbuhl/EmoGator/blob/{REVISION}/README.md"
LICENSE_URL = f"https://github.com/fredbuhl/EmoGator/blob/{REVISION}/LICENSE"
CATEGORIES = ("Adoration", "Amusement", "Anger", "Awe", "Confusion", "Contempt", "Contentment", "Desire",
              "Disappointment", "Disgust", "Distress", "Ecstasy", "Elation", "Embarrassment", "Fear", "Guilt",
              "Interest", "Neutral", "Pain", "Pride", "Realization", "Relief", "Romantic Love", "Sadness", "Serenity",
              "Shame", "Surprise (Negative)", "Surprise (Positive)", "Sympathy", "Triumph")
HELDOUT = frozenset(sorted((f"{index:06d}" for index in range(1, 358)),
                          key=lambda speaker: hashlib.sha256(f"emogator-speaker-dev-v1:{speaker}".encode()).hexdigest())[:20])
_NAME = re.compile(r"^(\d{6})-(\d{2})-([123])\.mp3$")


def identity(filename: str) -> tuple[str, int, str]:
    match = _NAME.fullmatch(filename)
    if not match or not 1 <= int(match[1]) <= 357 or not 1 <= int(match[2]) <= 30:
        raise ValueError(f"invalid official EmoGator filename: {filename}")
    return match[1], int(match[2]), match[3]


def load_tree(path: Path) -> dict[str, dict]:
    value = json.loads(path.read_text())
    if value.get("truncated"):
        raise ValueError("EmoGator pinned Git tree is truncated")
    rows = {item["path"]: item for item in value["tree"] if item["type"] == "blob"}
    audio = [item for name, item in rows.items() if name.startswith("data/mp3/") and name.endswith(".mp3")]
    if len(audio) != 32130:
        raise ValueError("EmoGator pinned tree must contain all 32,130 original MP3 files")
    counts = Counter()
    for item in audio:
        if not 0 < item["size"] <= 10_000_000:
            raise ValueError("EmoGator member exceeds bounded file size")
        speaker, category, instance = identity(PurePosixPath(item["path"]).name)
        counts[speaker] += 1
    if len(counts) != 357 or set(counts.values()) != {90}:
        raise ValueError("EmoGator contributor inventory does not match the pinned release")
    if "LICENSE" not in rows or "README.md" not in rows:
        raise ValueError("EmoGator source license and description must be preserved")
    return rows


def verify_blob(payload: bytes, entry: dict) -> None:
    blob = b"blob " + str(len(payload)).encode() + b"\0" + payload
    if len(payload) != entry["size"] or hashlib.sha1(blob).hexdigest() != entry["sha"]:
        raise ValueError(f"original bytes differ from pinned Git tree: {entry['path']}")


def download(root: Path, budget: StorageBudget) -> dict:
    archive, receipt_path = root / "emogator-original.tar.gz", root / "source-receipt.json"
    if archive.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("state") != "complete" or receipt["source_revision"] != REVISION or sha256_file(archive) != receipt["archive_sha256"]:
            raise ValueError("existing EmoGator archive or receipt differs; no silent overwrite")
        return receipt
    partial = archive.with_suffix(archive.suffix + ".part")
    if partial.exists():
        raise ValueError("partial EmoGator source download exists; retained for inspection")
    receipt = {"state": "downloading", "source_revision": REVISION, "url": URL, "license": "Apache-2.0",
               "license_url": LICENSE_URL, "readme_url": README_URL, "bytes": 0, "started_at": time.time()}
    digest = hashlib.sha256()
    _atomic_bytes(receipt_path, _json_bytes(receipt))
    try:
        with _open_url(URL) as response, partial.open("xb") as target:
            if "text/html" in response.headers.get("Content-Type", ""):
                raise ValueError("EmoGator archive request returned HTML")
            expected = int(response.headers.get("Content-Length", "0"))
            if expected > 700_000_000:
                raise ValueError("EmoGator source exceeds 700 MB download cap")
            budget.check(expected)
            while chunk := response.read(1024 * 1024):
                if receipt["bytes"] + len(chunk) > 700_000_000:
                    raise ValueError("EmoGator source download cap exceeded")
                budget.check(len(chunk))
                target.write(chunk)
                budget.used += len(chunk)
                digest.update(chunk)
                receipt["bytes"] += len(chunk)
                if receipt["bytes"] % (32 * 1024 * 1024) == 0:
                    _atomic_bytes(receipt_path, _json_bytes(receipt))
            target.flush()
            os.fsync(target.fileno())
        if expected and receipt["bytes"] != expected:
            raise ValueError("EmoGator download byte count disagrees with Content-Length")
        if not tarfile.is_tarfile(partial):
            raise ValueError("EmoGator download is not a tar archive")
        os.replace(partial, archive)
        receipt.update(state="complete", archive_sha256=digest.hexdigest(), completed_at=time.time(),
                       upstream_checksum_verified=False, member_git_hash_verification="required during preparation")
        _atomic_bytes(receipt_path, _json_bytes(receipt))
        return receipt
    except Exception as error:
        receipt.update(state="failed", error=str(error))
        _atomic_bytes(receipt_path, _json_bytes(receipt))
        raise


def prepare_member(name: str, payload: bytes, entry: dict, root: Path, budget: StorageBudget, receipt: dict):
    verify_blob(payload, entry)
    speaker, category, instance = identity(PurePosixPath(name).name)
    if sf.info(io.BytesIO(payload)).format != "MP3":
        raise ValueError("published MP3 member does not contain MPEG audio")
    original, original_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
    if original.shape[1] != 1:
        raise ValueError("non-mono EmoGator original has no approved channel policy")
    # Check and record, without normalizing or clipping source amplitudes. Silent
    # (all-zero) and nonfinite files are rejected by prepare_audio below.
    peak = float(np.max(np.abs(original))) if original.size else 0.0
    diagnostic = {"source_peak_abs": peak, "source_rms": float(np.sqrt(np.mean(original.astype(np.float64) ** 2))),
                  "source_zero_fraction": float(np.mean(original == 0)),
                  "source_at_or_over_full_scale_fraction": float(np.mean(np.abs(original) >= 1)),
                  "diagnostic_policy": "finite/mono/minimum-length/all-zero rejection; record clipping/silence statistics without gain changes"}
    prepared, trace = prepare_audio(payload)
    filename = PurePosixPath(name).stem + ".wav"
    audio_path, metadata_path = root / "prepared/audio" / filename, root / "prepared/provenance/utterances" / (filename + ".json")
    digest = hashlib.sha256(prepared).hexdigest()
    split = "dev" if speaker in HELDOUT else "train"
    trace.update(dataset="emogator", source_revision=REVISION, source_member=name,
                 original_git_blob_sha1=entry["sha"], source_receipt=receipt,
                 prepared_audio_sha256=digest, prepared_path=str(audio_path.resolve()),
                 contributor=speaker, intended_emotion=CATEGORIES[category - 1], category_id=category, instance=instance,
                 action_label="not separately annotated; emotion category does not establish laughter/crying/shouting type",
                 source_format="original published lossy MP3", **diagnostic)
    budget.write(audio_path, prepared)
    budget.write(metadata_path, _json_bytes(trace))
    row = ManifestRow(
        dataset="emogator", source_revision=REVISION, source_id=f"emogator:{PurePosixPath(name).name}",
        source_url=f"https://github.com/fredbuhl/EmoGator/blob/{REVISION}/{name}", audio_path=str(audio_path.resolve()), audio_sha256=digest,
        parent_recording_id=f"emogator:recording:{PurePosixPath(name).name}", parent_start_seconds=0,
        speaker_id=f"emogator:contributor:{speaker}", session_id=None, language="und",
        sample_rate_hz=16000, original_sample_rate_hz=original_rate, bandwidth_hz=8000,
        bandwidth_class="speech_band", native_recording=True, enhanced=False,
        bandwidth_evidence="Prepared 16 kHz reference has an 8 kHz Nyquist upper bound, not measured native bandwidth. Original lossy MP3 bytes, decoded native header and VHQ resampling are recorded; no native high-band reference is used.",
        duration_seconds=trace["prepared_frames"] / 16000, split=split,
        source_split="participant-heldout-dev" if split == "dev" else "participant-selected-train",
        license="Apache-2.0", license_url=LICENSE_URL,
        attribution="EmoGator dataset contributors and Fred Buhl; official fredbuhl/EmoGator repository release; original LICENSE and README retained",
        access_record=str(metadata_path.resolve()), gain_policy=GAIN_POLICY, resampler_policy=trace["resampler_policy"], teacher_cache_key=None)
    return row, trace


def acquire(root: Path, tree_path: Path, max_bytes: int, reserve_bytes: int) -> dict:
    root = root.resolve()
    budget = StorageBudget(root, max_bytes, reserve_bytes)
    tree = load_tree(tree_path)
    receipt = download(root, budget)
    rows, excluded, seen, source_formats, diagnostics = [], [], set(), Counter(), Counter()
    with tarfile.open(root / "emogator-original.tar.gz", "r|gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if PurePosixPath(member.name).is_absolute() or ".." in parts:
                raise ValueError("unsafe official archive member path")
            name = "/".join(parts[1:])
            if not member.isfile():
                if member.isdir():
                    continue
                raise ValueError("nonregular official archive member")
            if name not in tree or member.size != tree[name]["size"]:
                raise ValueError(f"archive contents differ from pinned Git tree: {name}")
            if name in seen:
                raise ValueError("duplicate archive member")
            seen.add(name)
            payload = archive.extractfile(member).read()
            verify_blob(payload, tree[name])
            if name in {"README.md", "LICENSE"}:
                budget.write(root / "provenance" / name, payload)
            if not name.startswith("data/mp3/"):
                continue
            try:
                row, trace = prepare_member(name, payload, tree[name], root, budget, receipt)
            except (ValueError, sf.LibsndfileError) as error:
                if "cap" in str(error) or "reserve" in str(error) or "disagree" in str(error) or "pinned Git" in str(error):
                    raise
                excluded.append({"source_member": name, "original_audio_sha256": hashlib.sha256(payload).hexdigest(), "reason": str(error)})
            else:
                rows.append(row)
                source_formats[f"{trace['original_rate_hz']}Hz/{trace['original_channels']}ch/{trace['original_subtype']}"] += 1
                diagnostics["with_samples_at_or_over_full_scale"] += trace["source_at_or_over_full_scale_fraction"] > 0
                diagnostics["over_90_percent_exact_zero_samples"] += trace["source_zero_fraction"] > 0.9
            if len(rows) % 500 == 0:
                _atomic_bytes(root / "progress.json", _json_bytes({"processed": len(rows) + len(excluded), "qualified_before_dedup": len(rows),
                              "excluded": len(excluded), "target_original_files": 32130, "updated_at": time.time()}))
    if seen != set(tree):
        raise ValueError("original archive does not contain the complete pinned Git tree")
    rows = deduplicate_prepared(rows, excluded)
    validate_manifest(rows)
    categories = Counter()
    for row in rows:
        if row.split == "train":
            _, category, _ = identity(row.source_id.split(":", 1)[1])
            categories[CATEGORIES[category - 1]] += 1
    summary = {"state": "complete", "preparation_version": 1, **summarize_manifest(rows),
               "source_revision": REVISION, "source_receipt": receipt,
               "tree_sha256": sha256_file(tree_path), "original_mp3_files": 32130,
               "train_category_utterances": dict(categories), "source_formats": dict(source_formats),
               "source_diagnostics_before_dedup": dict(diagnostics), "excluded": excluded,
               "heldout_contributors": sorted(HELDOUT), "split_policy": "Twenty contributors chosen by fixed SHA256 ordering; all their recordings held out.",
               "label_limit": "Nonverbal emotional bursts, with 30 intended emotion labels; separate action classes are not supplied.",
               "completed_at": time.time()}
    for split in ("train", "dev"):
        payload = b"".join((json.dumps(row.to_dict(), ensure_ascii=False) + "\n").encode() for row in rows if row.split == split)
        _atomic_bytes(root / "prepared" / f"{split}.jsonl", payload)
    _atomic_bytes(root / "prepared/provenance/complete.json", _json_bytes(summary))
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--tree", required=True, type=Path)
    parser.add_argument("--max-storage-gib", type=float, default=5)
    parser.add_argument("--reserve-gib", type=float, default=8)
    args = parser.parse_args(argv)
    result = acquire(args.root, args.tree, int(args.max_storage_gib * 1024**3), int(args.reserve_gib * 1024**3))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
