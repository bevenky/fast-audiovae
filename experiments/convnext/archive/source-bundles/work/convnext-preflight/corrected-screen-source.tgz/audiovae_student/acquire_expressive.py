"""Prepare reviewed expressive speech as traceable mono 16 kHz FLOAT audio.

Original archives/files stay intact. Whole utterances are decoded and, only when
necessary, resampled with SoXR VHQ. There is no loudness normalization, neural
restoration, crop encoding, evaluation download or model execution here. Labels
describe the source's intended performance, not a perceptual quality judgment.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import tarfile
import time
from typing import Iterator
import zipfile

import numpy as np
import soundfile as sf
import soxr

from .acquire import _atomic_bytes, _json_bytes, _open_url
from .data import DATASET_LICENSES, ManifestRow, load_manifest, summarize_manifest, validate_manifest


CREMA_REVISION = "1658cd342dff90010aa843eaeebd53610a08b1dc"
GAIN_POLICY = "no-additional-gain-normalization"
RESAMPLE_POLICY = "prepared-soxr-vhq-to-16000-v1"
NATIVE_POLICY = "prepared-native-16000-float-v1"
MAX_MEMBER_BYTES = 100_000_000
MIN_INPUT_SAMPLES = 1366
SOURCES = {
    "thorsten_emotional": {
        "revision": "emotional-v02", "language": "de",
        "filename": "thorsten-emotional_v02.tgz", "receipt": "thorsten-emotional-v02.json",
        "url": "https://www.openslr.org/resources/110/thorsten-emotional_v02.tgz",
        "review_url": "https://www.openslr.org/110/",
        "attribution": "Thorsten Mueller (voice), Dominik Kreutz (audio preparation); OpenSLR 110",
        "source_processing": "Publisher reports audio optimized and normalized to -24 dB; no further gain adjustment here.",
    },
    "jnv": {
        "revision": "ver3-2024-10-26", "language": "ja",
        "filename": "jnv_corpus_ver3.zip", "receipt": "jnv-ver3.json",
        "url": "https://ss-takashi.sakura.ne.jp/corpus/jnv/jnv_corpus_ver3.zip",
        "review_url": "https://sites.google.com/site/shinnosuketakamichi/research-topics/jnv_corpus",
        "attribution": "Detai Xin, Shinnosuke Takamichi, Hiroshi Saruwatari; JNV corpus",
        "source_processing": "Original published JNV nonverbal vocalization WAVs.",
    },
    "jvnv": {
        "revision": "ver1-2023-10-21", "language": "ja",
        "filename": "jvnv_ver1.zip", "receipt": "jvnv-ver1.json",
        "url": "https://ss-takashi.sakura.ne.jp/corpus/jvnv/jvnv_ver1.zip",
        "review_url": "https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus",
        "attribution": "Detai Xin, Junfeng Jiang, Shinnosuke Takamichi, Yuki Saito, Akiko Aizawa, Hiroshi Saruwatari; JVNV corpus",
        "source_processing": "Original published JVNV verbal speech and nonverbal expression WAVs.",
    },
    "crema_d": {
        "revision": CREMA_REVISION, "language": "en",
        "review_url": f"https://github.com/CheyneyComputerScience/CREMA-D/blob/{CREMA_REVISION}/README.md",
        "attribution": "Cao, Cooper, Keutmann, Gur, Nenkova and Verma; CREMA-D, IEEE Transactions on Affective Computing (2014)",
        "source_processing": "Publisher's processed AudioWAV files converted from recorded video; not MP3 or video alternatives.",
    },
}
EMOTIONS = {"ANG": "anger", "DIS": "disgust", "FEA": "fear", "HAP": "happiness", "NEU": "neutral", "SAD": "sadness"}
THORSTEN_STYLES = {"neutral", "disgusted", "angry", "amused", "surprised", "sleepy", "drunk", "whispering"}
JP_EMOTIONS = {"angry", "anger", "disgust", "fear", "happy", "happiness", "sad", "sadness", "surprise"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class StorageBudget:
    """Combined original and prepared files; no deleting unrelated artifacts."""

    def __init__(self, root: Path, max_bytes: int, reserve_bytes: int):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if max_bytes < 1 or reserve_bytes < 0:
            raise ValueError("invalid storage bounds")
        self.max_bytes, self.reserve_bytes = max_bytes, reserve_bytes
        self.used = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        self.check(0)

    def check(self, added: int):
        if added < 0 or self.used + added > self.max_bytes:
            raise ValueError("expressive storage byte cap would be exceeded")
        if shutil.disk_usage(self.root).free - added < self.reserve_bytes:
            raise ValueError("expressive filesystem reserve would be crossed")

    def write(self, path: Path, payload: bytes):
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("output must stay within the budgeted expressive root")
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError(f"existing source/prepared bytes disagree: {path}")
            return
        self.check(len(payload))
        _atomic_bytes(path, payload)
        self.used += len(payload)


def download_jvnv(root: Path, budget: StorageBudget) -> dict:
    """Fetch the official licensed archive, bounded at 2 GB and checked as ZIP."""
    spec = SOURCES["jvnv"]
    path, receipt_path = root / spec["filename"], root / spec["receipt"]
    if path.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("state") == "complete" and sha256_file(path) == receipt["archive_sha256"]:
            return receipt
        raise ValueError("existing JVNV download is inconsistent; do not silently overwrite")
    part = path.with_suffix(path.suffix + ".part")
    if part.exists():
        raise ValueError("partial JVNV download exists; retain it for inspection before retry")
    receipt = {"name": "jvnv-ver1", "filename": path.name, "url": spec["url"],
               "review_url": spec["review_url"], "license": DATASET_LICENSES["jvnv"],
               "source_revision": spec["revision"], "state": "downloading", "bytes": 0,
               "upstream_checksum_verified": False, "started_at": time.time()}
    digest = hashlib.sha256()
    _atomic_bytes(receipt_path, _json_bytes(receipt))
    try:
        with _open_url(spec["url"]) as response, part.open("xb") as target:
            if "text/html" in response.headers.get("Content-Type", ""):
                raise ValueError("official archive response is HTML, not ZIP")
            expected_bytes = int(response.headers.get("Content-Length", "0"))
            if expected_bytes > 2_000_000_000:
                raise ValueError("JVNV archive exceeds source download cap")
            budget.check(expected_bytes)
            while chunk := response.read(1024 * 1024):
                if receipt["bytes"] + len(chunk) > 2_000_000_000:
                    raise ValueError("JVNV source download cap exceeded")
                budget.check(len(chunk))
                target.write(chunk)
                budget.used += len(chunk)
                digest.update(chunk)
                receipt["bytes"] += len(chunk)
                if receipt["bytes"] % (32 * 1024 * 1024) == 0:
                    _atomic_bytes(receipt_path, _json_bytes(receipt))
            target.flush()
            os.fsync(target.fileno())
        if expected_bytes and receipt["bytes"] != expected_bytes:
            raise ValueError("JVNV archive length differs from HTTP Content-Length")
        if not zipfile.is_zipfile(part):
            raise ValueError("JVNV archive is not ZIP; excluded")
        os.replace(part, path)
        receipt.update(state="complete", archive_sha256=digest.hexdigest(), completed_at=time.time())
        _atomic_bytes(receipt_path, _json_bytes(receipt))
        return receipt
    except Exception as error:
        receipt.update(state="failed", error=str(error), updated_at=time.time())
        _atomic_bytes(receipt_path, _json_bytes(receipt))
        raise


@dataclass(frozen=True)
class Identity:
    source_id: str
    speaker: str
    session: str | None
    emotion: str
    split: str
    labels: dict


def source_identity(dataset: str, member: str) -> Identity:
    """Use published IDs; never infer an emotion/nonverbal action by listening."""
    path = PurePosixPath(member)
    if path.is_absolute() or ".." in path.parts or "\\" in member:
        raise ValueError("unsafe archive member path")
    if dataset == "thorsten_emotional":
        style = "whispering" if path.parent.name == "whisper" else path.parent.name
        if style not in THORSTEN_STYLES or not re.fullmatch(r"[a-f0-9]{32}\.wav", path.name):
            raise ValueError(f"unrecognized Thorsten source identity: {member}")
        # The same text hash in different styles denotes separate recordings.
        return Identity(f"{path.parent.name}/{path.name}", "thorsten:speaker:thorsten-mueller", None,
                        style, "train", {"intended_style": style, "text_id": path.stem})
    if dataset in {"jnv", "jvnv"}:
        pattern = (r"([FM][12])_([a-z]+)_(\d+)_([RF])\.wav" if dataset == "jnv"
                   else r"([FM][12])_([a-z]+)_(regular|free)_(\d+)\.wav")
        match = re.fullmatch(pattern, path.name)
        if not match or match[2] not in JP_EMOTIONS:
            raise ValueError(f"unrecognized {dataset} source identity: {member}")
        if dataset == "jnv":
            speaker, emotion, utterance, session = match.groups()
        else:
            speaker, emotion, session, utterance = match.groups()
        # Both corpora use four locally named speakers from the same lab. We
        # conservatively keep same-named speakers together across both corpora;
        # this does not claim that cross-corpus physical identities are proven.
        canonical_speaker = f"jnv-jvnv:conservative-speaker-group:{speaker}"
        return Identity(path.name, canonical_speaker, None, emotion,
                        "dev" if speaker == "M2" else "train",
                        {"intended_emotion": emotion, "session_type": session,
                         "utterance_id": utterance, "nonverbal": dataset == "jnv",
                         "includes_verbal_and_nonverbal": dataset == "jvnv",
                         "cross_corpus_identity": "same-local-ID conservative grouping, not independently verified"})
    if dataset == "crema_d":
        match = re.fullmatch(r"(\d{4})_([A-Z]{3})_([A-Z]{3})_(LO|MD|HI|XX)\.wav", path.name)
        if not match or match[3] not in EMOTIONS:
            raise ValueError(f"unrecognized CREMA-D source identity: {member}")
        actor, sentence, emotion, intensity = match.groups()
        return Identity(path.name, f"crema_d:actor:{actor}", None, EMOTIONS[emotion],
                        "dev" if int(actor) % 10 == 0 else "train",
                        {"intended_emotion": EMOTIONS[emotion], "intended_intensity": intensity,
                         "sentence_code": sentence, "labels_source": "official filename, not crowd-vote consensus"})
    raise ValueError(f"unsupported expressive source: {dataset}")


def prepare_audio(payload: bytes, *, stereo_first_channel: bool = False) -> tuple[bytes, dict]:
    """Whole-file decode and explicit VHQ resampling; preserve gain and finiteness."""
    info = sf.info(io.BytesIO(payload))
    if info.channels != 1 and not (info.channels == 2 and stereo_first_channel):
        raise ValueError("non-mono source requires a separately reviewed channel policy")
    if info.samplerate < 16000 or info.samplerate > 192000:
        raise ValueError("unsupported source sample rate")
    audio, rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)
    if not np.isfinite(audio).all():
        raise ValueError("nonfinite source samples")
    if info.channels == 2:
        audio = audio[:, 0]
    if rate != 16000:
        prepared = soxr.resample(audio, rate, 16000, quality="VHQ")
    else:
        prepared = audio
    prepared = np.asarray(prepared, dtype=np.float32)
    if not np.isfinite(prepared).all():
        raise ValueError("nonfinite resampler output")
    if len(prepared) < MIN_INPUT_SAMPLES:
        raise ValueError(f"source too short for the current teacher: {len(prepared)} < {MIN_INPUT_SAMPLES} samples")
    if np.count_nonzero(prepared) == 0:
        raise ValueError("all-zero source audio")
    if abs(len(prepared) - info.frames * 16000 / rate) > 1:
        raise ValueError("resampling duration disagreement")
    # libsndfile adds a wall-clock timestamp in FLOAT WAV PEAK chunks. Writing
    # standard IEEE_FLOAT/fact/data chunks directly makes file hashes stable
    # across restarts without changing any FP32 sample or quantizing to PCM16.
    samples = np.asarray(prepared, dtype="<f4").tobytes()
    chunks = (b"fmt " + struct.pack("<IHHIIHH", 16, 3, 1, 16000, 64000, 4, 32)
              + b"fact" + struct.pack("<II", 4, len(prepared))
              + b"data" + struct.pack("<I", len(samples)) + samples)
    encoded = b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks
    trace = {
        "original_audio_sha256": hashlib.sha256(payload).hexdigest(),
        "original_rate_hz": rate, "original_frames": info.frames,
        "original_channels": info.channels, "original_subtype": info.subtype,
        "prepared_rate_hz": 16000, "prepared_frames": len(prepared), "prepared_subtype": "FLOAT",
        "resampler_policy": RESAMPLE_POLICY if rate != 16000 else NATIVE_POLICY,
        "soxr_version": soxr.__version__ if rate != 16000 else None,
        "soundfile_version": sf.__version__, "gain_policy": GAIN_POLICY,
        "whole_utterance": True, "original_audio_retained": True,
    }
    if info.channels == 2:
        trace["channel_policy"] = "JNV original stereo: fixed channel 0 selected without mixing or gain scaling; both original channels retained in source archive"
    return encoded, trace


def prepare_member(dataset: str, member: str, payload: bytes, root: Path,
                   budget: StorageBudget, source_receipt: dict) -> tuple[ManifestRow, dict]:
    identity, spec = source_identity(dataset, member), SOURCES[dataset]
    if dataset == "jnv" and sf.info(io.BytesIO(payload)).samplerate != 48000:
        raise ValueError("JNV source differs from published 48 kHz corpus specification; quarantined rather than treating it as another recording")
    prepared, trace = prepare_audio(payload, stereo_first_channel=dataset == "jnv")
    digest = hashlib.sha256(prepared).hexdigest()
    path = root / "prepared/audio" / dataset / identity.source_id
    provenance_path = root / "prepared/provenance/utterances" / dataset / (identity.source_id + ".json")
    trace.update(dataset=dataset, source_revision=spec["revision"], source_member=member,
                 prepared_audio_sha256=digest, prepared_path=str(path.resolve()), labels=identity.labels,
                 source_processing=spec["source_processing"], source_receipt=source_receipt)
    budget.write(path, prepared)
    budget.write(provenance_path, _json_bytes(trace))
    row = ManifestRow(
        dataset=dataset, source_revision=spec["revision"], source_id=f"{dataset}:{identity.source_id}",
        source_url=source_receipt["url"], audio_path=str(path.resolve()), audio_sha256=digest,
        parent_recording_id=f"{dataset}:recording:{identity.source_id}", parent_start_seconds=0,
        speaker_id=identity.speaker, session_id=identity.session, language=spec["language"],
        sample_rate_hz=16000, original_sample_rate_hz=trace["original_rate_hz"],
        bandwidth_hz=8000, bandwidth_class="speech_band", native_recording=True, enhanced=False,
        bandwidth_evidence="Prepared 16 kHz speech-band reference has an 8 kHz Nyquist upper bound, not a measured physical bandwidth. Native source format and explicit resampling provenance are retained; no native high-band reference is used in this warmup.",
        duration_seconds=trace["prepared_frames"] / 16000, split=identity.split,
        source_split="speaker-heldout-dev" if identity.split == "dev" else "speaker-selected-train",
        license=DATASET_LICENSES[dataset], license_url=spec["review_url"],
        attribution=spec["attribution"], access_record=str(provenance_path.resolve()),
        gain_policy=GAIN_POLICY, resampler_policy=trace["resampler_policy"], teacher_cache_key=None,
    )
    return row, trace


def archive_members(path: Path) -> Iterator[tuple[str, bytes]]:
    """Read only WAV file contents, never extract archive paths or symlinks."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir() or not member.filename.lower().endswith(".wav") or member.filename.startswith("__MACOSX/"):
                    continue
                if member.file_size > MAX_MEMBER_BYTES or (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("oversized or symlink WAV archive member")
                yield member.filename, archive.read(member)
    else:
        with tarfile.open(path, "r|*") as archive:
            for member in archive:
                if not member.name.lower().endswith(".wav"):
                    continue
                if not member.isfile() or member.size > MAX_MEMBER_BYTES:
                    raise ValueError("oversized or non-regular WAV archive member")
                yield member.name, archive.extractfile(member).read()


def deduplicate_prepared(rows: list[ManifestRow], excluded: list[dict]) -> list[ManifestRow]:
    """Quarantine exact decoded duplicates; cross-split copies are all excluded."""
    groups = {}
    for row in rows:
        groups.setdefault(row.audio_sha256, []).append(row)
    retained = []
    for digest, group in groups.items():
        if len(group) == 1:
            retained.extend(group)
            continue
        cross_split = len({row.split for row in group}) > 1
        keep = [] if cross_split else group[:1]
        retained.extend(keep)
        for row in group[len(keep):]:
            excluded.append({"source_id": row.source_id, "prepared_audio_sha256": digest,
                             "reason": "exact decoded audio duplicate across splits; all copies excluded" if cross_split else "exact decoded audio duplicate; first source retained",
                             "duplicate_source_ids": [other.source_id for other in group],
                             "retained_source_id": keep[0].source_id if keep else None})
    return retained


def prepare_archive(dataset: str, root: Path, budget: StorageBudget) -> dict:
    if dataset not in {"thorsten_emotional", "jnv", "jvnv"}:
        raise ValueError("source is not an approved archive")
    spec = SOURCES[dataset]
    receipt = json.loads((root / spec["receipt"]).read_text())
    path = root / spec["filename"]
    if receipt.get("state") != "complete" or sha256_file(path) != receipt["archive_sha256"]:
        raise ValueError("original archive is incomplete or has changed")
    if receipt["url"] != spec["url"] or receipt["license"] != DATASET_LICENSES[dataset]:
        raise ValueError("archive receipt URL/license does not match the reviewed source")
    rows, excluded, emotions, seen = [], [], Counter(), set()
    for index, (member, payload) in enumerate(archive_members(path), 1):
        if member in seen:
            raise ValueError(f"duplicate archive member: {member}")
        seen.add(member)
        # Identity/path errors fail closed. Decode/length defects are recorded and
        # excluded; they never become silent substitutions or synthetic hours.
        identity = source_identity(dataset, member)
        try:
            row, trace = prepare_member(dataset, member, payload, root, budget, receipt)
        except (ValueError, sf.LibsndfileError) as error:
            if "storage" in str(error) or "reserve" in str(error) or "disagree" in str(error):
                raise
            excluded.append({"member": member, "original_audio_sha256": hashlib.sha256(payload).hexdigest(), "reason": str(error)})
            continue
        rows.append(row)
        emotions[identity.emotion] += row.duration_seconds
        if index % 100 == 0:
            _atomic_bytes(root / "prepared/progress.json", _json_bytes({"dataset": dataset, "processed": index, "qualified": len(rows), "excluded": len(excluded), "updated_at": time.time()}))
    rows = deduplicate_prepared(rows, excluded)
    emotions = Counter()
    for row in rows:
        emotions[source_identity(dataset, row.source_id.split(":", 1)[1]).emotion] += row.duration_seconds
    validate_manifest(rows)
    summary = {"dataset": dataset, "preparation_version": 2, "state": "complete", **summarize_manifest(rows), "seconds_by_intended_emotion": dict(emotions),
               "excluded": excluded, "source_receipt": receipt,
               "split_policy": "Thorsten train only; Japanese M2 conservative speaker group held out across JNV/JVNV; CREMA actors divisible by 10 held out."}
    for split in ("train", "dev"):
        payload = b"".join((json.dumps(r.to_dict(), ensure_ascii=False) + "\n").encode() for r in rows if r.split == split)
        _atomic_bytes(root / "prepared/manifests" / f"{dataset}-{split}.jsonl", payload)
    _atomic_bytes(root / "prepared/provenance" / f"{dataset}.json", _json_bytes(summary))
    return summary


def parse_lfs_pointer(payload: bytes, expected_git_blob: str | None = None) -> tuple[str, int]:
    if expected_git_blob is not None:
        blob = b"blob " + str(len(payload)).encode() + b"\0" + payload
        if hashlib.sha1(blob).hexdigest() != expected_git_blob:
            raise ValueError("CREMA-D Git blob hash disagrees with pinned tree")
    text = payload.decode("ascii")
    match = re.fullmatch(r"version https://git-lfs.github.com/spec/v1\noid sha256:([a-f0-9]{64})\nsize (\d+)\n", text)
    if not match:
        raise ValueError("invalid pinned CREMA-D LFS pointer")
    size = int(match[2])
    if not 0 < size <= MAX_MEMBER_BYTES:
        raise ValueError("CREMA-D audio exceeds per-member cap")
    return match[1], size


def crema_inventory(root: Path, tree_path: Path, budget: StorageBudget) -> list[dict]:
    """Read only AudioWAV pointers from a pinned official repository snapshot."""
    tree_bytes = tree_path.read_bytes()
    tree = json.loads(tree_bytes)
    expected = {item["path"]: item["sha"] for item in tree if item["path"].startswith("AudioWAV/") and item["path"].endswith(".wav")}
    if len(expected) != 7442:
        raise ValueError("CREMA-D tree must contain exactly the pinned 7,442 AudioWAV files")
    pointer_archive = root / "crema-pinned-lfs-pointers.tar.gz"
    url = f"https://codeload.github.com/CheyneyComputerScience/CREMA-D/tar.gz/{CREMA_REVISION}"
    if not pointer_archive.exists():
        with _open_url(url) as response:
            payload = response.read(40_000_001)
        if len(payload) > 40_000_000:
            raise ValueError("CREMA-D pointer snapshot exceeds 40 MB cap")
        budget.write(pointer_archive, payload)
    selected = {}
    with tarfile.open(pointer_archive, "r|gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            relative = "/".join(parts[1:])
            if relative not in expected:
                continue
            if not member.isfile() or member.size > 1024:
                raise ValueError("expected CREMA-D pointer is not a small regular file")
            pointer = archive.extractfile(member).read()
            digest, size = parse_lfs_pointer(pointer, expected[relative])
            if relative in selected:
                raise ValueError("duplicate CREMA-D AudioWAV pointer")
            selected[relative] = {"path": relative, "sha256": digest, "bytes": size,
                                  "git_blob_sha1": expected[relative]}
    if selected.keys() != expected.keys():
        raise ValueError("CREMA-D pointer snapshot does not match complete pinned audio tree")
    receipt = {"source_revision": CREMA_REVISION, "tree_sha256": hashlib.sha256(tree_bytes).hexdigest(),
               "pointer_archive_url": url, "pointer_archive_sha256": sha256_file(pointer_archive),
               "audio_records": len(selected), "audio_bytes": sum(x["bytes"] for x in selected.values()),
               "verification": "Git blob SHA1 matches pinned tree; each audio file must match its Git LFS SHA256 and byte count"}
    _atomic_bytes(root / "crema-pointer-inventory.json", _json_bytes(receipt))
    return sorted(selected.values(), key=lambda row: row["path"])


def acquire_crema(root: Path, tree_path: Path, budget: StorageBudget, workers: int = 8) -> dict:
    """Download official LFS WAV bytes with pinned SHA256 checks; resume by file."""
    if not 1 <= workers <= 8:
        raise ValueError("CREMA-D download concurrency must be 1..8")
    inventory = crema_inventory(root, tree_path, budget)
    # Bound the original collection before issuing thousands of audio requests.
    missing_bytes = sum(item["bytes"] for item in inventory if not (root / "crema-originals" / item["path"]).exists())
    budget.check(missing_bytes)

    def retrieve(item):
        url = f"https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/{CREMA_REVISION}/{item['path']}"
        original_path = root / "crema-originals" / item["path"]
        if original_path.exists():
            payload = original_path.read_bytes()
        else:
            for attempt in range(3):
                try:
                    with _open_url(url) as response:
                        payload = response.read(item["bytes"] + 1)
                    break
                except (OSError, TimeoutError):
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
        if len(payload) != item["bytes"] or hashlib.sha256(payload).hexdigest() != item["sha256"]:
            raise ValueError(f"CREMA-D original audio does not match pinned LFS bytes: {item['path']}")
        return item, url, original_path, payload

    rows, excluded, emotions = [], [], Counter()
    # Keep only at most eight downloaded utterances resident. The main thread
    # owns writes and provenance, so no concurrent budget accounting is needed.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        iterator = iter(inventory)
        pending = [pool.submit(retrieve, next(iterator)) for _ in range(min(workers, len(inventory)))]
        processed = 0
        while pending:
            item, url, original_path, payload = pending.pop(0).result()
            budget.write(original_path, payload)
            receipt = {"url": url, "source_revision": CREMA_REVISION, "license": DATASET_LICENSES["crema_d"],
                       "original_path": str(original_path.resolve()), "original_audio_sha256": item["sha256"],
                       "upstream_checksum_verified": True, "git_blob_sha1": item["git_blob_sha1"],
                       "upstream_checksum_kind": "pinned Git LFS SHA256"}
            identity = source_identity("crema_d", item["path"])
            try:
                row, trace = prepare_member("crema_d", item["path"], payload, root, budget, receipt)
            except (ValueError, sf.LibsndfileError) as error:
                if "storage" in str(error) or "reserve" in str(error) or "disagree" in str(error):
                    raise
                excluded.append({"member": item["path"], "original_audio_sha256": item["sha256"], "reason": str(error)})
            else:
                rows.append(row)
                emotions[identity.emotion] += row.duration_seconds
            processed += 1
            if processed % 100 == 0:
                _atomic_bytes(root / "prepared/crema-progress.json", _json_bytes({"processed": processed, "total": len(inventory),
                              "qualified": len(rows), "excluded": len(excluded), "seconds": sum(r.duration_seconds for r in rows), "updated_at": time.time()}))
            following = next(iterator, None)
            if following is not None:
                pending.append(pool.submit(retrieve, following))
    rows = deduplicate_prepared(rows, excluded)
    emotions = Counter()
    for row in rows:
        emotions[source_identity("crema_d", row.source_id.split(":", 1)[1]).emotion] += row.duration_seconds
    validate_manifest(rows)
    result = {"dataset": "crema_d", "preparation_version": 2, "state": "complete", **summarize_manifest(rows), "excluded": excluded,
              "seconds_by_intended_emotion": dict(emotions), "source_revision": CREMA_REVISION,
              "split_policy": "Actors divisible by 10 held out; remaining actors train. All source files retained with exact LFS verification."}
    for split in ("train", "dev"):
        payload = b"".join((json.dumps(r.to_dict(), ensure_ascii=False) + "\n").encode() for r in rows if r.split == split)
        _atomic_bytes(root / "prepared/manifests" / f"crema_d-{split}.jsonl", payload)
    _atomic_bytes(root / "prepared/provenance/crema_d.json", _json_bytes(result))
    return result


def complete_all(root: Path, tree_path: Path, max_bytes: int, reserve_bytes: int, workers: int = 8) -> dict:
    """Finish a resumable preparation, writing the readiness receipt last.

    The official JVNV download may already be running independently. We first
    prepare other sources, then wait up to 90 minutes for that verified archive.
    No training process should consume this corpus until the final receipt exists.
    """
    summaries = []
    for dataset in ("thorsten_emotional", "jnv", "crema_d", "jvnv"):
        summary_path = root / "prepared/provenance" / f"{dataset}.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            if summary.get("preparation_version") == 2 and summary.get("state") == "complete":
                summaries.append(summary)
                continue
        if dataset == "jvnv":
            receipt_path = root / SOURCES[dataset]["receipt"]
            deadline = time.monotonic() + 90 * 60
            while receipt_path.exists() and json.loads(receipt_path.read_text()).get("state") == "downloading":
                if time.monotonic() > deadline:
                    raise TimeoutError("JVNV download still incomplete after 90 minutes; source remains unready")
                time.sleep(10)
            if not receipt_path.exists():
                download_jvnv(root, StorageBudget(root, max_bytes, reserve_bytes))
        budget = StorageBudget(root, max_bytes, reserve_bytes)
        summary = acquire_crema(root, tree_path, budget, workers) if dataset == "crema_d" else prepare_archive(dataset, root, budget)
        summaries.append(summary)
    rows = []
    for dataset in SOURCES:
        for split in ("train", "dev"):
            rows.extend(load_manifest(root / "prepared/manifests" / f"{dataset}-{split}.jsonl"))
    validate_manifest(rows)
    result = {"state": "complete", "preparation_version": 2, **summarize_manifest(rows),
              "source_summaries": {row["dataset"]: str((root / "prepared/provenance" / f"{row['dataset']}.json").resolve()) for row in summaries},
              "excluded_records": sum(len(row["excluded"]) for row in summaries),
              "not_covered": ["No verified whistling-labelled source acquired", "No claim of 16 expressive training hours; report measured qualified train hours"],
              "completed_at": time.time()}
    for split in ("train", "dev"):
        payload = b"".join((json.dumps(row.to_dict(), ensure_ascii=False) + "\n").encode() for row in rows if row.split == split)
        _atomic_bytes(root / "prepared" / f"{split}.jsonl", payload)
    _atomic_bytes(root / "prepared/provenance/expressive-complete.json", _json_bytes(result))
    return result


def audit_nonverbal_labels(root: Path) -> dict:
    """Count actual train metadata without equating emotion with an action.

    JNV provides phonetic phrases and JVNV adds generic NV time intervals. They
    do not provide separate verified laughing/giggling/crying/shouting labels.
    Phrase candidate groups below are explicit interpretations for later review,
    never asserted acoustic action ground truth or used to expand source hours.
    """
    ready = json.loads((root / "prepared/provenance/expressive-complete.json").read_text())
    if ready.get("state") != "complete":
        raise ValueError("audit requires completed expressive acquisition")
    rows = load_manifest(root / "prepared/train.jsonl")
    candidates = {
        "laughter_like_phrase_candidate": {"あは", "あはは", "うふ", "うふふ", "えへ", "えへへ", "はは", "ははは", "ふふ", "ふふふ", "へへ", "へへへ"},
        "sniffle_or_sob_phrase_candidate": {"ぐすん"},
        "scream_like_phrase_candidate": {"きゃー", "ぎゃー", "ひい", "ひえー", "ひゃー", "ひぇぇっ"},
    }
    grouped = {}
    def add(name, row, nv_seconds=None, phrase=None):
        group = grouped.setdefault(name, {"train_utterances": 0, "whole_file_seconds": 0.0,
                                          "annotated_nv_seconds": 0.0, "nv_timed_utterances": 0,
                                          "speakers": set(), "phrases": set()})
        group["train_utterances"] += 1
        group["whole_file_seconds"] += row.duration_seconds
        group["speakers"].add(row.speaker_id)
        if nv_seconds is not None:
            group["annotated_nv_seconds"] += nv_seconds
            group["nv_timed_utterances"] += 1
        if phrase:
            group["phrases"].add(phrase)
    with zipfile.ZipFile(root / SOURCES["jnv"]["filename"]) as jnv, zipfile.ZipFile(root / SOURCES["jvnv"]["filename"]) as jvnv:
        jnv_phrases = dict(line.split("|", 1) for line in jnv.read("JNV/phrases.txt").decode().splitlines() if line.strip())
        jvnv_phrases = {parts[0]: parts[1] for line in jvnv.read("jvnv_v1/transcription.csv").decode().splitlines()
                       if len(parts := line.split("|", 2)) == 3}
        names = set(jvnv.namelist())
        for row in rows:
            if row.dataset == "thorsten_emotional" and ":whisper/" in row.source_id:
                add("explicit_whisper_style", row)
            if row.dataset not in {"jnv", "jvnv"}:
                continue
            stem = row.source_id.split(":", 1)[1].removesuffix(".wav")
            speaker, key = stem.split("_", 1)
            phrase, nv_seconds = None, None
            if row.dataset == "jnv":
                phrase = jnv_phrases.get(key)
                add("jnv_nonverbal_corpus_all", row)
            else:
                phrase = jvnv_phrases.get(key)
                label_name = f"jvnv_v1/nv_label/{speaker}/{stem}.txt"
                if label_name in names:
                    intervals = []
                    for line in jvnv.read(label_name).decode().splitlines():
                        start, end, label = line.split()
                        start, end = float(start), float(end)
                        if label != "NV" or start < 0 or end <= start or end > row.duration_seconds + 1 / 16000:
                            raise ValueError(f"invalid official NV interval: {label_name}")
                        intervals.append((start, end))
                    intervals.sort()
                    if any(right[0] < left[1] for left, right in zip(intervals, intervals[1:])):
                        raise ValueError(f"overlapping official NV intervals: {label_name}")
                    nv_seconds = sum(end - start for start, end in intervals)
                add("jvnv_generic_nv_annotation", row, nv_seconds)
            normalized = re.sub(r"[？?！!。、, .]", "", phrase or "")
            for group, phrases in candidates.items():
                if normalized in phrases:
                    add(group, row, nv_seconds, phrase)
    for group in grouped.values():
        group["speaker_groups"] = len(group.pop("speakers"))
        group["phrases"] = sorted(group["phrases"])
    result = {"scope": "qualified train rows only", "groups": grouped,
              "verified_action_taxonomy": "Only Thorsten whisper is an explicit action/style class. JNV/JVNV use emotion and phonetic/generic NV labels, not a verified action taxonomy.",
              "not_separately_verified": ["laughing versus giggling", "crying", "shouting", "screaming", "whistling"],
              "candidate_status": "manual interpretation of source phonetic phrase; requires audio/event-label review before claiming actual action coverage",
              "duration_scope": "whole_file_seconds includes speech and pauses; only annotated_nv_seconds measures publisher-marked NV intervals, which may include other NV types"}
    _atomic_bytes(root / "prepared/provenance/nonverbal-label-audit.json", _json_bytes(result))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download-jvnv", "prepare", "acquire-crema", "complete", "audit-labels"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--dataset", choices=tuple(SOURCES), default="thorsten_emotional")
    parser.add_argument("--max-storage-gib", type=float, default=8)
    parser.add_argument("--reserve-gib", type=float, default=8)
    parser.add_argument("--crema-tree", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    budget = StorageBudget(root, int(args.max_storage_gib * 1024**3), int(args.reserve_gib * 1024**3))
    if args.command == "download-jvnv":
        result = download_jvnv(root, budget)
    elif args.command == "audit-labels":
        result = audit_nonverbal_labels(root)
    elif args.command in {"acquire-crema", "complete"}:
        if args.crema_tree is None:
            parser.error("--crema-tree is required for acquire-crema")
        result = (acquire_crema(root, args.crema_tree, budget, args.workers) if args.command == "acquire-crema"
                  else complete_all(root, args.crema_tree, budget.max_bytes, budget.reserve_bytes, args.workers))
    else:
        result = prepare_archive(args.dataset, root, budget)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
