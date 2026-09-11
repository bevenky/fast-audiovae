"""Prepare reviewed human whistles from a pinned mirror, using CPU only.

The selection is reviewed against original Freesound pages. Encoded mirror
bytes are retained without substituting previews. Original Freesound byte
equality and microphone quality are not independently established.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import signal
import time

import soundfile as sf

from .acquire import _atomic_bytes, _json_bytes
from .acquire_expressive import GAIN_POLICY, StorageBudget, deduplicate_prepared, prepare_audio
from .acquire_fsd_vocal import Exclusions, clip_identity
from .data import ManifestRow, summarize_manifest, validate_manifest

REPO = "MoamenElSayed/freesound-commercial-50k"
REVISION = "ac5aed8cf1aeb97b26a375f24794d42e15f97163"
LICENSES = {
    "CC0-1.0": ("freesound_human_whistle_cc0", "https://creativecommons.org/publicdomain/zero/1.0/"),
    "CC-BY-3.0": ("freesound_human_whistle_cc_by_3", "https://creativecommons.org/licenses/by/3.0/"),
    "CC-BY-4.0": ("freesound_human_whistle_cc_by_4", "https://creativecommons.org/licenses/by/4.0/"),
}
METADATA_COLUMNS = ("title", "description", "tags", "username", "freesound_id", "license",
                    "attribution_required", "commercial_use")


class ClipRejected(ValueError):
    """An individual recording fails the reviewed audio contract."""


def validate_selection(selection: dict, expected_sha256: str) -> str:
    digest = hashlib.sha256(_json_bytes(selection)).hexdigest()
    if digest != expected_sha256:
        raise ValueError("Reviewed human-whistle selection SHA-256 changed")
    if (selection.get("format_version"), selection.get("preparation_version"),
            selection.get("repo"), selection.get("revision")) != (1, 1, REPO, REVISION):
        raise ValueError("Unrecognized human-whistle source or selection contract")
    forbidden = set(selection["forbidden_fsd_clip_ids"])
    seen = set()
    for item in selection["selected"]:
        if item["id"] in forbidden or item["id"] in seen:
            raise ValueError("Forbidden or duplicated Freesound clip ID")
        seen.add(item["id"])
        ids = clip_identity(item["id"], item["uploader"])
        if any(item[key] != value for key, value in ids.items()):
            raise ValueError("Freesound identity or uploader split changed")
        license_url = item["license_url"]
        # Source pages still use both official HTTP and HTTPS CC links. Keep
        # the recorded evidence unchanged while comparing their exact paths.
        canonical_url = "https://" + license_url[7:] if license_url.startswith("http://") else license_url
        if LICENSES.get(item["license"]) != (item["dataset"], canonical_url):
            raise ValueError("Reviewed license is outside the permitted source contract")
        if item["mirror_metadata"]["freesound_id"] != int(item["id"]):
            raise ValueError("Mirror and Freesound source IDs disagree")
    if not seen:
        raise ValueError("Empty human-whistle selection")
    return digest


class _CountedFile:
    def __init__(self, stream, fetcher):
        self.stream, self.fetcher = stream, fetcher

    def read(self, count=-1):
        if count < 0:
            raise ValueError("Unbounded Parquet audio reads are forbidden")
        self.fetcher.reserve_read(count)
        return self.stream.read(count)

    def seek(self, *args): return self.stream.seek(*args)
    def tell(self): return self.stream.tell()
    def readable(self): return True
    def writable(self): return False
    def seekable(self): return True
    @property
    def closed(self): return self.stream.closed


class ParquetClipFetcher:
    """One audio row group at a time; all downloaded bytes count across resumes."""
    def __init__(self, root: Path, selection_sha256: str, max_transfer_bytes=8 * 1024**3):
        from huggingface_hub import HfFileSystem
        self.fs = HfFileSystem()
        self.path = root / "prepared/provenance/transfer.json"
        self.limit, self.selection_hash = max_transfer_bytes, selection_sha256
        self.used = 0
        if self.path.exists():
            value = json.loads(self.path.read_bytes())
            if value["selection_sha256"] != selection_sha256 or value["max_transfer_bytes"] != self.limit:
                raise ValueError("Resume transfer policy changed")
            self.used = value["reserved_read_bytes"]
        self.key, self.rows, self.source = None, None, None

    def reserve_read(self, count):
        if self.used + count > self.limit:
            raise ValueError("Parquet transfer byte limit would be exceeded")
        self.used += count
        _atomic_bytes(self.path, _json_bytes({"selection_sha256": self.selection_hash,
            "max_transfer_bytes": self.limit, "reserved_read_bytes": self.used,
            "scope": "Parquet file ranges; reserved before read, so interrupted reads may be overcounted"}))

    def __call__(self, item):
        import pyarrow.parquet as pq
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        metadata = item["mirror_metadata"]
        key = (metadata["shard"], metadata["row_group"])
        if key != self.key:
            self.rows = None
            url = hf_hub_url(REPO, key[0], repo_type="dataset", revision=REVISION)
            upstream = get_hf_file_metadata(url)
            if upstream.commit_hash != REVISION:
                raise ValueError("Remote Parquet revision changed")
            before = self.used
            with self.fs.open(f"datasets/{REPO}@{REVISION}/{key[0]}", "rb",
                              cache_type="none", block_size=65536) as stream:
                parquet = pq.ParquetFile(_CountedFile(stream, self), pre_buffer=False)
                self.rows = parquet.read_row_group(key[1], columns=["audio", *METADATA_COLUMNS],
                                                   use_threads=False).to_pylist()
            self.source = {"repo": REPO, "revision": REVISION, "shard": key[0], "row_group": key[1],
                "shard_upstream_etag": upstream.etag, "shard_bytes": upstream.size,
                "transferred_group_bytes": self.used - before,
                "full_shard_hash_verified": False,
                "provenance": "Encoded audio bytes from pinned Parquet; no preview conversion or audio decoding during fetch"}
            self.key = key
        row = self.rows[metadata["row_within_group"]]
        if any(row[column] != metadata[column] for column in METADATA_COLUMNS):
            raise ValueError("Pinned Parquet row differs from the reviewed metadata")
        payload = row["audio"].get("bytes")
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("The selected Parquet row does not contain encoded audio bytes")
        return payload, dict(self.source, row_within_group=metadata["row_within_group"],
                             mirror_audio_path=row["audio"].get("path"), encoded_audio_bytes=len(payload))


def prepare_clip(item, payload, root, budget, receipt, selection_sha256):
    try:
        info = sf.info(io.BytesIO(payload))
    except sf.LibsndfileError as error:
        raise ClipRejected(f"Undecodable mirror audio: {error}") from error
    if info.channels not in {1, 2}:
        raise ClipRejected("Source requires an unreviewed channel policy")
    if info.frames / info.samplerate > 180:
        raise ClipRejected("Whistle exceeds the 180-second whole-utterance guard")
    if info.samplerate != item["sample_rate_hz"]:
        raise ClipRejected("Mirror sample rate differs from the original source page")
    if abs(info.frames / info.samplerate - item["duration_seconds"]) > 0.05:
        raise ClipRejected("Mirror duration differs from the original source page")
    try:
        prepared, trace = prepare_audio(payload, stereo_first_channel=True)
    except (ValueError, sf.LibsndfileError) as error:
        raise ClipRejected(str(error)) from error
    if info.channels == 2:
        trace["channel_policy"] = "Freesound stereo: fixed channel0, no mixing or gain scaling; both encoded original channels retained"
    suffix = {"WAV": ".wav", "WAVEX": ".wav", "FLAC": ".flac", "AIFF": ".aiff",
              "OGG": ".ogg", "MP3": ".mp3"}.get(info.format, ".audio")
    original = root / "originals" / (item["id"] + suffix)
    output = root / "prepared/audio" / (item["id"] + ".wav")
    provenance = root / "prepared/provenance/utterances" / (item["id"] + ".json")
    digest = hashlib.sha256(prepared).hexdigest()
    trace.update(preparation_version=1, dataset=item["dataset"], source_revision=REVISION,
        selected_metadata=item, selection_sha256=selection_sha256, source_receipt=receipt,
        original_path=str(original.resolve()), prepared_path=str(output.resolve()), prepared_audio_sha256=digest,
        original_encoding=info.format, original_freesound_byte_identity_verified=False,
        source_processing="Mirror encoded bytes retained; source page rate/duration matched; microphone processing and original Freesound byte equality unverified")
    budget.write(original, payload)
    budget.write(output, prepared)
    row = ManifestRow(dataset=item["dataset"], source_revision=REVISION, source_id=item["source_id"],
        source_url=item["source_url"], audio_path=str(output.resolve()), audio_sha256=digest,
        parent_recording_id=item["parent_recording_id"], parent_start_seconds=0,
        speaker_id=None, session_id=item["session_id"], language="und", sample_rate_hz=16000,
        original_sample_rate_hz=info.samplerate, bandwidth_hz=8000, bandwidth_class="speech_band",
        native_recording=False, enhanced=False,
        bandwidth_evidence="Prepared16k reference has8k Nyquist upper bound; original microphone bandwidth and prior processing unverified",
        duration_seconds=trace["prepared_frames"] / 16000, split=item["split"], source_split="mirror-train-uploader-split-v1",
        license=item["license"], license_url=item["license_url"],
        attribution=f"{item['uploader']}: {item['title']} (Freesound {item['id']}); {item['source_url']}",
        access_record=str(provenance.resolve()), gain_policy=GAIN_POLICY,
        resampler_policy=trace["resampler_policy"], teacher_cache_key=None)
    trace["manifest"] = row.to_dict()
    budget.write(provenance, _json_bytes(trace))
    return row, trace


def acquire(selection, root, *, expected_selection_sha256, exclusions=None, fetch=None,
            max_bytes=300 * 1024**2, reserve_bytes=8 * 1024**3, max_transfer_bytes=8 * 1024**3):
    digest = validate_selection(selection, expected_selection_sha256)
    root = Path(root).resolve()
    exclusions = exclusions or Exclusions()
    budget = StorageBudget(root, max_bytes, reserve_bytes)
    budget.write(root / "prepared/provenance/selection.json", _json_bytes(selection))
    policy = {"preparation_version": 1, "selection_sha256": digest, "exclusions": exclusions.identity(),
              "max_transfer_bytes": max_transfer_bytes, "max_retained_bytes": max_bytes}
    budget.write(root / "prepared/provenance/acquisition-policy.json", _json_bytes(policy))
    fetch = fetch or ParquetClipFetcher(root, digest, max_transfer_bytes)
    marker = root / "prepared/provenance/complete.json"
    previous = marker.read_bytes() if marker.exists() else None
    rows, quarantined = [], []

    def reject(item, reason):
        record = {"source_id": item["source_id"], "reason": reason}
        budget.write(root / "prepared/provenance/quarantined" / (item["id"] + ".json"),
                     _json_bytes({"selection_sha256": digest, "selected_metadata": item, "rejection": record}))
        quarantined.append(record)

    def progress(state, error=None):
        _atomic_bytes(root / "prepared/progress.json", _json_bytes({"state": state,
            "preparation_version": 1, "selected": len(selection["selected"]), "rows": len(rows),
            "quarantined": len(quarantined), "hours": sum(row.duration_seconds for row in rows) / 3600,
            "stored_bytes": budget.used, "updated_at": time.time(), "error": error}))

    try:
        progress("starting")
        order = sorted(selection["selected"], key=lambda item: (item["mirror_metadata"]["shard"],
                        item["mirror_metadata"]["row_group"], item["mirror_metadata"]["row_within_group"]))
        for item in order:
            rejected = root / "prepared/provenance/quarantined" / (item["id"] + ".json")
            if rejected.exists():
                saved = json.loads(rejected.read_bytes())
                if saved["selection_sha256"] != digest or saved["selected_metadata"] != item:
                    raise ValueError("Saved quarantine differs from reviewed selection")
                quarantined.append(saved["rejection"])
                continue
            if exclusions.metadata_matches(item):
                reject(item, "prior/reserved identity")
                continue
            page = Path(item["source_page_path"])
            if hashlib.sha256(page.read_bytes()).hexdigest() != item["source_page_sha256"]:
                raise ValueError("Reviewed source-page evidence changed")
            path = root / "prepared/provenance/utterances" / (item["id"] + ".json")
            if path.exists():
                receipt = json.loads(path.read_bytes())
                if receipt["selection_sha256"] != digest or receipt["selected_metadata"] != item:
                    raise ValueError("Saved whistle receipt differs from reviewed selection")
                row = ManifestRow.from_dict(receipt["manifest"])
                for key, expected in (("original_path", receipt["original_audio_sha256"]),
                                      ("prepared_path", row.audio_sha256)):
                    if hashlib.sha256(Path(receipt[key]).read_bytes()).hexdigest() != expected:
                        raise ValueError("Saved whistle audio SHA-256 mismatch")
            else:
                payload, source_receipt = fetch(item)
                if hashlib.sha256(payload).hexdigest() in exclusions.hashes:
                    reject(item, "prior/reserved original bytes")
                    continue
                try:
                    row, receipt = prepare_clip(item, payload, root, budget, source_receipt, digest)
                except ClipRejected as error:
                    reject(item, str(error))
                    progress("preparing")
                    continue
            if row.audio_sha256 in exclusions.hashes or receipt["original_audio_sha256"] in exclusions.hashes:
                reject(item, "prior/reserved audio hash")
            else:
                rows.append(row)
            progress("preparing")
        rows = deduplicate_prepared(rows, quarantined)
        if not rows:
            raise ValueError("No verified human-whistle recordings remain")
        validate_manifest(rows)
        summary = {"state": "complete", "preparation_version": 1, **summarize_manifest(rows),
            "selection_sha256": digest, "exclusion_policy": policy["exclusions"], "quarantined": quarantined,
            "source_revision": REVISION, "counts_by_target": {"human_whistling": len(rows)},
            "licenses": dict(Counter(row.license for row in rows)), "selection_policy": selection["selection_policy"]}
        for split in ("train", "dev"):
            budget.write(root / "prepared" / (split + ".jsonl"), b"".join(
                (json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n").encode()
                for row in rows if row.split == split))
        encoded = _json_bytes(summary)
        if previous is not None and previous != encoded:
            raise ValueError("Completed human-whistle audit changed")
        budget.write(marker, encoded)
        progress("complete")
        return summary
    except BaseException as error:
        progress("interrupted" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed", str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-training-manifest", type=Path, action="append", default=[])
    parser.add_argument("--reserved-manifest", type=Path, action="append", default=[])
    parser.add_argument("--reserved-evaluation-manifest", type=Path, action="append", default=[])
    args = parser.parse_args()
    def interrupted(signum, _frame):
        raise InterruptedError(f"Signal {signum}; verified receipts retained")
    signal.signal(signal.SIGTERM, interrupted)
    result = acquire(json.loads(args.selection.read_bytes()), args.output_dir,
        expected_selection_sha256=args.selection_sha256,
        exclusions=Exclusions(args.exclude_training_manifest, args.reserved_manifest, args.reserved_evaluation_manifest))
    print(json.dumps({key: result[key] for key in ("state", "rows", "hours", "hours_by", "quarantined")}, indent=2))


if __name__ == "__main__":
    main()
