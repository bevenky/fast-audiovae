from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from audiovae_student.acquire import _json_bytes
from audiovae_student.acquire_expressive import StorageBudget
from audiovae_student.acquire_fsd_vocal import (
    ALLOWED_LABELS, Exclusions, TARGET_LABELS, acquire, clip_identity,
    prepare_clip, read_selection, reviewed_license, select_candidates, uploader_split,
    _fetch_pinned_once, REVISION,
)
from audiovae_student.data import load_manifest, validate_manifest


VOCAB = {label: f"/m/{index}" for index, label in enumerate(sorted(ALLOWED_LABELS | {"Music", "Animal"}))}


def entry(identifier, label="Laughter", uploader="person", split="train", votes=None,
          license_url="http://creativecommons.org/licenses/by/3.0/", extra=()):
    labels = [label, "Human_voice", *extra]
    row = {"fname": str(identifier), "labels": ",".join(labels),
           "mids": ",".join(VOCAB[label] for label in labels), "split": split}
    info = {"uploader": uploader, "license": license_url, "title": "synthetic test"}
    ratings = {VOCAB[label]: [1, 1] if votes is None else votes}
    return row, info, ratings


def selection(entries, evaluation=()):
    catalog, info, ratings = [], {}, {}
    for row, item, votes in entries:
        catalog.append(row); info[row["fname"]] = item; ratings[row["fname"]] = votes
    return select_candidates(catalog, list(evaluation), info, ratings, VOCAB)


def waveform(frequency=440, rate=44100, channels=1):
    audio = (.2 * np.sin(np.arange(rate // 5) * 2 * np.pi * frequency / rate)).astype(np.float32)
    if channels == 2:
        audio = np.stack((audio, audio), axis=1)
    stream = io.BytesIO(); sf.write(stream, audio, rate, format="WAV", subtype="PCM_16")
    return stream.getvalue()


def selection_hash(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


class FsdVocalTests(unittest.TestCase):
    def test_pinned_download_verifies_revision_size_and_upstream_hash(self):
        payload = waveform()
        meta = SimpleNamespace(commit_hash=REVISION, size=len(payload), etag=hashlib.sha256(payload).hexdigest())
        hub = SimpleNamespace(get_hf_file_metadata=lambda *a, **k: meta, hf_hub_url=lambda *a, **k: "https://test.invalid/pinned.wav")
        with patch.dict("sys.modules", {"huggingface_hub": hub}):
            with patch("audiovae_student.acquire_fsd_vocal._open_url", return_value=io.BytesIO(payload)):
                got, receipt = _fetch_pinned_once("1")
            self.assertEqual(got, payload)
            self.assertTrue(receipt["upstream_content_hash_verified"])
            meta.etag = "a" * 64
            with patch("audiovae_student.acquire_fsd_vocal._open_url", return_value=io.BytesIO(payload)):
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    _fetch_pinned_once("1")
            meta.commit_hash = "b" * 40
            with self.assertRaisesRegex(ValueError, "revision"):
                _fetch_pinned_once("1")

    def test_strict_license_label_vote_and_official_clip_exclusions(self):
        data = selection([
            entry(1), entry(2, license_url="https://creativecommons.org/publicdomain/zero/1.0/"),
            entry(3, license_url="http://creativecommons.org/licenses/by-nc/3.0/"),
            entry(4, extra=("Music",)), entry(5, extra=("Animal",)),
            entry(6, votes=[1, .5]), entry(7, votes=[1, 1, -1]),
            entry(8, split="val"), entry(9), entry(10, votes=[1, 1, .5, 0]),
        ], evaluation=[{"fname": "9"}])
        self.assertEqual([item["id"] for item in data["selected"]], ["1", "2", "10"])
        self.assertEqual(data["forbidden_official_clip_ids"], ["8", "9"])
        self.assertEqual(data["excluded_counts"]["no_two_PP_votes_without_NP"], 2)
        self.assertIsNone(reviewed_license("https://creativecommons.org/licenses/by/3.0/?redirect=NC"))
        self.assertIsNone(reviewed_license("https://creativecommons.org.evil/licenses/by/3.0/"))

    def test_shared_uploader_canonicalization_and_no_false_speaker_identity(self):
        first = clip_identity("42", " Alice ")
        second = clip_identity("43", "ALICE")
        self.assertEqual(first["source_id"], first["parent_recording_id"])
        self.assertEqual(first["session_id"], second["session_id"])
        self.assertEqual(first["split"], second["split"])
        self.assertIsNone(first["speaker_id"])
        self.assertEqual(first["session_id"], "freesound:uploader:alice")
        with self.assertRaises(ValueError): clip_identity("../42", "alice")
        with self.assertRaises(ValueError): clip_identity("0042", "alice")
        # Unused publisher-validation clips do not blacklist their uploader's
        # different, eligible publisher-training recordings.
        data = selection([entry(1, uploader="same"), entry(2, uploader="same", split="val")])
        self.assertEqual([item["id"] for item in data["selected"]], ["1"])

    def test_unknown_annotation_values_and_changed_metadata_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "rating"):
            selection([entry(1, votes=[1, 1, .75])])
        row, info, votes = entry(1)
        row["mids"] = "/m/wrong,/m/wrong"
        with self.assertRaisesRegex(ValueError, "mapping"):
            selection([(row, info, votes)])
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "dev.csv").write_bytes(b"changed metadata")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                read_selection(Path(tmp))

    def test_original_retained_explicit_resample_no_gain_and_deterministic_preparation(self):
        item = selection([entry(1)])["selected"][0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); budget = StorageBudget(root, 10**7, 0)
            payload = waveform()
            row, receipt = prepare_clip(item, payload, root, budget, {"fixture": True}, "a" * 64)
            self.assertEqual(Path(receipt["original_path"]).read_bytes(), payload)
            self.assertEqual(row.duration_seconds, .2)
            self.assertEqual(row.original_sample_rate_hz, 44100)
            self.assertEqual(sf.info(row.audio_path).samplerate, 16000)
            self.assertEqual(row.resampler_policy, "prepared-soxr-vhq-to-16000-v1")
            self.assertEqual(row.gain_policy, "no-additional-gain-normalization")
            before = Path(row.audio_path).read_bytes()
            again, _ = prepare_clip(item, payload, root, budget, {"fixture": True}, "a" * 64)
            self.assertEqual(before, Path(again.audio_path).read_bytes())
            with self.assertRaisesRegex(ValueError, "published"):
                prepare_clip(item, waveform(rate=48000), root, budget, {}, "a" * 64)
            with self.assertRaisesRegex(ValueError, "published"):
                prepare_clip(item, waveform(channels=2), root, budget, {}, "a" * 64)

    def test_acquisition_is_resumable_verified_and_final_jsonl_and_audit_are_stable(self):
        users = {split: next(f"user{i}" for i in range(100) if uploader_split(f"user{i}") == split)
                 for split in ("train", "dev")}
        data = selection([entry(1, uploader=users["train"]), entry(2, uploader=users["dev"])])
        calls = []
        def fetch(identifier):
            calls.append(identifier)
            return waveform(400 + int(identifier) * 100), {"fixture": True}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = acquire(data, root, expected_selection_sha256=selection_hash(data), reserve_bytes=0, fetch=fetch)
            self.assertEqual(first["rows"], 2)
            self.assertEqual(calls, ["1", "2"])
            train, dev = load_manifest(root / "prepared/train.jsonl"), load_manifest(root / "prepared/dev.jsonl")
            self.assertEqual(len(train), 1); self.assertEqual(len(dev), 1)
            validate_manifest(train + dev)
            marker = root / "prepared/provenance/complete.json"
            before = marker.read_bytes()
            again = acquire(data, root, expected_selection_sha256=selection_hash(data), reserve_bytes=0, fetch=fetch)
            self.assertEqual(first, again)
            self.assertEqual(marker.read_bytes(), before)
            self.assertEqual(calls, ["1", "2"])
            (root / "originals/1.wav").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Resume original"):
                acquire(data, root, expected_selection_sha256=selection_hash(data), reserve_bytes=0, fetch=fetch)

    def test_review_and_storage_limits_fail_before_publish(self):
        data = selection([entry(1)])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "selection SHA"):
                acquire(data, root, expected_selection_sha256="b" * 64, reserve_bytes=0)
            with self.assertRaisesRegex(ValueError, "storage"):
                acquire(data, root, expected_selection_sha256=selection_hash(data), max_bytes=100,
                        reserve_bytes=0, fetch=lambda _: (waveform(), {}))
            self.assertFalse((root / "prepared/provenance/complete.json").exists())

    def test_exact_decoded_duplicates_are_not_counted_twice(self):
        data = selection([entry(1), entry(2)])
        with tempfile.TemporaryDirectory() as tmp:
            result = acquire(data, Path(tmp), expected_selection_sha256=selection_hash(data), reserve_bytes=0,
                             fetch=lambda _: (waveform(), {"fixture": True}))
            self.assertEqual(result["rows"], 1)
            self.assertEqual(len(result["quarantined"]), 1)
            self.assertAlmostEqual(result["hours"], .2 / 3600)


if __name__ == "__main__":
    unittest.main()
