"""Synthetic archive tests; no network requests or real corpus downloads."""

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from audiovae_student.acquire import (
    DownloadBudget, _fleurs_catalog, acquire_bootstrap, collect_archive, load_evaluation_exclusions,
)
from audiovae_student.data import load_manifest, validate_manifest


def audio_payload(seconds=3, format="WAV", rate=16000):
    target = io.BytesIO()
    sf.write(target, np.linspace(-0.1, 0.1, int(seconds * rate), dtype=np.float32), rate, format=format, subtype="PCM_16")
    return target.getvalue()


def archive_payload(members):
    result = io.BytesIO()
    with tarfile.open(fileobj=result, mode="w:gz") as archive:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return result.getvalue()


class AcquisitionTests(unittest.TestCase):
    def test_fleurs_preserves_bytes_unknown_identity_and_exclusions(self):
        first, second = audio_payload(3), audio_payload(4)
        catalog = _fleurs_catalog(b"1\tfirst.wav\ttext\ttext\tchars\t48000\tMALE\n2\tsecond.wav\ttext\ttext\tchars\t64000\tFEMALE\n")
        payload = archive_payload([("train/first.wav", first), ("train/second.wav", second)])
        exclusions = {"filenames": set(), "text_ids": {("hi_in", 1)}, "sha256": set()}
        with tempfile.TemporaryDirectory() as directory:
            rows, receipt = collect_archive(io.BytesIO(payload), dataset="fleurs", partition="train", language="hi_in",
                root=Path(directory), seconds=3, source_url="https://example.test/train.tar.gz",
                budget=DownloadBudget(1000000), catalog=catalog, exclusions=exclusions)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(Path(row.audio_path).read_bytes(), second)
            self.assertEqual(row.audio_sha256, hashlib.sha256(second).hexdigest())
            self.assertIsNone(row.speaker_id)
            self.assertEqual(row.session_id, "fleurs:unknown-session-group:hi_in")
            self.assertEqual(row.bandwidth_class, "speech_band")
            self.assertFalse(receipt["complete_archive_digest_verified"])

    def test_librispeech_real_ids_and_split_audit(self):
        wav = audio_payload(format="FLAC")
        with tempfile.TemporaryDirectory() as directory:
            rows = []
            for partition, filename in (("train-clean-100", "19-198-0001.flac"), ("dev-clean", "20-199-0001.flac")):
                payload = archive_payload([(f"LibriSpeech/{partition}/{filename}", wav + b"\x00" * (partition == "dev-clean"))])
                selected, _ = collect_archive(io.BytesIO(payload), dataset="librispeech", partition=partition, language="en",
                    root=Path(directory), seconds=2, source_url="https://example.test/source", budget=DownloadBudget(1000000))
                rows.extend(selected)
            validate_manifest(rows)
            self.assertEqual(rows[0].speaker_id, "librispeech:speaker:19")
            self.assertEqual(rows[0].session_id, "librispeech:chapter:19:198")
            self.assertEqual(rows[1].split, "dev")
            self.assertEqual(rows[0].parent_start_seconds, 0)

    def test_wrong_partition_and_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            kwargs = dict(dataset="librispeech", partition="train-clean-100", language="en", root=Path(directory),
                seconds=2, source_url="https://example.test/source", budget=DownloadBudget(1000000))
            for name in ("../19-198-0001.flac", "LibriSpeech/test-clean/19-198-0001.flac"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    collect_archive(io.BytesIO(archive_payload([(name, audio_payload(format="FLAC"))])), **kwargs)
            with self.assertRaisesRegex(ValueError, "partition"):
                collect_archive(io.BytesIO(b""), **{**kwargs, "partition": "test-clean"})

    def test_metadata_frame_disagreement_and_wrong_rate_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for rate, samples in ((16000, 100), (8000, 24000)):
                with self.subTest(rate=rate), self.assertRaises(ValueError):
                    collect_archive(io.BytesIO(archive_payload([("a.wav", audio_payload(rate=rate))])), dataset="fleurs",
                        partition="train", language="hi_in", root=Path(directory), seconds=2,
                        source_url="https://example.test/source", budget=DownloadBudget(1000000),
                        catalog={"a.wav": {"dataset_id": 1, "num_samples": samples, "gender": "MALE"}})

    def test_legacy_regression_metadata_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            path.write_text(json.dumps({"samples": [{"language": "hi_in", "dataset_id": 1919,
                "original_audio_path": "13563417804047448798.wav", "sha256": "checksum"}]}))
            exclusions = load_evaluation_exclusions([path])
            self.assertIn(("hi_in", 1919), exclusions["text_ids"])
            self.assertIn(("hi_in", "13563417804047448798.wav"), exclusions["filenames"])
            self.assertIn("checksum", exclusions["sha256"])

    def test_orchestrator_pins_train_only_and_writes_manifests(self):
        tsv = b"1\ta.wav\ttext\ttext\tchars\t48000\tMALE\n"
        wav = archive_payload([("train/a.wav", audio_payload())])
        urls = []
        def opener(url):
            urls.append(url)
            return io.BytesIO(tsv if url.endswith(".tsv") else wav)
        with tempfile.TemporaryDirectory() as directory, patch("audiovae_student.acquire._open_url", side_effect=opener):
            result = acquire_bootstrap(directory, fleurs_languages=["hi_in"], minutes_per_language=0.01,
                librispeech_train_minutes=0, librispeech_dev_minutes=0)
            rows = load_manifest(Path(directory) / "train.jsonl")
            self.assertEqual(len(rows), 1)
            self.assertEqual(result["summary"]["rows"], 1)
            self.assertTrue(all("70bb2e84b976b7e960aa89f1c648e09c59f894dd" in url for url in urls))
            self.assertTrue(all("train" in url for url in urls))
            self.assertTrue((Path(directory) / "provenance/sources.json").exists())

    def test_download_budget_and_missing_catalog(self):
        budget = DownloadBudget(4)
        budget.add(4)
        with self.assertRaisesRegex(ValueError, "budget"):
            budget.add(1)
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "metadata"):
            collect_archive(io.BytesIO(b""), dataset="fleurs", partition="train", language="hi_in",
                root=Path(directory), seconds=2, source_url="https://example.test/source", budget=DownloadBudget(100))


if __name__ == "__main__":
    unittest.main()
