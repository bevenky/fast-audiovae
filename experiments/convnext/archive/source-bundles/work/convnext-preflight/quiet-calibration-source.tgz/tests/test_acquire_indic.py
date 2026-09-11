import hashlib
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from audiovae_student.acquire_indic import Collector, LANGUAGES, REVISION, make_row, parse_args, source_identifiers
from audiovae_student.data import validate_manifest


def audio(rate=16000, channels=1):
    stream = io.BytesIO()
    sf.write(stream, np.zeros((rate * 3, channels), dtype=np.float32), rate, format="FLAC")
    return stream.getvalue()


def metadata(path="123456789_chunk_2.flac", speaker="S1234"):
    return {"audio_filepath": {"path": path}, "speaker_id": speaker, "duration": 3,
            "samples": 48000, "lang": "hi", "scenario": "Extempore", "task_name": "Source task"}


def row(root, payload=None, meta=None):
    return make_row(meta or metadata(), payload or audio(), language="hindi", audio_path=root / "sample.flac",
                    parquet_path="hindi/train-00000-of-00082.parquet", row_group=0, row_number=3,
                    access_record=root / "source-access.json")


def args(root):
    return SimpleNamespace(root=root, max_audio_gib=1, min_free_gib=0, reserved_manifest=[],
                           prior_training_manifest=[], reserved_evaluation_manifest=[], languages=["hindi"],
                           hours_per_language=2, speaker_seconds=5, minimum_clip_seconds=2.56)


class IndicAcquisitionTests(unittest.TestCase):
    def test_verified_native_shard_proof_survives_cleanup_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native_args = args(root); native_args.download_method = "native"
            collector = Collector(native_args)
            payload = b"synthetic original shard"
            digest = hashlib.sha256(payload).hexdigest()
            item = SimpleNamespace(path="hindi/train-00000.parquet", size=len(payload),
                                   lfs=SimpleNamespace(sha256=digest))
            def download(repo, filename, **kwargs):
                path = Path(kwargs["local_dir"]) / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                return str(path)
            collector.staging.download = download
            with collector._source_stream(None, item) as stream:
                self.assertEqual(stream.read(), payload)
                self.assertEqual(collector.source_files[item.path]["verified_sha256"], digest)
            self.assertTrue(stream.closed)
            self.assertEqual(collector.staging.entries, {})
            collector.snapshot("paused")
            collector.staging.close()
            resumed = Collector(native_args)
            self.assertEqual(resumed.source_files[item.path]["verified_sha256"], digest)
            self.assertEqual(resumed.source_files[item.path]["verified_bytes"], len(payload))
            resumed.staging.close()

    def test_native_mode_is_operational_and_preserves_existing_selection_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_args = args(root); original_args.download_method = "ranged"
            original = Collector(original_args)
            value, receipt = row(root)
            Path(value.audio_path).write_bytes(audio())
            path=root/"receipts/hindi/sample.json";path.parent.mkdir(parents=True);path.write_text(json.dumps(receipt))
            original._remember(value, receipt)
            original.snapshot("paused")
            policy=(root/"acquisition-policy.json").read_bytes()
            manifest=(root/"train.jsonl").read_bytes()
            native_args=args(root);native_args.download_method="native"
            native=Collector(native_args)
            self.assertEqual(native.rows,original.rows)
            self.assertEqual((root/"acquisition-policy.json").read_bytes(),policy)
            self.assertEqual((root/"train.jsonl").read_bytes(),manifest)
            native.staging.close()

    def test_completed_language_skips_hub_listing_even_without_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            collector=Collector(args(Path(tmp)))
            collector.hours["hindi"]=collector.args.hours_per_language
            with patch.dict("sys.modules", {"huggingface_hub": None, "pyarrow.parquet": None}):
                self.assertIsNone(collector.collect_language("hindi"))
            collector.source_files["hindi/train.parquet"]={"row_groups":2}
            collector.completed_groups["hindi/train.parquet"]=[0,1]
            self.assertTrue(collector._shard_complete("hindi/train.parquet"))
            collector.completed_groups["hindi/train.parquet"]=[0]
            self.assertFalse(collector._shard_complete("hindi/train.parquet"))

    def test_queued_language_exits_before_new_hub_requests_after_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            collector = Collector(args(Path(tmp)))
            collector.stop.set()
            # If any transport import executes, it fails: paused tasks must
            # return before constructing a filesystem or listing a repository.
            with patch.dict("sys.modules", {"huggingface_hub": None, "pyarrow.parquet": None}):
                self.assertIsNone(collector.collect_language("hindi"))

    def test_workers_one_through_eight_and_reject_unsupported_counts(self):
        for workers in range(1, 9):
            self.assertEqual(parse_args(["--root", "/synthetic", "--workers", str(workers)]).workers, workers)
        for workers in (0, -1, 9):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--root", "/synthetic", "--workers", str(workers)])

    def test_worker_increase_preserves_receipt_resume_and_completed_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial_args = args(root); initial_args.workers = 3
            initial = Collector(initial_args)
            result, receipt = row(root)
            Path(result.audio_path).write_bytes(audio())
            receipt_path = root / "receipts/hindi/sample.json"
            receipt_path.parent.mkdir(parents=True); receipt_path.write_text(json.dumps(receipt))
            initial._remember(result, receipt)
            initial.completed_groups = {"hindi/train-00000-of-00082.parquet": [0, 1]}
            initial.snapshot("paused")
            policy_bytes = (root / "acquisition-policy.json").read_bytes()
            manifest_bytes = (root / "train.jsonl").read_bytes()
            new_args = args(root); new_args.workers = 6
            resumed = Collector(new_args)
            self.assertEqual(resumed.rows, initial.rows)
            self.assertEqual(resumed.hours, initial.hours)
            self.assertEqual(resumed.speaker_seconds, initial.speaker_seconds)
            self.assertEqual(resumed.completed_groups, initial.completed_groups)
            self.assertEqual((root / "acquisition-policy.json").read_bytes(), policy_bytes)
            self.assertEqual((root / "train.jsonl").read_bytes(), manifest_bytes)
            self.assertFalse(resumed.candidate(metadata("123456789_chunk_9.flac"), "hindi"))

    def test_original_bytes_actual_frames_and_honest_bandwidth(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = audio()
            result, receipt = row(Path(tmp), payload)
        validate_manifest([result], training_only=True)
        self.assertEqual(result.audio_sha256, hashlib.sha256(payload).hexdigest())
        self.assertEqual(result.duration_seconds, 3)
        self.assertEqual(result.source_revision, REVISION)
        self.assertEqual(result.source_split, "train")
        self.assertFalse(result.native_recording)
        self.assertFalse(result.enhanced)
        self.assertEqual(result.bandwidth_class, "speech_band")
        self.assertEqual(receipt["source"]["scenario"], "Extempore")
        self.assertEqual(receipt["source"]["decoded_frames"], 48000)

    def test_unknown_full_recording_offset_is_not_fabricated(self):
        first = source_identifiers(metadata("123456789_chunk_2.flac"), "hindi")
        second = source_identifiers(metadata("123456789_chunk_9.flac"), "hindi")
        self.assertNotEqual(first["parent_recording_id"], second["parent_recording_id"])
        self.assertEqual(first["session_id"], second["session_id"])
        self.assertEqual(first["speaker_id"], "indicvoices:speaker:S1234")

    def test_actual_conversation_uuid_keeps_both_speaker_tracks_in_one_session(self):
        recording = "749ed384-2108-4dbf-aac1-0226046cdbc9"
        first = source_identifiers(metadata(recording + "_0_chunk_1.flac", "S_first"), "hindi")
        second = source_identifiers(metadata(recording + "_1_chunk_2.flac", "S_second"), "hindi")
        self.assertEqual(first["session_id"], second["session_id"])
        self.assertNotEqual(first["parent_recording_id"], second["parent_recording_id"])
        self.assertNotEqual(first["speaker_id"], second["speaker_id"])

    def test_changed_rate_channels_or_sample_metadata_fail_before_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for payload in (audio(48000), audio(channels=2)):
                with self.assertRaisesRegex(ValueError, "stop before any conversion"):
                    row(root, payload)
            incorrect = metadata()
            incorrect["samples"] = 47999
            with self.assertRaisesRegex(ValueError, "sample count"):
                row(root, meta=incorrect)
            self.assertFalse((root / "sample.flac").exists())

    def test_invalid_source_identifiers_are_rejected(self):
        for path in ("../123_chunk_1.flac", "unknown.flac"):
            with self.assertRaises(ValueError):
                source_identifiers(metadata(path), "hindi")
        changed = metadata()
        changed["lang"] = "ta"
        with self.assertRaisesRegex(ValueError, "language"):
            source_identifiers(changed, "hindi")
        self.assertEqual(len(LANGUAGES), 22)

    def test_cap_one_recording_and_atomic_receipt_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collector = Collector(args(root))
            result, receipt = row(root)
            self.assertTrue(collector.candidate(metadata(), "hindi"))
            Path(result.audio_path).write_bytes(audio())
            receipt_path = root / "receipts/hindi/sample.json"
            receipt_path.parent.mkdir(parents=True)
            receipt_path.write_text(json.dumps(receipt))
            resumed = Collector(args(root))
            self.assertEqual(len(resumed.rows), 1)
            self.assertFalse(resumed.candidate(metadata("123456789_chunk_5.flac"), "hindi"))
            self.assertFalse(resumed.candidate(metadata("987654321_chunk_1.flac"), "hindi"))
            self.assertTrue(resumed.candidate(metadata("987654321_chunk_1.flac", "S_other"), "hindi"))
            Path(result.audio_path).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "audio hash mismatch"):
                Collector(args(root))


if __name__ == "__main__":
    unittest.main()
