"""Synthetic archive tests; no network requests or real corpus downloads."""

import hashlib
from concurrent.futures import ThreadPoolExecutor
import io
import itertools
import json
from pathlib import Path
import tarfile
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from audiovae_student.acquire import (
    AudioStorageBudget, DownloadBudget, _fleurs_catalog, _run_source_tasks, acquire_bootstrap, allocate_language_seconds,
    collect_archive, discover_fleurs_languages, load_evaluation_exclusions, nonoverlapping_window_inventory,
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
    def test_fleurs_transcript_quotes_are_literal_tsv_data(self):
        # These unpaired quotation marks are valid transcript text, not quoting
        # syntax. Ordinary csv.reader would merge fields/records incorrectly.
        payload = (b'1\ta.wav\t"A short quotation\t"a short quotation\tchars\t48000\tMALE\n'
                   b'2\tb.wav\tAnother sentence.\tanother sentence\tchars\t64000\tFEMALE\n')
        catalog = _fleurs_catalog(payload)
        self.assertEqual(set(catalog), {"a.wav", "b.wav"})
        self.assertEqual(catalog["b.wav"]["num_samples"], 64000)
        with self.assertRaisesRegex(ValueError, "physical line 2"):
            _fleurs_catalog(payload.splitlines(keepends=True)[0] + b"malformed\trow\n")

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
            self.assertEqual(row.bandwidth_hz, 8000)
            self.assertIn("Nyquist upper bound", row.bandwidth_evidence)
            self.assertIn("not a measured", row.bandwidth_evidence)
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

    def test_metadata_mismatch_is_quarantined_without_weakening_audio_matching(self):
        bad, good = audio_payload(6), audio_payload(4)
        catalog = {"bad.wav": {"dataset_id": 1, "num_samples": 48000, "gender": "FEMALE"},
                   "good.wav": {"dataset_id": 2, "num_samples": 64000, "gender": "FEMALE"}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, receipt = collect_archive(io.BytesIO(archive_payload([("bad.wav", bad), ("good.wav", good)])),
                dataset="fleurs", partition="train", language="af_za", root=root, seconds=3,
                source_url="https://example.test/train", budget=DownloadBudget(1000000), catalog=catalog)
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0].source_id.endswith("good.wav"))
            self.assertEqual(rows[0].duration_seconds, 4)
            self.assertEqual(receipt["quarantined_records"], 1)
            self.assertTrue(receipt["quota_met"])
            records = [json.loads(line) for line in (root / receipt["quarantine_log"]).read_text().splitlines()]
            self.assertEqual(records[0]["filename"], "bad.wav")
            self.assertEqual(records[0]["expected_tsv_samples"], 48000)
            self.assertEqual(records[0]["decoded_audio_frames"], 96000)
            self.assertEqual(records[0]["audio_sha256"], hashlib.sha256(bad).hexdigest())
            self.assertFalse(any(path.name == "bad.wav" for path in root.rglob("*.wav")))
            self.assertEqual(Path(rows[0].audio_path).read_bytes(), good)

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

    def test_capacity_allocation_keeps_small_languages_and_redistributes(self):
        result = allocate_language_seconds({"small": 1800, "large": 18000, "middle": 7200}, 18000)
        self.assertEqual(result["small"], 1800)
        self.assertEqual(result["middle"], 7200)
        self.assertEqual(result["large"], 9000)
        self.assertAlmostEqual(sum(result.values()), 18000)
        with self.assertRaisesRegex(ValueError, "insufficient"):
            allocate_language_seconds({"small": 10}, 11, minimum_seconds=0)
        with self.assertRaisesRegex(ValueError, "minimum language"):
            allocate_language_seconds({"a": 3600, "b": 3600}, 100)

    def test_previous_training_utterance_excluded_without_excluding_unknown_group(self):
        first, second = audio_payload(3), audio_payload(4)
        catalog = _fleurs_catalog(b"1\ta.wav\ttext\ttext\tchars\t48000\tMALE\n2\tb.wav\ttext\ttext\tchars\t64000\tFEMALE\n")
        archive = archive_payload([("train/a.wav", first), ("train/b.wav", second)])
        with tempfile.TemporaryDirectory() as directory:
            kwargs = dict(dataset="fleurs", partition="train", language="hi_in", root=Path(directory),
                seconds=2, source_url="https://example.test/train", catalog=catalog)
            previous, _ = collect_archive(io.BytesIO(archive), budget=DownloadBudget(1000000), **kwargs)
            fresh, receipt = collect_archive(io.BytesIO(archive), budget=DownloadBudget(1000000),
                prior_training_rows=previous, **kwargs)
            self.assertEqual(len(fresh), 1)
            self.assertTrue(fresh[0].source_id.endswith("b.wav"))
            self.assertEqual(fresh[0].session_id, previous[0].session_id)
            self.assertEqual(receipt["skips"]["prior_training_source"], 1)

    def test_unique_window_inventory_counts_only_full_disjoint_scored_regions(self):
        wav = audio_payload(5.13)
        catalog = {"a.wav": {"dataset_id": 1, "num_samples": 82080, "gender": "MALE"}}
        with tempfile.TemporaryDirectory() as directory:
            rows, _ = collect_archive(io.BytesIO(archive_payload([("a.wav", wav)])), dataset="fleurs",
                partition="train", language="hi_in", root=Path(directory), seconds=1,
                source_url="https://example.test/train", catalog=catalog, budget=DownloadBudget(1000000))
            result = nonoverlapping_window_inventory(rows)
            self.assertEqual(result["windows"], 2)
            self.assertAlmostEqual(result["unique_scored_hours"], 5.12 / 3600)
            self.assertEqual(result["window_seconds"], 2.56)

    def test_separate_audio_root_reserve_and_cap_preserve_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "old.wav"
            existing.write_bytes(b"existing")
            budget = AudioStorageBudget(root, max_bytes=10)
            self.assertEqual(budget.used_bytes, 8)
            with self.assertRaisesRegex(ValueError, "storage cap"):
                budget.check(3)
            with patch("audiovae_student.acquire.shutil.disk_usage", return_value=SimpleNamespace(free=12)):
                budget = AudioStorageBudget(root, min_free_bytes=10)
                with self.assertRaisesRegex(ValueError, "reserve"):
                    budget.check(3)
            self.assertEqual(existing.read_bytes(), b"existing")

    def test_resume_reuses_pinned_metadata_and_completed_source_without_network(self):
        tsv = b"1\ta.wav\ttext\ttext\tchars\t48000\tMALE\n"
        archive = archive_payload([("a.wav", audio_payload())])
        def opener(url):
            return io.BytesIO(tsv if url.endswith(".tsv") else archive)
        with tempfile.TemporaryDirectory() as directory:
            root, audio = Path(directory) / "durable", Path(directory) / "scratch"
            kwargs = dict(root=root, audio_root=audio, fleurs_languages=["hi_in"], minutes_per_language=0.01,
                librispeech_train_minutes=0, librispeech_dev_minutes=0)
            with patch("audiovae_student.acquire._open_url", side_effect=opener):
                first = acquire_bootstrap(**kwargs)
            with patch("audiovae_student.acquire._open_url") as request:
                resumed = acquire_bootstrap(**kwargs, resume=True)
                request.assert_not_called()
            self.assertEqual(first["nonoverlapping_windows"], resumed["nonoverlapping_windows"])
            self.assertEqual(resumed["state"], "ready")
            self.assertTrue(Path(load_manifest(root / "train.jsonl")[0].audio_path).is_relative_to(audio.resolve()))
            with self.assertRaisesRegex(ValueError, "same saved acquisition"):
                acquire_bootstrap(**{**kwargs, "minutes_per_language": 1}, resume=True)

    def test_unmet_unique_window_gate_is_not_ready(self):
        tsv = b"1\ta.wav\ttext\ttext\tchars\t48000\tMALE\n"
        archive = archive_payload([("a.wav", audio_payload())])
        def opener(url):
            return io.BytesIO(tsv if url.endswith(".tsv") else archive)
        with tempfile.TemporaryDirectory() as directory, patch("audiovae_student.acquire._open_url", side_effect=opener):
            with self.assertRaisesRegex(ValueError, "below the required minimum"):
                acquire_bootstrap(directory, fleurs_languages=["hi_in"], minutes_per_language=0.01,
                    librispeech_train_minutes=0, librispeech_dev_minutes=0, minimum_unique_windows=2)
            report = json.loads((Path(directory) / "provenance/sources.json").read_text())
            self.assertEqual(report["state"], "insufficient_unique_data")
            self.assertEqual(report["nonoverlapping_windows"]["windows"], 1)

    def test_train_other_500_is_supported_with_real_partition_and_identity(self):
        archive = archive_payload([("LibriSpeech/train-other-500/19/198/19-198-0001.flac", audio_payload(format="FLAC"))])
        with tempfile.TemporaryDirectory() as directory:
            rows, _ = collect_archive(io.BytesIO(archive), dataset="librispeech", partition="train-other-500",
                language="en", root=Path(directory), seconds=1, source_url="https://example.test/train-other-500.tar.gz",
                budget=DownloadBudget(1000000))
            self.assertEqual(rows[0].source_split, "train-other-500")
            self.assertIn("d1a0fd59409feb2c614ce4d30c387708", rows[0].source_revision)

    def test_all_language_discovery_rejects_incomplete_inventory(self):
        with patch("audiovae_student.acquire._open_url", return_value=io.BytesIO(b'[{"type":"directory","path":"data/hi_in"}]')):
            with self.assertRaisesRegex(ValueError, "exactly 102"):
                discover_fleurs_languages()

    def test_parallel_languages_and_libri_preserve_order_bytes_and_main_thread_manifests(self):
        import audiovae_student.acquire as module
        barrier = threading.Barrier(4)
        lock = threading.Lock()
        active = peak = 0
        payloads = {
            "hi_in": archive_payload([("a.wav", audio_payload(3))]),
            "en_us": archive_payload([("b.wav", audio_payload(4))]),
            "train-clean-100": archive_payload([("LibriSpeech/train-clean-100/19-20-0001.flac", audio_payload(5, format="FLAC"))]),
            "train-other-500": archive_payload([("LibriSpeech/train-other-500/21-22-0001.flac", audio_payload(6, format="FLAC"))]),
        }
        def opener(url):
            nonlocal active, peak
            if url.endswith(".tsv"):
                return io.BytesIO(b"1\ta.wav\ttext\ttext\tchars\t48000\tMALE\n" if "/hi_in/" in url
                                 else b"1\tb.wav\ttext\ttext\tchars\t64000\tMALE\n")
            key = next(key for key in payloads if key in url)
            with lock:
                active += 1
                peak = max(peak, active)
            barrier.wait(timeout=5)
            class Response(io.BytesIO):
                def __exit__(self, *args):
                    nonlocal active
                    with lock:
                        active -= 1
                    return super().__exit__(*args)
            return Response(payloads[key])
        original_atomic = module._atomic_bytes
        main_thread = threading.get_ident()
        def checked_atomic(path, payload):
            if path.parent.name == "manifests" or path.name == "sources.json":
                self.assertEqual(threading.get_ident(), main_thread)
            original_atomic(path, payload)
        with tempfile.TemporaryDirectory() as directory:
            kwargs = dict(root=directory, fleurs_languages=["hi_in", "en_us"], minutes_per_language=0.01,
                          librispeech_train_minutes=0.01, librispeech_other_minutes=0.01,
                          librispeech_dev_minutes=0)
            with patch.object(module, "_open_url", side_effect=opener), patch.object(module, "_atomic_bytes", side_effect=checked_atomic), \
                    patch.object(module.time, "monotonic", side_effect=lambda: next(ticks)):
                ticks = itertools.count(0, 16)
                result = acquire_bootstrap(**kwargs, workers=4)
            self.assertEqual(peak, 4)
            rows = load_manifest(Path(directory) / "train.jsonl")
            self.assertEqual([row.source_id for row in rows], ["hi_in:train:a.wav", "en_us:train:b.wav", "19-20-0001", "21-22-0001"])
            self.assertEqual(result["execution"]["workers"], 4)
            self.assertNotIn("workers", result["acquisition_configuration"])
            self.assertEqual(result["audio_storage_bytes"], sum(Path(row.audio_path).stat().st_size for row in rows))
            progress = list((Path(directory) / "provenance/progress").glob("*.json"))
            self.assertEqual(len(progress), 4)
            self.assertTrue(all("total_downloaded_bytes" in json.loads(path.read_text()) for path in progress))
            with patch.object(module, "_open_url") as request:
                resumed = acquire_bootstrap(**kwargs, workers=1, resume=True)
                request.assert_not_called()
            self.assertEqual(resumed["nonoverlapping_windows"], result["nonoverlapping_windows"])
            self.assertEqual(resumed["downloaded_bytes"], result["downloaded_bytes"])

    def test_concurrent_reads_reserve_one_shared_transfer_cap(self):
        barrier = threading.Barrier(4)
        class Response:
            def read(self, count):
                barrier.wait(timeout=5)
                return b"x" * count
        budget = DownloadBudget(20, initial_bytes=1)
        with ThreadPoolExecutor(max_workers=4) as pool:
            values = list(pool.map(lambda _: budget.read(Response(), 6), range(4)))
        self.assertEqual(sum(map(len, values)), 19)
        self.assertEqual(budget.read_bytes, 20)
        with self.assertRaisesRegex(ValueError, "budget"):
            budget.read(io.BytesIO(b"x"), 1)

    def test_concurrent_writes_cannot_race_past_storage_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            budget = AudioStorageBudget(root, max_bytes=100)
            barrier = threading.Barrier(2)
            def write(index):
                payload = bytes([index]) * 60
                barrier.wait(timeout=5)
                try:
                    budget.store(root / f"{index}.wav", payload, hashlib.sha256(payload).hexdigest())
                    return True
                except ValueError as error:
                    self.assertIn("storage cap", str(error))
                    return False
            with ThreadPoolExecutor(max_workers=2) as pool:
                result = list(pool.map(write, (1, 2)))
            self.assertEqual(sum(result), 1)
            self.assertEqual(budget.used_bytes, 60)
            self.assertEqual(sum(path.stat().st_size for path in root.iterdir()), 60)
            existing = next(root.iterdir())
            original = existing.read_bytes()
            with self.assertRaisesRegex(ValueError, "different bytes"):
                budget.store(existing, b"replacement", hashlib.sha256(b"replacement").hexdigest())
            self.assertEqual(existing.read_bytes(), original)

    def test_failed_atomic_write_does_not_consume_storage_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            budget = AudioStorageBudget(root, max_bytes=3)
            with patch("audiovae_student.acquire._atomic_bytes", side_effect=OSError("disk write failed")):
                with self.assertRaisesRegex(OSError, "disk write failed"):
                    budget.store(root / "a.wav", b"abc", hashlib.sha256(b"abc").hexdigest())
            self.assertEqual(budget.used_bytes, 0)
            budget.store(root / "a.wav", b"abc", hashlib.sha256(b"abc").hexdigest())
            self.assertEqual(budget.used_bytes, 3)

    def test_failed_source_stops_pending_work_but_persists_completed_sibling(self):
        barrier = threading.Barrier(2)
        stop = threading.Event()
        recorded, started = [], []
        main_thread = threading.get_ident()
        def acquire(task):
            started.append(task)
            barrier.wait(timeout=5)
            if task == "fail":
                raise ValueError("broken source")
            self.assertTrue(stop.wait(timeout=5))
            return ["finished"], {"source": task}
        def record(rows, receipt):
            self.assertEqual(threading.get_ident(), main_thread)
            recorded.append((rows, receipt))
        with self.assertRaisesRegex(ValueError, "broken source"):
            _run_source_tasks(["fail", "complete", "must-not-start"], acquire, record,
                              workers=2, stop_event=stop)
        self.assertEqual(set(started), {"fail", "complete"})
        self.assertEqual(recorded, [(["finished"], {"source": "complete"})])
        self.assertTrue(stop.is_set())

    def test_resume_preserves_prior_transfer_and_legacy_incomplete_prefix_once(self):
        import audiovae_student.acquire as module
        tsvs = {"hi_in": b"1\ta.wav\ttext\ttext\tchars\t48000\tMALE\n",
                "en_us": b"1\tb.wav\ttext\ttext\tchars\t64000\tMALE\n"}
        payloads = {"hi_in": archive_payload([("a.wav", audio_payload(3))]),
                    "en_us": archive_payload([("b.wav", audio_payload(4))])}
        def opener(url):
            language = "hi_in" if "/hi_in/" in url else "en_us"
            if url.endswith(".tsv"):
                return io.BytesIO(tsvs[language])
            if language == "en_us":
                raise OSError("interrupted source")
            return io.BytesIO(payloads[language])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kwargs = dict(root=root, fleurs_languages=["hi_in", "en_us"], minutes_per_language=0.01,
                          librispeech_train_minutes=0, librispeech_dev_minutes=0)
            with patch.object(module, "_open_url", side_effect=opener):
                with self.assertRaisesRegex(OSError, "interrupted source"):
                    acquire_bootstrap(**kwargs)
            report_path = root / "provenance/sources.json"
            saved = json.loads(report_path.read_text())
            self.assertEqual(saved["state"], "failed")
            self.assertEqual(len(saved["source_receipts"]), 1)
            original = next((root / "audio").rglob("*.wav"))
            original_bytes = original.read_bytes()
            # Emulate the previous serial implementation's separate progress.
            saved.pop("execution")
            report_path.write_text(json.dumps(saved))
            (root / "provenance/download-progress.json").write_text(json.dumps({"dataset": "fleurs",
                "partition": "train", "language": "en_us", "downloaded_prefix_bytes": 123}))
            expected_prior = saved["downloaded_bytes"] + 123
            calls = []
            def resumed_opener(url):
                calls.append(url)
                self.assertIn("/en_us/", url)
                return io.BytesIO(payloads["en_us"])
            limit = expected_prior + len(payloads["en_us"]) // 2
            with patch.object(module, "_open_url", side_effect=resumed_opener):
                with self.assertRaisesRegex(ValueError, "budget"):
                    acquire_bootstrap(**kwargs, resume=True, workers=4, max_download_bytes=limit)
            failed = json.loads(report_path.read_text())
            self.assertEqual(failed["execution"]["prior_downloaded_bytes"], expected_prior)
            self.assertEqual(failed["downloaded_bytes"], limit)
            self.assertEqual(original.read_bytes(), original_bytes)
            with patch.object(module, "_open_url", side_effect=resumed_opener):
                result = acquire_bootstrap(**kwargs, resume=True, workers=2, max_download_bytes=10**7)
            self.assertEqual(result["execution"]["prior_downloaded_bytes"], limit)
            self.assertEqual(result["downloaded_bytes"], limit + len(payloads["en_us"]))
            self.assertEqual(result["state"], "ready")
            self.assertEqual(len(load_manifest(root / "train.jsonl")), 2)

    def test_workers_are_bounded_before_any_network_access(self):
        with tempfile.TemporaryDirectory() as directory, patch("audiovae_student.acquire._open_url") as request:
            for workers in (0, 17, True, 1.5):
                with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, "workers"):
                    acquire_bootstrap(directory, workers=workers)
            request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
