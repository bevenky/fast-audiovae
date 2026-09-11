import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from audiovae_student.data import (
    ManifestRow, ManifestValidationError, build_training_manifest, load_manifest,
    main, summarize_manifest, validate_manifest, validate_mixture,
)


def row(source_id="a", **overrides):
    value = {
        "dataset": "fleurs", "source_revision": "pinned-release-1", "source_id": source_id,
        "source_url": f"https://example.test/{source_id}", "audio_path": f"audio/{source_id}.wav",
        "audio_sha256": hashlib.sha256(source_id.encode()).hexdigest(),
        "parent_recording_id": f"fleurs:recording:{source_id}", "parent_start_seconds": 0,
        "speaker_id": f"fleurs:speaker:{source_id}", "session_id": f"fleurs:session:{source_id}",
        "language": "hi", "sample_rate_hz": 16000, "original_sample_rate_hz": 16000,
        "bandwidth_hz": 7600, "bandwidth_class": "speech_band",
        "bandwidth_evidence": "source documentation: original 16 kHz release; conservative 7.6 kHz loss limit",
        "native_recording": True, "enhanced": False, "duration_seconds": 10,
        "split": "train", "source_split": "train", "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/", "attribution": "Dataset authors",
        "access_record": "provenance/source-terms.json", "gain_policy": "unchanged-v1",
        "resampler_policy": "pinned-resampler-v1", "teacher_cache_key": None,
    }
    value.update(overrides)
    return ManifestRow.from_dict(value)


class DataTests(unittest.TestCase):
    def test_approved_quotas_and_bandwidth_totals(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "data-mixture.json"
        config = json.loads(path.read_text())
        validate_mixture(config)
        self.assertEqual(sum(item["pilot_hours"] for item in config["sources"]), 100)
        self.assertEqual(sum(item["main_hours"] for item in config["sources"]), 1000)
        self.assertEqual(sum(config["main_bandwidth_allocation_hours"].values()), 1000)
        self.assertNotIn("gigaspeech", [item["dataset"] for item in config["sources"]])
        config["sources"][0]["pilot_hours"] = 11
        with self.assertRaisesRegex(ManifestValidationError, "pilot quotas"):
            validate_mixture(config)

    def test_roundtrip_and_summary_without_audio_access(self):
        original = row()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(json.dumps(original.to_dict()) + "\n")
            loaded = load_manifest(path)
        self.assertEqual(loaded, [original])
        self.assertAlmostEqual(summarize_manifest(loaded)["hours"], 10 / 3600)
        self.assertEqual(summarize_manifest(loaded)["cached_teacher_rows"], 0)

    def test_every_identity_protects_reserved_clips_even_with_stale_split(self):
        reserved = row("reserved", split="train")
        for key in ("parent_recording_id", "audio_sha256", "speaker_id", "session_id"):
            with self.subTest(key=key), self.assertRaisesRegex(ManifestValidationError, "leakage"):
                build_training_manifest([row("candidate", **{key: getattr(reserved, key)})], reserved_rows=[reserved])

    def test_cross_split_speaker_and_session_leakage(self):
        for key in ("speaker_id", "session_id"):
            train = row()
            test = row("b", split="test", source_split="test", **{key: getattr(train, key)})
            with self.subTest(key=key), self.assertRaisesRegex(ManifestValidationError, "leakage"):
                build_training_manifest([train, test])

    def test_fleurs_test_relabeling_and_forbidden_source(self):
        for overrides, message in (
            ({"source_split": "test"}, "held-out"),
            ({"dataset": "Giga-Speech", "license": "Apache-2.0"}, "GigaSpeech"),
            ({"source_url": "https://huggingface.co/datasets/speechcolab/gigaspeech"}, "GigaSpeech"),
            ({"license": "CC-BY-NC-4.0"}, "unexpected data license"),
            ({"dataset": "unreviewed"}, "unreviewed"),
        ):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ManifestValidationError, message):
                build_training_manifest([row(**overrides)])

    def test_original_and_restored_overlap_cannot_count_twice(self):
        original = row(dataset="indicvoices", sample_rate_hz=44100, original_sample_rate_hz=44100,
                       bandwidth_hz=18000, bandwidth_class="native_fullband")
        restored = row("restored", dataset="indicvoices_r", sample_rate_hz=48000,
                       original_sample_rate_hz=44100, bandwidth_hz=18000, bandwidth_class="enhanced",
                       native_recording=False, enhanced=True, parent_recording_id=original.parent_recording_id)
        with self.assertRaisesRegex(ManifestValidationError, "overlapping training segments"):
            build_training_manifest([original, restored])

    def test_microphone_copy_and_duplicate_file_rejected(self):
        first = row()
        with self.assertRaisesRegex(ManifestValidationError, "overlapping training segments"):
            validate_manifest([first, row("mic2", parent_recording_id=first.parent_recording_id)])
        with self.assertRaisesRegex(ManifestValidationError, "duplicate training audio"):
            validate_manifest([first, row("copy", audio_sha256=first.audio_sha256)])

    def test_adjacent_segments_are_allowed_but_not_split_leakage(self):
        first = row()
        second = row("second", parent_recording_id=first.parent_recording_id, parent_start_seconds=10)
        validate_manifest([first, second], training_only=True)
        heldout = row("heldout", parent_recording_id=first.parent_recording_id,
                      parent_start_seconds=20, split="dev", source_split="dev")
        with self.assertRaisesRegex(ManifestValidationError, "parent leakage"):
            validate_manifest([first, heldout])

    def test_upsampling_and_headers_do_not_establish_fullband(self):
        for overrides in (
            {"sample_rate_hz": 48000, "bandwidth_hz": 18000, "bandwidth_class": "native_fullband"},
            {"sample_rate_hz": 48000, "original_sample_rate_hz": 48000,
             "bandwidth_hz": 18000, "bandwidth_class": "native_fullband", "bandwidth_evidence": "header"},
            {"sample_rate_hz": 48000, "original_sample_rate_hz": 48000,
             "bandwidth_hz": 18000, "bandwidth_class": "native_fullband", "native_recording": False},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ManifestValidationError):
                row(**overrides)

    def test_enhancement_and_metadata_are_explicit(self):
        for overrides in (
            {"enhanced": True}, {"duration_seconds": float("nan")}, {"sample_rate_hz": True},
            {"speaker_id": None, "session_id": None}, {"source_revision": "main"},
            {"audio_sha256": "not-a-hash"}, {"teacher_cache_key": "unpinned"},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ManifestValidationError):
                row(**overrides)

    def test_enhancement_can_generate_bandwidth_but_never_be_native(self):
        restored = row("restored", dataset="indicvoices_r", sample_rate_hz=48000,
                       original_sample_rate_hz=16000, bandwidth_hz=18000,
                       bandwidth_class="enhanced", native_recording=False, enhanced=True)
        validate_manifest([restored], training_only=True)
        self.assertEqual(summarize_manifest([restored])["hours_by"]["bandwidth_class"].keys(), {"enhanced"})

    def test_dataset_allocations_preserve_language_selection(self):
        for overrides in (
            {"dataset": "common_voice_scripted_26_en", "language": "hi", "license": "CC0-1.0"},
            {"dataset": "voxpopuli_en", "language": "de", "license": "CC0-1.0"},
            {"dataset": "mls_non_english", "language": "en-US"},
        ):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ManifestValidationError, "English"):
                build_training_manifest([row(**overrides)])

    def test_training_builder_filters_only_after_complete_leakage_check(self):
        train, test = row(), row("test", split="test", source_split="test")
        self.assertEqual(build_training_manifest([train, test]), [train])
        with self.assertRaisesRegex(ManifestValidationError, "training-only"):
            validate_manifest([train, test], training_only=True)

    def test_jsonl_diagnostic_includes_line_number(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.jsonl"
            path.write_text(json.dumps(row().to_dict()) + "\n{}\n")
            with self.assertRaisesRegex(ManifestValidationError, ":2: schema mismatch"):
                load_manifest(path)

    def test_cli_reserved_manifest_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path, reserved_path = Path(directory) / "train.jsonl", Path(directory) / "reserved.jsonl"
            path.write_text(json.dumps(row().to_dict()) + "\n")
            reserved_path.write_text(json.dumps(row("other", speaker_id=row().speaker_id, split="test").to_dict()) + "\n")
            with self.assertRaises(SystemExit) as error:
                main(["validate", str(path), "--reserved-manifest", str(reserved_path), "--training-only"])
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
