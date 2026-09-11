"""Synthetic audio and frozen stub teacher only; no data/model download or GPU."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F

from audiovae_student.cache import load_cache
from audiovae_student.data import ManifestRow
from audiovae_student.prepare_targets import prepare_targets, read_verified_audio, select_rows
from audiovae_student.teacher import FrozenAudioVAE2


class CountingTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.calls = 0

    def encode(self, audio, sample_rate):
        self.calls += 1
        padded = F.pad(audio, (0, (-audio.shape[-1]) % 640))
        return padded.reshape(1, 1, -1, 640).mean(-1).cumsum(-1).expand(1, 64, -1).contiguous()

    def decode(self, z, sr_cond):
        return z[:, :1].repeat_interleave(1920, dim=-1)


def teacher():
    return FrozenAudioVAE2(CountingTeacher(), {
        "config": {"synthetic_fixture": True},
        "source_sha256": "a" * 64, "checkpoint_sha256": "b" * 64,
        "posterior": "raw_mu", "sample_rate_in": 16000, "sample_rate_out": 48000,
        "latent_channels": 64, "encoder_hop": 640, "decoder_hop": 1920,
        "sr_cond": 48000, "dtype": "float32",
    })


def source(directory, identifier, split="train", length=1921, rate=16000, channels=1):
    audio = np.linspace(-0.01, 0.02 + int(identifier) / 1000, length, dtype=np.float32)
    if channels != 1:
        audio = np.repeat(audio[:, None], channels, axis=1)
    path = directory / f"{identifier}.wav"
    sf.write(path, audio, rate, subtype="FLOAT")
    return ManifestRow.from_dict({
        "dataset": "fleurs", "source_revision": "synthetic-test-v1", "source_id": str(identifier),
        "source_url": "https://example.test/synthetic", "audio_path": path.name,
        "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "parent_recording_id": f"synthetic:parent:{identifier}", "parent_start_seconds": 0,
        "speaker_id": f"synthetic:speaker:{identifier}", "session_id": f"synthetic:session:{identifier}",
        "language": "hi" if split == "train" else "en", "sample_rate_hz": rate,
        "original_sample_rate_hz": rate, "bandwidth_hz": min(7600, rate / 2),
        "bandwidth_class": "speech_band", "bandwidth_evidence": "synthetic test wave, no real source",
        "native_recording": True, "enhanced": False, "duration_seconds": length / rate,
        "split": split, "source_split": "train", "license": "CC-BY-4.0",
        "license_url": "https://example.test/license", "attribution": "Synthetic test fixture",
        "access_record": "synthetic test only", "gain_policy": "none: preserve original amplitude",
        "resampler_policy": "none: original mono 16000 Hz", "teacher_cache_key": None,
    })


def manifest(directory, rows):
    path = directory / "manifest.jsonl"
    path.write_text("".join(json.dumps(row.to_dict()) + "\n" for row in rows))
    return path


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()
        torch.set_num_threads(self.old_threads)

    def test_file_hash_rate_channels_duration_and_policy_are_checked(self):
        row = source(self.directory, 1)
        audio = read_verified_audio(row, manifest_directory=self.directory)
        expected, _ = sf.read(self.directory / row.audio_path, dtype="float32")
        np.testing.assert_array_equal(audio.numpy().reshape(-1), expected)
        self.assertEqual(audio.shape, (1, 1, 1921))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            read_verified_audio(replace(row, audio_sha256="0" * 64), manifest_directory=self.directory)
        with self.assertRaisesRegex(ValueError, "duration"):
            read_verified_audio(replace(row, duration_seconds=1), manifest_directory=self.directory)
        with self.assertRaisesRegex(ValueError, "unsupported preprocessing"):
            read_verified_audio(replace(row, gain_policy="normalize"), manifest_directory=self.directory)
        with self.assertRaisesRegex(ValueError, "duration limit"):
            read_verified_audio(row, manifest_directory=self.directory, max_utterance_seconds=0.01)
        with self.assertRaisesRegex(ValueError, "requires original mono"):
            read_verified_audio(source(self.directory, 2, rate=24000), manifest_directory=self.directory)
        with self.assertRaisesRegex(ValueError, "actual file must be mono"):
            read_verified_audio(source(self.directory, 3, channels=2), manifest_directory=self.directory)

    def test_bound_reserves_dev_and_never_crops_utterances(self):
        rows = [source(self.directory, n) for n in range(3)] + [source(self.directory, 4, split="dev")]
        selected = select_rows(rows, max_records=2, max_hours=None)
        self.assertEqual([row.source_id for row in selected], ["0", "4"])
        self.assertEqual(len(select_rows(rows, max_records=None, max_hours=2 * rows[0].duration_seconds / 3600)), 2)
        with self.assertRaisesRegex(ValueError, "initial complete"):
            select_rows(rows, max_records=None, max_hours=rows[0].duration_seconds / 3600)
        with self.assertRaisesRegex(ValueError, "bound target"):
            select_rows(rows, max_records=None, max_hours=None)

    def test_continuous_cache_index_and_resume_without_target_reexecution(self):
        rows = [source(self.directory, 1), source(self.directory, 2, split="dev")]
        path, model = manifest(self.directory, rows), teacher()
        index_path, output_dir = self.directory / "index.json", self.directory / "cache"
        summary = prepare_targets(path, model, output_dir=output_dir, index_path=index_path, max_records=2)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["generated_records"], 2)
        self.assertEqual(model.model.calls, 4)  # Two startup passes, two retained utterances.
        policy = summary["teacher"]["target_preparation_policy"]
        self.assertEqual(policy["warmup_encode_decode_passes"], 2)
        self.assertEqual(policy["warmup_source_sha256"], rows[0].audio_sha256)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
        self.assertFalse(model.model.training)
        index = json.loads(index_path.read_text())
        self.assertEqual(set(index), {"format_version", "train", "dev"})
        for split in ("train", "dev"):
            self.assertEqual(len(index[split]), 1)
            entry = index[split][0]
            self.assertFalse(Path(entry["path"]).is_absolute())
            record = load_cache(self.directory / entry["path"], expected_cache_key=entry["cache_key"])
            self.assertEqual(record.metadata["identity"]["source"]["split"], split)
            self.assertEqual(record.input_samples, 1921)
            self.assertEqual(record.teacher_audio.shape[-1], 4 * 1920)
            self.assertEqual(record.valid_output_samples, 3 * 1921)
        resumed = prepare_targets(path, model, output_dir=output_dir, index_path=index_path, max_records=2)
        self.assertEqual(resumed["resumed_records"], 2)
        self.assertEqual(resumed["generated_records"], 0)
        self.assertEqual(model.model.calls, 6)  # Resume repeats startup only, not cached targets.
        self.assertEqual(json.loads(index_path.read_text()), index)

    def test_interrupted_run_keeps_complete_index_and_resumes(self):
        rows = [source(self.directory, 1), source(self.directory, 2, split="dev")]
        path, model = manifest(self.directory, rows), teacher()
        index_path, output_dir = self.directory / "index.json", self.directory / "cache"

        def interrupt(_event):
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            prepare_targets(path, model, output_dir=output_dir, index_path=index_path, max_records=2, progress=interrupt)
        index = json.loads(index_path.read_text())
        self.assertEqual(len(index["train"]), 1)
        self.assertEqual(index["dev"], [])
        self.assertEqual(list(self.directory.rglob("*.tmp")), [])
        summary = prepare_targets(path, model, output_dir=output_dir, index_path=index_path, max_records=2)
        self.assertEqual(summary["generated_records"], 1)
        self.assertEqual(summary["resumed_records"], 1)
        self.assertEqual(model.model.calls, 6)

    def test_interruption_between_cache_and_index_recovers_orphan(self):
        rows = [source(self.directory, 1), source(self.directory, 2, split="dev")]
        path, model = manifest(self.directory, rows), teacher()
        index_path, output_dir = self.directory / "index.json", self.directory / "cache"
        from audiovae_student import prepare_targets as prep
        original = prep._atomic_json

        def interrupt_on_index(path, value):
            if path == index_path.resolve():
                raise KeyboardInterrupt
            return original(path, value)

        with patch.object(prep, "_atomic_json", side_effect=interrupt_on_index):
            with self.assertRaises(KeyboardInterrupt):
                prepare_targets(path, model, output_dir=output_dir, index_path=index_path, max_records=2)
        self.assertFalse(index_path.exists())
        self.assertEqual(len(list(output_dir.glob("*.pt"))), 1)
        summary = prepare_targets(path, model, output_dir=output_dir, index_path=index_path, max_records=2)
        self.assertEqual(summary["resumed_records"], 1)
        self.assertEqual(model.model.calls, 6)

    def test_corrupt_cache_and_changed_source_fail_closed(self):
        rows = [source(self.directory, 1), source(self.directory, 2, split="dev")]
        path, model = manifest(self.directory, rows), teacher()
        index_path, output_dir = self.directory / "index.json", self.directory / "cache"
        prepare_targets(path, model, output_dir=output_dir, index_path=index_path, max_records=2)
        (self.directory / rows[0].audio_path).write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "source audio SHA-256"):
            prepare_targets(path, model, output_dir=output_dir, index_path=index_path, max_records=2)
        entry = json.loads(index_path.read_text())["train"][0]
        cache_path = self.directory / entry["path"]
        payload = torch.load(cache_path, weights_only=True)
        payload["teacher_audio"][..., 0] += 1
        torch.save(payload, cache_path)
        with self.assertRaisesRegex(ValueError, "tensor SHA-256"):
            prepare_targets(path, model, output_dir=output_dir, index_path=index_path, max_records=2)
        self.assertEqual(model.model.calls, 4)

    def test_split_leakage_fails_before_teacher_runs(self):
        first, second = source(self.directory, 1), source(self.directory, 2, split="dev")
        second = replace(second, speaker_id=first.speaker_id)
        path, model = manifest(self.directory, [first, second]), teacher()
        with self.assertRaisesRegex(ValueError, "speaker leakage"):
            prepare_targets(path, model, output_dir=self.directory / "cache",
                            index_path=self.directory / "index.json", max_records=2)
        self.assertEqual(model.model.calls, 0)


if __name__ == "__main__":
    unittest.main()
