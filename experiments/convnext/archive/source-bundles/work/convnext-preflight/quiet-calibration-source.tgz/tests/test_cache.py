"""Synthetic fixtures only. No benchmark recordings, real models or GPU use."""

from dataclasses import replace
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from audiovae_student.cache import (
    load_cache, prepare_utterance_cache, sample_training_crop, save_cache,
)
from audiovae_student.data import ManifestRow, ManifestValidationError
from audiovae_student.teacher import FrozenAudioVAE2


class ContinuousStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.encode_calls = 0
        self.decode_calls = 0

    def encode(self, audio, sample_rate):
        self.encode_calls += 1
        padded = F.pad(audio, (0, (-audio.shape[-1]) % 640))
        frame_means = padded.reshape(1, 1, -1, 640).mean(-1)
        # Context-dependent output detects accidental independent crop encoding.
        return frame_means.cumsum(-1).expand(1, 64, -1).contiguous()

    def decode(self, z, sr_cond):
        self.decode_calls += 1
        return z[:, :1].cumsum(-1).repeat_interleave(1920, dim=-1)


def teacher():
    provenance = {
        "config": {"test_fixture": True},
        "source_sha256": hashlib.sha256(b"synthetic source").hexdigest(),
        "checkpoint_sha256": hashlib.sha256(b"synthetic weights").hexdigest(),
        "posterior": "raw_mu", "sample_rate_in": 16000, "sample_rate_out": 48000,
        "latent_channels": 64, "encoder_hop": 640, "decoder_hop": 1920,
        "sr_cond": 48000, "dtype": "float32", "test_fixture": True,
    }
    return FrozenAudioVAE2(ContinuousStub(), provenance)


def source(length, **overrides):
    # A fictional source identity exercises the reviewed FLEURS policy without
    # loading, modifying or training on any real dataset or benchmark clip.
    values = {
        "dataset": "fleurs", "source_revision": "synthetic-fixture-only", "source_id": "synthetic-a",
        "source_url": "https://example.test/synthetic", "audio_path": "synthetic-never-read.wav",
        "audio_sha256": hashlib.sha256(b"synthetic source bytes").hexdigest(),
        "parent_recording_id": "synthetic:parent:a", "parent_start_seconds": 0,
        "speaker_id": "synthetic:speaker:a", "session_id": "synthetic:session:a", "language": "hi",
        "sample_rate_hz": 16000, "original_sample_rate_hz": 16000, "bandwidth_hz": 7600,
        "bandwidth_class": "speech_band", "bandwidth_evidence": "synthetic fixture, speech-band-only loss",
        "native_recording": True, "enhanced": False, "duration_seconds": length / 16000,
        "split": "train", "source_split": "train", "license": "CC-BY-4.0",
        "license_url": "https://example.test/fixture-license", "attribution": "Synthetic unit test",
        "access_record": "synthetic fixture only; no access event", "gain_policy": "unchanged",
        "resampler_policy": "none: already 16k", "teacher_cache_key": None,
    }
    values.update(overrides)
    return ManifestRow.from_dict(values)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.old_threads)

    def test_continuous_targets_generated_once_before_cropping(self):
        model = teacher()
        audio = torch.arange(35 * 640, dtype=torch.float32).reshape(1, 1, -1) / 100000
        cache = prepare_utterance_cache(audio, source(audio.shape[-1]), model)
        crop = sample_training_crop(cache, 30, scored_frames=2)
        self.assertEqual(model.model.encode_calls, 1)
        self.assertEqual(model.model.decode_calls, 1)
        self.assertEqual(crop.context_frames, 29)
        self.assertEqual(crop.context_start_frame, 1)
        torch.testing.assert_close(crop.latents, cache.latents[..., 1:32], atol=0, rtol=0)
        torch.testing.assert_close(crop.teacher_audio[..., crop.scored_slice], cache.teacher_audio[..., 30 * 1920:32 * 1920], atol=0, rtol=0)
        self.assertEqual(crop.loss_mask().sum().item(), 2 * 1920)

    def test_real_start_has_no_fabricated_left_context(self):
        audio = torch.ones(1, 1, 3 * 640)
        cache = prepare_utterance_cache(audio, source(audio.shape[-1]), teacher())
        crop = sample_training_crop(cache, 0, scored_frames=2)
        self.assertEqual(crop.context_frames, 0)
        self.assertTrue(crop.starts_at_utterance_start)
        self.assertEqual(crop.latents.shape[-1], 2)
        self.assertEqual(crop.scored_slice, slice(0, 3840))
        crop = sample_training_crop(cache, 1, scored_frames=2)
        self.assertEqual(crop.context_frames, 1)
        self.assertEqual(crop.scored_slice, slice(1920, 5760))

    def test_partial_tail_excludes_context_and_all_padding(self):
        audio = torch.ones(1, 1, 641)
        cache = prepare_utterance_cache(audio, source(641), teacher())
        self.assertEqual(cache.teacher_audio.shape[-1], 3840)
        self.assertEqual(cache.valid_output_samples, 1923)
        crop = sample_training_crop(cache, 1, scored_frames=4)
        self.assertEqual(crop.latents.shape[-1], 5)
        self.assertEqual(crop.teacher_audio.shape[-1], 9600)
        self.assertEqual(crop.valid_scored_samples, 3)
        self.assertEqual(crop.scored_slice, slice(1920, 1923))
        self.assertEqual(crop.reference_scored_slice, slice(640, 641))
        self.assertEqual(crop.loss_mask().sum().item(), 3)
        self.assertEqual(torch.count_nonzero(crop.latents[..., 2:]).item(), 0)
        self.assertEqual(torch.count_nonzero(crop.reference16k[..., 641:]).item(), 0)
        torch.testing.assert_close(crop.teacher_audio[..., crop.scored_slice], cache.teacher_audio[..., 1920:1923], atol=0, rtol=0)

    def test_atomic_roundtrip_weights_only_and_no_overwrite(self):
        cache = prepare_utterance_cache(torch.ones(1, 1, 641), source(641), teacher())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.pt"
            save_cache(cache, path)
            with patch("audiovae_student.cache.torch.load", wraps=torch.load) as loader:
                actual = load_cache(path, expected_cache_key=cache.cache_key)
                self.assertTrue(loader.call_args.kwargs["weights_only"])
            torch.testing.assert_close(actual.latents, cache.latents, atol=0, rtol=0)
            self.assertEqual(actual.metadata, cache.metadata)
            with self.assertRaises(FileExistsError):
                save_cache(cache, path)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])
            with self.assertRaisesRegex(ValueError, "requested cache key"):
                load_cache(path, expected_cache_key="wrong")
            save_cache(cache, path, overwrite=True)

    def test_mutated_payload_or_metadata_rejected(self):
        cache = prepare_utterance_cache(torch.ones(1, 1, 640), source(640), teacher())
        altered = replace(cache, teacher_audio=cache.teacher_audio + 1)
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            altered.validate()
        metadata = deepcopy(cache.metadata)
        metadata["identity"]["source"]["gain_policy"] = "changed"
        with self.assertRaisesRegex(ValueError, "cache key"):
            replace(cache, metadata=metadata).validate()

    def test_input_and_teacher_identity_change_keys(self):
        audio = torch.ones(1, 1, 640)
        first = prepare_utterance_cache(audio, source(640), teacher())
        changed_audio = prepare_utterance_cache(audio * 2, source(640), teacher())
        changed_source = prepare_utterance_cache(audio, source(640, resampler_policy="different-version"), teacher())
        model = teacher()
        model._provenance["config"]["test_revision"] = 2
        changed_teacher = prepare_utterance_cache(audio, source(640), model)
        self.assertEqual(len({record.cache_key for record in (first, changed_audio, changed_source, changed_teacher)}), 4)
        with self.assertRaisesRegex(ValueError, "teacher_cache_key"):
            prepare_utterance_cache(audio * 2, source(640, teacher_cache_key=first.cache_key), teacher())
        repeated = prepare_utterance_cache(audio, source(640, teacher_cache_key=first.cache_key), teacher())
        self.assertEqual(repeated.cache_key, first.cache_key)

    def test_eval_or_reserved_recordings_cannot_enter_training(self):
        audio = torch.ones(1, 1, 640)
        reserved = source(640, split="regression", source_split="test")
        with self.assertRaisesRegex(ManifestValidationError, "leakage"):
            prepare_utterance_cache(audio, source(640), teacher(), reserved_rows=[reserved])
        evaluation = prepare_utterance_cache(audio, reserved, teacher())
        with self.assertRaisesRegex(ValueError, "regression cache"):
            sample_training_crop(evaluation, 0)

    def test_caller_mutation_does_not_change_record_or_crop(self):
        audio = torch.ones(1, 1, 640)
        cache = prepare_utterance_cache(audio, source(640), teacher())
        crop = sample_training_crop(cache, 0)
        audio.zero_()
        crop.latents.zero_()
        cache.validate()
        self.assertEqual(cache.reference16k.sum().item(), 640)
        self.assertNotEqual(cache.latents.sum().item(), 0)

    def test_optional_reference_and_invalid_inputs(self):
        audio = torch.ones(1, 1, 640)
        cache = prepare_utterance_cache(audio, source(640), teacher(), include_reference=False)
        self.assertIsNone(cache.reference16k)
        self.assertIsNone(sample_training_crop(cache, 0).reference16k)
        for start, scored, context in ((-1, 64, 29), (1, 64, 29), (0, 0, 29), (0, 64, 28), (True, 64, 29)):
            with self.subTest(values=(start, scored, context)), self.assertRaises(ValueError):
                sample_training_crop(cache, start, scored_frames=scored, context_frames=context)
        for value in (audio.half(), torch.zeros(640), torch.empty(1, 1, 0), torch.full_like(audio, float("nan"))):
            with self.subTest(shape=value.shape), self.assertRaises(ValueError):
                prepare_utterance_cache(value, source(640), teacher())
        with self.assertRaisesRegex(ValueError, "duration"):
            prepare_utterance_cache(audio, source(1280), teacher())


if __name__ == "__main__":
    unittest.main()
