"""CPU-only source authentication and exact full-geometry replay fixtures."""
from dataclasses import replace
import hashlib
from pathlib import Path
import sys

import pytest
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "fast-audiovae/experiments/convnext"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "fast-audiovae/experiments/convnext/tests"))

from audiovae_student.cache import TrainingCrop
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.quiet_audio import QuietAudioConfig
from audiovae_student.teacher import SOURCE_SHA256, CHECKPOINT_SHA256
from full_source_teacher_head import (authenticated_source, capture_full_source_pre_tanh,
                                      crop_window, full_source_cache_key, tensor_bytes_sha)
from test_data import row


def fixture(tmp_path):
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Sequential(torch.nn.Upsample(scale_factor=1920, mode="nearest"),
                                            torch.nn.Conv1d(64, 1, 1), torch.nn.Tanh())

        def forward(self, z):
            return self.model(z)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = Decoder()

        def encode(self, audio, sample_rate):
            assert sample_rate == 16000
            assert not torch.backends.cudnn.enabled
            return audio[..., ::640].repeat(1, 64, 1).contiguous()

    class Teacher:
        def __init__(self):
            self.model = Model().eval()
            self.model.requires_grad_(False)
            self.device = torch.device("cpu")
            self.provenance = {"source_sha256": SOURCE_SHA256, "checkpoint_sha256": CHECKPOINT_SHA256}
            self.calls = []

        def decode(self, z):
            self.calls.append(tuple(z.shape))
            return self.model.decoder(z)

    teacher = Teacher()
    audio = torch.linspace(-.8, .8, 3200).reshape(1, 1, -1)
    path = tmp_path / "audio.wav"
    sf.write(path, audio[0, 0].numpy(), 16000, subtype="FLOAT")
    manifest = row("full-source", audio_path=str(path), audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   resampler_policy="none", gain_policy="unchanged", duration_seconds=.2)
    data = {"rows": {manifest.source_id: manifest.to_dict()}, "counts": {manifest.source_id: 3200}}
    inventory = {"samples16k": 3200, "source_audio_sha256": manifest.audio_sha256,
                 "prepared_audio_sha256": tensor_bytes_sha(audio)}
    before = state_fingerprint(teacher.model.state_dict())
    with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
        z = teacher.model.encode(audio, 16000)
        target = teacher.decode(z)
    key = full_source_cache_key(inventory, before, z, target)
    crop = TrainingCrop(crop_window(z, 1, 4), crop_window(target, 1920, 7680),
        crop_window(audio, 640, 2560), key, manifest.source_id, 2, 1, 1, 3, 5760)
    teacher.calls.clear()
    return teacher, crop, data, before, path


def test_full_source_encoder_decoder_cache_key_and_all_context_exact(tmp_path):
    torch.set_num_threads(1)
    teacher, crop, data, before, _ = fixture(tmp_path)
    result = capture_full_source_pre_tanh(teacher, crop, data, QuietAudioConfig(),
                                        expected_teacher_state_sha256=before)
    assert teacher.calls == [(1, 64, 5), (1, 64, 5)]
    assert torch.equal(result["post_tanh"], crop.teacher_audio)
    assert torch.equal(torch.tanh(result["pre_tanh"]), crop.teacher_audio)
    assert result["receipt"]["cache_key_recreated"] == crop.cache_key
    assert result["receipt"]["entire_cached_latents_exact"]
    assert result["receipt"]["entire_cached_post_target_exact"]
    assert result["receipt"]["scored_samples"] == 5754
    assert state_fingerprint(teacher.model.state_dict()) == before
    assert not teacher.model.decoder.model[-2]._forward_hooks


def test_source_and_reference_authentication_fail_before_model_calls(tmp_path):
    teacher, crop, data, before, path = fixture(tmp_path)
    bad = replace(crop, reference16k=crop.reference16k + .01)
    with pytest.raises(ValueError, match="reference"):
        capture_full_source_pre_tanh(teacher, bad, data, QuietAudioConfig(), expected_teacher_state_sha256=before)
    assert teacher.calls == []
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="file hash"):
        authenticated_source(data, crop)


def test_exact_latent_target_and_full_cache_key_failures(tmp_path):
    teacher, crop, data, before, _ = fixture(tmp_path)
    with pytest.raises(RuntimeError, match="latents versus cache"):
        capture_full_source_pre_tanh(teacher, replace(crop, latents=crop.latents + .001), data,
                                    QuietAudioConfig(), expected_teacher_state_sha256=before)
    with pytest.raises(RuntimeError, match="cache key"):
        capture_full_source_pre_tanh(teacher, replace(crop, cache_key="0" * 64), data,
                                    QuietAudioConfig(), expected_teacher_state_sha256=before)
    target = crop.teacher_audio.clone()
    target[..., 0] += 1e-6  # Unscored context is still required to match exactly.
    with pytest.raises(RuntimeError, match="context/padding"):
        capture_full_source_pre_tanh(teacher, replace(crop, teacher_audio=target), data,
                                    QuietAudioConfig(), expected_teacher_state_sha256=before)
    assert not teacher.model.decoder.model[-2]._forward_hooks
    assert state_fingerprint(teacher.model.state_dict()) == before
