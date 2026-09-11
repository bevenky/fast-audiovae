"""Teacher integrity and interface checks, with optional pinned-asset coverage.

Set AUDIOVAE2_SOURCE and AUDIOVAE2_CHECKPOINT to existing local files to enable
the original model integration check. Tests never download data or use a GPU.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from audiovae_student import teacher as teacher_module
from audiovae_student.teacher import FrozenAudioVAE2


class StubAudioVAE2(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.register_buffer("running_value", torch.tensor(1.0))
        self.last_sample_rate = None
        self.last_cond = None

    def encode(self, audio, sample_rate):
        self.last_sample_rate = sample_rate
        padded = F.pad(audio, (0, (-audio.shape[-1]) % 640))
        mu = padded.reshape(audio.shape[0], 1, -1, 640).mean(-1) * self.weight
        return mu.expand(-1, 64, -1).contiguous()

    def decode(self, z, sr_cond):
        self.last_cond = sr_cond
        return z[:, :1].repeat_interleave(1920, dim=-1) * self.weight


class TeacherContractTests(unittest.TestCase):
    def setUp(self):
        self.teacher = FrozenAudioVAE2(StubAudioVAE2(), {"config": {"nested": [1]}})

    def test_sample_accounting_and_explicit_conditioning(self):
        for length in (0, 1, 639, 640, 641, 1279, 1280, 1281):
            with self.subTest(length=length):
                audio = torch.ones(2, 1, length)
                z = self.teacher.encode(audio)
                frames = (length + 639) // 640
                self.assertEqual(tuple(z.shape), (2, 64, frames))
                raw = self.teacher.decode(z)
                self.assertEqual(tuple(raw.shape), (2, 1, 1920 * frames))
                trimmed = self.teacher.reconstruct(audio)
                self.assertEqual(tuple(trimmed.shape), (2, 1, 3 * length))
                torch.testing.assert_close(trimmed, raw[..., :3 * length], atol=0, rtol=0)
        self.assertEqual(self.teacher.model.last_sample_rate, 16000)
        torch.testing.assert_close(self.teacher.model.last_cond, torch.tensor([48000, 48000], dtype=torch.int32))

    def test_mu_is_not_scaled_sampled_or_normalized(self):
        audio = torch.full((1, 1, 641), 0.25)
        z = self.teacher.encode(audio)
        expected = torch.tensor([0.5, 0.5 / 640]).reshape(1, 1, 2).expand(1, 64, 2)
        torch.testing.assert_close(z, expected, atol=0, rtol=0)

    def test_parent_training_mode_does_not_enable_teacher_training(self):
        nn.Sequential(self.teacher).train()
        self.assertFalse(self.teacher.training)
        self.assertTrue(all(not m.training for m in self.teacher.modules()))
        self.assertTrue(all(not p.requires_grad for p in self.teacher.parameters()))

    def test_targets_can_be_used_by_student_autograd_under_autocast(self):
        audio = torch.ones(1, 1, 640, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            z = self.teacher.encode(audio)
            target = self.teacher.decode(z)
        self.assertEqual(z.dtype, torch.float32)
        self.assertEqual(target.dtype, torch.float32)
        self.assertFalse(z.requires_grad)
        self.assertFalse(z.is_inference())
        student_weight = nn.Parameter(torch.tensor(1.0))
        ((z * student_weight).square().mean() + (target * student_weight).square().mean()).backward()
        self.assertIsNotNone(student_weight.grad)
        self.assertIsNone(audio.grad)
        self.assertTrue(all(p.grad is None for p in self.teacher.parameters()))

    def test_rejects_invalid_inputs(self):
        for value in (torch.zeros(1, 640), torch.zeros(1, 2, 640), torch.zeros(0, 1, 640)):
            with self.subTest(shape=tuple(value.shape)), self.assertRaises(ValueError):
                self.teacher.encode(value)
        for dtype in (torch.float16, torch.float64, torch.int32):
            with self.subTest(dtype=dtype), self.assertRaises(TypeError):
                self.teacher.encode(torch.zeros(1, 1, 640, dtype=dtype))
        with self.assertRaises(TypeError):
            self.teacher.encode([0.0])
        with self.assertRaises(ValueError):
            self.teacher.encode(torch.full((1, 1, 640), float("nan")))
        with self.assertRaises(ValueError):
            self.teacher.decode(torch.full((1, 64, 1), float("inf")))
        with self.assertRaises(ValueError):
            self.teacher.encode(torch.empty(1, 1, 640, device="meta"))
        with self.assertRaises(ValueError):
            self.teacher.decode(torch.zeros(1, 24, 1))

    def test_rejects_mutated_teacher_state(self):
        audio = torch.zeros(1, 1, 640)
        self.teacher.model.train()
        with self.assertRaisesRegex(RuntimeError, "eval"):
            self.teacher.encode(audio)
        self.teacher.train()
        self.teacher.model.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            self.teacher.encode(audio)
        self.teacher.model.requires_grad_(False).half()
        with self.assertRaisesRegex(RuntimeError, "float32"):
            self.teacher.encode(audio)

    def test_rejects_invalid_teacher_outputs(self):
        with patch.object(self.teacher.model, "encode", return_value=torch.zeros(1, 64, 2)):
            with self.assertRaisesRegex(RuntimeError, "shape"):
                self.teacher.encode(torch.zeros(1, 1, 640))
        with patch.object(self.teacher.model, "decode", return_value=torch.full((1, 1, 1920), float("nan"))):
            with self.assertRaisesRegex(RuntimeError, "nonfinite"):
                self.teacher.decode(torch.zeros(1, 64, 1))

    def test_provenance_is_serializable_and_defensive(self):
        provenance = self.teacher.provenance
        provenance["config"]["nested"].append(2)
        self.assertEqual(self.teacher.provenance["config"]["nested"], [1])
        self.assertEqual(json.loads(json.dumps(self.teacher.provenance))["device"], "cpu")

    def test_backend_policy_changes_identity_and_overrides_inherited_policy(self):
        from audiovae_student.cache import _identity_hash
        current = self.teacher.provenance
        legacy = dict(current)
        del legacy["encoder_backend_policy"]
        self.assertNotEqual(_identity_hash(current), _identity_hash(legacy))
        self.assertEqual(current["encoder_backend_policy"]["active_backend"], "unchanged_cpu")
        current["encoder_backend_policy"]["cuda_cudnn"]["enabled"] = True
        wrapped = FrozenAudioVAE2(self.teacher.model, current)
        self.assertFalse(wrapped.provenance["encoder_backend_policy"]["cuda_cudnn"]["enabled"])
        cuda_policy = teacher_module._encoder_backend_policy(torch.device("cuda"))
        self.assertEqual(cuda_policy["active_backend"], "cuda_without_cudnn")
        self.assertNotEqual(_identity_hash(cuda_policy), _identity_hash(self.teacher.provenance["encoder_backend_policy"]))

    def test_cpu_encode_leaves_cudnn_policy_untouched(self):
        with patch.object(torch.backends.cudnn, "flags", side_effect=AssertionError("CPU must not change cuDNN")):
            self.teacher.encode(torch.ones(1, 1, 640))
            self.teacher.decode(torch.ones(1, 64, 1))

    def test_cuda_backend_context_restores_flags_on_success_and_error_without_gpu(self):
        # Exercise only backend state management, without CUDA tensors/kernels.
        def flags():
            return tuple(getattr(torch.backends.cudnn, name) for name in
                         ("enabled", "benchmark", "deterministic", "allow_tf32"))
        before = flags()
        for fail in (False, True):
            try:
                with teacher_module._encoder_backend_context(torch.device("cuda")):
                    self.assertEqual(flags(), (False, False, True, False))
                    if fail:
                        raise RuntimeError("synthetic encoder failure")
            except RuntimeError as error:
                self.assertEqual(str(error), "synthetic encoder failure")
            self.assertEqual(flags(), before)


class TeacherAssetTests(unittest.TestCase):
    def test_wrong_source_is_rejected_before_code_or_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "teacher.py"
            checkpoint = Path(directory) / "weights.pth"
            source.write_text("raise RuntimeError('must not execute')")
            checkpoint.write_bytes(b"untrusted checkpoint")
            with patch.object(teacher_module, "_load_source") as import_source, patch.object(torch, "load") as load:
                with self.assertRaisesRegex(ValueError, "source SHA-256 mismatch"):
                    FrozenAudioVAE2.from_files(source, checkpoint)
                import_source.assert_not_called()
                load.assert_not_called()

    def test_wrong_checkpoint_is_rejected_before_source_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "teacher.py"
            checkpoint = Path(directory) / "weights.pth"
            payload = b"raise RuntimeError('must not execute')"
            source.write_bytes(payload)
            checkpoint.write_bytes(b"wrong checkpoint")
            with patch.object(teacher_module, "SOURCE_SHA256", hashlib.sha256(payload).hexdigest()):
                with patch.object(teacher_module, "_load_source") as import_source, patch.object(torch, "load") as load:
                    with self.assertRaisesRegex(ValueError, "checkpoint SHA-256 mismatch"):
                        FrozenAudioVAE2.from_files(source, checkpoint)
                    import_source.assert_not_called()
                    load.assert_not_called()

    def test_missing_files_never_trigger_download(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(FileNotFoundError):
            FrozenAudioVAE2.from_files(Path(directory) / "missing.py", Path(directory) / "missing.pth")

    def test_unsupported_device_fails_before_file_access(self):
        with self.assertRaisesRegex(ValueError, "cpu or cuda"):
            FrozenAudioVAE2.from_files("missing.py", "missing.pth", device="mps")

    def test_cuda_tf32_policy_fails_before_file_access(self):
        previous = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            with self.assertRaisesRegex(ValueError, "allow_tf32=False"):
                FrozenAudioVAE2.from_files("missing.py", "missing.pth", device="cuda")
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous


@unittest.skipUnless(os.environ.get("AUDIOVAE2_SOURCE") and os.environ.get("AUDIOVAE2_CHECKPOINT"), "local pinned assets not supplied")
class ActualTeacherIntegrationTests(unittest.TestCase):
    def test_original_teacher_contract_and_raw_mu_parity(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            teacher = FrozenAudioVAE2.from_files(os.environ["AUDIOVAE2_SOURCE"], os.environ["AUDIOVAE2_CHECKPOINT"])
            self.assertEqual(teacher.provenance["checkpoint_sha256"], teacher_module.CHECKPOINT_SHA256)
            self.assertEqual(teacher.provenance["config"]["decoder_dim"], 2048)
            audio = torch.linspace(-0.1, 0.1, 641).reshape(1, 1, -1)
            z = teacher.encode(audio)
            with torch.no_grad():
                expected_mu = teacher.model.encoder(F.pad(audio, (0, 639)))["mu"]
                expected_wave = teacher.model.decode(expected_mu)
            torch.testing.assert_close(z, expected_mu, atol=0, rtol=0)
            raw = teacher.decode(z)
            torch.testing.assert_close(raw, expected_wave, atol=0, rtol=0)
            self.assertEqual(tuple(raw.shape), (1, 1, 3840))
            result = teacher.reconstruct(audio)
            self.assertEqual(tuple(result.shape), (1, 1, 1923))
            torch.testing.assert_close(result, expected_wave[..., :1923], atol=0, rtol=0)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                mixed_context_z = teacher.encode(audio)
                mixed_context_wave = teacher.decode(mixed_context_z)
            torch.testing.assert_close(mixed_context_z, expected_mu, atol=0, rtol=0)
            torch.testing.assert_close(mixed_context_wave, expected_wave, atol=0, rtol=0)
            nn.Conv1d(64, 4, 1)(z).sum().backward()
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
