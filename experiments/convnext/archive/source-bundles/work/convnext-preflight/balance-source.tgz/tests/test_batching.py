"""Batch/serial objective and gradient parity, using synthetic inputs on CPU."""

from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from audiovae_student.batching import batched_scored_crop_loss
from audiovae_student.cache import DECODER_HOP, ENCODER_HOP, TrainingCrop
from audiovae_student.corpus_training import scored_crop_loss
from audiovae_student.losses import WarmupLossConfig, WarmupReconstructionLoss
from audiovae_student.model import StudentConfig, StudentDecoder


def crop(context, scored=3, valid=None, *, reference=True, source="example"):
    frames = context + scored
    return TrainingCrop(
        torch.randn(1, 64, frames), torch.randn(1, 1, frames * DECODER_HOP) * 0.03,
        torch.randn(1, 1, frames * ENCODER_HOP) * 0.03 if reference else None,
        "a" * 64, source, context, 0, context, scored,
        scored * DECODER_HOP if valid is None else valid,
    )


class BatchedLossTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(21)
        self.model = StudentDecoder(StudentConfig(hidden_channels=8, expansion_channels=16,
                                                   head_channels=12, dilations=(1, 2),
                                                   layer_scale_init=0.3))
        self.criterion = WarmupReconstructionLoss(WarmupLossConfig(
            teacher_fft_sizes=(32, 64), reference_fft_sizes_16k=(16, 32)))

    def tearDown(self):
        torch.set_num_threads(self.old_threads)

    def test_variable_history_partial_tail_and_missing_reference_match_serial(self):
        crops = [crop(0, reference=False), crop(29), crop(7, valid=2 * DECODER_HOP + 123),
                 crop(2, scored=2, reference=False), crop(0)]
        expected = [scored_crop_loss(self.model, item, self.criterion, "cpu") for item in crops]
        with patch.object(self.model, "forward", wraps=self.model.forward) as forward:
            actual = batched_scored_crop_loss(self.model, crops, self.criterion, "cpu")
        self.assertEqual(forward.call_count, 1)
        self.assertEqual(forward.call_args.args[0].shape, (5, 64, 32))
        self.assertEqual(actual.examples, 5)
        self.assertGreater(actual.padded_latent_frames, 0)
        for name in actual.mean:
            target = torch.stack([value[name] for value in expected]).mean()
            torch.testing.assert_close(actual.mean[name], target, atol=3e-5, rtol=2e-6)
        covered = sorted(index for group in actual.groups for index in group.indices)
        self.assertEqual(covered, list(range(len(crops))))
        for group in actual.groups:
            for name, value in group.values.items():
                target = torch.stack([expected[i][name] for i in group.indices]).mean()
                torch.testing.assert_close(value, target, atol=3e-5, rtol=2e-6)

    def test_gradient_matches_original_equal_per_example_objective(self):
        items = [crop(0), crop(29), crop(3, valid=DECODER_HOP + 99, reference=False)]
        serial = deepcopy(self.model)
        for item in items:
            (scored_crop_loss(serial, item, self.criterion, "cpu")["total"] / len(items)).backward()
        result = batched_scored_crop_loss(self.model, items, self.criterion, "cpu")
        result.mean["total"].backward()
        for (name, actual), (expected_name, expected) in zip(self.model.named_parameters(), serial.named_parameters()):
            self.assertEqual(name, expected_name)
            self.assertIsNotNone(actual.grad, name)
            self.assertIsNotNone(expected.grad, name)
            torch.testing.assert_close(actual.grad, expected.grad, atol=3e-4, rtol=3e-4, msg=name)

    def test_context_and_padding_targets_never_enter_fft(self):
        items = [crop(0), crop(5, valid=DECODER_HOP + 57)]
        clean = batched_scored_crop_loss(self.model, items, self.criterion, "cpu")
        altered = []
        for item in items:
            teacher, reference = item.teacher_audio.clone(), item.reference16k.clone()
            teacher[..., :item.scored_slice.start] = float("nan")
            teacher[..., item.scored_slice.stop:] = float("nan")
            reference[..., :item.reference_scored_slice.start] = float("nan")
            reference[..., item.reference_scored_slice.stop:] = float("nan")
            altered.append(replace(item, teacher_audio=teacher, reference16k=reference))
        actual = batched_scored_crop_loss(self.model, altered, self.criterion, "cpu")
        for name in actual.mean:
            torch.testing.assert_close(actual.mean[name], clean.mean[name], atol=0, rtol=0)

    def test_short_example_prefix_matches_when_batched_with_longer_history(self):
        short, long = crop(0), crop(29)
        singles = [scored_crop_loss(self.model, item, self.criterion, "cpu") for item in (short, long)]
        packed = batched_scored_crop_loss(self.model, [short, long], self.criterion, "cpu")
        self.assertEqual(len(packed.groups), 1)  # Different offsets, same scored length.
        for name in packed.mean:
            torch.testing.assert_close(packed.mean[name], (singles[0][name] + singles[1][name]) / 2,
                                       atol=3e-5, rtol=2e-6)

    def test_unequal_microbatches_preserve_effective_batch_weighting(self):
        items = [crop(0), crop(4), crop(7, reference=False)]
        first = batched_scored_crop_loss(self.model, items[:2], self.criterion, "cpu")
        last = batched_scored_crop_loss(self.model, items[2:], self.criterion, "cpu")
        together = batched_scored_crop_loss(self.model, items, self.criterion, "cpu")
        for name in together.mean:
            torch.testing.assert_close(together.mean[name], first.mean[name] * 2 / 3 + last.mean[name] / 3,
                                       atol=3e-5, rtol=2e-6)

    def test_empty_batch_and_invalid_crop_metadata_fail_before_forward(self):
        good = crop(0)
        bad = [replace(good, context_frames=2), replace(good, valid_scored_samples=7),
               replace(good, valid_scored_samples=good.scored_frames * DECODER_HOP + 3),
               replace(good, reference16k=torch.zeros(1, 1, 17))]
        with patch.object(self.model, "forward", wraps=self.model.forward) as forward:
            with self.assertRaisesRegex(ValueError, "at least one"):
                batched_scored_crop_loss(self.model, [], self.criterion, "cpu")
            for item in bad:
                with self.assertRaises(ValueError):
                    batched_scored_crop_loss(self.model, [item], self.criterion, "cpu")
        forward.assert_not_called()


if __name__ == "__main__":
    unittest.main()
