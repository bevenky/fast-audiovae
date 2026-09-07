"""Small encoder tests use generated tensors and never load model assets."""
import types
import unittest
import warnings

try:
    import torch
except ModuleNotFoundError:
    raise unittest.SkipTest("Install the encoder extra to run these tests")

from fast_audiovae.encoder import prepare_encoder


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self, *args, padding=0, output_padding=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.__padding = padding
        self.__output_padding = output_padding

    def forward(self, x):
        return super().forward(torch.nn.functional.pad(
            x, (self.__padding * 2 - self.__output_padding, 0)))


class Snake1d(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)


class CausalEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            wn = torch.nn.utils.weight_norm
            self.input = wn(CausalConv1d(1, 4, 7, padding=3))
            self.snake = Snake1d(4)
            self.depthwise = wn(CausalConv1d(4, 4, 7, padding=9, dilation=3, groups=4))
            self.pointwise = wn(CausalConv1d(4, 4, 1))

    def forward(self, x):
        return self.pointwise(self.depthwise(self.snake(self.input(x))))


class EncoderTests(unittest.TestCase):
    def model(self):
        return CausalEncoder().eval()

    def test_outputs_are_exact_and_original_nonlinear_layers_remain(self):
        model = self.model()
        original_snake, original_dw = model.snake.forward, model.depthwise.forward
        inputs = [torch.randn(2, 1, length) for length in (1, 7, 639, 640, 641)]
        with torch.no_grad():
            expected = [model(x) for x in inputs]
        threads = torch.get_num_threads()
        handle = prepare_encoder(model)
        self.assertEqual(handle.pointwise_layers, ("pointwise",))
        self.assertEqual(len(handle.folded_layers), 3)
        self.assertEqual(model.snake.forward, original_snake)
        self.assertEqual(model.depthwise.forward, original_dw)
        with torch.inference_mode():
            for x, ref in zip(inputs, expected):
                self.assertTrue(torch.equal(model(x), ref))
        self.assertEqual(torch.get_num_threads(), threads)
        handle.restore()

    def test_restore_is_idempotent_and_does_not_unfold_weights(self):
        model = self.model()
        original_root, original_pw = model.forward, model.pointwise.forward
        handle = prepare_encoder(model)
        handle.restore()
        handle.restore()
        self.assertEqual(model.forward, original_root)
        self.assertEqual(model.pointwise.forward, original_pw)
        self.assertNotIn("forward", model.__dict__)
        self.assertNotIn("forward", model.pointwise.__dict__)
        self.assertFalse(hasattr(model.pointwise, "weight_g"))
        second = prepare_encoder(model)
        self.assertEqual(second.folded_layers, ())
        second.restore()

    def test_invalid_preparation_leaves_weights_and_forwards_untouched(self):
        for mutation in (lambda m: m.train(), lambda m: m.snake.train(), lambda m: m.double()):
            model = self.model()
            original = model.pointwise.forward
            mutation(model)
            with self.assertRaises(ValueError):
                prepare_encoder(model)
            self.assertTrue(hasattr(model.pointwise, "weight_g"))
            self.assertEqual(model.pointwise.forward, original)
        with self.assertRaises(ValueError):
            prepare_encoder(torch.nn.Sequential())

    def test_duplicate_and_existing_pointwise_patch_are_rejected(self):
        model = self.model()
        handle = prepare_encoder(model)
        with self.assertRaises(RuntimeError):
            prepare_encoder(model)
        handle.restore()
        model = self.model()
        model.pointwise.forward = types.MethodType(lambda self, x: x, model.pointwise)
        with self.assertRaises(ValueError):
            prepare_encoder(model)
        self.assertTrue(hasattr(model.input, "weight_g"))

    def test_inference_input_and_model_guards(self):
        model = self.model()
        handle = prepare_encoder(model)
        x = torch.randn(1, 1, 17)
        with self.assertRaises(RuntimeError):
            model(x)
        with torch.no_grad():
            for invalid in (x.double(), x[:, 0], torch.empty(1, 1, 0), torch.zeros(1, 2, 17),
                            torch.empty(1, 1, 17, device="meta")):
                with self.assertRaises(ValueError):
                    model(invalid)
            with torch.autocast("cpu"), self.assertRaises(RuntimeError):
                model(x)
            model.snake.train()
            with self.assertRaises(ValueError):
                model(x)
            model.eval().double()
            with self.assertRaises(ValueError):
                model(x)
        handle.restore()

    def test_noncontiguous_input_keeps_upstream_behavior(self):
        model = self.model()
        x = torch.randn(1, 1, 34)[..., ::2]
        self.assertFalse(x.is_contiguous())
        with torch.no_grad():
            ref = model(x)
        handle = prepare_encoder(model)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(x), ref))
        handle.restore()


if __name__ == "__main__":
    unittest.main()
