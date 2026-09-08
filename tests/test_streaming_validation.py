"""Exercise the validation driver with synthetic stateful decoders only."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location(
    "streaming_validation_driver", Path(__file__).parents[1] / "tools" / "validate_streaming.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
driver.np = np


def full(z):
    return np.repeat(np.cumsum(z[:, :1, :], axis=-1), driver.HOP, axis=-1)


class Stream:
    def __init__(self):
        self.reset()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def reset(self):
        self.value = np.float32(0)

    def decode_chunk(self, z):
        frames = np.cumsum(z[:, :1, :], axis=-1) + self.value
        self.value = frames[0, 0, -1]
        return np.repeat(frames, driver.HOP, axis=-1)

    def flush(self):
        return np.empty((1, 1, 0), dtype=np.float32)


class Decoder:
    def streaming_decode(self):
        return Stream()


class StreamingValidationTests(unittest.TestCase):
    def latent(self):
        return np.broadcast_to(np.arange(1, 9, dtype=np.float32), (1, 64, 8)).copy()

    def test_boundaries_include_final_partial(self):
        self.assertEqual(list(driver.chunks(7, (5,))), [(0, 5), (5, 7)])
        self.assertEqual(list(driver.chunks(8, (1, 3))), [(0, 1), (1, 4), (4, 5), (5, 8)])
        with self.assertRaises(ValueError):
            list(driver.chunks(5, (0,)))

    def test_stateful_reference_passes_all_protocols(self):
        z = self.latent()
        checks = driver.validate_case(Decoder(), full, "synthetic", z, full(z))
        self.assertEqual(len(checks), 10)
        self.assertTrue(all(check["passed"] for check in checks))
        self.assertTrue(all(check["exact"] for check in checks))

    def test_stateless_decoder_fails(self):
        class Stateless(Stream):
            def decode_chunk(self, z):
                return full(z)

        class BrokenDecoder:
            def streaming_decode(self):
                return Stateless()

        checks = driver.validate_case(BrokenDecoder(), full, "synthetic", self.latent())
        self.assertTrue(any(not check["passed"] for check in checks))

    def test_same_model_policy_keeps_upstream_error_visible(self):
        z = self.latent()
        old_reference = full(z) + np.float32(0.1)
        checks = driver.validate_case(Decoder(), full, "synthetic", z, old_reference,
                                      reference_policy="same-model")
        upstream = next(row for row in checks if row["check"] == "full_vs_stored_upstream")
        self.assertFalse(upstream["passed"])
        self.assertFalse(upstream["gate"])
        self.assertGreater(upstream["max_abs"], 0)
        self.assertTrue(all(row["passed"] for row in checks if row["gate"]))

    def test_shared_state_fails_interleaving(self):
        class SharedDecoder:
            def __init__(self):
                self.shared = Stream()

            def streaming_decode(self):
                self.shared.reset()
                return self.shared

        checks = driver.validate_case(SharedDecoder(), full, "synthetic", self.latent())
        self.assertTrue(any(not check["passed"] for check in checks
                            if check["check"].startswith("independent_stream")))

    def test_invalid_waves_and_boundary_error_fail(self):
        reference = full(self.latent())
        actual = reference.copy()
        actual[..., driver.HOP] += 1
        result = driver.discrepancy(reference, actual, boundaries=[driver.HOP])
        self.assertFalse(result["passed"])
        self.assertEqual(result["boundary_max_abs"], 1)
        actual[..., 0] = np.nan
        self.assertFalse(driver.discrepancy(reference, actual)["passed"])
        self.assertFalse(driver.discrepancy(reference, reference.astype(np.float64))["passed"])
        self.assertFalse(driver.discrepancy(reference, reference[..., :-1])["passed"])

    def test_reused_audio_output_is_rejected(self):
        class Aliased(Stream):
            def __init__(self):
                super().__init__()
                self.buffer = np.empty((1, 1, driver.HOP), dtype=np.float32)

            def decode_chunk(self, z):
                self.buffer[:] = super().decode_chunk(z)
                return self.buffer

        with self.assertRaisesRegex(ValueError, "previously returned"):
            driver.decode_chunks(Aliased(), self.latent(), (1,))

    def test_npz_input_contract(self):
        z = self.latent()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.npz"
            np.savez(path, clip__z=z, clip__ref=full(z))
            cases = driver.load_cases(path)
            self.assertEqual(cases[0][0], "clip")
            np.testing.assert_array_equal(cases[0][1], z)
            np.savez(path, clip__z=z.astype(np.float64))
            with self.assertRaisesRegex(ValueError, "latent"):
                driver.load_cases(path)


if __name__ == "__main__":
    unittest.main()
