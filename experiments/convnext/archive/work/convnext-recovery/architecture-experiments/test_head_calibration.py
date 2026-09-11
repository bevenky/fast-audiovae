from __future__ import annotations

import unittest

import torch

from head_calibration import covariance_ridge_fit, minimum_norm_anchor


class HeadCalibrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(402)
        self.w = torch.randn(5, 9, generator=generator, dtype=torch.float64)
        self.h = torch.randn(9, 4, generator=generator, dtype=torch.float64)
        self.t = torch.randn(5, 4, generator=generator, dtype=torch.float64)
        self.x = torch.randn(9, 37, generator=generator, dtype=torch.float64)
        self.y = torch.randn(5, 37, generator=generator, dtype=torch.float64)
        self.a = self.x @ self.x.T / self.x.shape[1]
        self.b = self.x @ (self.y - self.w @ self.x).T / self.x.shape[1]
        q = torch.linalg.qr(self.h, mode="reduced")[0]
        self.p = torch.eye(9, dtype=torch.float64) - q @ q.T

    def test_full_anchor_satisfies_constraint_and_minimum_norm(self):
        result = minimum_norm_anchor(self.w, self.h, self.t)
        torch.testing.assert_close(result.weight @ self.h, self.t, rtol=1e-12, atol=1e-12)
        feasible = self.w @ self.p
        self.assertLess(abs(float((result.delta_fp64 * feasible).sum())), 1e-11)
        self.assertLessEqual(float(result.delta_fp64.square().sum()),
                             float((result.delta_fp64 + feasible).square().sum()))
        self.assertEqual(result.report["feature_rank"], 4)

    def test_shared_anchor_preserves_phase_differences(self):
        result = minimum_norm_anchor(self.w, self.h, self.t, mode="shared")
        original = self.w @ self.h
        modified = result.weight @ self.h
        torch.testing.assert_close(modified - modified.mean(1, keepdim=True),
                                   original - original.mean(1, keepdim=True),
                                   rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(modified.mean(1), self.t.mean(1), rtol=1e-12, atol=1e-12)
        self.assertEqual(int(torch.linalg.matrix_rank(result.delta_fp64)), 1)

    def test_identity_target_has_exactly_zero_minimum_edit(self):
        target = self.w @ self.h
        for mode in ("full", "shared"):
            result = minimum_norm_anchor(self.w, self.h, target, mode=mode)
            self.assertTrue(torch.equal(result.weight, self.w))
            self.assertEqual(float(result.delta_fp64.norm()), 0)

    def test_covariance_fit_is_constrained_optimum(self):
        result = covariance_ridge_fit(self.w, self.h, self.t, self.a, self.b,
                                      shrinkage=0.01)
        delta = result.delta_fp64
        c = self.a + 0.01 * self.a.trace() / 9 * torch.eye(9, dtype=torch.float64)
        gradient = delta @ c - self.b.T
        torch.testing.assert_close(result.weight @ self.h, self.t, rtol=1e-12, atol=1e-12)
        self.assertLess(float((gradient @ self.p).norm()), 1e-11)
        objective = lambda x: float(0.5 * (x @ c * x).sum() - (x * self.b.T).sum())
        for sign in (-1, 1):
            self.assertGreater(objective(delta + sign * 0.1 * self.w @ self.p), objective(delta))

    def test_covariance_anchor_ignores_residual_fit_and_is_optimum(self):
        first = covariance_ridge_fit(self.w, self.h, self.t, self.a, self.b,
                                    shrinkage=0.1, anchor_only=True)
        second = covariance_ridge_fit(self.w, self.h, self.t, self.a, self.b * 123,
                                     shrinkage=0.1, anchor_only=True)
        self.assertTrue(torch.equal(first.weight, second.weight))
        c = self.a + 0.1 * self.a.trace() / 9 * torch.eye(9, dtype=torch.float64)
        self.assertLess(float((first.delta_fp64 @ c @ self.p).norm()), 1e-11)

    def test_covariance_anchor_identity_is_zero(self):
        result = covariance_ridge_fit(self.w, self.h, self.w @ self.h, self.a, self.b,
                                      shrinkage=1.0, anchor_only=True)
        self.assertTrue(torch.equal(result.weight, self.w))
        self.assertEqual(float(result.delta_fp64.norm()), 0)

    def test_covariance_shared_fit_preserves_phase_differences(self):
        result = covariance_ridge_fit(self.w, self.h, self.t, self.a, self.b,
                                      shrinkage=0.001, mode="shared")
        old, new = self.w @ self.h, result.weight @ self.h
        torch.testing.assert_close(new - new.mean(1, keepdim=True), old - old.mean(1, keepdim=True),
                                   rtol=1e-12, atol=1e-12)

    def test_fp32_install_has_separate_rounding_report(self):
        w = self.w.float()
        result = minimum_norm_anchor(w, self.h, self.t)
        self.assertEqual(result.weight.dtype, torch.float32)
        self.assertEqual(result.weight.device.type, "cpu")
        observed = float((result.weight.double() @ self.h - self.t).abs().max())
        self.assertEqual(observed, result.report["installed_coefficient_rounding_constraint_max_abs"])
        self.assertLess(result.report["constraint_residual_fp64_max_abs"], 1e-11)
        self.assertGreater(observed, result.report["constraint_residual_fp64_max_abs"])

    def test_inputs_and_global_rng_unchanged(self):
        values = [self.w, self.h, self.t, self.a, self.b]
        copies = [v.clone() for v in values]
        rng = torch.random.get_rng_state().clone()
        minimum_norm_anchor(self.w, self.h, self.t)
        covariance_ridge_fit(self.w, self.h, self.t, self.a, self.b, shrinkage=0.01)
        for value, copy in zip(values, copies):
            self.assertTrue(torch.equal(value, copy))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_phase_permutation_equivariance(self):
        permutation = [3, 0, 2, 1]
        first = minimum_norm_anchor(self.w, self.h, self.t, mode="shared")
        second = minimum_norm_anchor(self.w, self.h[:, permutation], self.t[:, permutation], mode="shared")
        torch.testing.assert_close(first.weight, second.weight, rtol=1e-12, atol=1e-12)

    def test_invalid_shapes_rank_and_finiteness(self):
        with self.assertRaisesRegex(ValueError, "four phases"):
            minimum_norm_anchor(self.w, self.h[:, :3], self.t[:, :3])
        singular = self.h.clone()
        singular[:, 3] = singular[:, 0]
        with self.assertRaisesRegex(ValueError, "rank four"):
            minimum_norm_anchor(self.w, singular, self.t)
        bad = self.t.clone()
        bad[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            minimum_norm_anchor(self.w, self.h, bad)
        with self.assertRaisesRegex(ValueError, "mode"):
            minimum_norm_anchor(self.w, self.h, self.t, mode="wrong")

    def test_bad_covariance_or_unplanned_grid_rejected(self):
        with self.assertRaisesRegex(ValueError, "predeclared"):
            covariance_ridge_fit(self.w, self.h, self.t, self.a, self.b, shrinkage=0.02)
        bad = self.a.clone()
        bad[1, 0] += 1
        with self.assertRaisesRegex(ValueError, "symmetric"):
            covariance_ridge_fit(self.w, self.h, self.t, bad, self.b, shrinkage=0.01)
        with self.assertRaisesRegex(ValueError, "positive mean"):
            covariance_ridge_fit(self.w, self.h, self.t, self.a * 0, self.b, shrinkage=0.01)


if __name__ == "__main__":
    unittest.main()
