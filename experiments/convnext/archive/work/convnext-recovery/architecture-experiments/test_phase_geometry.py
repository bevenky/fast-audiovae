"""Independent known-phase fixture for the fresh eight-second anchor geometry."""
import unittest

import torch


class PhaseGeometryTest(unittest.TestCase):
    def test_known_sample_positions_and_excluded_prefix(self):
        torch.set_num_threads(1)
        # Each phase selects its corresponding feature, so waveform row r in
        # phase p has the independently known value 10000*p+r.
        known_features = torch.zeros(2048, 4, dtype=torch.float64)
        known_features[:4] = torch.eye(4, dtype=torch.float64)
        known_target = (torch.arange(480, dtype=torch.float64)[:, None]
                        + 10000 * torch.arange(4, dtype=torch.float64)[None, :])
        weight = torch.zeros(480, 2048, dtype=torch.float64)
        weight[:, :4] = known_target
        h = known_features.repeat(1, 200)[None]
        waveform = known_target.T.repeat(200, 1).reshape(1, 1, -1)
        self.assertEqual(tuple(h.shape), (1, 2048, 800))
        self.assertEqual(tuple(waveform.shape), (1, 1, 384000))
        # The first four seconds deliberately disagree with stationary silence.
        h[..., :400] = -999
        waveform[..., :192000] = -777
        observed_h = h[0, :, 400:].reshape(2048, 100, 4).mean(1)
        observed_t = waveform[0, 0, 192000:].reshape(100, 4, 480).mean(0).T
        self.assertTrue(torch.equal(observed_h, known_features))
        self.assertTrue(torch.equal(observed_t, known_target))
        self.assertTrue(torch.equal(weight @ observed_h, observed_t))
        # Verify physical sample ordering, including joins between phases and
        # complete 40 ms cycles, independently of the matrix reshape.
        for sample in (0, 1, 479, 480, 959, 960, 1439, 1440, 1919, 1920, 191999):
            phase = (sample // 480) % 4
            expected = 10000 * phase + sample % 480
            self.assertEqual(float(waveform[0, 0, 192000 + sample]), expected)


if __name__ == "__main__":
    unittest.main()
