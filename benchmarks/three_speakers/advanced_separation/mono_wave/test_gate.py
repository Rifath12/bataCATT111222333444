import unittest

import numpy as np

from gate import fit_mixture, select_tracks


class MixtureGateTest(unittest.TestCase):
    def setUp(self):
        x = np.arange(256) * (2 * np.pi / 256)
        self.truth = np.stack([np.sin(i * x) * 0.12 for i in (2, 5, 11)])
        self.mix = self.truth.sum(axis=0)
        self.fallback = (0.35 * self.truth).astype(np.float32)

    def test_positive_rescaling_and_dc_are_invariant(self):
        candidates = self.truth * np.array([1e-7, 8, 0.13])[:, None] + 0.45
        selected, source, scores = select_tracks(self.mix + 0.5, self.fallback, candidates)
        self.assertEqual(source, "second_pass")
        self.assertGreater(scores[source], 0.9999)
        np.testing.assert_allclose(selected, self.truth, atol=1e-7)
        np.testing.assert_allclose(selected.mean(axis=1), 0.0, atol=1e-7)

    def test_three_near_copies_are_rejected_despite_exact_reconstruction(self):
        candidates = self.mix + np.array([1.0e-3, 1.0e-3, 1.0e-3])[:, None] * (
            self.truth - self.truth.mean(axis=0)
        )
        fitted = fit_mixture(self.mix, candidates)
        self.assertGreater(fitted.explained, 0.9999)
        self.assertGreater(float(fitted.gains.min()), 0)
        self.assertLess(fitted.independence, 0.01)
        selected, source, _ = select_tracks(self.mix, self.fallback, candidates)
        self.assertEqual(source, "baseline")
        np.testing.assert_array_equal(selected, self.fallback)

    def test_a_negligible_third_source_is_rejected(self):
        weak_mix = self.truth[0] + self.truth[1] + self.truth[2] * 0.001
        fitted = fit_mixture(weak_mix, self.truth)
        self.assertGreater(fitted.explained, 0.9999)
        self.assertGreater(float(fitted.gains.min()), 0)
        self.assertEqual(select_tracks(weak_mix, self.fallback, self.truth)[1], "baseline")

    def test_silent_third_track_keeps_reconstruction_diagnostic(self):
        only_two = np.stack([self.truth[0], self.truth[1], np.zeros(256)])
        observed = only_two.sum(axis=0)
        fitted = fit_mixture(observed, only_two)
        self.assertGreater(fitted.explained, 0.9999)
        self.assertEqual(fitted.independence, 0.0)
        self.assertEqual(select_tracks(observed, self.fallback, only_two)[1], "baseline")

    def test_nonfinite_second_pass_retains_valid_first_pass(self):
        invalid = self.truth.copy()
        invalid[0, 0] = np.inf
        selected, source, _ = select_tracks(self.mix, self.fallback, invalid, first_pass=self.truth)
        self.assertEqual(source, "first_pass")
        np.testing.assert_allclose(selected, self.truth, atol=1e-7)

    def test_silence_retains_baseline_and_invalid_input_is_rejected(self):
        self.assertEqual(select_tracks(self.mix * 0, self.fallback, self.truth)[1], "baseline")
        with self.assertRaises(ValueError):
            select_tracks([], self.fallback[:, :0], self.truth[:, :0])
        with self.assertRaises(ValueError):
            select_tracks(self.mix, self.fallback[:, :-1], self.truth)
        with self.assertRaises(ValueError):
            select_tracks(self.mix, self.fallback * np.nan, self.truth)


if __name__ == "__main__":
    unittest.main()
