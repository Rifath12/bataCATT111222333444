"""Verify the adaptive array profile's block boundaries and speaker masks."""

import unittest

import numpy as np

from array_covariance import AdaptiveArrayMask
from spatial_render import SpatialOverlapMask


def fixture():
    rng = np.random.default_rng(125)
    spectra, bins, channels = 85, 9, 8
    random = rng.standard_normal((spectra, bins, channels, 2), dtype=np.float32)
    x = random[..., 0] + 1j * random[..., 1]
    vectors = rng.standard_normal((4, bins, channels, channels, 2), dtype=np.float32)
    complex_vectors = vectors[..., 0] + 1j * vectors[..., 1]
    cov = complex_vectors @ complex_vectors.conj().swapaxes(-1, -2)
    for j in range(3):
        cov[j] += 3 * cov[3]
    mask = np.ones((3, spectra), dtype=bool)
    mask[1, :10] = False
    mask[2, :10] = False
    mask[0, 37:45] = False
    mask[1, 37:45] = False
    mask[2, 15:20] = False
    y = np.complex64(0.13 - 0.04j) * np.ones((spectra, 3, bins), dtype=np.complex64)
    return AdaptiveArrayMask(cov, mask, SpatialOverlapMask(cov, mask)), x, y


class AdaptiveArrayTest(unittest.TestCase):
    def test_partitioning_realizes_same_timeline(self):
        engine, x, y = fixture()
        actual = y.copy()
        expected = engine.apply(x, y.copy(), 0, len(x))
        boundaries = [0, 16, 30, 42, 43, 47, 65, len(x)]
        for start, stop in zip(boundaries, boundaries[1:]):
            left = max(0, start - engine.context_frames)
            right = min(len(x), stop + engine.context_frames)
            engine.apply(x[left:right], actual[start:stop], start, stop, spectrum_start=left)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-6)

    def test_solo_and_inactive_outputs_stay_separate(self):
        engine, x, y = fixture()
        separated = engine.apply(x, y.copy(), 0, len(x))
        solo = engine.activity.sum(axis=0) < 2
        np.testing.assert_array_equal(separated[solo], y[solo])
        np.testing.assert_array_equal(separated[15:20, 2, :], np.zeros_like(separated[15:20, 2, :]))
        self.assertTrue(np.isfinite(separated).all())
        self.assertGreater(np.sum(np.abs(separated[15:20, :2, :])), 1.0)

    def test_silent_overlap_emits_exact_silence(self):
        engine, x, y = fixture()
        active = engine.activity.sum(axis=0) >= 2
        segregated = engine.apply(np.zeros_like(x), y.copy(), 0, len(x))
        np.testing.assert_array_equal(segregated[active], np.zeros_like(segregated[active]))


if __name__ == "__main__":
    unittest.main()
