"""Check the sparse overlap transform against the original PCM calculation."""

import unittest

import numpy as np

from asr_mask import FRAME, overlap_asr_tracks
from asr_mask_original import overlap_asr_tracks as dense


class SparseParity(unittest.TestCase):
    def test_boundary_and_overlap_parity(self):
        rng = np.random.default_rng(6218)
        for length in [31, 216, 400, 3200, 160031, 318087]:
            n = (length + FRAME - 1) // FRAME
            mix = rng.normal(size=length).astype("float32")
            stems = rng.normal(size=(3, length)).astype("float32")
            for fraction in (0, 0.25, 0.75, 1):
                active = rng.random((3, n)) < fraction
                with self.subTest(samples=length, occupancy=fraction):
                    a = dense(mix, stems, active)
                    b = overlap_asr_tracks(mix, stems, active)
                    np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
