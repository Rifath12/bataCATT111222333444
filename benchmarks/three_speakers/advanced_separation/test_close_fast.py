"""Verify the optional accelerated trees against the portable inference path."""
from importlib.util import find_spec
import unittest

import numpy as np

from boosted_close import BoostedMask


@unittest.skipUnless(find_spec("numba"), "Optional numba dependency is unavailable")
class FastTreeTests(unittest.TestCase):
    def test_matches_numpy_for_features_and_waveforms(self):
        portable = BoostedMask.load()
        fast = BoostedMask.load(fast=True)
        rng = np.random.default_rng(8441)
        features = rng.normal(0, 3, (3000, 22)).astype(np.float32)
        for tree, row in zip(portable.trees, features):
            first = int(tree.column[0])
            if first >= 0:
                row[first] = tree.threshold[0]
        np.testing.assert_array_equal(fast.predict(features), portable.predict(features))
        wave = rng.normal(0, .08, (3, 5487)).astype(np.float32)
        transfer = np.tile(np.eye(3, dtype=np.float32), (513, 1, 1))
        transfer[:, 0, 2] = .027
        np.testing.assert_array_equal(fast.refine(wave, transfer, wave), portable.refine(wave, transfer, wave))


if __name__ == "__main__":
    unittest.main()
