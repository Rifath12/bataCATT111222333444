"""Regression checks for speaker identities and three-way activity."""

import unittest

import numpy as np

from timing_refinement import match_views, refine


class TimingRefinementTest(unittest.TestCase):
    def test_matches_alternate_slot_permutation(self):
        frame = np.zeros((200, 3), dtype=bool)
        frame[10:35, 0] = True
        frame[65:90, 1] = True
        frame[150:180, 2] = True
        src = frame.astype(np.float32)
        a, b = match_views(src, src[:, [1, 2, 0]], frame)
        np.testing.assert_array_equal(a, src)
        np.testing.assert_array_equal(b, src)

    def test_silent_speaker_cannot_be_recognized(self):
        original = np.zeros((100, 3), dtype=bool)
        original[10:40, 0] = True
        original[55:80, 1] = True
        posterior = original.astype(np.float32)
        self.assertIsNone(match_views(posterior, posterior, original))

    def test_ambiguous_view_assignment_stays_unmodified(self):
        original = np.zeros((100, 3), dtype=bool)
        original[10:40, 0] = True
        original[10:40, 1] = True
        original[55:80, 2] = True
        posterior = original.astype(np.float32)
        self.assertIsNone(match_views(posterior, posterior, original))

    def test_refinement_preserves_exact_three_and_keeps_core(self):
        original = np.zeros((200, 3), dtype=bool)
        original[10:50, 0] = True
        original[11:45, 1] = True
        original[40:60, 2] = True
        posterior = original.astype(np.float32)
        posterior[20:40, 2] = 1.0  # Would introduce unverified triple overlaps.
        posterior[40:42, 2] = 0.0  # Would remove an existing triple overlap.
        posterior[44:58, 0] = 1.0
        posterior[10:12, 0] = 0.0
        change = refine(original, posterior, posterior)
        np.testing.assert_array_equal(change.all(1), original.all(1))
        self.assertFalse(change[10, 0])
        self.assertTrue(change[50, 0])


if __name__ == "__main__":
    unittest.main()
