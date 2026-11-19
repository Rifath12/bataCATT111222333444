"""Small renderer regressions that run without downloading model weights."""

import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

from partial_enrollment import activity_affinity, assignment_is_tied, short_solo_cue
from render import enrollment, output_tracks


class ShortEnrollmentTest(unittest.TestCase):
    def test_short_fragments_are_trimmed_and_pooled(self):
        wave = np.ones(120 * 160, dtype=np.float32)
        fragments = [(0, 28), (40, 67), (80, 108)]
        cue = short_solo_cue(wave, fragments)
        self.assertEqual(len(cue), 64000)
        self.assertEqual(np.count_nonzero(cue), (22 + 21 + 22) * 160)
        self.assertIsNone(short_solo_cue(wave, fragments[:2]))

    def test_activity_finds_nonidentity_permutation(self):
        a = np.zeros((3, 150), dtype=bool)
        a[0, :85] = True
        a[1, 40:115] = True
        a[2, 75:] = True
        sources = np.repeat(a[[2, 0, 1]], 80, axis=1).astype(np.float32)
        scores = activity_affinity(sources[:, 10 * 80 :], a, 10)
        rows, columns = linear_sum_assignment(-scores)
        np.testing.assert_array_equal(columns, [1, 2, 0])
        self.assertFalse(assignment_is_tied(scores, rows, columns))

    def test_evidence_free_assignments_are_flagged(self):
        a = np.ones((3, 50), dtype=bool)
        sources = np.ones((3, 4000), dtype=np.float32)
        scores = activity_affinity(sources, a, 0)
        self.assertTrue(assignment_is_tied(scores, *linear_sum_assignment(-scores)))

    def test_clean_anchor_caller_contract(self):
        activity = np.repeat(np.eye(3, dtype=bool), 100, axis=1)
        wave = np.ones(48000, dtype=np.float32)
        existing = enrollment(wave, activity, activity.sum(0))
        self.assertEqual(existing.shape, (3, 64000))
        partial, missing = enrollment(
            wave, activity, activity.sum(0), allow_partial=True
        )
        self.assertEqual(missing, [])
        np.testing.assert_array_equal(partial, existing)

    def test_renderer_uses_activity_when_embeddings_are_unavailable(self):
        activity = np.zeros((3, 300), dtype=bool)
        activity[0, :175] = True
        activity[1, 100:240] = True
        activity[2, 155:250] = True
        reference = np.repeat(activity, 160, axis=1).astype(np.float32)
        reference *= np.array([0.10, 0.11, 0.12], np.float32)[:, None]
        mixture = reference.sum(axis=0)

        class PermutedSeparation:
            def __init__(self, model, device):
                pass

            def infer(self, mix, speakers):
                self_start = (100 - 30) * 80
                return (
                    reference[[2, 0, 1], ::2][:, self_start : self_start + len(mix)],
                    None,
                    None,
                )

        fake = SimpleNamespace(
            Separation=PermutedSeparation,
            load_separator=lambda *a: None,
            VoiceEmbeddings=lambda path: lambda samples: None,
            resample=lambda x, src, dst: x[::2] if dst < src else np.repeat(x, 2),
            normalized_dot=lambda a, b: 0.0,
        )
        options = SimpleNamespace(project=Path("."), models=Path("."), workers=1)
        with patch.dict(sys.modules, process_meeting=fake):
            tracks, metadata = output_tracks(mixture, activity, "dprnn", options)
        np.testing.assert_array_equal(tracks, reference)
        self.assertEqual(metadata["speakers_without_long_solo"], [1, 2])
        self.assertEqual(metadata["speakers_without_usable_embedding"], [0, 1, 2])
        self.assertEqual(metadata["tied_identity_windows"], 0)

    def test_no_overlap_export_even_when_two_speakers_have_only_short_turns(self):
        activity = np.zeros((3, 100), dtype=bool)
        activity[0, 0:80] = True
        activity[1, 80:90] = True
        activity[2, 90:] = True
        wave = np.linspace(-0.2, 0.2, 16000, dtype=np.float32)
        for method in ("dprnn", "target"):
            tracks, stats = output_tracks(wave, activity, method, SimpleNamespace())
            np.testing.assert_allclose(tracks, wave * np.repeat(activity, 160, axis=1))
            self.assertEqual(stats["overlap_windows"], 0)
            np.testing.assert_array_equal(tracks.sum(axis=0), wave)


if __name__ == "__main__":
    unittest.main()
