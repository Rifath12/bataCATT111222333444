"""Regression tests for sample-aligned waveform quality scoring."""

import json
import unittest

import numpy as np

from waveform_acceptance import evaluate_case, fraction_db, signal_metrics, summarize


RATE = 8000


def voices():
    time = np.arange(RATE, dtype=np.float64) / RATE
    return np.stack([np.sin(2 * np.pi * freq * time) for freq in (300, 310, 330)])


class WaveformAcceptanceTests(unittest.TestCase):
    def test_known_target_voice_and_artifact_decomposition(self):
        refs = voices()
        time = np.arange(RATE) / RATE
        score = signal_metrics(refs, refs[0] + 0.2 * refs[1] + 0.1 * np.sin(2 * np.pi * 370 * time), 0)
        self.assertAlmostEqual(score["target_aligned_fraction"], 1 / 1.05)
        self.assertAlmostEqual(score["competing_voice_fraction"], 0.04 / 1.05)
        self.assertAlmostEqual(score["other_error_fraction"], 0.01 / 1.05)
        self.assertAlmostEqual(score["si_sdr_db"], 10 * np.log10(20))

    def test_muted_or_inaudible_tracks_fail(self):
        refs = voices()
        active = np.ones_like(refs, dtype=bool)
        result = evaluate_case("quiet", RATE, refs, 0.001 * refs, active)
        self.assertGreater(result["voices"]["exact_three"]["A"]["target_aligned_fraction"], 0.9999)
        self.assertEqual(result["voices"]["exact_three"]["A"]["retained_active_fraction"], 0)
        self.assertFalse(result["passes_exact_three"]["98"])
        silent = evaluate_case("zero", RATE, refs, np.zeros_like(refs), active)
        self.assertTrue(silent["voices"]["exact_three"]["A"]["issued_silence"])
        self.assertEqual(silent["voices"]["exact_three"]["A"]["target_aligned_fraction"], 0)

    def test_fixed_labels_do_not_repair_switches(self):
        refs = voices()
        active = np.ones_like(refs, dtype=bool)
        swapped = refs[[1, 0, 2]]
        result = evaluate_case("swapped", RATE, refs, swapped, active)
        self.assertFalse(result["passes_exact_three"]["96"])
        self.assertEqual(result["exact_three_windows"][0]["label_diagnostic"]["best_diagnostic_labels"], ["B", "A", "C"])
        self.assertLess(result["voices"]["exact_three"]["A"]["target_aligned_fraction"], 1e-10)
        json.dumps(summarize([result]), allow_nan=False)

    def test_quiet_reference_and_grouped_pass_counts(self):
        refs = voices() * np.array([1.0, 0.3, 0.6])[:, None]
        active = np.ones_like(refs, dtype=bool)
        good = evaluate_case("good", RATE, refs, refs, active, mixture=refs.sum(axis=0), group="shared")
        bad = evaluate_case("bad", RATE, refs, refs[[0, 2, 1]], active, group="shared")
        self.assertTrue(good["passes_exact_three"]["98"])
        self.assertEqual(good["quietest_exact_three_speaker"], "B")
        self.assertTrue(good["voices"]["exact_three"]["B"]["si_sdri_db"] > 100)
        summary = summarize([good, bad])
        qualifiers = summary["passes_at_fidelity_target_pct"]["98"]
        self.assertEqual(qualifiers["voices_passed"], 4)
        self.assertEqual(qualifiers["cases_passed"], 1)
        self.assertEqual(qualifiers["groups_passed"], 0)
        self.assertEqual(qualifiers["groups_eligible"], 1)
        self.assertIsNone(qualifiers["independent_group_interval_95"])

    def test_missing_audio_and_short_reference_masks(self):
        refs = voices()
        active = np.ones_like(refs, dtype=bool)
        candidate = refs.copy()
        candidate[0, :RATE // 5] = 0.0
        result = evaluate_case("dropped", RATE, refs, candidate, active)
        self.assertAlmostEqual(result["voices"]["exact_three"]["A"]["retained_active_fraction"], 0.8)
        self.assertFalse(result["passes_exact_three"]["96"])
        active[0, :1000] = False
        with self.assertRaisesRegex(ValueError, "subset"):
            evaluate_case("bad_masks", RATE, refs, candidate, active, exact_three=np.ones(RATE, dtype=bool))

    def test_threshold_conversion(self):
        self.assertAlmostEqual(fraction_db(0.96), 13.8021124171)
        self.assertAlmostEqual(fraction_db(0.98), 16.9019608003)

    def test_off_target_output_is_measured(self):
        refs = voices()
        active = np.ones_like(refs, dtype=bool)
        active[0, RATE // 2:] = False
        outputs = refs.copy()
        outputs[0, RATE // 2:] = 0.1 * refs[1, RATE // 2:]
        score = evaluate_case("off_target", RATE, refs, outputs, active)
        a = score["voices"]["reference_active"]["A"]
        self.assertAlmostEqual(a["off_target_output_db_relative_to_reference"], -20.0)
        self.assertAlmostEqual(a["other_active_while_target_inactive_s"], 0.5)


if __name__ == "__main__":
    unittest.main()
