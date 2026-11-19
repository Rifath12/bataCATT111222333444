"""Regressions for the physical microphone calibration trial."""

import unittest

import numpy as np

from calibrate_wiring import run


RATE = 16_000


def three_takes(leak=0.0):
    t = np.arange(RATE * 2) / RATE
    takes = []
    for voice, frequency in enumerate((211, 293, 367)):
        wave = 0.08 * np.sin(2 * np.pi * frequency * t)
        channel_gain = np.full(3, leak)
        channel_gain[voice] = 1.0
        takes.append(wave[:, None] * channel_gain)
    return takes


class CalibrationRegressions(unittest.TestCase):
    def test_separate_capture_passes_even_with_quiet_speech(self):
        report = run(three_takes(), RATE, [20.0])
        self.assertEqual(report["all_remixes_pass"], {"96": True, "98": True})
        self.assertEqual(len(report["remixes"]), 4)
        self.assertAlmostEqual(report["exact_three_seconds_per_remix"], 2.0)

    def test_quiet_speaker_failure_cannot_hide_in_other_voices(self):
        report = run(three_takes(0.02), RATE, [20.0])
        balanced, quiet = report["remixes"][:2]
        self.assertTrue(balanced["passes_exact_three"]["98"])
        self.assertFalse(quiet["passes_exact_three"]["96"])
        self.assertEqual(quiet["quietest_exact_three_speaker"], "A")
        self.assertAlmostEqual(
            quiet["voices"]["exact_three"]["A"]["target_aligned_fraction"], 1/(1+0.08)
        )

    def test_duplicate_voices_are_rejected(self):
        original = three_takes()
        same = np.repeat(original[0][:, :1], 3, axis=1)
        with self.assertRaisesRegex(ValueError, "appear duplicated"):
            run([same, same, same], RATE, [12])

    def test_wrong_clock_and_silent_direct_mic_are_rejected(self):
        takes = three_takes()
        with self.assertRaisesRegex(ValueError, "16 kHz"):
            run(takes, 8000, [12])
        takes[0][:, 0] = 0.03
        with self.assertRaisesRegex(ValueError, "audible speech"):
            run(takes, RATE, [12])


if __name__ == "__main__":
    unittest.main()
