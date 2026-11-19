from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import soundfile as sf

from rolling_close import dominant_speech, estimate_transfer, render


class RollingMicrophonesTest(unittest.TestCase):
    def test_silence_and_equal_channels_are_ineligible(self):
        rng = np.random.default_rng(20261120)
        voice = rng.normal(0., .05, 160000).astype(np.float32)
        for audio in (np.zeros((3, voice.size), np.float32), np.stack([voice] * 3)):
            self.assertEqual([len(group) for group in dominant_speech(audio)], [0, 0, 0])

    def test_update_requires_repeated_evidence_and_preserves_channel_order(self):
        rng = np.random.default_rng(3)
        direct = rng.normal(0., .06, 131072).astype(np.float32)
        # Interleave quiet and active seconds. Only physical A is active.
        direct[(np.arange(direct.size) // 16000) % 2 == 0] = 0
        # The calibrator requires speech-band energy. Use an explicit 1 kHz carrier.
        direct *= .25
        direct += np.where((np.arange(direct.size) // 16000) % 2 == 1,
                           .08 * np.sin(np.arange(direct.size) * 2 * np.pi / 16), 0).astype(np.float32)
        evidence = dominant_speech(np.stack([direct, .15 * direct, .05 * direct]))
        self.assertGreaterEqual(len(evidence[0]), 3)
        self.assertEqual([len(k) for k in evidence[1:]], [0, 0])
        initial = np.broadcast_to(np.eye(3, dtype=np.float32), (513, 3, 3)).copy()
        initial[:, 1, 0] = .002
        initial[:, 2, 0] = .001
        unchanged, _ = estimate_transfer(initial, initial, [evidence])
        np.testing.assert_array_equal(unchanged, initial)
        changed, details = estimate_transfer(initial, initial, [evidence, evidence])
        self.assertGreater(np.median(changed[:, 1, 0]), .002)
        self.assertEqual(details[0]["blocks"], 2)
        np.testing.assert_array_equal(changed[:, :, 1:], initial[:, :, 1:])
        np.testing.assert_array_equal(changed.diagonal(axis1=1, axis2=2),
                                      initial.diagonal(axis1=1, axis2=2))

    def test_recorded_waveform_copies_are_rejected(self):
        rng = np.random.default_rng(23)
        with TemporaryDirectory() as temp:
            p = Path(temp)
            paths = [p / f"{c}.wav" for c in "ABC"]
            x = rng.normal(0, .1, 8192).astype(np.float32)
            initial = np.broadcast_to(np.eye(3, dtype=np.float32), (513, 3, 3)).copy()
            np.save(p / "initial.npy", initial)
            for sign in (-1., 1.):
                for path, signal in zip(paths, (x, sign*x, x[::-1]), strict=True):
                    sf.write(path, signal, 16000, subtype="FLOAT")
                with self.assertRaisesRegex(ValueError, "copies of the same signal"):
                    render(paths, p / "initial.npy", p / "rejected")
                self.assertFalse((p / "rejected").exists())

    def test_short_identity_and_failed_stage(self):
        with TemporaryDirectory() as temp:
            p = Path(temp)
            initial = np.broadcast_to(np.eye(3, dtype=np.float32), (513, 3, 3)).copy()
            np.save(p / "input.npy", initial)
            paths = [p / f"{c}.wav" for c in "ABC"]
            for i, path in enumerate(paths):
                sf.write(path, np.full(159, .03 * i, np.float32), 16000, subtype="FLOAT")
            report = render(paths, p / "input.npy", p / "export")
            self.assertEqual(report["samples"], 159)
            for path in paths:
                np.testing.assert_array_equal(sf.read(path)[0], sf.read(p / "export" / f"speaker_{path.name}")[0])
            with self.assertRaises(FileExistsError):
                render(paths, p / "input.npy", p / "export")
            sf.write(paths[2], np.full(159, np.nan, np.float32), 16000, subtype="FLOAT")
            with self.assertRaises(ValueError):
                render(paths, p / "input.npy", p / "failed")
            self.assertFalse((p / "failed").exists())


if __name__ == "__main__":
    unittest.main()
