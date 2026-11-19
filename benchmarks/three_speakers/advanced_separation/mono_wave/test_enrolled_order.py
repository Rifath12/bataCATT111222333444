import unittest

import numpy as np

from enrolled_order import SAMPLE_RATE, assign_tracks, fit_templates


class EnrollmentTests(unittest.TestCase):
    def test_recovers_known_order_across_scale_and_sign(self):
        t = np.arange(8192) / SAMPLE_RATE
        voices = np.stack([.04 * np.sin(2 * np.pi * f * t) for f in (240, 740, 1600)])
        profiles = fit_templates([[v] for v in voices], SAMPLE_RATE)
        order, margin = assign_tracks(profiles, voices[[2, 0, 1]] * np.array([1.4, -.2, .6])[:, None], SAMPLE_RATE)
        np.testing.assert_array_equal(order, [1, 2, 0])
        self.assertGreater(margin, 0)

    def test_declines_missing_enrollment_and_silent_output(self):
        with self.assertRaises(ValueError):
            fit_templates([[np.zeros(8192)]] * 3, SAMPLE_RATE)
        with self.assertRaises(ValueError):
            assign_tracks(np.zeros((3, 12)), np.zeros((3, 8192)), SAMPLE_RATE)

    def test_declines_other_rates(self):
        with self.assertRaises(ValueError):
            fit_templates([[], [], []], 8000)


class WaveCliTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        import soundfile as sf

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        t = np.arange(16000) / SAMPLE_RATE
        tones = np.stack([.12 * np.sin(2 * np.pi * f * t) for f in (250, 930, 1680)])
        self.enrollment = [self.folder / f"ref_{n}.wav" for n in range(3)]
        self.tracks = [self.folder / f"slot_{n}.wav" for n in range(3)]
        for n, path in enumerate(self.enrollment):
            sf.write(path, tones[n], SAMPLE_RATE, subtype="PCM_16")
        for i, n in enumerate((2, 0, 1)):
            sf.write(self.tracks[i], tones[n] * [.5, -1., .8][i], SAMPLE_RATE,
                     subtype=["PCM_24", "FLOAT", "PCM_16"][i])
        self.out = self.folder / "labeled"

    def invoke(self):
        from pathlib import Path
        import subprocess
        import sys

        return subprocess.run(
            [sys.executable, str(Path(__file__).with_name("enrolled_order.py")), "--tracks", *self.tracks,
             "--enroll-a", self.enrollment[0], "--enroll-b", self.enrollment[1],
             "--enroll-c", self.enrollment[2], "--out", self.out],
            capture_output=True, text=True, check=False)

    def test_cli_preserves_input_containers_and_refuses_to_overwrite(self):
        import hashlib
        import json

        before = [p.read_bytes() for p in self.tracks]
        called = self.invoke()
        self.assertEqual(called.returncode, 0, called.stderr)
        mapping = json.loads((self.out / "mapping.json").read_text())
        for name, expected in zip("ABC", (1, 2, 0), strict=True):
            row = mapping["label_mapping"][name]
            self.assertEqual(row["input_track_number"], expected + 1)
            audio = (self.out / row["output_wav"]).read_bytes()
            self.assertEqual(audio, before[expected])
            self.assertEqual(row["sha256"], hashlib.sha256(audio).hexdigest())
        sentinel = self.out / "Speaker_A.wav"
        after = sentinel.read_bytes()
        self.assertNotEqual(self.invoke().returncode, 0)
        self.assertEqual(sentinel.read_bytes(), after)
        self.assertEqual(before, [p.read_bytes() for p in self.tracks])

    def test_cli_declines_tied_profiles_and_misaligned_inputs(self):
        import shutil
        import soundfile as sf

        saved = self.enrollment[0].read_bytes()
        shutil.copyfile(self.enrollment[1], self.enrollment[0])
        call = self.invoke()
        self.assertNotEqual(call.returncode, 0)
        self.assertIn("unique", call.stderr)
        self.assertFalse(self.out.exists())
        self.enrollment[0].write_bytes(saved)
        wave, sr = sf.read(self.tracks[2])
        sf.write(self.tracks[2], wave[:-1], sr)
        self.assertNotEqual(self.invoke().returncode, 0)
        self.assertFalse(self.out.exists())
        sf.write(self.tracks[2], wave, 8000)
        self.assertNotEqual(self.invoke().returncode, 0)
        self.assertFalse(self.out.exists())
