"""Field calibration and waveform export regressions."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import errno
from unittest.mock import patch

import numpy as np
import soundfile as sf

from boosted_close import BoostedMask
from close_refinement import load_parameters, refine
import close_waveform as close


class FieldCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        source = Path(self.temp.name)
        self.paths = [source / f"{speaker}.wav" for speaker in close.NAMES]
        self.output = source / "output"
        self.wave = np.random.default_rng(93).normal(0, .06, (3, 3329)).astype(np.float32)
        for name, data in zip(self.paths, self.wave, strict=True):
            sf.write(name, data, close.RATE, subtype="FLOAT")
        self.transfer = np.tile(np.eye(3, dtype=np.float32), (513, 1, 1))

    def read_outputs(self, directory):
        return np.stack([sf.read(directory / f"speaker_{speaker}.wav", dtype="float32")[0]
                         for speaker in close.NAMES])

    def test_missing_all_calibration_preserves_physical_audio(self):
        def no_calibration(audio, baseline):
            self.fail("A learned mask should not run without any calibrated incoming path")

        close.render(self.paths, self.output, self.transfer, refinement=no_calibration)
        np.testing.assert_array_equal(self.read_outputs(self.output), self.wave)

    def test_missing_receiver_preserves_only_that_microphone_in_all_modes(self):
        self.transfer[:, 0, 1] = .04
        self.transfer[:, 1, 2] = .08
        model = BoostedMask.load()
        parameters = load_parameters()
        for name, correct in [
            ("standard", None),
            ("gentle", lambda audio, base: refine(audio, base, parameters)),
            ("boosted", lambda audio, base: model.refine(audio, self.transfer, base)),
        ]:
            with self.subTest(name=name):
                base = close.suppress(self.wave, self.transfer, close.FROZEN_PROFILE)
                expected = correct(self.wave, base) if correct else base
                directory = self.output / name
                close.render(self.paths, directory, self.transfer, refinement=correct)
                actual = self.read_outputs(directory)
                np.testing.assert_array_equal(actual[:2], expected[:2])
                np.testing.assert_array_equal(actual[2], self.wave[2])

    def test_copied_and_inverted_mix_fail_before_publishing(self):
        for sign in (1, -1):
            sf.write(self.paths[2], self.wave[0] * sign, close.RATE, subtype="FLOAT")
            with self.assertRaisesRegex(ValueError, "A and C contain copies"):
                close.render(self.paths, self.output, self.transfer)
            self.assertFalse(any(self.output.glob("speaker_*.wav")))

    def test_common_silence_and_matching_intro_are_valid(self):
        silence = np.zeros(self.wave.shape[1], dtype=np.float32)
        for path in self.paths[1:]:
            sf.write(path, silence, close.RATE, subtype="FLOAT")
        close.render(self.paths, self.output / "silence", self.transfer)
        tail_difference = self.wave[0].copy()
        tail_difference[-1] += .1
        sf.write(self.paths[2], tail_difference, close.RATE, subtype="FLOAT")
        with patch.object(close, "CORE_SAMPLES", 1024):
            close.render(self.paths, self.output / "intro", self.transfer)
        np.testing.assert_array_equal(self.read_outputs(self.output / "intro")[2], tail_difference)

    def test_concurrent_publication_keeps_existing_audio(self):
        original = close.os.link
        def other_export(source, destination):
            if destination.name == "speaker_B.wav":
                destination.write_bytes(b"existing")
            return original(source, destination)
        with patch.object(close.os, "link", side_effect=other_export):
            with self.assertRaises(FileExistsError):
                close.render(self.paths, self.output, self.transfer)
        self.assertEqual((self.output / "speaker_B.wav").read_bytes(), b"existing")
        self.assertFalse((self.output / "speaker_A.wav").exists())
        self.assertFalse((self.output / "speaker_C.wav").exists())

    def test_export_on_volume_without_hard_links(self):
        with patch.object(close.os, "link", side_effect=OSError(errno.EOPNOTSUPP, "unsupported")):
            close.render(self.paths, self.output, self.transfer)
        np.testing.assert_array_equal(self.read_outputs(self.output), self.wave)

    def test_invalid_model_output_has_no_published_waveforms(self):
        self.transfer[:, 1, 0] = .07
        for name, wrong in [("short", lambda audio, base: base[:, :-1]),
                            ("nonfinite", lambda audio, base: base * np.nan)]:
            with self.subTest(name=name):
                directory = self.output / name
                with self.assertRaisesRegex(ValueError, "shortened or nonfinite"):
                    close.render(self.paths, directory, self.transfer, refinement=wrong)
                self.assertFalse(any(directory.glob("speaker_*.wav")))


if __name__ == "__main__":
    unittest.main()
