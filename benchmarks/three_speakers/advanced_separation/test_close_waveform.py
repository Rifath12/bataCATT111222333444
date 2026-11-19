"""Regression checks for dedicated-mic waveform rendering."""
from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

import close_waveform as close


def transfer(leakage=0.0):
    model = np.tile(np.eye(3, dtype=np.float32), (close.FFT // 2 + 1, 1, 1))
    model[:, 1, 0] = leakage
    model[:, 2, 1] = leakage * 0.75
    return model


class CloseWaveformTest(unittest.TestCase):
    def test_identity_preserves_even_tiny_inputs(self):
        samples = np.random.default_rng(31).normal(size=(3, 71)).astype(np.float32)
        np.testing.assert_array_equal(close.suppress(samples, transfer(), close.FROZEN_PROFILE), samples)

    def test_empty_dominant_speech_falls_back_to_identity(self):
        p = np.zeros((3, 2))
        models, diagnostics = close.fit_power_transfer([], p, p)
        for result in models.values():
            np.testing.assert_array_equal(result, transfer())
        assert all(row["selected_dominant"] == 0 for row in diagnostics)

    def test_render_preserves_sample_alignment_and_block_boundaries(self):
        rng = np.random.default_rng(319)
        wave = (rng.normal(0, 0.03, size=(3, 11731))).astype(np.float32)
        wave[1] += wave[0] * 0.20
        wave[2] += wave[1] * 0.15
        target = close.suppress(wave, transfer(0.075), close.FROZEN_PROFILE)
        with TemporaryDirectory() as work:
            root = Path(work)
            paths = [root / f"input_{label}.wav" for label in close.NAMES]
            for path, samples in zip(paths, wave, strict=True):
                sf.write(path, samples, close.RATE, subtype="FLOAT")
            output = root / "out"
            with patch.object(close, "CORE_SAMPLES", 4096):
                close.render(paths, output, transfer(0.075))
            produced = np.array([sf.read(output / f"speaker_{label}.wav", dtype="float32")[0] for label in close.NAMES])
            np.testing.assert_allclose(produced, target, rtol=1e-5, atol=3e-7)
            self.assertEqual(produced.shape, wave.shape)
            with self.assertRaises(FileExistsError):
                close.render(paths, output, transfer())

    def test_reject_linked_duplicate_inputs(self):
        with TemporaryDirectory() as work:
            root = Path(work)
            paths = [root / f"input_{label}.wav" for label in close.NAMES]
            for path in paths[:2]:
                sf.write(path, np.zeros(128), close.RATE)
            os.link(paths[0], paths[2])
            with self.assertRaises(ValueError):
                close.validate_inputs(paths)


if __name__ == "__main__":
    unittest.main()
