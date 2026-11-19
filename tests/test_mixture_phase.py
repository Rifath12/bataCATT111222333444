from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from scipy.signal import istft, stft
import soundfile as sf

from mixture_phase import restore_wavs


class PhaseRestorationTest(unittest.TestCase):
    def test_streaming_pcm_matches_whole_file_transform_including_boundaries(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            rng = np.random.default_rng(12)
            sources = rng.normal(0, .04, (3, 30017)).astype(np.float32)
            sources[2, :10000] = 0
            mix = sources.sum(axis=0)
            estimates = sources * np.array([[1.7],[.6],[.8]], np.float32)
            filenames = [f"Speaker_{name}.wav" for name in "ABC"]
            sf.write(root / "mixed.wav", mix, 8000, subtype="PCM_16")
            for fname, y in zip(filenames, estimates, strict=True):
                sf.write(root / fname, y, 8000, subtype="PCM_16")
            out = root / "new"
            info = restore_wavs(root, out, filenames, block=1024)
            reloaded = np.stack([sf.read(root / name, dtype="float32")[0] for name in filenames])
            observation = sf.read(root / "mixed.wav", dtype="float32")[0]
            spec = dict(nperseg=256, noverlap=192)
            mags = np.abs(stft(reloaded, **spec)[2])
            weights = mags / np.maximum(1e-13, mags.sum(axis=0))
            expected = istft(weights * stft(observation, **spec)[2], **spec)[1][:,:len(mix)]
            expected *= info["playback_gain"]
            actual = np.stack([sf.read(out / name, dtype="float32")[0] for name in filenames])
            np.testing.assert_allclose(actual, expected, atol=1.01/32768)
            np.testing.assert_array_equal(actual[2,:9500], 0)
            with self.assertRaises(FileExistsError):
                restore_wavs(root, out, filenames)

    def test_short_silent_input_and_unsafe_names(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("Speaker_A.wav", "mixed.wav"):
                sf.write(root / name, np.zeros(100), 8000)
            restore_wavs(root, root / "out", ["Speaker_A.wav"])
            np.testing.assert_array_equal(sf.read(root / "out" / "Speaker_A.wav")[0], np.zeros(100))
            with self.assertRaises(ValueError):
                restore_wavs(root, root / "bad", ["../escaping.wav"])
