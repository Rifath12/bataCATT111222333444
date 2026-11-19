import unittest
import numpy as np

from audio_input import fold_channels


class FoldChannelsTests(unittest.TestCase):
    def setUp(self):
        self.wave = np.random.default_rng(12).normal(0, .04, 98347).astype(np.float32)

    def test_opposite_polarity_preserves_speech(self):
        np.testing.assert_array_equal(fold_channels(np.column_stack([self.wave, -self.wave])), self.wave)

    def test_silent_channel_does_not_halve_the_occupied_channel(self):
        for position in (0, 1):
            data = np.zeros((len(self.wave), 2), np.float32)
            data[:, position] = self.wave
            np.testing.assert_array_equal(fold_channels(data), self.wave)

    def test_mono_and_ordinary_stereo(self):
        np.testing.assert_array_equal(fold_channels(self.wave[:, None]), self.wave)
        stereo = np.column_stack([self.wave, self.wave[::-1]])
        np.testing.assert_array_equal(fold_channels(stereo), stereo.mean(axis=1))
        silence = np.zeros((100, 3), dtype=np.float32)
        np.testing.assert_array_equal(fold_channels(silence), silence[:, 0])
