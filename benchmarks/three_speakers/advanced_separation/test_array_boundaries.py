"""Regressions for clipped turns, bounded decoding audio and FFT block boundaries."""

from pathlib import Path
import tempfile
import unittest

import array_context as context
import numpy as np
import soundfile as sf
import spatial_render as render


def turn(start, end, speaker="A"):
    return dict(speaker=speaker, start=start, end=end)


class ContextRegressions(unittest.TestCase):
    def test_clip_at_each_recording_edge(self):
        rows = [turn(-0.4, -0.01), turn(-0.2, 0.15, "B"), turn(0.95, 1.3, "C")]
        timing = list(context.valid_turns(rows, 1.0))
        self.assertEqual(timing, [turn(0.0, 0.15, "B"), turn(0.95, 1.0, "C")])
        mask = context.activity(rows, 16000)
        self.assertFalse(mask[:, 0].any())
        self.assertEqual(mask[:, 1].sum(), 15)
        self.assertEqual(mask[:, 2].sum(), 5)
        self.assertEqual(list(context.spans(timing[1:], 1.0, 0.42, 0.18, 17)), [(0.77, 1.0)])

    def test_reject_invalid_labels_and_times_even_if_not_active(self):
        for rows in ([turn(0, float("nan"))], [turn(2, 1)], [turn(2, 3, "AB")]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                context.activity(rows, 16000)

    def test_dedup_padding_at_duration_cap(self):
        rows = [turn(0.3, 15.9), turn(16.01, 20.0)]
        windows = list(context.spans(rows, 21.0, 0.42, 0.18, 17))
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0][1], windows[1][0])
        self.assertGreater(windows[0][1], 15.9)
        self.assertLess(windows[1][0], 16.01)

    def test_nested_turns_produce_no_overlapping_decodes(self):
        rows = [turn(0.0, 18.0), turn(3, 17.5), turn(18.2, 19.4), turn(18.3, 20.1)]
        windows = list(context.spans(rows, 21.0, 0.42, 0.18, 17))
        self.assertTrue(all(a < b for a, b in windows))
        self.assertTrue(all(right <= next_left for (_, right), (next_left, _) in zip(windows, windows[1:])))
        self.assertEqual(windows[0][0], 0.0)
        self.assertEqual(windows[-1][1], 20.28)

    def test_streaming_power_is_identical_including_mid_block_crop(self):
        rng = np.random.default_rng(630)
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/"beam.flac"
            sf.write(path, rng.normal(0.0, .12, 180_019), 16000, subtype="PCM_16")
            with sf.SoundFile(path) as source:
                for length in (159, 160, 10001, 160000, 180019):
                    source.seek(0)
                    frames = (length + context.FRAME - 1) // context.FRAME
                    y = np.pad(source.read(length, dtype="float32"), (0, frames * context.FRAME - length))
                    expected = context.uniform_filter1d(np.mean(y.reshape(frames, context.FRAME) ** 2, axis=1), 7)
                    np.testing.assert_array_equal(expected, context.beam_power(source, length))


class RendererRegressions(unittest.TestCase):
    @staticmethod
    def mask_and_frames():
        rng = np.random.default_rng(91)
        shape = (13, 18, 8)
        x = rng.normal(size=shape) + 1j*rng.normal(size=shape)
        angles = rng.normal(size=(4, shape[1], 8, 8)) + 1j*rng.normal(size=(4, shape[1], 8, 8))
        cov = angles @ angles.conj().swapaxes(-1, -2) + np.eye(8)
        active = np.ones((3, 13), dtype=bool)
        active[1:, 1:3] = False
        y = np.broadcast_to(x[:, :, 0][:, None, :], (13, 3, 18)).copy()
        return render.SpatialOverlapMask(cov, active), x.astype("complex64"), y.astype("complex64")

    def test_partition_has_same_likelihood_as_full_input(self):
        mask, x, y = self.mask_and_frames()
        expected = mask.apply(x, y.copy(), 0, len(x))
        got = y.copy()
        for start, stop in [(0, 4), (4, 9), (9, len(x))]:
            lo, hi = max(0, start - 1), min(len(x), stop + 1)
            mask.apply(x[lo:hi], got[start:stop], start, stop, spectrum_start=lo)
        np.testing.assert_allclose(expected, got, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(got[1:3], y[1:3])
        without_halo = mask.apply(x[4:9], y.copy()[4:9], 4, 9)
        self.assertGreater(np.abs(expected[4] - without_halo[0]).max(), 1e-3)

    def test_static_beam_preserves_clock_and_identity_at_chunk_and_eof(self):
        rng = np.random.default_rng(48)
        wave = rng.normal(0, .15, (139123, 2)).astype("float32")
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            sf.write(root/"source.wav", wave, 16000, subtype="FLOAT")
            weights = np.zeros((3, 321, 2), dtype="complex64")
            weights[0, :, 0] = 1
            weights[1, :, 1] = 1
            weights[2, :, :] = 0.5
            with sf.SoundFile(root/"source.wav") as source:
                frames = (len(source) + render.HOP - 1) // render.HOP
                gain = render.write_continuous_beams(source, weights, frames, root/"new.wav")
            actual = sf.read(root/"new.wav", dtype="float32")[0]
            self.assertEqual(gain, 1.0)
            self.assertEqual(len(actual), len(wave))
            np.testing.assert_allclose(actual[:, :2], wave, rtol=1e-6, atol=5e-7)


if __name__ == "__main__":
    unittest.main()
