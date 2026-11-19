from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import soundfile as sf

from meeting_roster import EnrolledRoster
from process_meeting import RATE, Separation, track_id


class FixedOutputs(Separation):
    def __init__(self, blocks, **kwargs):
        super().__init__(None, "cpu", **kwargs)
        self.blocks = iter(blocks)

    def infer(self, y, hints):
        result = next(self.blocks)
        return result, len(result) or None, len(result) or None


class RosterTest(unittest.TestCase):
    def test_declared_roster_exports_silence_and_complete_clock(self):
        wave = np.zeros((RATE + 1,), np.float32)
        engine = Separation(None, "cpu", participants=3)
        chunks = []
        tracks, rows = engine.separate(wave, 3031, 100, 72, False,
                                       on_commit=lambda row, outputs: chunks.append(outputs))
        self.assertEqual(list(tracks), ["spk_A", "spk_B", "spk_C"])
        self.assertTrue(all(row["speaker_ids"] == [] for row in rows))
        for key in tracks:
            np.testing.assert_array_equal(tracks[key], wave)
            np.testing.assert_array_equal(np.concatenate([c[key] for c in chunks]), tracks[key])

    def test_three_source_override_never_creates_a_fourth_slot_after_a_gap(self):
        rng = np.random.default_rng(27)
        blocks = [rng.normal(size=(3, RATE)).astype(np.float32),
                  np.zeros((0, RATE), np.float32), rng.normal(size=(3, RATE)).astype(np.float32)]
        engine = FixedOutputs(blocks)
        result, rows = engine.separate(np.zeros(3 * RATE, np.float32), RATE, 0, 0, False, hints=3)
        self.assertEqual(set(result), {track_id(i) for i in range(3)})
        self.assertTrue(rows[2]["identity_needs_review"])

    def test_single_stream_override_can_follow_more_than_one_participant(self):
        rng = np.random.default_rng(17)
        blocks = [rng.normal(size=(1, RATE)).astype(np.float32) for _ in range(3)]
        engine = FixedOutputs(blocks)
        tracks, _ = engine.separate(np.zeros(3 * RATE, np.float32), RATE, 0, 0, False, hints=1)
        self.assertEqual(set(tracks), {track_id(i) for i in range(3)})

    def test_common_solo_recording_formats_preserve_voice_identity(self):
        with TemporaryDirectory() as temporary:
            paths = [Path(temporary) / f"{s}.wav" for s in "ABC"]
            rates = (44100, 48000, 16000)
            low = []
            for path, rate, hz in zip(paths, rates, (250, 720, 1560), strict=True):
                wave = .1 * np.sin(2 * np.pi * hz * np.arange(rate) / rate)
                if rate == 44100:
                    wave = np.column_stack((wave, -wave))
                if rate == 48000:
                    wave = np.column_stack((np.zeros_like(wave), wave))
                sf.write(path, wave, rate, subtype="FLOAT")
                low.append(.1 * np.sin(2 * np.pi * hz * np.arange(RATE) / RATE))
            router = EnrolledRoster(paths)
            ordered, _ = router.route(np.asarray(low, np.float32)[[2,0,1]], 0, RATE)
            np.testing.assert_allclose(ordered, np.asarray(low, np.float32), atol=1e-6)

    def test_enrolled_voices_can_take_turns_in_a_single_decoder_slot(self):
        with TemporaryDirectory() as temporary:
            t = np.arange(32768) / 16000
            waves = [.1 * np.sin(2 * np.pi * hz * t) for hz in (250, 720, 1560)]
            paths = [Path(temporary) / f"{s}.wav" for s in "ABC"]
            for file, wave in zip(paths, waves, strict=True):
                sf.write(file, wave, 16000, subtype="FLOAT")
            router = EnrolledRoster(paths)
            low = [w[::2].astype(np.float32) for w in waves]
            mixed = np.concatenate((low[2],low[0],low[1]))[None]
            ordered, tiles = router.route(mixed, 0, mixed.shape[1])
            np.testing.assert_allclose(ordered.sum(axis=0), mixed[0], atol=1e-7)
            for n, expected in enumerate((2, 0, 1)):
                seg = ordered[:, n * len(low[0]) + RATE//2:(n+1)*len(low[0]) - RATE//2]
                self.assertEqual(np.argmax(np.mean(seg**2, axis=1)), expected)
            self.assertTrue(tiles)
            tiny = router.route(np.zeros((0, 4), np.float32), 0, 4)
            self.assertEqual(tiny[0].shape, (3, 4))
            with self.assertRaisesRegex(ValueError, "distinct"):
                EnrolledRoster([paths[0]] * 3)
