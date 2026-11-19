from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from headset_spill import RoomPresence, pilot_excerpts, read_pilots


class HeadsetSpillTest(unittest.TestCase):
    def test_identity_calibration_preserves_all_samples(self):
        rng = np.random.default_rng(1544)
        takes = []
        for speaker in range(3):
            audio = rng.normal(0, .002, (3, 32768)).astype(np.float32)
            audio[speaker] += rng.normal(0, .07, 32768).astype(np.float32)
            takes.append(audio)
        model = RoomPresence.fit(takes)
        raw = np.sum(takes, axis=0)
        direct = np.broadcast_to(np.eye(3, dtype=np.float32), (513, 3, 3))
        np.testing.assert_array_equal(model.reduce(raw, raw, direct), raw)
        self.assertEqual(model.reduce(raw[:, :0], raw[:, :0], direct).shape, (3, 0))

    def test_pilots_must_match_the_physical_microphone(self):
        mono = np.linspace(-.08, .08, 8192, dtype=np.float32)
        wrong = np.stack([mono, mono * 2, mono])
        with self.assertRaisesRegex(ValueError, "strongest"):
            pilot_excerpts([wrong, wrong, wrong])

    def test_rejects_files_with_invalid_channel_layout(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = [Path(temp) / f"{i}.wav" for i in range(3)]
            for path in paths:
                sf.write(path, np.zeros(8192), 16000)
            with self.assertRaisesRegex(ValueError, "channels"):
                read_pilots(paths)

if __name__ == "__main__":
    unittest.main()
