"""Offline solo calibration and routing regression checks."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import soundfile as sf

from calibrate_close import calibrate, save
from close_waveform import RATE, stored_transfer


class SoloCalibrationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        rng = np.random.default_rng(9)
        self.coefficients = .1 + .1*np.eye(3, dtype=np.float32)
        np.fill_diagonal(self.coefficients, 1.)
        self.coefficients[2, 0] = .31
        self.paths = []
        for idx, label in enumerate("ABC"):
            voice = rng.normal(0, .02, 3 * RATE).astype(np.float32)
            path = self.root / f"{label}.wav"
            sf.write(path, voice[:, None] * self.coefficients[None, :, idx], RATE, subtype="FLOAT")
            self.paths.append(path)

    def test_recovers_path_powers_and_reloads_in_renderer(self):
        measured, report = calibrate(self.paths)
        np.testing.assert_allclose(np.median(measured, axis=0), self.coefficients**2, atol=2e-6)
        self.assertEqual(report["matrix_axes"], ["frequency", "microphone", "speaker"])
        self.assertTrue(all(t["selected_windows"] >= 60 for t in report["takes"]))
        out = self.root / "model.npy"
        save(measured, report, out)
        np.testing.assert_array_equal(stored_transfer(out), measured)
        persisted = json.loads(out.with_suffix(".json").read_text())
        self.assertEqual(persisted.pop("transfer_sha256"), hashlib.sha256(out.read_bytes()).hexdigest())
        self.assertEqual(persisted, report)
        with self.assertRaises(FileExistsError):
            save(measured, report, out)
        self.assertTrue(out.with_suffix(".json").exists())

    def test_rejects_swapped_repeated_or_short_takes(self):
        for paths in ([self.paths[1], self.paths[0], self.paths[2]], [self.paths[0]]*3):
            with self.assertRaises(ValueError):
                calibrate(paths)
        sf.write(self.paths[0], np.zeros((RATE,3)), RATE)
        with self.assertRaisesRegex(ValueError,"60 active"):
            calibrate(self.paths)

    def test_rejects_silence_dc_wrong_layout_rate_and_nonfinite_tail(self):
        for data, rate in ((np.zeros((3*RATE,3)), RATE), (np.ones((3*RATE,3))*.1, RATE),
                           (np.zeros((3*RATE,2)),RATE), (np.zeros((3*RATE,3)),8000)):
            with self.subTest(rate=rate, shape=data.shape):
                sf.write(self.paths[0], data, rate, subtype="FLOAT")
                with self.assertRaises(ValueError):
                    calibrate(self.paths)
        samples = np.zeros((3*RATE,3), dtype=np.float32)
        samples[-1] = np.nan
        sf.write(self.paths[0], samples, RATE, subtype="FLOAT")
        with self.assertRaisesRegex(ValueError,"nonfinite"):
            calibrate(self.paths)

    def test_stale_matrix_or_report_is_preserved(self):
        matrix, report = calibrate(self.paths)
        for suffix, other in ((".npy",".json"), (".json",".npy")):
            out = self.root / "stale.npy"
            orphan = out.with_suffix(suffix)
            orphan.write_bytes(b"preserve")
            with self.assertRaises(FileExistsError):
                save(matrix, report, out)
            self.assertEqual(orphan.read_bytes(), b"preserve")
            self.assertFalse(out.with_suffix(other).exists())
            orphan.unlink()

    def test_racing_calibration_writers_publish_one_matching_pair(self):
        matrix, report = calibrate(self.paths)
        out = self.root / "shared.npy"

        def write(_):
            try:
                save(matrix, report, out)
                return True
            except FileExistsError:
                return False

        with ThreadPoolExecutor(max_workers=4) as workers:
            self.assertEqual(sum(workers.map(write,range(8))), 1)
        digest = json.loads(out.with_suffix(".json").read_text())["transfer_sha256"]
        self.assertEqual(digest, hashlib.sha256(out.read_bytes()).hexdigest())
        np.testing.assert_array_equal(stored_transfer(out), matrix)

    def test_rejects_a_replaced_solo_calibration(self):
        matrix, report = calibrate(self.paths)
        out = self.root / "setup.npy"
        save(matrix, report, out)
        matrix[:, 0, 1] *= .5
        np.save(out, matrix, allow_pickle=False)
        with self.assertRaisesRegex(ValueError, "does not match"):
            stored_transfer(out)

    def test_warns_on_paths_outside_the_model_range(self):
        voice, _ = sf.read(self.paths[0], dtype="float32")
        voice[:,1] = .60 * voice[:,0]
        sf.write(self.paths[0], voice, RATE, subtype="FLOAT")
        matrix, report = calibrate(self.paths)
        self.assertEqual(report["takes"][0]["above_model_bleed_limit"], ["B"])
        np.testing.assert_allclose(matrix[:,1,0], .14, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
