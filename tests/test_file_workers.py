import threading
import unittest

import numpy as np

from process_meeting import Separation, ordered_inference


class FileWorkerTests(unittest.TestCase):
    def test_windows_are_consumed_in_order_with_bounded_lookahead(self):
        available = []
        second_ready = threading.Event()

        def windows():
            for value in range(30):
                available.append(value)
                yield value

        def infer(value):
            if value == 0 and not second_ready.wait(timeout=5):
                raise RuntimeError("Second window did not start concurrently")
            if value == 1:
                second_ready.set()
            return value

        stream = ordered_inference(infer, windows(), 2)
        self.assertEqual(next(stream), 0)
        self.assertLessEqual(len(available), 3)
        self.assertEqual([*stream], list(range(1, 30)))

    def test_worker_failure_propagates_in_commit_order(self):
        def infer(value):
            if value == 2:
                raise LookupError("failed window")
            return value

        result = ordered_inference(infer, range(7), 3)
        self.assertEqual(next(result), 0)
        self.assertEqual(next(result), 1)
        with self.assertRaisesRegex(LookupError, "failed window"):
            next(result)

    def test_paced_and_accelerator_workers_are_rejected(self):
        wave = np.zeros(20, np.float32)
        for device, replay, workers in [("cpu", True, 2), ("cuda", False, 2), ("cpu", False, 0)]:
            with self.subTest(device=device, replay=replay, workers=workers):
                engine = Separation(None, device)
                with self.assertRaisesRegex(ValueError, "file workers"):
                    engine.separate(wave, 10, 0, 0, replay, file_workers=workers)
