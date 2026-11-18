import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from process_meeting import (
    RATE, Separation, _BYTE_ALPHABET, active_spans, characters,
    has_distinct_sources, overlap_candidate, write_listening_report,
)


class PrefixSeparator(Separation):
    def infer(self, y, hints):
        output = y[None] * float(y.mean())
        return output, 1, 1


class SeparationChecks(unittest.TestCase):
    def test_margin_only_considers_adjacent_overlap(self):
        scores = {1: -2.0, 2: 3.5, 3: 3.0, 4: 3.1}
        self.assertEqual(overlap_candidate(scores, 2, 0.0), 2)
        self.assertEqual(overlap_candidate(scores, 2, 0.6), 3)
        self.assertEqual(overlap_candidate(scores, 1, 3.0), 1)
        self.assertEqual(overlap_candidate(scores, 4, 3.0), 4)

    def test_silent_and_duplicate_tracks_fail_overlap_guard(self):
        x = np.array([1, 0, 1, 0], np.float32)
        y = np.array([0, 1, 0, 1], np.float32)
        self.assertTrue(has_distinct_sources(x + y, np.stack((x, y))))
        self.assertFalse(has_distinct_sources(x + y, np.stack((x, x))))
        self.assertFalse(has_distinct_sources(x + y, np.stack((x, y * 0.01))))

    def test_no_speech_yields_no_tracks(self):
        engine = Separation(None, "cpu")
        tracks, commits = engine.separate(np.zeros(RATE, np.float32), 800, 200, 200, False)
        self.assertEqual(tracks, {})
        self.assertTrue(all(row["decoded_source_count"] is None for row in commits))
        self.assertEqual(active_spans(np.empty(0, dtype=np.float32)), [])

    def test_committed_prefix_is_independent_of_future_samples(self):
        engine = PrefixSeparator(None, "cpu")
        original = np.full(1500, 0.1, np.float32)
        changed = original.copy()
        changed[700:] = 0.8
        a, ac = engine.separate(original, 500, 100, 200, False)
        b, bc = engine.separate(changed, 500, 100, 200, False)
        np.testing.assert_array_equal(a["spk_A"][:500], b["spk_A"][:500])
        self.assertEqual(ac[0]["input_available_through"], 700)
        self.assertEqual(ac[-1]["sample_end"], len(original))
        self.assertEqual(len(ac), len(bc))

    def test_character_bytes_can_span_multiple_tokens(self):
        encoded = "".join(chr(_BYTE_ALPHABET[x]) for x in "四十".encode("utf-8"))
        out = characters([encoded[:2], encoded[2:]], [0.0, 0.08], 1.0, 1.5)
        self.assertEqual("".join(ch["text"] for ch in out), "四十")
        self.assertEqual(out[0]["start_ms"], 1000)
        self.assertTrue(all(1000 <= r["start_ms"] <= r["end_ms"] <= 1500 for r in out))

    def test_report_escapes_labels_transcript_and_filenames(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            info = {"input_file": "clip<&.wav",
                    "review_windows": 3,
                    "speakers": {"spk_A": {"display_name": "<A>",
                                            "file": "Speaker_A_张 三.wav"}}}
            rows = [{"speaker": "<A>", "start_ms": 0, "end_ms": 800, "text": "你好<script>"}]
            write_listening_report(directory, info, rows)
            html = (directory / "listen.html").read_text(encoding="utf-8")
            self.assertIn("clip&lt;&amp;.wav", html)
            self.assertIn("Speaker_A_%E5%BC%A0%20%E4%B8%89.wav", html)
            self.assertIn("你好&lt;script&gt;", html)
            self.assertNotIn("<script>", html)


if __name__ == "__main__":
    unittest.main()
