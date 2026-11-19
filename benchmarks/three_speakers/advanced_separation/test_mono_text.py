"""Regressions for normalized character edits and JSONL handling."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import soundfile as sf

from mono_text import aligned_tracks, amend, export_rows, key, read_rows, whole_turn_matches
from recognize import text_scores


def utterance(content, speaker="A"):
    return {"speaker": speaker, "start": 1.0, "end": 3.0, "text": content}


class TranscriptTests(unittest.TestCase):
    def test_interior_edit_preserves_display_punctuation_case_and_markup(self):
        row = dict(utterance("今日，去找Ａlice！<noise>"), characters=[{"text": "旧"}], tokens=["foo"], timestamps=[1.2])
        views = {key(row): ("今日要找alice", "今日要找alice")}
        result, changed = amend([row], views)
        self.assertEqual(result, [utterance("今日，要找Ａlice！<noise>")])
        self.assertEqual(changed, 1)

    def test_single_whole_turn_edit_requires_exact_agreement(self):
        row = utterance("早上天气很好！")
        for pair, expected in [
            (("今天气很好", "今天气很好"), "早上天气很好！"),
            (("早上天气很好啊", "早上天气很好呢"), "早上天气很好！"),
            (("早上天气很好啊", "早上天气很好啊"), "早上天气很好啊！"),
        ]:
            with self.subTest(pair=pair):
                result, edits = whole_turn_matches([row], {key(row):pair})
                self.assertEqual(result[0]["text"], expected)
                self.assertEqual(edits, int(expected != row["text"]))

    def test_export_jsonl_and_separate_speakers(self):
        with TemporaryDirectory() as work:
            path = Path(work)
            export_rows([utterance("你好。"), utterance("你好！", "B")], path)
            self.assertEqual(read_rows(path/"transcript.jsonl"), [utterance("你好。"), utterance("你好！", "B")])
            self.assertIn("你好。", (path/"speaker_A.txt").read_text())
            self.assertIn("B  你好！", (path/"transcript.txt").read_text())
            self.assertEqual((path/"speaker_C.txt").read_text(), "")

    def test_reject_duplicate_and_invalid_labels(self):
        with TemporaryDirectory() as work:
            path = Path(work)/"invalid.jsonl"
            for rows in ([utterance("甲"), utterance("乙")], [utterance("丙", "AB")]):
                path.write_text("".join(json.dumps(row)+"\n" for row in rows))
                with self.assertRaisesRegex(ValueError, "invalid.jsonl"):
                    read_rows(path)

    def test_reject_non_aligned_input_tracks(self):
        with TemporaryDirectory() as work:
            path = Path(work)
            for name, length in zip("ABC", [512,512,511]):
                sf.write(path/f"speaker_{name}.wav", np.zeros(length), 16000)
            with self.assertRaisesRegex(ValueError, "identical"):
                aligned_tracks(path)

    def test_no_reference_has_undefined_cer_and_exposes_insertions(self):
        self.assertIsNone(text_scores([], [])["cpCER"])
        result = text_scores([], [utterance("你好")])
        self.assertIsNone(result["cpCER"])
        self.assertEqual(result["totals"]["insertions"], 2)


if __name__ == "__main__":
    unittest.main()
