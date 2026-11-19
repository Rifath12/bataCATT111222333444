"""Regression tests for time clipping, speaker attribution and sparse edit scoring."""

from random import Random
from tempfile import TemporaryDirectory
from pathlib import Path
import unittest

from audit_metrics import (
    Token,
    activity_scores,
    audit,
    clock,
    constrained_matches,
    score_characters,
    spans,
    textgrid,
    timed_characters,
)


def turn(speaker, start, end, text=""):
    return dict(speaker=speaker, start=start, end=end, text=text)


def dense_edits(reference, hypothesis, tolerance):
    """Simple quadratic dynamic program used only as the test oracle."""
    scores = list(range(len(hypothesis) + 1))
    for i, a in enumerate(reference, 1):
        old, scores = scores, [i]
        for j, b in enumerate(hypothesis, 1):
            options = [old[j] + 1, scores[j - 1] + 1]
            if abs(a.tick - b.tick) <= tolerance:
                options.append(old[j - 1] + (a.char != b.char))
            scores.append(min(options))
    return scores[-1]


class AuditTest(unittest.TestCase):
    def test_sparse_chain_matches_dense_edit_oracle(self):
        random = Random(616)
        for _ in range(300):
            ref, hyp = (
                [Token(random.choice("甲乙丙"), random.randrange(-3, 7), False)
                 for _ in range(random.randrange(8))]
                for _ in range(2)
            )
            tolerance = random.randrange(5)
            links = list(constrained_matches(ref, hyp, tolerance))
            self.assertTrue(all(a < b and c < d for (a, c), (b, d) in zip(links, links[1:])))
            self.assertTrue(all(abs(ref[i].tick - hyp[j].tick) <= tolerance for i, j in links))
            edits = len(ref) + len(hyp) - sum(1 + (ref[i].char == hyp[j].char) for i, j in links)
            self.assertEqual(edits, dense_edits(ref, hyp, tolerance))

    def test_activity_sweep_deduplicates_one_speaker_and_scores_wrong_labels(self):
        reference = [turn("X", -1, 1.7), turn("X", 0.1, 0.6), turn("Y", 1, 1.3),
                     turn("Z", 1.1, 1.3)]
        predicted = [turn("A", 0, 1.7), turn("B", 1.1, 1.22), turn("D", 1.1, 1.3)]
        result = activity_scores(spans(reference, predicted, clock(1.5)), {"A": "X", "B": "Y"})
        triple = result["count_3"]
        self.assertAlmostEqual(triple["clock_s"], 0.2)
        self.assertAlmostEqual(triple["reference_speaker_s"], 0.6)
        self.assertAlmostEqual(triple["speaker_recall"], 0.32 / 0.6)
        self.assertAlmostEqual(triple["confused_speaker_s"], 0.2)
        self.assertAlmostEqual(triple["missed_speaker_s"], 0.08)
        self.assertEqual(triple["exact_speaker_set_accuracy"], 0)

    def test_global_mapping_uses_activity_and_exposes_swapped_transcripts(self):
        ref = [turn("X", 0, 1, "甲"), turn("Y", 1, 2, "乙")]
        timing = [turn("A", 0, 1), turn("B", 1, 2)]
        wrong_text = [turn("A", 0, 1, "乙"), turn("B", 1, 2, "甲")]
        result = audit(ref, ref, wrong_text, timing, 2, [0.1])
        self.assertEqual(result["activity_mapping"], {"A": "X", "B": "Y"})
        self.assertEqual(result["cpCER"]["edits"], 0)
        self.assertEqual(result["text_fixed_activity_mapping"]["edits"], 2)
        self.assertEqual(result["timed"]["0.1"]["full"]["edits"], 2)

    def test_external_mapping_and_activity_only_exact_window(self):
        ref = [turn("X", 0, 2), turn("Y", 0.1, 1.9), turn("Z", 0.2, 1.8)]
        hyp = [turn("A", 0, 2), turn("B", 0.1, 1.9), turn("C", 0.4, 1.0)]
        fixed = {"A": "X", "B": "Y", "C": "Z"}
        report = audit(ref, ref, None, hyp, 2, [0.1], speaker_map=fixed)
        self.assertNotIn("cpCER", report)
        self.assertEqual(report["mapping_policy"], "fixed")
        (window,) = report["exact_three_windows"]["RTTM"]
        self.assertEqual((window["start_s"], window["end_s"]), (0.2, 1.8))
        self.assertAlmostEqual(window["exact_set_fraction"], 0.6 / 1.6)
        detection = report["exact_three_detection"]["RTTM"]
        self.assertAlmostEqual(detection["recall"], 0.6 / 1.6)
        self.assertEqual(detection["precision"], 1)
        with self.assertRaises(ValueError):
            audit(ref, ref, None, hyp, 2, [], speaker_map={"A": "X", "B": "X"})

    def test_time_constrained_scores_ignore_duplicate_text_far_away(self):
        reference = [turn("X", 0, 1, "甲甲")]
        hypothesis = [turn("A", 2, 3, "甲甲")]
        result = audit(reference, reference, hypothesis, hypothesis, 4, [0.5])
        timed = result["timed"]["0.5"]
        self.assertEqual(timed["full"]["edits"], 4)
        self.assertEqual(timed["count_1"]["deletions"], 2)
        self.assertEqual(timed["count_0"]["insertions"], 2)

    def test_empty_and_extra_speakers_are_counted_without_fake_rate(self):
        self.assertIsNone(score_characters([], [])["CER"])
        self.assertIsNone(score_characters([], [turn("A", 0, 1, "甲")])["CER"])
        self.assertEqual(score_characters([], [turn("A", 0, 1, "甲")])["edits"], 1)
        self.assertEqual(score_characters([turn("X", 0, 1, "乙")], [])["edits"], 1)
        trace = audit([], [turn("X", 0, 1)], [turn("A", 0, 1, "甲")],
                      [turn("A", 0, 1)], 1, [0.5])
        self.assertEqual(trace["text_fixed_activity_mapping"]["edits"], 1)
        self.assertEqual(trace["timed"]["0.5"]["full"]["edits"], 1)

    def test_supplied_single_character_timestamp_is_preserved(self):
        row = turn("X", 0, 1, "甲")
        row["characters"] = [dict(text="甲", start_ms=50, end_ms=100)]
        self.assertEqual(timed_characters([row])["X"], [Token("甲", 75000, True)])
        row["characters"][0]["start_ms"] = -1
        self.assertEqual(timed_characters([row])["X"], [Token("甲", 500000, False)])

    def test_textgrid_reader_handles_bom_escaped_quotes_and_exponents(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.TextGrid"
            path.write_text(
                'item [1]:\nclass = "IntervalTier"\nname = "X"\n'
                'intervals [1]:\nxmin = 1e-3\nxmax = 1e+0\n'
                'text = "他叫我""等等""。\n我们继续。"\n',
                encoding="utf-8-sig",
            )
            labels = textgrid(path)
        self.assertEqual(labels, [turn("X", 0.001, 1, '他叫我"等等"。\n我们继续。')])


if __name__ == "__main__":
    unittest.main()
