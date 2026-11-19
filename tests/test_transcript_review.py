import unittest

from transcript_review import attach_separation_evidence


class TranscriptReviewTest(unittest.TestCase):
    def test_intersects_tiles_and_preserves_existing_review(self):
        rows = [{"start_ms": 50, "end_ms": 450, "speaker_id": "spk_A", "review_required": False},
                {"start_ms": 250, "end_ms": 300, "speaker_id": "spk_C", "review_required": True}]
        commits = [{"sample_start": 0, "sample_end": 200, "decoded_source_count": 1,
                    "identity_tiles": [
                        {"start_sample": 0, "end_sample": 120, "speaker_ids": ["spk_A"], "identity_needs_review": True},
                        {"start_sample": 120, "end_sample": 200, "speaker_ids": ["spk_B"]}]},
                   {"sample_start": 200, "sample_end": 500, "speaker_ids": ["spk_A", "spk_C"],
                    "decoded_source_count": 2, "identity_needs_review": ["spk_B"]}]
        attach_separation_evidence(rows, commits, 1000)
        self.assertEqual(rows[0]["separation_evidence"], {
            "assigned_activity_ms": 320, "single_source_passthrough_ms": 70,
            "ambiguous_identity_ms": 70, "overlap_count_changed": False})
        self.assertTrue(rows[0]["review_required"])
        self.assertTrue(rows[1]["review_required"])

    def test_word_timing_alone_cannot_clear_unknown_ownership(self):
        rows = [{"start_ms": 0, "end_ms": 20, "speaker_id": "spk_A", "review_required": False}]
        attach_separation_evidence(rows, [], 8000)
        self.assertTrue(rows[0]["review_required"])
        self.assertEqual(rows[0]["separation_evidence"]["assigned_activity_ms"], 0)
