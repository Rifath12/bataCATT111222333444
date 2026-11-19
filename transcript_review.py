"""Carry the waveform separator's evidence into each speaker-attributed segment."""
from bisect import bisect_right


def attach_separation_evidence(rows, commits, sample_rate):
    ends = [commit["sample_end"] for commit in commits]
    for row in rows:
        begin = round(row["start_ms"] * sample_rate / 1000)
        end = round(row["end_ms"] * sample_rate / 1000)
        speaker = row["speaker_id"]
        covered = passthrough = uncertain = 0
        changed = False
        index = bisect_right(ends, begin)
        while index < len(commits) and commits[index]["sample_start"] < end:
            commit = commits[index]
            for tile in commit.get("identity_tiles", [commit]):
                start_sample = tile.get("start_sample", commit["sample_start"])
                end_sample = tile.get("end_sample", commit["sample_end"])
                count = max(0, min(end, end_sample) - max(begin, start_sample))
                if count == 0 or speaker not in tile["speaker_ids"]:
                    continue
                covered += count
                if commit["decoded_source_count"] == 1:
                    passthrough += count
                flag = tile.get("identity_needs_review", False)
                needs_review = speaker in flag if isinstance(flag, list) else flag
                if needs_review:
                    uncertain += count
                changed |= bool(commit.get("count_review"))
            index += 1
        row["separation_evidence"] = {
            "assigned_activity_ms": round(covered * 1000 / sample_rate),
            "single_source_passthrough_ms": round(passthrough * 1000 / sample_rate),
            "ambiguous_identity_ms": round(uncertain * 1000 / sample_rate),
            "overlap_count_changed": changed,
        }
        row["review_required"] = bool(row.get("review_required") or uncertain or changed or not covered)
