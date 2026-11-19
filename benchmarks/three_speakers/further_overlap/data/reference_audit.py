"""Check the independent Mandarin character alignments against official TextGrid sentences."""

import argparse
from collections import Counter
import json
from pathlib import Path
from rapidfuzz.distance import Levenshtein

from recognize import normalized, textgrid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    files, pools, exact_chars, duration, triple_seconds = {}, set(), 0, 0.0, 0.0
    bins = Counter()
    for r in json.loads(a.manifest.read_text()):
        session = r["session"]
        if session not in files:
            official = textgrid(a.references / f"{session}.TextGrid")
            files[session] = {}
            for speaker in r["session_speaker_ids"]:
                words = json.loads(
                    (a.references / f"{session}_{speaker}.alignment.json").read_text()
                )
                ordered = [r for r in official if r["speaker"] == speaker]
                official_text = normalized(
                    "".join(
                        x["text"] for x in sorted(ordered, key=lambda x: x["start"])
                    )
                )
                word_text = normalized(
                    "".join(
                        x["text"]
                        for x in words["word_alignments"]
                        if x["start"] < x["end"]
                    )
                )
                files[session][speaker] = dict(
                    official_chars=len(official_text),
                    timed_chars=len(word_text),
                    edits=Levenshtein.distance(official_text, word_text),
                    source_word_intervals=len(words["word_alignments"]),
                    timed_text_markup_stripped=True,
                )
        pools.update(r["speaker_ids"])
        exact_chars += r["midpoint_three_chars"]
        triple_seconds += r["exact_three_s"]
        duration += r["duration_s"]
        bins.update({"clips": 1, "characters": r["text_chars"]})
    whole = Counter()
    for speakers in files.values():
        for row in speakers.values():
            whole.update(
                {k: row[k] for k in ("official_chars", "timed_chars", "edits")}
            )
    result = dict(
        number_source_meetings=len(files),
        unique_speakers=len(pools),
        total_duration_s=round(duration, 2),
        exact_three_s=round(triple_seconds, 2),
        true_triple_characters=int(exact_chars),
        alignment_audit_source_totals=dict(whole),
        **dict(bins),
        full_source_speakers=files,
    )
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "full_source_speakers"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
