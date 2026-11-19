"""Evaluate released and direct transcripts on boundary-complete AISHELL-4 clips."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from duplicate_leakage import apply, encounters
from recognize import text_scores


def read_jsonl(path):
    return [json.loads(row) for row in path.read_text().splitlines() if row.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--duplicate-settings", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    priors = json.loads(a.duplicate_settings.read_text())
    totals = defaultdict(Counter)
    detail = []
    for item in json.loads(a.manifest.read_text()):
        name = Path(item["audio"]).stem
        work = a.root / name
        text = json.loads((a.root / item["text_reference"]).read_text())
        cropped = read_jsonl(work / "stem" / "transcript.jsonl")
        fast = apply(cropped, encounters(cropped, work / "mask"), priors)
        result = {}
        for arm, rows in [
            ("direct_stem", cropped),
            ("released_fast_transcript", fast),
            ("older_context", read_jsonl(work / "current" / "transcript.jsonl")),
            (
                "released_refined_context",
                read_jsonl(work / "refined" / "transcript.jsonl"),
            ),
        ]:
            score = text_scores(text, rows)
            assert score["totals"]["reference_chars"] == item["reference_characters"]
            result[arm] = score
            totals[arm].update(score["totals"])
        detail.append(
            dict(
                id=name,
                seconds=item["duration_s"],
                exact_three_seconds=item["exact_three_s"],
                scores=result,
            )
        )
    for data in totals.values():
        data["CER"] = data["errors"] / data["reference_chars"]
    a.out.write_text(json.dumps(dict(clips=detail, pooled=totals), indent=2) + "\n")
    print(dict(totals))


if __name__ == "__main__":
    main()
