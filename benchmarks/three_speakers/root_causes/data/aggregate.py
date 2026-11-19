"""Pool fresh-excerpt timing/ASR counts and record the separate full meeting."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

from diarize import masks, reference
from multiview_score import counts, measures


def read(path):
    return json.loads(path.read_text())


def collect(root, timing_only):
    extra = root / "new_excerpts"
    records = []
    combined = None
    totals = Counter()
    for entry in read(extra / "manifest.json"):
        directory = extra / Path(entry["audio"]).stem
        predicted = read(directory / "timing.json")["segments"]
        duration = entry["duration_s"]
        actual = masks(reference(extra / entry["ref"]), duration)[1].T
        hypothesis = masks(predicted, duration)[1].T
        frame_counts = counts(SimpleNamespace(actual=actual), hypothesis)
        combined = frame_counts.copy() if combined is None else combined + frame_counts
        row = dict(
            session=entry["session"], duration_s=duration, timing=measures(frame_counts)
        )
        if not timing_only:
            scores = read(directory / "sensevoice" / "scores.json")
            row.update(
                render=read(directory / "render.json"),
                speaker_scores=scores["speaker_scores"],
                character_counts=scores["totals"],
                cpCER=scores["cpCER"],
                asr_wall_s=scores["inference_s"],
            )
            totals.update(scores["totals"])
        records.append(row)
    report = dict(excerpts=records, pooled_excerpt_timing=measures(combined))
    if not timing_only:
        meeting_dir = root / "R8008_M8017"
        textgrid = root / "R8008_M8017.TextGrid"
        report.update(
            pooled_excerpt_character_counts=totals,
            pooled_excerpt_cpCER=totals["errors"] / totals["reference_chars"],
            full_meeting=dict(
                id=meeting_dir.name,
                render=read(meeting_dir / "render.json"),
                asr=read(meeting_dir / "sensevoice" / "scores.json"),
                text_reference_sha256=sha256(textgrid.read_bytes()).hexdigest(),
            ),
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(
        json.dumps(collect(args.root, args.timing_only), indent=2) + "\n"
    )
