"""Calibrate room-array activity on minutes 2–4, then score the remainder."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter, minimum_filter1d

from array_enrollment import timestamp_rows
from diarize import masks, reference, score


def smooth(probability, threshold, gap, expansion):
    row = probability >= threshold
    if gap > 1:
        row = minimum_filter1d(maximum_filter1d(row, gap), gap)
    if expansion > 1:
        row = maximum_filter1d(row, expansion)
    return row


def fit(probability, actual):
    best = (-1, ())
    for threshold in np.linspace(0.07, 0.69, 16):
        for gap in (1, 9, 19, 31, 47, 65):
            for expansion in (1, 9, 17, 25, 41):
                prediction = smooth(probability, threshold, gap, expansion)
                denominator = actual.sum() + prediction.sum()
                f1 = 2 * (actual & prediction).sum() / max(denominator, 1)
                if f1 > best[0]:
                    best = (float(f1), (float(threshold), gap, expansion))
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trial", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    trial = json.loads((args.trial / "diarization.json").read_text(encoding="utf-8"))
    duration = trial["duration_s"]
    enrollment = trial["enrollment_seconds"]
    cutoff = enrollment + 120
    begin, end = enrollment * 100, cutoff * 100
    posterior = median_filter(np.load(args.trial / "confidence.npy"), size=(1, 9))
    ref = reference(args.reference)
    _, actual = masks(ref, duration)
    parameters = [
        fit(p[begin:end], label[begin:end]) for p, label in zip(posterior, actual)
    ]
    hypothesis = np.stack(
        [smooth(p, *choice[1]) for p, choice in zip(posterior, parameters)]
    )
    hypothesis[:, :end] = False
    rows = timestamp_rows(hypothesis, duration)

    def held_out(spans):
        return [
            dict(s, start=max(cutoff, s["start"]) - cutoff, end=s["end"] - cutoff)
            for s in spans
            if s["end"] > cutoff
        ]

    result = dict(
        duration_s=duration,
        enrollment_seconds=enrollment,
        calibration_seconds=120,
        application_start_s=cutoff,
        params=parameters,
        segments=rows,
        evaluation=score(held_out(ref), held_out(rows), duration - cutoff),
    )
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result["evaluation"], flush=True)


if __name__ == "__main__":
    main()
