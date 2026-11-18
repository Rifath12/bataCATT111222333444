"""Tune activity thresholds on a marked prefix and evaluate the remaining meeting."""

from pathlib import Path
import argparse
from itertools import product
import json

import numpy as np
from scipy.ndimage import median_filter, maximum_filter1d, minimum_filter1d
from scipy.optimize import linear_sum_assignment

from diarize import masks, reference, score


def postprocess(probabilities, rank, thresholds, close, expand):
    active = probabilities >= np.asarray(thresholds)[rank]
    if close > 1:
        active = minimum_filter1d(
            maximum_filter1d(active, close, axis=0), close, axis=0
        )
    if expand > 1:
        active = maximum_filter1d(active, expand, axis=0)
    return active


def to_segments(active, length):
    rows = []
    for speaker, line in enumerate(active.T):
        edges = np.flatnonzero(np.diff(np.pad(line.astype(np.int8), (1, 1))))
        rows.extend(
            dict(
                speaker=str(speaker),
                start=0.01 * int(a),
                end=min(length, 0.01 * int(b)),
            )
            for a, b in zip(edges[::2], edges[1::2])
            if 0.01 * int(a) < length
        )
    return sorted(rows, key=lambda r: (r["start"], r["speaker"]))


def fit_prefix(probability, activity, recall_target):
    affinity = activity.T.astype(np.float32) @ probability
    actual, estimated = linear_sum_assignment(-affinity)
    activity = activity[:, actual[np.argsort(estimated)]]
    multi = activity.sum(axis=1) >= 2
    true_count = activity.sum()
    true_overlap = activity[multi].sum()
    ranked = np.argsort(np.argsort(-probability, axis=1), axis=1)
    best = (-float("inf"), ())
    third_thresholds = (0.12, 0.22, 0.35, 0.48) if recall_target else (0.22, 0.35, 0.48)
    for a, b, c, closing, padding in product(
        (0.50, 0.60, 0.70, 0.80, 0.88, 0.94),
        (0.38, 0.50, 0.60, 0.69),
        third_thresholds,
        (1, 7, 15),
        (1, 7, 15),
    ):
        predicted = postprocess(probability, ranked, (a, b, c), closing, padding)
        intersection = activity & predicted
        f1 = 2 * intersection.sum() / max(1, true_count + predicted.sum())
        ovf1 = (
            2
            * intersection[multi].sum()
            / max(1, true_overlap + predicted[multi].sum())
        )
        overlap_recall = intersection[multi].sum() / max(1, true_overlap)
        shortfall = max(0.0, recall_target - overlap_recall)
        value = float(f1 + 0.25 * ovf1 - 8.0 * shortfall)
        if value > best[0]:
            best = (value, ((a, b, c), closing, padding))
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probabilities", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--rttm", type=Path, required=True)
    p.add_argument("--prefix", type=int, default=240)
    p.add_argument("--overlap-recall-target", type=float, default=0.0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    raw = np.load(args.probabilities)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    duration = baseline["duration_s"]
    step = float(raw["step"])
    if not np.isclose(step, 0.01):
        raise ValueError("This calibration expects 10 ms native probabilities")
    cutoff = args.prefix
    cut_frames = cutoff * 100
    ref = reference(args.rttm)
    _, labels = masks(ref, duration)
    probabilities = raw["probabilities"][: labels.shape[1]]
    silence = np.zeros(
        (labels.shape[1] - len(probabilities), probabilities.shape[1]), np.float32
    )
    probabilities = np.vstack((probabilities, silence))
    native_seconds = {
        int(k.rsplit("_", 1)[1]) - 1: v
        for k, v in baseline["native_output_speaker_seconds"].items()
    }
    chosen = sorted(sorted(native_seconds, key=native_seconds.get, reverse=True)[:3])
    usable = median_filter(probabilities[:, chosen], size=(5, 1))
    statistics, params = fit_prefix(
        usable[:cut_frames], labels[:, :cut_frames].T, args.overlap_recall_target
    )
    rank = np.argsort(np.argsort(-usable, axis=1), axis=1)
    active = postprocess(usable, rank, *params)
    rows = to_segments(active, duration)

    def held_out(spans):
        return [
            dict(x, start=max(cutoff, x["start"]) - cutoff, end=x["end"] - cutoff)
            for x in spans
            if x["end"] > cutoff
        ]

    result = dict(
        duration_s=duration,
        inference_wall_s=float(raw["wall_s"]),
        setup_annotated_prefix_s=cutoff,
        development_overlap_recall_target=args.overlap_recall_target,
        rank_thresholds=params,
        tuning_score=statistics,
        segments=rows,
        evaluation=score(held_out(ref), held_out(rows), duration - cutoff),
        baseline_same_holdout=score(
            held_out(ref), held_out(baseline["segments"]), duration - cutoff
        ),
    )
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print({k: v for k, v in result.items() if k != "segments"}, flush=True)


if __name__ == "__main__":
    main()
