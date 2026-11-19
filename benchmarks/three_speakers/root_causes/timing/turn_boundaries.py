"""Correct early onsets and late offsets while protecting brief speaker interjections."""

import argparse
import json
from pathlib import Path

import numpy as np

from calibrate_nemotron import to_segments
from diarize import masks


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return zip(edges[::2], edges[1::2])


def correct(active, config):
    """Apply the fitted boundary adjustment without audio or reference annotations."""
    result = active.copy()
    onset, offset = config["onset_frames"], config["offset_frames"]
    for speaker, column in enumerate(active.T):
        for lo, hi in runs(column):
            if hi - lo < max(12, 2 * (onset + offset)):
                continue
            if lo > 0:
                result[lo : lo + onset, speaker] = False
            if hi < len(active):
                result[hi - offset : hi, speaker] = False
    # Restoring previous overlap preserves short third-speaker turns before
    # narrowing only intervals at a handoff between different speaker pairs.
    triple = active.sum(1) == 3
    result[triple] = True
    amount = config["handoff_frames"]
    if not amount:
        return result
    for lo, hi in runs(triple):
        if lo == 0 or hi == len(active):
            continue
        left, right = active[lo - 1], active[hi]
        if left.sum() != 2 or right.sum() != 2 or np.array_equal(left, right):
            continue
        arriving = ~left & right
        leaving = left & ~right
        cut_in, cut_out = min(hi, lo + amount), max(lo, hi - amount)
        if cut_in > cut_out:
            cut_in = cut_out = (lo + hi) // 2
        result[lo:cut_in, arriving] = False
        result[cut_out:hi, leaving] = False
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("timing", type=Path)
    p.add_argument("--profile", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    input_timing = json.loads(a.timing.read_text(encoding="utf-8"))
    duration = input_timing["duration_s"]
    if duration <= 0:
        p.error("Expected a positive recording duration")
    labels, mask = masks(input_timing["segments"], duration)
    if not labels:
        output = []
    else:
        if len(labels) > 3:
            p.error("Expected at most three speakers")
        estimate = correct(mask.T, json.loads(a.profile.read_text()))
        output = to_segments(estimate, duration)
        for seg in output:
            seg["speaker"] = labels[int(seg["speaker"])]
    rendered = dict(input_timing, segments=output)
    a.out.write_text(
        json.dumps(rendered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
