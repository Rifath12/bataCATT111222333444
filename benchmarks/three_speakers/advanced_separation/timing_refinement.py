"""Refine anonymous A/B/C boundaries using two matching 10-ms mono posteriors."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter, minimum_filter1d, uniform_filter1d
from scipy.optimize import linear_sum_assignment
import soundfile as sf


LABELS = tuple("ABC")
STEP = 0.01


def unique_assignment(scores, tolerance=1e-5):
    rows, cols = linear_sum_assignment(-scores)
    best = scores[rows, cols].sum()
    for row, col in zip(rows, cols, strict=True):
        excluded = -scores.copy()
        excluded[row, col] = np.inf
        next_rows, next_cols = linear_sum_assignment(excluded)
        if best - scores[next_rows, next_cols].sum() <= tolerance:
            return None
    return rows, cols


def match_views(first, second, anchors):
    """Align both views with the original A/B/C marks, without voice references."""
    centered = first - first.mean(axis=0)
    shifted = second - second.mean(axis=0)
    affinity = centered.T @ shifted
    affinity /= np.maximum(
        np.linalg.norm(centered, axis=0)[:, None] * np.linalg.norm(shifted, axis=0),
        1e-8,
    )
    view_assignment = unique_assignment(affinity)
    if view_assignment is None:
        return None
    slots, shifted_slots = view_assignment
    aligned = second[:, shifted_slots[np.argsort(slots)]]
    candidates = centered.T @ anchors
    anchor_assignment = unique_assignment(candidates)
    if anchor_assignment is None:
        return None
    slots, labels = anchor_assignment
    order = slots[np.argsort(labels)]
    if np.any(candidates[order, np.arange(len(order))] <= 1e-5):
        # A silent or disconnected label cannot support a safe posterior assignment.
        return None
    return first[:, order], aligned[:, order]


def refine(original, first, second):
    """Use the frozen M8013 profile, preserving every exact-three decision."""
    if original.shape != first.shape or original.shape != second.shape:
        raise ValueError("Speaker timing and both posteriors must share a sample clock")
    a = median_filter(first, size=(5, 1), mode="nearest")
    b = median_filter(second, size=(5, 1), mode="nearest")
    near = maximum_filter1d(original, size=91, axis=0)
    strong = near & (np.minimum(a, b) >= 0.50)
    strong &= uniform_filter1d(strong.astype(np.float32), 5, axis=0) >= 0.60
    proposal = original | strong

    # Exact-three corrections need different evidence and are outside this profile.
    existing_three = original.all(axis=1)
    introduced_three = proposal.all(axis=1) & ~existing_three
    proposal[introduced_three] = original[introduced_three]
    interior = minimum_filter1d(original, size=5, axis=0)
    weak_edge = original & ~interior & (np.maximum(a, b) < 0.30)
    weak_edge &= ~existing_three[:, None]
    return proposal & ~weak_edge


def original_activity(rows, frames, duration):
    out = np.zeros((frames, len(LABELS)), dtype=bool)
    centers = (np.arange(frames) + 0.5) * STEP
    for row in rows:
        start, end = float(row["start"]), float(row["end"])
        if row["speaker"] not in LABELS or not 0 <= start <= end <= duration + 0.02:
            raise ValueError("Timing must contain valid A/B/C intervals in seconds")
        left, right = np.searchsorted(centers, [start, min(end, duration)])
        out[left:right, LABELS.index(row["speaker"])] = True
    return out


def segments(activity, duration):
    output = []
    for label, column in zip(LABELS, activity.T, strict=True):
        edges = np.flatnonzero(np.diff(np.r_[False, column, False]))
        for left, right in zip(edges[::2], edges[1::2], strict=True):
            output.append({
                "speaker": label,
                "start": round(int(left) * STEP, 6),
                "end": min(round(int(right) * STEP, 6), duration),
            })
    return sorted(output, key=lambda row: (row["start"], row["speaker"]))


def read_views(cache, frames, duration):
    with np.load(cache, allow_pickle=False) as data:
        if abs(float(data["duration_s"]) - duration) > STEP:
            raise ValueError("Cache must come from the same original audio")
        base, alternate = data["base"], data["alternate"]
        offset = int(data["offset"])
        if base.ndim != 2 or alternate.ndim != 2 or offset < 0:
            raise ValueError("Expected two frame-by-speaker probability arrays")
        if len(base) < frames or len(alternate) - offset < frames:
            raise ValueError("Cache does not span the full audio")
        if min(base.shape[1], alternate.shape[1]) < len(LABELS):
            raise ValueError("Cache must expose three model slots per view")
        mass = np.zeros(base.shape[1])
        for start, end, slot in data["native"]:
            mass[int(slot)] += end - start
        slots = np.sort(np.argsort(-mass, kind="stable")[: len(LABELS)])
        first, second = base[:frames, slots], alternate[offset:offset + frames]
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Nonfinite values in timing cache")
    return first, second


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audio", "timing", "cache", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    info = sf.info(args.audio)
    if (info.samplerate, info.channels) != (16000, 1) or not info.frames:
        parser.error("expected a nonempty 16 kHz mono source")
    frames = (info.frames + 159) // 160
    old = json.loads(args.timing.read_text(encoding="utf-8"))
    original = original_activity(old["segments"], frames, info.duration)
    first, second = read_views(args.cache, frames, info.duration)
    paired = match_views(first, second, original)
    refined = refine(original, *paired) if paired is not None else original
    result = {
        "duration_s": info.duration,
        "speaker_ids": list(LABELS),
        "method": "two_view_edge_refinement",
        "applied": paired is not None,
        "segments": segments(refined, info.duration),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
