"""Local activity evidence for labeling separated streams when a voice has little solo speech."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def short_solo_cue(wave, runs, rate=16000, maximum_seconds=4):
    """Gather short solo turns. Return None when the model lacks 0.6 s of evidence."""
    hop = rate // 100
    limit = int(maximum_seconds * rate)
    pieces = []
    samples = 0
    for lo, hi in sorted(runs, key=lambda span: span[1] - span[0], reverse=True):
        if hi - lo < 16:
            continue
        start = (lo + 3) * hop
        stop = min((hi - 3) * hop, len(wave), start + limit - samples)
        if stop > start:
            pieces.append(wave[start:stop])
            samples += stop - start
        if samples == limit:
            break
    if samples < 0.6 * rate:
        return None
    cue = np.concatenate(pieces)
    return np.pad(cue, (0, limit - len(cue)))


def activity_affinity(tracks, activity, start_frame):
    """Contrast estimated source power across a speaker's active and inactive frames."""
    hop = 80  # 10 ms at the 8 kHz separator rate
    count = min(tracks.shape[1] // hop, activity.shape[1] - start_frame)
    scores = np.zeros((len(activity), len(tracks)), dtype=np.float32)
    if count < 2:
        return scores
    powers = (tracks[:, : count * hop].reshape(len(tracks), count, hop) ** 2).mean(
        axis=2
    )
    total = powers.sum(axis=0)
    voiced = total > max(1e-9, np.max(total) * 1e-3)
    shares = powers / np.maximum(total, 1e-12)
    local = activity[:, start_frame : start_frame + count]
    for slot, timeline in enumerate(local):
        on = timeline & voiced
        off = ~timeline & voiced & local.any(axis=0)
        if on.any() and off.any():
            reliability = min(1.0, min(on.sum(), off.sum()) / 12)
            scores[slot] = reliability * (
                shares[:, on].mean(axis=1) - shares[:, off].mean(axis=1)
            )
    return scores


def assignment_is_tied(scores, rows, columns, tolerance=1e-6):
    """Does a different labeling explain this window just as well?"""
    best = scores[rows, columns].sum()
    if len(rows) < 2:
        return False
    for row, col in zip(rows, columns, strict=True):
        costs = -scores.copy()
        costs[row, col] = np.inf
        alternate = linear_sum_assignment(costs)
        if best - scores[alternate].sum() <= tolerance:
            return True
    return False
