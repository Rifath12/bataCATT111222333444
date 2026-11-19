"""Adaptive close-microphone activity from speech posteriors and microphone leakage."""

import numpy as np
from scipy.ndimage import (
    binary_opening,
    uniform_filter1d,
)
from scipy.signal import convolve


RATE = 16000
HOP = 512


def estimate_leakage(power, probability):
    """Use audio-only, confidently dominant frames to estimate a fixed leakage matrix."""
    power = uniform_filter1d(power, 3, axis=-1, mode="nearest")
    floor = np.empty(3)
    scale = np.empty(3)
    for index, (track, posterior) in enumerate(zip(power, probability, strict=True)):
        low = track[posterior < 0.04]
        high = track[posterior >= 0.7]
        floor[index] = np.quantile(low if len(low) else track, 0.25)
        scale[index] = max(1e-9, np.quantile(high if len(high) else track, 0.9) - floor[index])
    energies = np.maximum(power - floor[:, None], 0) / scale[:, None]
    mix = np.eye(3)
    for source in range(3):
        other = np.delete(energies, source, axis=0)
        own = energies[source]
        dominant = (probability[source] > 0.85) & (own > 0.055) & (other.max(axis=0) < 0.25 * own)
        if np.count_nonzero(dominant) >= 20:
            mix[:, source] = np.clip(np.quantile(energies[:, dominant] / own[dominant], 0.40, axis=1), 0, 0.30)
            mix[source, source] = 1.0
    gram = mix.T @ mix
    projected = mix.T @ energies
    clean = np.maximum(np.linalg.solve(mix, energies), 0.0)
    step = 1.0 / np.linalg.eigvalsh(gram).max()
    for _ in range(12):
        clean = np.maximum(clean + step * (projected - gram @ clean), 0.0)
    ratio = clean / np.maximum(energies, 1e-7)
    return energies, clean, ratio, mix


def fill_short_gaps(mask, gap_frames):
    """Bridge interior gaps without eroding speech at either end of the recording."""
    if gap_frames <= 0:
        return mask.copy()
    repaired = mask.copy()
    for line in repaired:
        edges = np.flatnonzero(np.diff(np.r_[False, line, False]))
        for left, right in zip(edges[1:-1:2], edges[2::2], strict=True):
            if right - left <= gap_frames:
                line[left:right] = True
    return repaired


def temporal_mask(core, open_frames=1, gap_frames=4, hang_frames=3, pre_frames=3):
    if open_frames > 1:
        core = np.array([binary_opening(row, structure=np.ones(open_frames, bool)) for row in core])
    core = fill_short_gaps(core, gap_frames)
    taps = np.ones((1, pre_frames + hang_frames + 1), dtype=np.int16)
    if taps.shape[-1] == 1:
        return core
    added = convolve(core.astype(np.int16), taps, mode="full", method="direct")
    return added[:, pre_frames:pre_frames + core.shape[1]] > 0


def activity_mask(power, probability, *, threshold, absolute, own_fraction, gap, pre, hang, open_min, rule="adaptive", estimates=None):
    if rule == "adaptive":
        energy, clean, fraction, _ = estimates if estimates is not None else estimate_leakage(power, probability)
        # Keep quiet local speech as long as enough power is unique to its mic.
        trigger = (probability >= threshold) & (clean >= absolute) & (fraction >= own_fraction)
    elif rule == "static":
        smooth = uniform_filter1d(power, 3, axis=-1, mode="nearest")
        scales = [
            max(1e-9, np.quantile(row[p >= 0.7], .9) if np.any(p >= .7) else row.max())
            for row, p in zip(smooth, probability, strict=True)
        ]
        energy = smooth / np.asarray(scales)[:, None]
        trigger = (probability >= threshold) & (energy >= np.maximum(absolute, own_fraction * energy.max(axis=0)))
    else:
        raise ValueError(f"Unsupported rule {rule}")
    return temporal_mask(trigger, open_min, gap, hang, pre)

