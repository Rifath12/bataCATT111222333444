"""Conservatively suppress foreign harmonics in three mono output tracks."""

import numpy as np
from scipy.ndimage import median_filter, uniform_filter, uniform_filter1d
from scipy.signal import find_peaks, stft, istft

from debleed import WINDOW


def pitches(tracks, frames, rate=8000):
    """Normalized cross-correlation pitch in overlapping Hann-weighted frames."""
    low = round(rate / 430)
    high = round(rate / 65)
    window_size = 280
    hop = WINDOW["nperseg"] - WINDOW["noverlap"]
    result = np.empty((tracks.shape[0], frames))
    confidence = result.copy()
    for index, row in enumerate(tracks):
        pad = np.pad(row, (window_size // 2, window_size // 2 + high + hop))
        windows = np.lib.stride_tricks.sliding_window_view(pad, window_size + high + 1)[
            ::hop
        ][:frames]
        weight = np.hanning(window_size)
        # The weighting suppresses the edge clicks common in block-based separators.
        left = windows[:, :window_size]
        left = (left - left.mean(axis=-1, keepdims=True)) * weight
        right = np.lib.stride_tricks.sliding_window_view(windows, window_size, axis=-1)[
            :, low : high + 1
        ]
        right = (right - right.mean(axis=-1, keepdims=True)) * weight
        numerator = np.einsum("nt,nlt->nl", left, right)
        denominator = np.sqrt(
            np.einsum("nt,nt->n", left, left)[:, None]
            * np.einsum("nlt,nlt->nl", right, right)
        )
        similarity = numerator / np.maximum(denominator, 1e-10)
        # Choose the first close-to-best peak to avoid octave multiples of the lag.
        for frame, cc in enumerate(similarity):
            candidates, _ = find_peaks(cc)
            if not len(candidates):
                candidates = np.array([int(np.argmax(cc))])
            best = cc[candidates].max()
            peak = (
                int(candidates[cc[candidates] >= max(0, best - 0.09)][0])
                if best >= 0
                else int(candidates[0])
            )
            conf = max(0.0, best)
            offset = 0.0
            if 0 < peak < len(cc) - 1:
                a, b, c = cc[peak - 1 : peak + 2]
                offset = (0.5 * (c - a) / max(2 * b - a - c, 1e-8)).clip(-0.5, 0.5)
            result[index, frame] = rate / (low + peak + offset)
            confidence[index, frame] = conf
    frequency = np.where(confidence > 0.42, result, 0)
    smoothed = median_filter(frequency, size=(1, 5), mode="nearest")
    close = np.abs(frequency - smoothed) <= 0.17 * np.maximum(frequency, 1)
    confidence *= np.where(close, 1.0, 0.33)
    return result, confidence


def harmonic_features(tracks):
    spec = stft(tracks, **WINDOW)[2]
    freq, conf = pitches(tracks, spec.shape[-1])
    bins = np.arange(spec.shape[1], dtype=np.float32)[None, :, None] * (
        8000 / WINDOW["nperseg"]
    )
    harmonic = np.clip(np.round(bins / freq[:, None, :]), 1, 70)
    deviation = (bins - harmonic * freq[:, None, :]) / (13 + 0.012 * bins)
    evidence = np.exp(-0.5 * deviation**2)
    smooth = np.maximum(0.0, uniform_filter(np.abs(spec) ** 2, size=(1, 3, 5)))
    level = np.sqrt(np.mean(smooth, axis=1))
    level = np.minimum(
        1.0, level / np.maximum(0.18 * level.max(axis=1, keepdims=True), 1e-8)
    )
    reliability = np.clip((conf - 0.45) / 0.45, 0.0, 1.0) * level
    evidence *= reliability[:, None, :]
    # Above 2.6 kHz peaks are too densely spaced and this filter should disengage.
    evidence *= np.clip((2700 - bins) / 700, 0.0, 1.0)
    other = np.stack([np.max(evidence[np.arange(3) != j], axis=0) for j in range(3)])
    strength = np.sqrt(smooth / (smooth.sum(axis=0) + 1e-12))
    competitor = np.stack(
        [np.max((evidence * strength)[np.arange(3) != j], axis=0) for j in range(3)]
    )
    return (spec, evidence, other, competitor)


def apply_harmonic(features, strength, winner_margin, energy_cond, length):
    spec, evidence, other, competitor = features
    foreign = competitor if energy_cond else other
    result = np.clip((foreign - winner_margin * evidence), 0.0, 1.0)
    result = np.clip(uniform_filter1d(result, size=5, axis=-1), 0, 1)
    weight = 1 - strength * result
    return istft(spec * weight, **WINDOW)[1][..., :length].astype(np.float32)
