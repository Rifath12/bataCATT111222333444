"""Reconstruct overlapping speech for ASR using mixture phase and bounded attenuation."""

import numpy as np
from scipy import signal


FRAME = 160
WINDOW = 400
HOP = 100
CHUNK_SIZE = 10 * 16000
FFT_SETTINGS = dict(nperseg=WINDOW, noverlap=WINDOW - HOP, nfft=512)


def overlap_intervals(flags):
    changes = np.flatnonzero(np.diff(np.r_[False, flags, False]))
    starts, ends = changes[::2], changes[1::2]
    if not len(starts):
        return []
    merged = [(starts[0], ends[0])]
    for left, right in zip(starts[1:], ends[1:]):
        # Merge nearby spans to avoid starting and reading a second transform.
        if left - merged[-1][1] <= 2 * WINDOW:
            merged[-1] = (merged[-1][0], right)
        else:
            merged.append((left, right))
    return merged


def reconstruct(wave, estimates, strength):
    if len(wave) < WINDOW:
        pad = WINDOW - len(wave)
        wave = np.pad(wave, (0, pad))
        estimates = np.pad(estimates, ((0, 0), (0, pad)))
    spectrum = signal.stft(wave, **FFT_SETTINGS)[2]
    spectra = signal.stft(estimates, **FFT_SETTINGS, axis=-1)[2]
    powers = np.abs(spectra) ** 2
    total = powers.sum(axis=0)
    share = np.divide(
        powers, total, out=np.full_like(powers, 1 / len(estimates)), where=total > 1e-16
    )
    gains = (1 - strength) + strength * np.sqrt(share)
    return signal.istft(gains * spectrum, **FFT_SETTINGS)[1]


def overlap_asr_tracks(wave, separated, activity, strength=0.8):
    """Return 16 kHz tracks with the original single-speaker frames preserved."""
    output = np.zeros_like(separated)
    for start in range(0, len(wave), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(wave))
        active = np.repeat(
            activity[:, start // FRAME : (end + FRAME - 1) // FRAME], FRAME, axis=1
        )[:, : end - start]
        block = output[:, start:end]
        block[:] = wave[start:end] * active
        overlap = active.sum(axis=0) > 1
        for left, right in overlap_intervals(overlap):
            # Crop along the original STFT hop grid with a complete analysis
            # window on both sides of the region used in the output.
            begin = max(0, ((start + left - WINDOW) // HOP) * HOP)
            finish = min(len(wave), end + WINDOW, start + right + WINDOW)
            restored = reconstruct(
                wave[begin:finish], separated[:, begin:finish], strength
            )
            support = overlap[left:right]
            destination = block[:, left:right]
            samples = restored[:, start + left - begin : start + right - begin]
            destination[:, support] = (samples * active[:, left:right])[:, support]
    return output
