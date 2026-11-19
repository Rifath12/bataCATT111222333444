"""Reconstruct overlapping speech for ASR using mixture phase and bounded attenuation."""

import numpy as np
from scipy import signal


def overlap_asr_tracks(wave, separated, activity, strength=0.8):
    """Return 16 kHz tracks with the original single-speaker frames preserved."""
    frame = 160
    window = 400
    hop = 100
    options = dict(nperseg=window, noverlap=window - hop, nfft=512)
    output = np.zeros_like(separated)
    # Limit temporary spectral arrays for long meetings. Block edges are aligned
    # to the global STFT frame grid and surrounded by more than a full window.
    chunk_size = 10 * 16000
    for start in range(0, len(wave), chunk_size):
        end = min(start + chunk_size, len(wave))
        active = np.repeat(
            activity[:, start // frame : (end + frame - 1) // frame], frame, axis=1
        )[:, : end - start]
        block = output[:, start:end]
        block[:] = wave[start:end] * active
        overlap = active.sum(axis=0) > 1
        if not overlap.any():
            continue
        left = max(0, start - window)
        right = min(len(wave), end + window)
        mixed = wave[left:right]
        estimates = separated[:, left:right]
        if len(mixed) < window:
            pad = window - len(mixed)
            mixed = np.pad(mixed, (0, pad))
            estimates = np.pad(estimates, ((0, 0), (0, pad)))
        spectrum = signal.stft(mixed, **options)[2]
        spectra = signal.stft(estimates, **options, axis=-1)[2]
        powers = np.abs(spectra) ** 2
        total = powers.sum(axis=0)
        share = np.divide(
            powers,
            total,
            out=np.full_like(powers, 1 / len(estimates)),
            where=total > 1e-16,
        )
        gains = (1 - strength) + strength * np.sqrt(share)
        restored = signal.istft(gains * spectrum, **options)[1][
            :, start - left : end - left
        ]
        block[:, overlap] = (restored * active)[:, overlap]
    return output
