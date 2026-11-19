"""Mono spectral reconstruction for speaking turns with simultaneous speakers."""

import numpy as np
from scipy import ndimage, signal


PROFILES = ("previous", "projected", "phase_adaptive")
FRAME = 160
WINDOW = 400
STFT_OPTIONS = dict(nperseg=WINDOW, noverlap=300, nfft=512)


def separated_overlap(wave, estimates, activity, profile):
    """Keep directly observed solo speech. Reconstruct only overlap samples."""
    if profile not in PROFILES:
        raise ValueError(f"Unrecognized spectral reconstruction profile: {profile}")
    active = np.repeat(activity, FRAME, axis=1)[:, : len(wave)]
    overlap = active.sum(axis=0) > 1
    output = wave[None, :] * active
    if not overlap.any():
        return output
    pad = max(0, WINDOW - len(wave))
    mix = np.pad(wave, (0, pad)) if pad else wave
    stems = np.pad(estimates, ((0, 0), (0, pad))) if pad else estimates
    mixture = signal.stft(mix, **STFT_OPTIONS)[2]
    source = signal.stft(stems, **STFT_OPTIONS, axis=-1)[2]
    powers = np.abs(source) ** 2
    power_sum = powers.sum(axis=0, keepdims=True)
    shares = np.divide(
        powers,
        power_sum,
        out=np.full_like(powers, 1 / len(source)),
        where=power_sum > 1e-16,
    )
    previous = (0.2 + 0.8 * np.sqrt(shares)) * mixture

    if profile == "previous":
        spectrum = previous
    else:
        smoothed = ndimage.uniform_filter(powers, size=(1, 3, 5), mode="nearest")
        sums = smoothed.sum(axis=0, keepdims=True)
        shares = np.divide(
            smoothed,
            sums,
            out=np.full_like(smoothed, 1 / len(source)),
            where=sums > 1e-16,
        )
        residual = mixture - source.sum(axis=0)
        projected = source + shares * residual
        # A complex projection preserves the measured mixture and each model phase.
        # Its reliability drops when competing predictions nearly cancel or disagree
        # with the mixture spectrum. In those bins use the original soft mask.
        if profile == "projected":
            fraction = 0.45
        else:
            inconsistency = np.abs(residual) / (
                np.abs(mixture) + np.abs(source).sum(axis=0) + 1e-9
            )
            concentration = shares.max(axis=0)
            fraction = 0.8 * np.clip(1 - 1.6 * inconsistency, 0.0, 1.0) * concentration
        spectrum = (1 - fraction) * previous + fraction * projected
    combined = signal.istft(spectrum, **STFT_OPTIONS)[1][:, : len(wave)]
    output[:, overlap] = (combined * active)[:, overlap]
    return output
