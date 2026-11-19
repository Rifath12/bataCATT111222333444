"""Estimate and subtract bounded cross-talk from three aligned mono estimates."""

import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter1d
from scipy.signal import stft, istft

WINDOW = dict(nperseg=256, noverlap=192)


def blur(values, size=(1, 5, 9)):
    return np.maximum(0, uniform_filter(values, size=size, mode="nearest"))


def scale_to_mixture(mixture, tracks):
    gram = tracks.astype(np.float64) @ tracks.T
    rhs = tracks.astype(np.float64) @ mixture
    ridge = max(np.trace(gram) * 1e-3 / 3, 1e-12)
    scale = np.linalg.solve(gram + np.eye(3) * ridge, rhs)
    return tracks * np.clip(scale, 0.15, 4.0)[:, None]


def analyze(mixture, tracks):
    """Cache spectra, soft dominance and smooth leakage transfer functions."""
    tracks = np.asarray(tracks, np.float32)
    mixture = np.asarray(mixture, np.float32)
    if tracks.shape != (3, len(mixture)):
        raise ValueError("Expected waveform and exactly three aligned tracks.")
    if len(mixture) < WINDOW["nperseg"]:
        return None
    fitted = scale_to_mixture(mixture, tracks)
    specs = stft(fitted.astype(np.float32), **WINDOW)[2]
    mix = stft(mixture, **WINDOW)[2]
    energy = blur(np.abs(specs) ** 2)
    floor = max(float(energy.max()) * 1e-5, 1e-12)
    share = energy / (energy.sum(axis=0) + floor)
    return specs, mix, energy, share, floor


def corrections(analysis, confidence, phase, gate):
    """Use dominated bins to estimate leakage and preserve uncertain ownership."""
    specs, mix, energy, share, floor = analysis
    active = np.square(np.maximum(0.0, share - confidence))
    divisor = gaussian_filter1d(
        (active * np.abs(specs) ** 2).sum(axis=-1), 3.0, axis=-1
    )
    prior = np.maximum(0.01 * np.max(divisor, axis=-1, keepdims=True), floor)
    ratios = np.zeros((3, 3, specs.shape[1]), np.complex64)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            dot = (active[j] * specs[i] * specs[j].conj()).sum(axis=-1)
            h = gaussian_filter1d(dot, 3.0) / (divisor[j] + prior[j])
            if phase == "real":
                h = h.real
            ratios[i, j] = h / np.maximum(1, np.abs(h) / 0.32)
    # Estimate leakage only when the supposed donor dominates. Conservative
    # filtering avoids suppressing an independently audible target in the bin.
    if gate == "selective":
        pair = np.clip(share[None, ...] - share[:, None, ...], 0, 0.9)
        gate_weight = pair / 0.9
    else:
        gate_weight = np.broadcast_to(np.sqrt(share)[None, ...], (3, *share.shape))
    transfer = ratios[..., None] * specs[None, ...] * gate_weight
    leak = transfer.sum(axis=1)
    leak_power = blur(np.sum(np.abs(transfer) ** 2, axis=1))
    err = mix - specs.sum(axis=0)
    err_energy = blur(np.abs(err) ** 2, size=(5, 9))
    stationary = np.median(err_energy, axis=-1, keepdims=True)
    return leak, leak_power, stationary


def apply_filter(analysis, correction, strength, shrink, noise):
    specs, mix, energy, share, floor = analysis
    leak, power, stationary = correction
    result = specs - strength * leak
    estimated_leak = shrink * power + noise * stationary[None, ...]
    desired = np.maximum(energy - estimated_leak, 0.0)
    gain = np.sqrt(desired / (energy + floor))
    return (result * np.clip(gain, 0.45, 1)).astype(np.complex64)


def debleed(mix, tracks, *, confidence, phase, gate, strength, shrink, noise):
    if len(mix) < WINDOW["nperseg"]:
        return np.asarray(tracks, np.float32).copy()
    bins = analyze(mix, tracks)
    cleanup = corrections(bins, confidence, phase, gate)
    out = apply_filter(bins, cleanup, strength, shrink, noise)
    return istft(out, **WINDOW)[1][..., : len(mix)].astype(np.float32)
