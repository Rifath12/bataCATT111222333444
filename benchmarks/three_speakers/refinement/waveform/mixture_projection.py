"""Refine time-domain source estimates against a single observed mono mixture."""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import istft, stft


def refine_tracks(mixture, tracks, *, frequency_fit=True, fraction=0.5):
    """Use regularized complex gain fitting and energy-weighted residual allocation.

    The input mixture and estimated tracks must have the same sampling rate and
    time axis. Source order is preserved. The transform uses 32 ms windows at
    the separator's native 8 kHz rate.
    """
    mixture = np.asarray(mixture, dtype=np.float32)
    tracks = np.asarray(tracks, dtype=np.float32)
    if mixture.ndim != 1 or tracks.ndim != 2 or tracks.shape[1] != mixture.size:
        raise ValueError("Expected mono signal and aligned [source, sample] tracks")
    if not 0 <= fraction <= 1:
        raise ValueError("Residual fraction must lie in [0,1]")
    if len(mixture) < 256 or not tracks.size:
        return tracks.copy()

    gram = (tracks @ tracks.T).astype(np.float64)
    ridge = max(1e-12, np.trace(gram) * 0.001 / len(tracks))
    scales = np.linalg.solve(gram + np.eye(len(tracks)) * ridge, tracks @ mixture)
    estimated = (scales[:, None] * tracks).astype(np.float32)
    spec_kw = dict(nperseg=256, noverlap=192)
    bins = stft(estimated, **spec_kw)[2].astype(np.complex128)
    observation = stft(mixture, **spec_kw)[2].astype(np.complex128)

    if frequency_fit:
        # Smooth the local normal equations, rather than unstable pointwise gains.
        cov = np.einsum("sft,rft->fsr", bins.conj(), bins)
        rhs = np.einsum("sft,ft->fs", bins.conj(), observation - bins.sum(axis=0))
        cov = gaussian_filter1d(cov, 2, axis=0)
        rhs = gaussian_filter1d(rhs, 2, axis=0)
        power = np.diagonal(cov, axis1=1, axis2=2).real
        regularizer = 0.20 * power + 0.04 * power.sum(axis=1, keepdims=True) + 1e-10
        diag = np.arange(len(tracks))
        cov[:, diag, diag] += regularizer
        gain_delta = np.linalg.solve(cov, rhs[..., None])[..., 0]
        gain_delta /= np.maximum(1, np.abs(gain_delta) / 0.7)
        bins *= 1 + gain_delta.T[:, :, None]

    remainder = observation - bins.sum(axis=0)
    energy = np.abs(bins) ** 2
    total = energy.sum(axis=0)
    allocation = energy / np.maximum(1e-18, total)
    # Strongly unsupported residuals in a bin may be room noise.
    reliability = total / np.maximum(1e-18, total + np.abs(remainder) ** 2)
    bins += fraction * allocation * reliability * remainder
    restored = istft(bins, **spec_kw)[1][:, : mixture.size]
    return restored.astype(np.float32)
