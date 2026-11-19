"""Detect three-track reconstruction collapse without isolated references."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls


@dataclass(frozen=True)
class MixtureFit:
    explained: float
    gains: np.ndarray


def fit_mixture(mixture, tracks):
    """Fit positive per-track gains and measure explained centered PCM energy."""
    mixture = np.asarray(mixture, dtype=np.float64)
    tracks = np.asarray(tracks, dtype=np.float64)
    if mixture.ndim != 1 or tracks.ndim != 2 or tracks.shape[1] != len(mixture):
        raise ValueError("Expected one sample-aligned mixture and multiple tracks")
    if not np.all(np.isfinite(mixture)) or not np.all(np.isfinite(tracks)):
        return MixtureFit(0.0, np.zeros(tracks.shape[0]))
    mixture = mixture - np.mean(mixture)
    tracks = tracks - tracks.mean(axis=1, keepdims=True)
    energy = float(mixture @ mixture)
    if energy <= 1e-12:
        return MixtureFit(0.0, np.zeros(tracks.shape[0]))
    try:
        gains, residual = nnls(tracks.T, mixture)
    except RuntimeError:
        return MixtureFit(0.0, np.zeros(tracks.shape[0]))
    return MixtureFit(float(np.clip(1.0 - residual ** 2 / energy, 0.0, 1.0)), gains)


def select_tracks(mixture, baseline, candidate, *, first_pass=None, min_explained=0.90):
    """Select a well-supported candidate; otherwise retain the existing stems.

    Candidate slots must already be matched to stable speakers when names
    are needed. The mixture test detects gross model collapse and cannot
    certify correct transcription or rule out voice merging.
    """
    if not 0.0 <= min_explained <= 1.0:
        raise ValueError("min_explained must lie in [0, 1]")
    baseline = np.asarray(baseline, dtype=np.float32)
    if baseline.shape != (3, len(mixture)):
        raise ValueError("Expected three stable baseline tracks")
    metrics = {}
    for title, alternative in (("second_pass", candidate), ("first_pass", first_pass)):
        if alternative is None:
            continue
        alternative = np.asarray(alternative, dtype=np.float32)
        if alternative.shape != baseline.shape:
            raise ValueError("Expected aligned candidate tracks with the same shape as baseline")
        fitted = fit_mixture(mixture, alternative)
        metrics[title] = fitted.explained
        if fitted.explained >= min_explained and np.all(fitted.gains > 0):
            calibrated = alternative * fitted.gains.astype(np.float32)[:, None]
            return calibrated, title, metrics
    return baseline, "baseline", metrics
