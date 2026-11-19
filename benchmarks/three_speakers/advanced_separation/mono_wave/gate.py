"""Detect collapsed three-track separation without isolated references."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls


@dataclass(frozen=True)
class MixtureFit:
    explained: float
    gains: np.ndarray
    fractions: np.ndarray
    independence: float


def fit_mixture(mixture, tracks):
    """Measure positive reconstruction after subtracting per-signal DC offsets."""
    mixture = np.asarray(mixture, dtype=np.float64)
    tracks = np.asarray(tracks, dtype=np.float64)
    if (
        mixture.ndim != 1 or not mixture.size or tracks.ndim != 2
        or not tracks.shape[0] or tracks.shape[1] != mixture.size
    ):
        raise ValueError("Expected a nonempty sample-aligned mixture and multiple tracks")
    empty = np.zeros(tracks.shape[0])
    failure = MixtureFit(0.0, empty, empty, 0.0)
    if not np.all(np.isfinite(mixture)) or not np.all(np.isfinite(tracks)):
        return failure
    mixture = mixture - mixture.mean()
    tracks = tracks - tracks.mean(axis=1, keepdims=True)
    mixture_norm = np.linalg.norm(mixture)
    norms = np.linalg.norm(tracks, axis=1)
    live = norms > 0
    if (
        mixture_norm <= 1e-6 or not np.isfinite(mixture_norm)
        or not np.all(np.isfinite(norms)) or not np.any(live)
    ):
        return failure

    # In the normalized problem a coefficient measures the track's calibrated
    # L2 norm relative to the mixture, independent of the model's output scale.
    design = tracks[live] / norms[live, None]
    try:
        eigenvalues = np.linalg.eigvalsh(design @ design.T)
        weights, residual = nnls(design.T, mixture / mixture_norm)
    except (RuntimeError, np.linalg.LinAlgError):
        return failure
    independence = max(0.0, float(eigenvalues[0] / eigenvalues[-1])) if live.all() else 0.0
    fractions = np.zeros(tracks.shape[0])
    gains = np.zeros(tracks.shape[0])
    fractions[live] = weights
    gains[live] = weights * (mixture_norm / norms[live])
    return MixtureFit(float(np.clip(1.0 - residual**2, 0.0, 1.0)), gains, fractions, independence)


def select_tracks(mixture, baseline, candidate, *, first_pass=None, min_explained=0.90):
    """Require positive, independently supported contributions from three aligned tracks."""
    if not 0.0 <= min_explained <= 1.0:
        raise ValueError("min_explained must lie in [0, 1]")
    mixture = np.asarray(mixture)
    if mixture.ndim != 1 or not mixture.size:
        raise ValueError("Expected a nonempty mono mixture")
    baseline = np.asarray(baseline, dtype=np.float32)
    if baseline.shape != (3, mixture.size) or not np.all(np.isfinite(baseline)):
        raise ValueError("Expected three finite stable baseline tracks")
    metrics = {}
    for title, alternative in (("second_pass", candidate), ("first_pass", first_pass)):
        if alternative is None:
            continue
        alternative = np.asarray(alternative, dtype=np.float64)
        if alternative.shape != baseline.shape:
            raise ValueError("Expected aligned candidate tracks with the same shape as baseline")
        fitted = fit_mixture(mixture, alternative)
        metrics[title] = fitted.explained
        if (
            fitted.explained < min_explained or fitted.independence < 0.01
            or np.min(fitted.fractions) < 0.01
        ):
            continue
        centered = alternative - alternative.mean(axis=1, keepdims=True)
        calibrated = centered * fitted.gains[:, None]
        if np.all(np.isfinite(calibrated)) and np.max(np.abs(calibrated)) <= np.finfo(np.float32).max:
            return calibrated.astype(np.float32), title, metrics
    return baseline, "baseline", metrics
