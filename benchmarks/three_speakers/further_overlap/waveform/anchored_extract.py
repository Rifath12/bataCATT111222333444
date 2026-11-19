"""Experimental mono dark-voice recovery using same-speaker enrollment and a second extraction pass."""

import numpy as np
from scipy.signal import resample_poly


def scale_estimates(estimates, mixture):
    """Align output levels before subtracting one voice from the physical mixture."""
    gram = estimates.astype(np.float64) @ estimates.T
    ridge = max(1.0e-8, np.trace(gram) * 0.025 / len(estimates))
    gains = np.linalg.solve(gram + ridge * np.eye(len(estimates)), estimates @ mixture)
    return estimates * np.clip(gains, 0.0, 2.0).astype("float32")[:, None]


def call_extractor(
    extractor, mixture, anchors, window_samples=56000, fade_samples=6400, whole=False
):
    """Bound memory on long inputs and retain enrolled identities across chunks."""
    upsampled = resample_poly(mixture, 2, 1).astype("float32")
    if whole or len(upsampled) <= window_samples:
        restored = extractor.extract(upsampled, anchors)
    else:
        restored = np.zeros((len(anchors), len(upsampled)), dtype="float32")
        normalization = np.zeros(len(upsampled), dtype="float32")
        step = window_samples - fade_samples
        ramp = np.sin(np.linspace(0.0, np.pi / 2, fade_samples, dtype="float32")) ** 2
        i = 0
        while i < len(upsampled):
            end = min(i + window_samples, len(upsampled))
            wave = upsampled[i:end]
            valid = len(wave)
            if valid < 2048:
                wave = np.pad(wave, (0, 2048 - valid))
            out = extractor.extract(wave, anchors)[:, :valid]
            combined = np.ones(valid, dtype="float32")
            if i:
                combined[:fade_samples] = ramp[: min(valid, fade_samples)]
            if end != len(upsampled):
                combined[-fade_samples:] = 1.0 - ramp
            restored[:, i:end] += out * combined
            normalization[i:end] += combined
            if end == len(upsampled):
                break
            i += step
        restored /= np.maximum(normalization, 1.0e-6)[None]
    return resample_poly(restored, 1, 2, axis=-1)[:, : len(mixture)].astype("float32")


def observe(extractor, mixture, anchors, subtract, whole=False):
    direct = call_extractor(extractor, mixture, anchors, whole=whole)
    fitted = scale_estimates(direct, mixture)
    energy = np.mean(np.square(fitted), axis=-1)
    residual = direct.copy()
    for target in np.argsort(energy)[:-1]:
        strong = np.flatnonzero(energy > 1.5 * energy[target])
        if not len(strong):
            continue
        rest = mixture - subtract * fitted[strong].sum(axis=0)
        correction = call_extractor(
            extractor, rest, anchors[target : target + 1], whole=whole
        )[0]
        if np.all(np.isfinite(correction)):
            residual[target] = correction
    return direct, residual


def choose_track(neural_tracks, target_tracks, anchors, embed, margin=0.10):
    """Keep an anchored candidate only when it provides a stronger identity margin."""
    from scipy.optimize import linear_sum_assignment

    def similarities(tracks):
        result = np.full((len(anchors), len(tracks)), -1.0, dtype=np.float32)
        for col, track in enumerate(tracks):
            vector = embed(track)
            if vector is not None:
                for row, anchor in enumerate(anchors):
                    if anchor is not None:
                        result[row, col] = vector @ anchor
        return result

    base = similarities(neural_tracks)
    target = similarities(target_tracks)
    rows, cols = linear_sum_assignment(-base)
    output = neural_tracks[cols].copy()
    for row, col in zip(rows, cols, strict=True):
        others = np.arange(len(rows)) != row
        direct_margin = target[row, row] - max(target[others, row])
        base_margin = base[row, col] - max(base[others, col])
        if (
            target[row, row] > 0.30
            and direct_margin > 0.12
            and direct_margin > base_margin + margin
        ):
            output[row] = target_tracks[row]
    return output
