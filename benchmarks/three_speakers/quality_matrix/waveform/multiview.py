"""Reliability-limited combination of identical mixtures separated in two acoustic views."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import istft, stft
from scipy.ndimage import uniform_filter

from debleed import WINDOW


def energy(array):
    return np.maximum(0.0, uniform_filter(np.abs(array) ** 2, size=(1, 5, 9)))


def align(baseline, second):
    baseline = np.array(baseline, np.float32)
    second = np.array(second, np.float32)
    a = baseline / np.maximum(np.linalg.norm(baseline, axis=1, keepdims=True), 1e-10)
    b = second / np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-10)
    similarity = a @ b.T
    i, j = linear_sum_assignment(-np.abs(similarity))
    second = second[j] * np.sign(similarity[i, j])[:, None]
    # Use only global scaling to preserve audible gain through every aligned channel.
    gain = np.sum(baseline * second, axis=-1) / np.maximum(
        np.sum(second**2, axis=-1), 1e-10
    )
    second *= np.clip(gain, 0.35, 2)[:, None]
    return second, np.abs(similarity[i, j])


def features(baseline, second):
    second, credibility = align(baseline, second)
    x = stft(baseline, **WINDOW)[2]
    y = stft(second, **WINDOW)[2]
    xx = energy(x)
    yy = energy(y)
    cov = uniform_filter(x * y.conj(), size=(1, 5, 9))
    agreement = np.clip(np.real(cov) / (np.sqrt(xx * yy) + 1e-10), 0.0, 1.0)
    exp = max(float(xx.max()), float(yy.max()), 1e-8) * 1e-5
    own = xx / (xx.sum(axis=0) + exp)
    rival = yy / (yy.sum(axis=0) + exp)
    return x, y, xx, yy, agreement, own, rival, credibility


def combine(feat, length, technique, blend):
    x, y, xx, yy, agreement, own, rival, credibility = feat
    confidence = np.clip((credibility - 0.25) / 0.5, 0, 1)[:, None, None]
    if technique == "local":
        mix = np.clip(0.30 + 0.70 * (rival - own), 0.08, 0.85)
        return istft(
            (x + confidence * blend * mix * (y - x)).astype(np.complex64), **WINDOW
        )[1][:, :length]
    if technique == "consensus":
        weight = 1 - confidence * blend * (1 - agreement) * np.clip(
            1 - rival, 0.0, 1.0
        ) * np.clip(1 - own, 0.05, 0.8)
        return istft((x * weight).astype(np.complex64), **WINDOW)[1][:, :length]
    if technique == "stable_add":
        mix = (
            np.sqrt(np.minimum(xx, yy) / np.maximum(np.maximum(xx, yy), 1e-12))
            * agreement
        )
        return istft(
            (x + confidence * blend * mix * (y - x)).astype(np.complex64), **WINDOW
        )[1][:, :length]
    raise ValueError(technique)
