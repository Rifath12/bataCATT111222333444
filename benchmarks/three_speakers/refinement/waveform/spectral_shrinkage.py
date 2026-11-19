"""Estimate a bounded spectral gain from separated tracks and a mono mixture."""

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.optimize import linear_sum_assignment
from scipy.signal import istft, stft
from sklearn.ensemble import HistGradientBoostingRegressor

STFT_ARGS = dict(nperseg=256, noverlap=192)


def observations(mix, tracks):
    gram = (tracks @ tracks.T).astype(np.float64)
    ridge = max(1e-12, np.trace(gram) * 0.001 / len(tracks))
    scales = np.linalg.solve(gram + np.eye(len(tracks)) * ridge, tracks @ mix)
    bins = stft((tracks * scales[:, None]).astype(np.float32), **STFT_ARGS)[2]
    mixture = stft(mix, **STFT_ARGS)[2]
    power = np.abs(bins) ** 2
    floor = max(1e-12, power.max() * 1e-8)
    total = power.sum(axis=0, keepdims=True)
    feature = [
        np.log10((power + floor) / (total + floor * 3)),
        np.log10((power + floor) / (np.abs(mixture) ** 2 + floor)),
        np.log10((power + floor) / (power.mean(axis=(1, 2), keepdims=True) + floor)),
        np.broadcast_to(
            np.log10((total + floor) / (np.abs(mixture) ** 2 + floor)), power.shape
        ),
        np.broadcast_to(np.linspace(0, 1, power.shape[1])[None, :, None], power.shape),
    ]
    for smooth in ((1, 7, 5), (1, 15, 15)):
        blurred = uniform_filter(power, smooth)
        feature.append(
            np.log10(
                (blurred + floor) / (blurred.sum(axis=0, keepdims=True) + 3 * floor)
            )
        )
    strength = np.abs(bins) / (power.mean(axis=(1, 2), keepdims=True) ** 0.5 + 1e-8)
    return bins, np.stack(feature, axis=-1).astype(np.float32), strength


def fit(cases):
    rng = np.random.default_rng(197)
    xs, ys, ws = [], [], []
    for mix, tracks, refs in cases:
        bins, x, strength = observations(mix, tracks)
        score = (
            tracks
            @ refs.T
            / np.maximum(
                1e-8,
                np.outer(np.linalg.norm(tracks, axis=1), np.linalg.norm(refs, axis=1)),
            )
        )
        i, j = linear_sum_assignment(-np.abs(score))
        ref_spec = stft(refs[j[np.argsort(i)]], **STFT_ARGS)[2]
        denominator = np.abs(bins) ** 2
        target = (ref_spec * bins.conj()).real / np.maximum(1e-14, denominator)
        target = np.clip(target, 0, 1.5)
        # Equalize source contributions and softly emphasize audible time-frequency bins.
        weights = np.clip(strength, 0.05, 5.0)
        select = rng.random(size=weights.shape) < 0.18
        xs.append(x[select])
        ys.append(target[select])
        ws.append(weights[select])
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=55,
        max_leaf_nodes=14,
        min_samples_leaf=100,
        l2_regularization=1,
        early_stopping=False,
        random_state=197,
    )
    return model.fit(
        np.concatenate(xs), np.concatenate(ys), sample_weight=np.concatenate(ws)
    )


def refine(mix, tracks, model, strength):
    bins, x, _ = observations(mix, tracks)
    weight = model.predict(x.reshape(-1, x.shape[-1])).reshape(bins.shape)
    gain = 1 + strength * (np.clip(weight, 0.08, 1.35) - 1)
    result = istft(bins * gain, **STFT_ARGS)[1]
    return result[:, : len(mix)].astype(np.float32)
