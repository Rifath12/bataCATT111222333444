"""Single-pass, 8 kHz spectral refinement for three aligned SepFormer estimates."""

import json
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.signal import istft, stft

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
    return bins, np.stack(feature, axis=-1).astype(np.float32)


class SpectralRefiner:
    """Evaluate frozen scalar regression trees using their numeric JSON representation."""

    def __init__(self, path=None):
        path = (
            Path(path)
            if path is not None
            else Path(__file__).with_name("refine_three.json")
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["format"] != "spectral_gains_v1" or data["features"] != 7:
            raise ValueError("Incompatible spectral gain model")
        self.baseline = data["baseline"]
        self.strength = data["strength"]
        self.trees = data["trees"]

    def predict(self, features):
        # Store one bit per time-frequency bin. Many trees share threshold comparisons.
        masks = {}
        result = np.full(len(features), self.baseline, np.float64)
        for nodes in self.trees:
            leaf_count = sum(len(node) == 1 for node in nodes)
            width = (leaf_count - 1).bit_length()
            if width == 0:
                result += nodes[0][0]
                continue
            # The binary ID of the reached leaf records which constant to add.
            bits = np.zeros((width, (len(features) + 7) // 8), dtype=np.uint8)
            gain = []
            pending = [(0, None)]
            while pending:
                idx, path_mask = pending.pop()
                node = nodes[idx]
                if len(node) == 1:
                    leaf_id = len(gain)
                    gain.append(node[0])
                    for bit in range(width):
                        if leaf_id & (1 << bit):
                            bits[bit] |= path_mask
                    continue
                feature, cutoff, left, right = node
                key = (feature, cutoff)
                if key not in masks:
                    masks[key] = np.packbits(
                        features[:, feature].astype(np.float64) <= cutoff
                    )
                split = masks[key]
                if path_mask is None:
                    pending.extend(((right, ~split), (left, split)))
                else:
                    pending.extend(
                        ((right, path_mask & ~split), (left, path_mask & split))
                    )
            levels = np.unpackbits(bits, axis=1)[:, : len(features)]
            ids = levels[0].astype(np.uint8 if width <= 8 else np.int32)
            for bit in range(1, width):
                ids |= levels[bit].astype(ids.dtype, copy=False) << bit
            result += np.asarray(gain)[ids]
        return result

    def __call__(self, mixture, tracks):
        if tracks.shape != (3, len(mixture)):
            raise ValueError("Expected three sample-aligned 8 kHz separated tracks")
        if len(mixture) < STFT_ARGS["nperseg"]:
            return tracks.copy()
        bins, features = observations(mixture, tracks)
        weights = self.predict(features.reshape(-1, features.shape[-1])).reshape(
            bins.shape
        )
        gain = 1 + self.strength * (np.clip(weights, 0.08, 1.35) - 1)
        return istft(bins * gain, **STFT_ARGS)[1][:, : len(mixture)].astype(np.float32)
