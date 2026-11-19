"""Optional, bounded close-mic correction learned from an earlier meeting."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter

RATE = 16000
FFT = 512
HOP = 128
SHARE = 0.35
CONTEXT_SAMPLES = 4096
MODEL_PATH = Path(__file__).with_name("boosted_m8014.json")


def stft(audio):
    return signal.stft(audio, fs=RATE, nperseg=FFT, noverlap=FFT-HOP, axis=-1)[2]


def istft(spectrum, length):
    return signal.istft(spectrum, fs=RATE, nperseg=FFT, noverlap=FFT-HOP,
                        time_axis=-1, freq_axis=-2)[1][..., :length]


def power_smooth(z, frequencies, time):
    return uniform_filter(np.abs(z) ** 2, size=(1, frequencies, time), mode="nearest")


def log_power(x):
    return np.log(np.maximum(x, 2e-12)) * .25


def prepare(audio, calibration):
    z = stft(audio)
    # The two transform widths share the same hop and waveform clock.
    wide = signal.stft(audio, fs=16000, nperseg=1024, noverlap=896, axis=-1)[2]
    immediate = power_smooth(z, 1, 3)
    envelope = power_smooth(z, 7, 9)
    broad = power_smooth(z, 17, 33)
    fine = power_smooth(wide, 5, 11)[:, ::2, :z.shape[-1]]
    phase = {}
    for mic in range(3):
        for other in range(3):
            if other == mic:
                continue
            c = uniform_filter(z[mic] * z[other].conj(), size=(1, 13), mode="nearest")
            d = np.sqrt(np.maximum(immediate[mic] * immediate[other], 1e-22))
            phase[mic, other] = np.maximum(0, np.minimum(np.abs(c) / d, 2))
    priors = calibration[::2]
    estimated_bleed = np.einsum("fij,jft->ift", priors - np.eye(3)[None], envelope, optimize=True)
    return z, immediate, envelope, broad, fine, phase, priors, estimated_bleed


def channel_features(tensors, mic):
    z, immediate, envelope, broad, fine, phase, priors, bleed = tensors
    others = [idx for idx in range(3) if idx != mic]
    freq = np.broadcast_to(np.linspace(0, 1, z.shape[1])[:, None], z[mic].shape)
    full_power = (envelope.mean(1, keepdims=True))
    samples = [freq, np.full_like(freq, mic / 2), log_power(immediate[mic]),
               log_power(envelope[mic]), log_power(broad[mic]), log_power(fine[mic]),
               log_power(immediate[mic])-log_power(envelope[mic]),
               log_power(bleed[mic])-log_power(envelope[mic])]
    for other in others:
        leak = priors[:, mic, other, None] * envelope[other]
        mic_envelope = log_power(envelope[other]) - log_power(envelope[mic])
        mic_abs = log_power(immediate[other]) - log_power(immediate[mic])
        samples.extend([
            mic_envelope, mic_abs, log_power(broad[other]) - log_power(broad[mic]),
            log_power(fine[other]) - log_power(fine[mic]),
            log_power(leak) - log_power(envelope[mic]), phase[mic, other],
            np.broadcast_to(log_power(full_power[other])-log_power(full_power[mic]), freq.shape),
        ])
    return np.stack(samples, axis=-1).reshape(-1, len(samples)).astype(np.float32)


@dataclass(frozen=True)
class Tree:
    column: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    value: np.ndarray

    def predict(self, features):
        index = np.zeros(len(features), dtype=np.int32)
        active = np.arange(len(features), dtype=np.int32)
        while active.size:
            node = index[active]
            active = active[self.column[node] >= 0]
            node = index[active]
            index[active] = np.where(
                features[active, self.column[node]] <= self.threshold[node],
                self.left[node], self.right[node],
            )
        return self.value[index]


@dataclass(frozen=True)
class BoostedMask:
    context_samples = CONTEXT_SAMPLES

    intercept: float
    trees: tuple[Tree, ...]
    fast_predict: Callable | None = None

    @classmethod
    def load(cls, path=MODEL_PATH, *, fast=False):
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("format") != 1 or document.get("features") != 22:
            raise ValueError("Unsupported close-mic correction model")
        intercept = float(document["intercept"])
        if not np.isfinite(intercept) or not 1 <= len(document["trees"]) <= 160:
            raise ValueError("Malformed close-mic correction model")
        trees = []
        for encoded in document["trees"]:
            values = np.asarray(encoded, dtype=np.float64)
            if values.ndim != 2 or values.shape[1] != 5 or not 1 <= len(values) <= 255 or not np.isfinite(values).all():
                raise ValueError("Malformed close-mic correction tree")
            columns, thresholds, left, right, outputs = values.T
            indices = np.arange(len(values))
            branch = columns >= 0
            if (
                np.any(columns != columns.astype(np.int32))
                or np.any((columns < -1) | (columns >= 22))
                or np.any(left[branch] != left[branch].astype(np.int32))
                or np.any(right[branch] != right[branch].astype(np.int32))
                or np.any(left[branch] <= indices[branch])
                or np.any(right[branch] <= indices[branch])
                or np.any(left[branch] >= len(values))
                or np.any(right[branch] >= len(values))
            ):
                raise ValueError("Invalid branch in close-mic correction tree")
            trees.append(Tree(columns.astype(np.int32), thresholds, left.astype(np.int32),
                              right.astype(np.int32), outputs))
        predictor = None
        if fast:
            try:
                from boosted_fast import compile_predictor
            except ImportError as exc:
                raise RuntimeError("Install numba to enable --fast-trees") from exc
            predictor = compile_predictor(trees, intercept)
        return cls(intercept, tuple(trees), predictor)

    def predict(self, features):
        if self.fast_predict is not None:
            return self.fast_predict(features)
        prediction = np.full(len(features), self.intercept, dtype=np.float64)
        for tree in self.trees:
            prediction += tree.predict(features)
        return prediction

    def refine(self, microphones, transfer, baseline):
        """Blend the fixed room-trained mask with synchronized baseline tracks."""
        if microphones.shape != baseline.shape or microphones.ndim != 2 or microphones.shape[0] != 3:
            raise ValueError("Expected three synchronized microphone and baseline tracks")
        if microphones.shape[1] == 0:
            return baseline.copy()
        if transfer.shape != (513, 3, 3) or not np.isfinite(transfer).all():
            raise ValueError("Expected the calibrated close-mic power transfer matrix")
        length = microphones.shape[1]
        padded = np.pad(microphones, ((0, 0), (0, max(0, 1024-length))))
        tensors = prepare(padded, transfer)
        complex_audio = tensors[0]
        out = []
        for mic in range(3):
            features = channel_features(tensors, mic)
            gains = self.predict(features).reshape(complex_audio[mic].shape)
            gains = uniform_filter(gains, size=(1, 3), mode="nearest")
            out.append(complex_audio[mic] * gains)
        selected = istft(np.stack(out), length).astype(np.float32)
        return baseline + SHARE * (selected-baseline)
