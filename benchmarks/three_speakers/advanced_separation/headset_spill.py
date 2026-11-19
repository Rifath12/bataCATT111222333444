"""Reduce sustained off-speaker audio on separately recorded physical headsets."""
from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import signal
from scipy.fft import dct
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.spatial import cKDTree
import soundfile as sf

from close_waveform import FROZEN_PROFILE, RATE, render, stored_transfer

FFT = 512
HOP = 128
CONTEXT = 16384
BANDS = (120, 270, 490, 800, 1250, 1950, 3000, 4550, 6500)
FREQUENCIES = np.fft.rfftfreq(FFT, 1 / RATE)


def spectrum(audio):
    padded = np.pad(audio, ((0, 0), (0, max(0, FFT - audio.shape[1]))))
    return signal.stft(padded, nperseg=FFT, noverlap=FFT - HOP)[2]


def log_ratio(numerator, denominator):
    return np.clip(np.log((numerator + 1e-12) / (denominator + 1e-12)), -12, 12)


def presence_features(z):
    """Per-mic spectral level, shape and coherence with neighboring headsets."""
    power = np.abs(z) ** 2
    bands = np.stack([
        power[:, (FREQUENCIES >= low) & (FREQUENCIES < high)].mean(axis=1)
        for low, high in zip(BANDS[:-1], BANDS[1:], strict=True)
    ], axis=1)
    local = uniform_filter1d(bands, 7, axis=-1, mode="nearest")
    broad = uniform_filter1d(bands, 31, axis=-1, mode="nearest")
    rows = []
    for target in range(3):
        others = [index for index in range(3) if index != target]
        competing = local[others].max(axis=0)
        competing_broad = broad[others].max(axis=0)
        direct = local[target]
        ratio = log_ratio(direct, competing)
        wide = log_ratio(broad[target], competing_broad)
        parts = [
            ratio.mean(axis=0), ratio.min(axis=0), ratio.max(axis=0), np.median(ratio, axis=0),
            wide.mean(axis=0), wide.max(axis=0), log_ratio(direct.mean(axis=0), competing.mean(axis=0)),
            log_ratio(broad[target].mean(axis=0), competing_broad.mean(axis=0)),
            log_ratio(local[others[0]].mean(axis=0), local[others[1]].mean(axis=0)),
        ]
        parts.extend(dct(ratio, type=2, norm="ortho", axis=0)[:5])
        parts.extend(dct(log_ratio(direct, direct.mean(axis=0)), type=2, norm="ortho", axis=0)[1:5])
        for other in others:
            cross = uniform_filter1d(z[target] * z[other].conj(), 13, axis=-1, mode="nearest")
            normalizer = (uniform_filter1d(power[target], 13, axis=-1, mode="nearest")
                          * uniform_filter1d(power[other], 13, axis=-1, mode="nearest"))
            coherence = np.clip(abs(cross) ** 2 / np.maximum(normalizer, 1e-17), 0, 1)
            for lo, hi in ((120, 700), (700, 2300), (2300, 6500)):
                mask = (FREQUENCIES >= lo) & (FREQUENCIES < hi)
                parts.append(6 * coherence[mask].mean(axis=0))
        rows.append(np.stack(parts, axis=-1))
    return np.stack(rows).astype(np.float32)


def pilot_excerpts(takes):
    if len(takes) != 3:
        raise ValueError("Supply solo takes for A, B and C")
    banks = []
    for who, audio in enumerate(takes):
        if audio.ndim != 2 or audio.shape[0] != 3 or not np.isfinite(audio).all():
            raise ValueError("Each solo take needs three synchronized, finite headset channels")
        energy = np.var(audio, axis=1, dtype=np.float64)
        if energy[who] <= max(np.delete(energy, who)):
            raise ValueError(f"Solo {'ABC'[who]} is not strongest on its own headset")
        bank = []
        for start in range(0, audio.shape[1] - 8192 + 1, 4096):
            excerpt = audio[:, start:start + 8192]
            level = np.sqrt(np.mean(excerpt[who] ** 2))
            if level > .002:
                bank.append(excerpt * (.05 / level))
        if len(bank) < 3:
            raise ValueError("Each speaker needs at least three qualifying half-second solo excerpts")
        banks.append(bank)
    return banks


@dataclass
class RoomPresence:
    trees: list
    labels: list
    centers: np.ndarray
    scales: np.ndarray

    @classmethod
    def fit(cls, takes):
        banks = pilot_excerpts(takes)
        rng = np.random.default_rng(14328)
        rows = [[], [], []]
        labels = [[], [], []]
        for index in range(224):
            combination = index % 7 + 1
            active = np.array([bool((combination >> speaker) & 1) for speaker in range(3)])
            levels = np.zeros(3)
            levels[active] = rng.uniform(-27, 0, active.sum())
            levels -= max(levels[active])
            donors = np.stack([bank[rng.integers(len(bank))] for bank in banks])
            mixture = ((10 ** (levels / 20) * active)[:, None, None] * donors).sum(0).astype(np.float32)
            features = presence_features(spectrum(mixture))
            for mic in range(3):
                batch = features[mic, 4:-4:4]
                rows[mic].append(batch)
                labels[mic].append(np.full(len(batch), active[mic], dtype=np.float32))
        trees, centers, scales = [], [], []
        for mic in range(3):
            values = np.concatenate(rows[mic])
            center = np.median(values, axis=0)
            scale = np.maximum(.18, np.std(values, axis=0))
            centers.append(center)
            scales.append(scale)
            trees.append(cKDTree((values - center) / scale))
        return cls(trees, [np.concatenate(group) for group in labels], np.array(centers), np.array(scales))

    def presence(self, z):
        rows = []
        for mic, features in enumerate(presence_features(z)):
            normalized = (features - self.centers[mic]) / self.scales[mic]
            distances, neighbors = self.trees[mic].query(normalized, k=27, workers=1)
            weights = 1 / np.maximum(distances, .15) ** 2
            score = np.average(self.labels[mic][neighbors], axis=1, weights=weights)
            rows.append(np.where(distances[:, 5] > 6.5, 1., score))
        return np.stack(rows)

    def reduce(self, audio, previous, transfer):
        if previous.shape != audio.shape or audio.ndim != 2 or audio.shape[0] != 3:
            raise ValueError("Expected matching ordered A/B/C microphone and waveform arrays")
        if not audio.shape[1]:
            return previous.copy()
        z = spectrum(audio)
        observed = uniform_filter1d(np.abs(z) ** 2, 15, axis=-1, mode="nearest")
        cross = transfer[::2] - np.eye(3, dtype=np.float32)[None]
        foreign = np.einsum("fij,jft->ift", np.maximum(cross, 0), observed, optimize=True)
        speech = slice(4, 209)
        ratio = foreign[:, speech].sum(axis=1) / np.maximum(observed[:, speech].sum(axis=1), 1e-13)
        support = np.clip((10 * np.log10(np.maximum(ratio, 1e-10)) + 18.) / 12., 0., 1.)
        nearby = maximum_filter1d(self.presence(z), 19, axis=-1, mode="nearest")
        absent = np.clip((.5 - nearby) / .25, 0., 1.)
        gain = uniform_filter1d(1. - .6 * absent * support, 7, axis=-1, mode="nearest")
        # Preserve speech around a turn boundary. Both room evidence and the physical
        # transfer must support reducing the track for the full surrounding 1.024 seconds.
        gain = maximum_filter1d(gain, 129, axis=-1, mode="nearest")
        gain = uniform_filter1d(np.maximum(.5, gain), 9, axis=-1, mode="nearest")
        sample_positions = np.arange(audio.shape[1]) / HOP
        by_sample = np.stack([np.interp(sample_positions, np.arange(row.size), row) for row in gain])
        return (previous * by_sample).astype(np.float32)


def read_pilots(paths):
    if len({path.resolve() for path in paths}) != 3:
        raise ValueError("Supply a different solo recording for each speaker")
    result = []
    for path in paths:
        with sf.SoundFile(path) as recording:
            if recording.channels != 3 or recording.samplerate != RATE:
                raise ValueError("Each solo WAV needs three 16 kHz channels in fixed A/B/C order")
            # Use at most the first minute of each pilot so fitting memory is bounded.
            result.append(recording.read(60 * RATE, dtype="float32").T)
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("microphones", nargs=3, type=Path, metavar="MIC_WAV")
    parser.add_argument("--pilots", nargs=3, required=True, type=Path, metavar="SOLO_WAV")
    parser.add_argument("--power-transfer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    transfer = stored_transfer(args.power_transfer)
    started = perf_counter()
    model = RoomPresence.fit(read_pilots(args.pilots))
    fit_seconds = perf_counter() - started
    render(args.microphones, args.out, transfer, FROZEN_PROFILE,
           refinement=lambda raw, standard: model.reduce(raw, standard, transfer),
           context_samples=CONTEXT)
    report = {"sample_rate": RATE, "microphone_order": "ABC", "pilot_fit_seconds": fit_seconds,
              "processing_seconds": perf_counter() - started, "max_reduction_db": 6.0206,
              "mode": "standard_with_sustained_off_speaker_reduction"}
    (args.out / "headset_spill.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote speaker_A.wav, speaker_B.wav and speaker_C.wav in {report['processing_seconds']:.2f} s")


if __name__ == "__main__":
    main()
