"""Trial with room-array channels and marked speaker turns in the first 120 seconds."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import fft
from scipy.ndimage import uniform_filter1d, median_filter
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier
import soundfile as sf

from diarize import reference, masks, score


RATE = 16000
HOP = 160
FFT = 512
WINDOW = np.sqrt(np.hanning(FFT + 1)[:-1]).astype(np.float32)
BANDS = tuple(
    (a * FFT // RATE, b * FFT // RATE)
    for a, b in (
        (200, 500),
        (500, 900),
        (900, 1500),
        (1500, 2300),
        (2300, 3300),
        (3300, 4500),
        (4500, 6200),
    )
)


def stft(file, start, end):
    lo = start * HOP - (FFT // 2 - HOP // 2)
    hi = (end - 1) * HOP + (FFT // 2 + HOP // 2)
    file.seek(max(lo, 0))
    signal = file.read(
        max(0, min(hi, file.frames) - max(lo, 0)), dtype="float32", always_2d=True
    )
    signal = np.pad(signal, ((max(0, -lo), max(0, hi - file.frames)), (0, 0)))
    windows = sliding_window_view(signal, FFT, axis=0)[::HOP]
    return fft.rfft(windows * WINDOW, axis=-1).transpose(0, 2, 1)


def templates(spectrum, activity):
    total = activity.sum(0)
    groups = np.r_[activity & (total == 1), (total == 0)[None, :]]
    result = []
    for index, mask in enumerate(groups):
        # Ignore the first/last 100 ms of each marked turn.
        good = uniform_filter1d(mask.astype(np.float32), 21) > 0.99
        data = spectrum[good].transpose(1, 0, 2)
        if data.shape[1] < 100:
            raise ValueError(
                f"Less than one second of usable enrollment for template {index}"
            )
        power = (np.abs(data) ** 2).sum(-1)
        coherence = data / np.sqrt(np.maximum(power[..., None], 1e-12))
        weight = np.sqrt(
            power / np.maximum(np.median(power, axis=1, keepdims=True), 1e-12)
        )
        weight = np.clip(weight, 0.03, 8)
        channels = data.shape[-1]
        cov = coherence.swapaxes(-1, -2).conj() @ (coherence * weight[..., None])
        cov = cov / np.maximum(weight.sum(1)[:, None, None], 1e-12)
        cov = 0.96 * cov + 0.04 * np.eye(channels, dtype=np.float32) / channels
        _, logdet = np.linalg.slogdet(cov)
        result.append(
            (np.linalg.inv(cov).astype(np.complex64), logdet.astype(np.float32))
        )
    return result


def probabilities(spectrum, fitted):
    vector = spectrum.transpose(1, 0, 2)
    power = (np.abs(vector) ** 2).sum(-1)
    unit = vector / np.sqrt(np.maximum(power[..., None], 1e-12))
    logits = np.stack(
        [
            -logdet[:, None]
            - unit.shape[-1]
            * np.log(np.maximum(((unit @ inv) * unit.conj()).real.sum(-1), 1e-5))
            for inv, logdet in fitted
        ]
    )
    return softmax(logits / 2.0, axis=0).transpose(0, 2, 1)


def features(spectrum, posterior):
    power = np.abs(spectrum) ** 2
    channel_energies = np.stack([power[:, a:b, :].mean(1) for a, b in BANDS], axis=1)
    absolute = np.log(np.maximum(channel_energies, 1e-10))
    relative = absolute - absolute.mean(2, keepdims=True)
    band_mix = power.mean(2)
    estimated = np.stack(
        [(posterior[:, :, a:b] * band_mix[:, a:b]).mean(-1).T for a, b in BANDS], axis=1
    )
    source_energy = np.log(np.maximum(estimated, 1e-10))
    source_relative = source_energy - source_energy.mean(2, keepdims=True)
    array = np.concatenate(
        [
            x.reshape(len(spectrum), -1)
            for x in (absolute.mean(2), relative, source_energy, source_relative)
        ],
        axis=1,
    )
    return np.hstack([array, uniform_filter1d(array, 21, axis=0)])


def timestamp_rows(activity, duration):
    segments = []
    for speaker, line in enumerate(activity):
        groups = np.flatnonzero(np.diff(np.pad(line.astype(np.int8), (1, 1))))
        for a, b in zip(groups[::2], groups[1::2]):
            segments.append(
                {
                    "speaker": str(speaker),
                    "start": float(a) * 0.01,
                    "end": min(duration, float(b) * 0.01),
                }
            )
    return sorted(segments, key=lambda x: (x["start"], x["speaker"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("array", type=Path)
    p.add_argument("--prefix-rttm", type=Path, required=True)
    p.add_argument("--eval-rttm", type=Path)
    p.add_argument("--prefix-seconds", type=int, default=120)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    began = time.perf_counter()
    p0 = args.prefix_seconds
    enroll = [
        dict(r, end=min(r["end"], p0))
        for r in reference(args.prefix_rttm)
        if r["start"] < p0
    ]
    label, train = masks(enroll, p0)
    if len(label) != 3:
        raise ValueError("The enrollment prefix must contain exactly three speakers")
    args.out.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(args.array) as file:
        if file.samplerate != RATE or file.channels < 2:
            raise ValueError("Expected a synchronized 16 kHz multichannel recording")
        total = int(np.ceil(file.frames / HOP))
        duration = file.frames / RATE
        warm_frames = round(p0 * 100)
        spectrum = stft(file, 0, warm_frames)
        spatial = templates(spectrum, train)
        training = features(spectrum, probabilities(spectrum, spatial))
        classifiers = []
        for target in train:
            model = HistGradientBoostingClassifier(
                max_iter=180, max_leaf_nodes=15, l2_regularization=5.0, random_state=1
            )
            classifiers.append(model.fit(training[::2], target[::2]))
        confidence = np.zeros((3, total), np.float32)
        for lo in range(warm_frames, total, 1200):
            hi = min(lo + 1200, total)
            a = max(0, lo - 15)
            b = min(total, hi + 15)
            spectrum = stft(file, a, b)
            observations = features(spectrum, probabilities(spectrum, spatial))[
                lo - a : hi - a
            ]
            confidence[:, lo:hi] = np.vstack(
                [m.predict_proba(observations)[:, 1] for m in classifiers]
            )
            if (lo - warm_frames) % 24000 == 0:
                print(
                    f"Room-array labeling {hi * 0.01:.0f}/{duration:.0f} s", flush=True
                )
        active = median_filter(confidence, size=(1, 9)) >= 0.50
        rows = timestamp_rows(active, duration)
        result = {
            "input_channels": file.channels,
            "enrollment_seconds": p0,
            "duration_s": duration,
            "inference_wall_s": time.perf_counter() - began,
            "segments": rows,
        }
        if args.eval_rttm:

            def held_out(ranges):
                return [
                    dict(s, start=max(p0, s["start"]) - p0, end=s["end"] - p0)
                    for s in ranges
                    if s["end"] > p0
                ]

            result["evaluation"] = score(
                held_out(reference(args.eval_rttm)), held_out(rows), duration - p0
            )
        (args.out / "diarization.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        np.save(args.out / "confidence.npy", confidence)
        print({k: v for k, v in result.items() if k != "segments"}, flush=True)


if __name__ == "__main__":
    main()
