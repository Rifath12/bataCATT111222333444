"""Measure three-speaker diarization with a streaming Sortformer model."""

import argparse
from pathlib import Path
import json
import sys
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import onnxruntime as ort
from scipy import fft
from scipy.ndimage import median_filter
import soundfile as sf

from diarize import reference, score


def audio_features(audio, mel_basis):
    emphasized = audio.copy()
    emphasized[1:] -= 0.97 * audio[:-1]
    windows = sliding_window_view(np.pad(emphasized, (256, 256)), 512)[::160]
    window = np.zeros(512, np.float32)
    window[56:456] = np.hanning(401)[:-1]
    out = np.empty((len(windows), 128), np.float32)
    for start in range(0, len(windows), 4096):
        end = start + 4096
        spectrum = fft.rfft(windows[start:end] * window)
        power = spectrum.real**2 + spectrum.imag**2
        out[start:end] = np.log(power @ mel_basis.T + 2**-24)
    return out


def predict(args, wave):
    sys.path.insert(0, str(args.adapter.resolve() / "src"))
    from sortformer import Sortformer, CHUNK_LEN, SUBSAMPLING

    # The adapter's default ORT session picks all CPU cores. Explicitly bound it.
    session_options = ort.SessionOptions

    def bounded_options():
        options = session_options()
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = 1
        return options

    ort.SessionOptions = bounded_options
    try:
        model = Sortformer(
            str(args.model), provider="CPUExecutionProvider", log_severity=3
        )
    finally:
        ort.SessionOptions = session_options
    features = audio_features(wave, model.mel_basis)
    stride = CHUNK_LEN * SUBSAMPLING
    total = (len(features) + stride - 1) // stride
    rows = []
    began = time.perf_counter()
    for n, start in enumerate(range(0, len(features), stride), start=1):
        feature = features[start : start + stride]
        size = len(feature)
        if size < stride:
            feature = np.pad(feature, ((0, stride - size), (0, 0)))
        result = model._streaming_update(feature[None], size)[0]
        rows.append(result[: (size + SUBSAMPLING - 1) // SUBSAMPLING])
        if n == 1 or n % 20 == 0 or n == total:
            print(
                f"Sortformer {n}/{total}, {time.perf_counter() - began:.1f} s",
                flush=True,
            )
    return np.concatenate(rows, axis=0)


def decode(probs, duration, threshold=0.5):
    active = median_filter(probs, size=(3, 1)) >= threshold
    # The session count is known. Retain the three persistent streaming IDs.
    identities = np.argsort(-active.sum(axis=0), kind="stable")[:3]
    segments = []
    for speaker, index in enumerate(sorted(identities)):
        bounds = np.flatnonzero(
            np.diff(np.pad(active[:, index].astype(np.int8), (1, 1)))
        )
        for a, b in zip(bounds[::2], bounds[1::2]):
            start, end = float(a * 0.08), min(duration, float(b * 0.08))
            if end > start:
                segments.append({"start": start, "end": end, "speaker": str(speaker)})
    return sorted(segments, key=lambda x: (x["start"], x["speaker"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio", type=Path)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--rttm", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    audio, rate = sf.read(args.audio, dtype="float32")
    if rate != 16000 or audio.ndim != 1:
        raise ValueError("Expected one mono 16 kHz file")
    t0 = time.perf_counter()
    probs = predict(args, audio)
    duration = len(audio) / rate
    segments = decode(probs, duration)
    result = {
        "duration_s": duration,
        "inference_wall_s": time.perf_counter() - t0,
        "known_session_speaker_count": 3,
        "segments": segments,
    }
    if args.rttm:
        result["evaluation"] = score(reference(args.rttm), segments, duration)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    np.save(args.output.with_suffix(".npy"), probs)
    print({k: v for k, v in result.items() if k != "segments"}, flush=True)


if __name__ == "__main__":
    main()
