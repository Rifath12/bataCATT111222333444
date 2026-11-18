"""Intersect speaker-timing predictions with an independent speech detector."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from silero_vad import get_speech_timestamps, load_silero_vad
import soundfile as sf
import torch

from diarize import masks, reference, score


def gate(signal, sampling_rate):
    peak = float(np.max(np.abs(signal)))
    gain = min(8.0, 0.98 / max(peak, 1e-6))
    model = load_silero_vad(onnx=True)
    return get_speech_timestamps(
        torch.from_numpy(signal * gain),
        model,
        sampling_rate=sampling_rate,
        min_speech_duration_ms=100,
        min_silence_duration_ms=70,
        speech_pad_ms=20,
        threshold=0.5,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio", type=Path)
    p.add_argument("--diarization", type=Path, required=True)
    p.add_argument("--rttm", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    y, rate = sf.read(args.audio, dtype="float32")
    if y.ndim != 1 or rate != 16000:
        p.error("Expected a 16 kHz mono WAV")
    started = time.perf_counter()
    speech = gate(y, rate)
    diarization = json.loads(args.diarization.read_text(encoding="utf-8"))
    duration = len(y) / rate
    labels, before = masks(diarization["segments"], duration)
    gate_segments = [
        dict(start=x["start"] / rate, end=x["end"] / rate, speaker="speech")
        for x in speech
    ]
    _, active = masks(gate_segments, duration)
    updated = []
    for label, line in zip(labels, before & active[0]):
        changes = np.flatnonzero(np.diff(np.pad(line.astype(np.int8), (1, 1))))
        updated.extend(
            dict(
                start=float(a * 0.01), end=min(duration, float(b * 0.01)), speaker=label
            )
            for a, b in zip(changes[::2], changes[1::2])
        )
    gated = dict(
        diarization,
        segments=sorted(updated, key=lambda a: (a["start"], a["speaker"])),
        vad_wall_s=time.perf_counter() - started,
    )
    if args.rttm:
        gated["evaluation"] = score(reference(args.rttm), updated, duration)
    else:
        gated.pop("evaluation", None)
    args.out.write_text(json.dumps(gated, indent=2) + "\n", encoding="utf-8")
    print({k: v for k, v in gated.items() if k != "segments"}, flush=True)


if __name__ == "__main__":
    main()
