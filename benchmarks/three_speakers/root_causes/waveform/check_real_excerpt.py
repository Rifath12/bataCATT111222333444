"""Re-render only the three-voice windows of a previous real meeting with the frozen filter."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import soundfile as sf
import torch

from acoustic_repair import dereverberate
from diarize import masks
from process_meeting import VoiceEmbeddings, normalized_dot, resample
from render import HOP, RATE, enrollment, merge_overlap, scale_sources
from test_wave import low_memory_separator


def infer(wave, mix, separator, voice, anchors, speakers, length):
    tracks = scale_sources(separator(wave), mix)
    embeds = [voice(track) for track in tracks]
    scores = np.asarray(
        [
            [
                normalized_dot(anchors[i], e)
                if anchors[i] is not None and e is not None
                else 0.0
                for e in embeds
            ]
            for i in speakers
        ]
    )
    first, second = linear_sum_assignment(-scores)
    ordered = np.zeros((len(speakers), length), np.float32)
    for i, k in zip(first, second, strict=True):
        result = resample(tracks[k], 8000, RATE)
        ordered[i, : min(len(result), length)] = result[:length]
    return ordered


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["input", "timing", "previous", "out", "weights", "voice"]:
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    y, sr = sf.read(a.input, dtype="float32")
    if sr != RATE or y.ndim != 1:
        raise ValueError("Expected mono meeting audio at 16 kHz")
    names, activity = masks(json.loads(a.timing.read_text())["segments"], len(y) / RATE)
    originals = []
    for name in names:
        waveform, rate = sf.read(a.previous / f"speaker_{name}.wav", dtype="float32")
        if rate != RATE or len(waveform) != len(y):
            raise ValueError("Inconsistent previous output dimensions")
        originals.append(waveform)
    outputs = np.stack(originals)
    count = activity.sum(axis=0)
    cues = enrollment(y, activity, count)
    voice = VoiceEmbeddings(a.voice)
    anchors = [voice(resample(c, RATE, 8000)) for c in cues]
    separator = low_memory_separator(a.weights)
    changes = []
    for start, end in merge_overlap(count):
        for lo in range(start, end, 300):
            hi = min(lo + 300, end)
            if (count[lo:hi] == 3).sum() < 5:
                continue
            begin, finish = lo * HOP, min(len(y), hi * HOP)
            left, right = max(0, begin - 4800), min(len(y), finish + 4800)
            fragment = y[left:right]
            speakers = np.flatnonzero(activity[:, lo:hi].any(axis=1))
            mix = resample(fragment, RATE, 8000)
            control = infer(
                mix, mix, separator, voice, anchors, speakers, len(fragment)
            )
            filtered = infer(
                dereverberate(mix),
                mix,
                separator,
                voice,
                anchors,
                speakers,
                len(fragment),
            )
            peak_difference = 0.0
            for i, speaker in enumerate(speakers):
                active = np.repeat(activity[speaker, lo:hi], HOP)[: finish - begin]
                core = slice(begin - left, finish - left)
                original = outputs[speaker, begin:finish]
                peak_difference = max(
                    peak_difference,
                    float(np.max(np.abs(control[i, core] * active - original))),
                )
                original[:] = filtered[i, core] * active
            print(
                f"{lo / 100:.2f}–{hi / 100:.2f} baseline maximum absolute difference {peak_difference:.7f}",
                flush=True,
            )
            if peak_difference > 2e-4:
                raise RuntimeError(
                    "The original renderer did not reproduce within PCM rounding tolerance"
                )
            changes.append(
                {
                    "start_s": lo / 100,
                    "end_s": hi / 100,
                    "baseline_max_abs_difference": peak_difference,
                }
            )
    if not changes:
        raise RuntimeError("No three-speaker windows")
    if np.abs(outputs).max() >= 0.98:
        raise RuntimeError("Global gain changes would invalidate the cached comparison")
    a.out.mkdir(parents=True, exist_ok=True)
    for name, wave in zip(names, outputs, strict=True):
        sf.write(a.out / f"speaker_{name}.wav", wave, RATE, subtype="PCM_16")
    (a.out / "window_check.json").write_text(json.dumps(changes, indent=2) + "\n")


if __name__ == "__main__":
    main()
