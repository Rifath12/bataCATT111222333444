"""Measure speaker-specific energy and crosstalk during reference-solo meeting frames."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.optimize import linear_sum_assignment

from diarize import masks, reference
from scipy.ndimage import minimum_filter1d


def levels(file):
    energy = []
    with sf.SoundFile(file) as f:
        for y in f.blocks(16000 * 8, dtype="float32", always_2d=True):
            if f.samplerate != 16000:
                raise ValueError("16 kHz input needed")
            y = y[:, 0]
            n = len(y) // 160
            energy.extend(np.mean(y[: n * 160].reshape(n, 160) ** 2, axis=1))
    return np.asarray(energy)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mono", type=Path, required=True)
    p.add_argument("--rttm", type=Path, required=True)
    p.add_argument("--timing", type=Path, required=True)
    p.add_argument("--beams", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    power = levels(a.mono)
    duration = sf.info(a.mono).duration
    ids, truth = masks(reference(a.rttm), duration)
    names, system = masks(json.loads(a.timing.read_text())["segments"], duration)
    rows, cols = linear_sum_assignment(-truth.astype("f") @ system.astype("f").T)
    data = {}
    for i, j in zip(rows, cols):
        name = names[j]
        beam = levels(a.beams / f"beam_{name}.flac")
        samples = truth[:, : len(power)]
        averages = {}
        for k, spk in enumerate(ids):
            use = minimum_filter1d(
                (samples[k] & (samples.sum(0) == 1)).astype("B"), 15
            ).astype(bool)
            averages[spk] = {
                "solo_s": float(use.sum()) * 0.01,
                "reference_mic_rms_dbfs": float(
                    10 * np.log10(power[use].mean() + 1e-18)
                ),
                "gain_db": float(
                    10
                    * np.log10((beam[use].sum() + 1e-12) / (power[use].sum() + 1e-12))
                ),
            }
        target = averages[ids[i]]["gain_db"]
        improvement = np.mean(
            [target - e["gain_db"] for k, e in averages.items() if k != ids[i]]
        )
        data[name] = {
            "matched_id": ids[i],
            "relative_rejection_of_other_solo_speakers_db": float(improvement),
            "per_true_speaker": averages,
        }
    a.out.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
