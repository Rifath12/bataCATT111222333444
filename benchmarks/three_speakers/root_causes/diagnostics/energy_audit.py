"""Measure irreversible zero samples in delivered tracks during close-mic-supported reference speech."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from oracle_matrix import occupancy
from recognize import textgrid


def envelopes(path):
    energy = []
    zeros = []
    with sf.SoundFile(path) as f:
        if (f.samplerate, f.channels) != (16000, 1):
            raise ValueError(f"Unsupported input {path}")
        for wave in f.blocks(160 * 1000, dtype="int16"):
            n = len(wave) // 160
            a = wave[: n * 160].reshape(n, 160)
            energy.extend(
                10 * np.log10(1e-9 + (a.astype("f") ** 2).mean(axis=1) / (32768.0**2))
            )
            zeros.extend(~(a != 0).any(axis=1))
    return np.asarray(energy), np.asarray(zeros)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for x in ("reference", "tracks", "headsets", "mixture", "deployed_score", "out"):
        p.add_argument("--" + x.replace("_", "-"), type=Path, required=True)
    a = p.parse_args()
    ref = textgrid(a.reference)
    mapping = {
        r["reference"]: r["system"]
        for r in json.loads(a.deployed_score.read_text())["speaker_scores"]
    }
    mix, _ = envelopes(a.mixture)
    labels = sorted(mapping)
    active = occupancy(ref, labels, len(mix)).astype(bool)
    report = {"resolution_s": 0.01, "speakers": {}, "totals": {}}
    for num, source in enumerate(labels):
        slot = mapping[source]
        close, _ = envelopes(next(a.headsets.glob(f"*_{source}.wav")))
        stem, zeros = envelopes(a.tracks / f"speaker_{slot}.wav")
        if len(close) != len(mix) or len(stem) != len(mix):
            raise ValueError("Unaligned lengths")
        solo = active[num] & (active.sum(axis=0) == 1)
        silence = ~active.any(axis=0)
        # Verify alignment of energy changes for solo speech, in each 10ms stream.
        c = np.diff(close, prepend=close[0])
        far = np.diff(mix, prepend=mix[0])
        cors = []
        for shift in range(-15, 16):
            u = solo[20:-20]
            cors.append(
                np.corrcoef(c[20:-20][u], far[20 + shift : len(far) - 20 + shift][u])[
                    0, 1
                ]
            )
        lag = (int(np.argmax(cors)) - 15) * 10
        # Shift the nearer recorder to the room clock without wrapping samples.
        frames = lag // 10
        if frames > 0:
            close = np.pad(close[:-frames], (frames, 0), constant_values=-90)
        if frames < 0:
            close = np.pad(close[-frames:], (0, -frames), constant_values=-90)
        # Require both independent close-talk and mixed microphone energy, well above room floor.
        min_close = max(-55.0, np.quantile(close[silence], 0.95) + 6)
        min_far = max(-60.0, np.quantile(mix[silence], 0.95) + 6)
        clear = (close > min_close) & (mix > min_far) & active[num]
        speaker_report = {
            "headset_speech_db_threshold": round(float(min_close), 2),
            "room_speech_db_threshold": round(float(min_far), 2),
            "solo_envelope_lag_ms": lag,
        }
        for typ, condition in [("solo", clear & solo), ("overlap", clear & ~solo)]:
            d0 = int(condition.sum())
            dz = int((condition & zeros).sum())
            speaker_report[typ] = {"frames": d0, "all_zero_frames": dz}
            pair = report["totals"].setdefault(typ, {"frames": 0, "all_zero_frames": 0})
            pair["frames"] += d0
            pair["all_zero_frames"] += dz
        report["speakers"][source] = speaker_report
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(report["totals"])


if __name__ == "__main__":
    main()
