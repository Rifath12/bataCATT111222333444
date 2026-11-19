"""Score three ordered speaker WAVs against a known-source single-microphone conversation."""

from argparse import ArgumentParser
from itertools import permutations
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from waveform_acceptance import signal_metrics, gain_coverage, threshold_pass

RATE = 16_000
SPEAKERS = "ABC"


def world_order(reference, tracks, active):
    """Find one reference-only diagnostic order over the entire recording."""
    scores = np.zeros((3, 3))
    for i in range(3):
        wanted = reference[i, active[i]]
        wanted = wanted - wanted.mean()
        for j in range(3):
            proposed = tracks[j, active[i]]
            proposed = proposed - proposed.mean()
            scores[i, j] = (wanted @ proposed) ** 2 / max(
                1e-20, (wanted @ wanted) * (proposed @ proposed)
            )
    return max(
        permutations(range(3)), key=lambda p: sum(scores[i, j] for i, j in enumerate(p))
    )


def evaluate(manifest, reference, tracks, active):
    result = {}
    for count in (1, 2, 3):
        result[count] = {
            "cases": 0,
            "joint96": 0,
            "joint98": 0,
            "joint96_with_inactive_quiet": 0,
            "events": [],
        }
    for event in manifest["events"]:
        start, stop = event["start"], event["end"]
        members = [SPEAKERS.index(s) for s in event["members"]]
        n = len(members)
        r = reference[:, start:stop]
        y = tracks[:, start:stop]
        a = active[:, start:stop]
        common = a[members].all(axis=0)
        if np.count_nonzero(common) < 480:
            raise ValueError(f"Too little simultaneous speech for {event['id']}")
        values = []
        for i in members:
            v = signal_metrics(r[:, common], y[i, common], i)
            v.update(gain_coverage(r[i], y[i], common, RATE // 10, 480))
            values.append(v)
        flags = [threshold_pass(values, 0.96), threshold_pass(values, 0.98)]
        missing = [i for i in range(3) if i not in members]
        unwanted_db = None
        if missing:
            reference_power = np.mean(r[members][:, common] ** 2)
            worst_unwanted = np.max(np.mean(y[missing][:, common] ** 2, axis=1))
            unwanted_db = float(
                10 * np.log10(max(1e-20, worst_unwanted) / max(1e-20, reference_power))
            )
        chunk = result[n]
        chunk["joint96_with_inactive_quiet"] += int(
            flags[0] and (unwanted_db is None or unwanted_db <= -20)
        )
        chunk["cases"] += 1
        for flag, name in zip(flags, ("joint96", "joint98"), strict=True):
            chunk[name] += int(flag)
        chunk["events"].append(
            {
                "id": event["id"],
                "members": event["members"],
                "common_active_seconds": common.sum() / RATE,
                "all_at_96": flags[0],
                "all_at_98": flags[1],
                "loudest_absent_speaker_output_db_relative_to_active_reference": unwanted_db,
                "alignments": [v["target_aligned_fraction"] for v in values],
                "retention": [v["retained_active_fraction"] for v in values],
            }
        )
    # An input-only silent portion is known from scoreable PCM, not just a VAD guess.
    exact_zero = np.all(reference == 0, axis=0)
    floor = 1e-20
    summary = {
        "events_by_speaker_count": result,
        "all_target_zero_seconds": float(exact_zero.sum() / RATE),
        "peak_on_all_zero_input": float(np.max(np.abs(tracks[:, exact_zero])))
        if exact_zero.any()
        else None,
    }
    summary["spill_on_other_speakers_db"] = {}
    for i, who in enumerate(SPEAKERS):
        other_only = (reference[i] == 0) & active.any(axis=0)
        inactive_power = np.mean(tracks[i, other_only] ** 2)
        target_power = np.mean(reference[i, active[i]] ** 2)
        summary["spill_on_other_speakers_db"][who] = float(
            10 * np.log10(max(floor, inactive_power) / max(floor, target_power))
        )
    return summary


def load_audio(paths, expected):
    saved = []
    rates = []
    for f in paths:
        with sf.SoundFile(f) as stream:
            rate = stream.samplerate
            if (
                rate not in (8_000, RATE)
                or stream.channels != 1
                or stream.frames * RATE != expected * rate
            ):
                raise ValueError(
                    f"Unexpected WAV duration, sample clock or channel count in {f}"
                )
            rates.append(rate)
            v = stream.read(dtype="float64")
            if not np.isfinite(v).all():
                raise ValueError(f"Nonfinite WAV samples in {f}")
            saved.append(v if rate == RATE else resample_poly(v, 2, 1))
    if len(set(rates)) != 1:
        raise ValueError("The three output microphone sample rates differ")
    return np.stack(saved), rates[0]


def score(condition, paths):
    manifest = json.loads((condition.parent / "mono_manifest.json").read_text())
    with np.load(condition / "meeting.npz", allow_pickle=False) as f:
        ref = f["references"].astype(np.float64)
        activity = f["activity"]
    tracks, stored_rate = load_audio(paths, ref.shape[1])
    order = list(world_order(ref, tracks, activity))
    out = {
        "stored_output_rate": stored_rate,
        "evaluation_rate": RATE,
        "room": manifest["room"],
        "condition": condition.name,
        "seconds": ref.shape[1] / RATE,
        "fixed_ABC": evaluate(manifest, ref, tracks, activity),
        "global_diagnostic_order_of_input_slots": order,
    }
    out["diagnostic_best_global_order"] = (
        out["fixed_ABC"]
        if order == [0, 1, 2]
        else evaluate(manifest, ref, tracks[order], activity)
    )
    return out


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("condition", type=Path)
    p.add_argument("wavs", nargs=3, type=Path)
    p.add_argument("--out", type=Path, required=True)
    options = p.parse_args()
    report = score(options.condition, options.wavs)
    options.out.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        options.condition.parent.name,
        options.condition.name,
        {
            n: (s["joint96"], s["joint98"], s["cases"])
            for n, s in report["fixed_ABC"]["events_by_speaker_count"].items()
        },
    )
