"""Evaluate the second diarization view on the original long-meeting suite."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from diarize import masks, reference
from improve_overlap import Decoder
from multiview_score import Clip, align, counts, fit_length, measures


def meeting(work, variants, session, profile):
    if session == "R8008_M8013":
        pcm = work / "session/mono.wav"
        original = work / "nemotron_probabilities.npz"
        native = work / "nemotron_raw.rttm"
        ref = work / f"{session}.rttm"
    else:
        part = work / "independent_3speaker"
        pcm, original, native = (
            part / session / name
            for name in ("mono.wav", "probabilities.npz", "native.rttm")
        )
        ref = part / f"references/Test_Ali_far/{session}.rttm"
    seconds = sf.info(pcm).duration
    actual = masks(reference(ref), seconds)[1].T
    with np.load(original) as a, np.load(variants / session / "view.npz") as b:
        first = fit_length(a["probabilities"], len(actual))
        second = fit_length(b["probabilities"][928:], len(actual))
        a_wall, b_wall = float(a["wall_s"]), float(b["wall_s"])
    activity = np.zeros_like(first, dtype=bool)
    centers = (np.arange(len(actual)) + 0.5) * 0.01
    for segment in reference(native):
        lo, hi = np.searchsorted(centers, [segment["start"], segment["end"]])
        activity[lo:hi, int(segment["speaker"].rsplit("_", 1)[1]) - 1] = True
    chosen, anchor, secondary = align(first, second, activity.sum(axis=0))
    previous = Decoder(anchor).decode(profile)
    clip = Clip(
        session,
        session,
        actual,
        activity[:, chosen],
        previous,
        None,
        None,
        None,
        a_wall,
        b_wall,
    )
    return clip, anchor, secondary


def frames(meeting_data, weight, profile):
    clip, first, second = meeting_data
    # Shrinking toward the unchanged raw input keeps stable labels in long clips.
    combined = first * (1.0 - weight) + second * weight
    return counts(clip, Decoder(combined).decode(profile))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    profile = json.loads(
        (args.previous / "independent_3speaker/profile_balanced.json").read_text()
    )
    dev = meeting(args.previous, args.views, "R8008_M8013", profile)
    settings = []
    for alpha in (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 1.0):
        result = measures(frames(dev, alpha, profile))
        objective = (
            result["exact_three_F1"]
            + 0.25 * result["three_way_speaker_recall"]
            - 0.5 * result["DER"]
        )
        settings.append((objective, alpha, result))
    _, weight, developed = max(settings, key=lambda row: row[0])
    result = {"development": developed, "alternate_weight": weight, "holdout": {}}
    totals = {}
    for session in ("R8008_M8014", "R8008_M8015"):
        current = meeting(args.previous, args.views, session, profile)
        clip = current[0]
        lines = {
            "native": counts(clip, clip.native),
            "previous_calibration": counts(clip, clip.previous),
            "two_view": frames(current, weight, profile),
        }
        result["holdout"][session] = {
            name: measures(raw) for name, raw in lines.items()
        }
        for name, raw in lines.items():
            totals[name] = totals.get(name, 0) + raw
    result["pooled"] = {name: measures(raw) for name, raw in totals.items()}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(weight, result["pooled"], flush=True)


if __name__ == "__main__":
    main()
