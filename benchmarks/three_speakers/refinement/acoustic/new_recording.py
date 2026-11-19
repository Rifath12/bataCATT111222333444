"""Run a development-fitted, file-adapted three-speaker verifier on additional recordings.

The analysis uses only mono audio, native diarization and a second shifted mono pass.
References, when supplied, are opened after emitting A/B/C timing for that recording.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from threadpoolctl import threadpool_limits

from calibrate_nemotron import to_segments
from diarize import masks, reference
from improve_overlap import Decoder
from frozen_dictionary import load_frozen, save_frozen
from long_views import meeting
from multiview_score import Clip, align, counts, fit_length, measures
from verify_long import DEV, evidence, make_frames, optimize


def inputs(path, profile):
    audio = path / "mono.wav"
    rate = sf.info(audio)
    if rate.samplerate != 16000 or rate.channels != 1:
        raise ValueError(f"Expected a mono 16 kHz file: {audio}")
    size = int(np.ceil(rate.duration / 0.01))
    with np.load(path / "probabilities.npz") as base, np.load(path / "view.npz") as alt:
        if not np.isclose(float(base["step"]), 0.01) or not np.isclose(
            float(alt["step"]), 0.01
        ):
            raise ValueError("Expected 10 ms model scores")
        first = fit_length(base["probabilities"], size)
        second = fit_length(alt["probabilities"][928:], size)
    activity = np.zeros_like(first, dtype=bool)
    t = (np.arange(size) + 0.5) * 0.01
    for seg in reference(path / "native.rttm"):
        lo, hi = np.searchsorted(t, [seg["start"], seg["end"]])
        activity[lo:hi, int(seg["speaker"].rsplit("_", 1)[1]) - 1] = True
    ids, first, second = align(first, second, activity.sum(0))
    clip = Clip(
        path.name,
        path.name,
        None,
        activity[:, ids],
        Decoder(first).decode(profile),
        None,
        None,
        None,
        0.0,
        0.0,
    )
    return clip, first, second, rate.duration


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path)
    p.add_argument("--views", type=Path)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--frozen", type=Path, help="Use previously fitted numeric trees")
    mode.add_argument(
        "--save-frozen", type=Path, help="Save the development fit as portable JSON"
    )
    p.add_argument("--input-dir", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--score", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.frozen is None and (a.work is None or a.views is None):
        p.error("Supply --frozen or both --work and --views")
    scores = {"sessions": {}, "pooled": {}}
    pooled = {}
    with threadpool_limits(2):
        if a.frozen:
            model, threshold, profile = load_frozen(a.frozen)
        else:
            profile = json.loads(
                (a.work / "independent_3speaker/profile_balanced.json").read_text()
            )
            dev = meeting(a.work, a.views, DEV, profile)
            model, threshold, _, _ = optimize(
                *dev, *evidence(*dev, a.work / "session/mono.wav")
            )
            if a.save_frozen:
                save_frozen(a.save_frozen, model, threshold, profile)
        scores["threshold"] = threshold
        for path in a.input_dir:
            clip, first, second, duration = inputs(path, profile)
            proposed, x = evidence(clip, first, second, path / "mono.wav")
            estimate = (
                make_frames(
                    clip,
                    first,
                    second,
                    proposed,
                    model.predict_proba(x)[:, 1],
                    threshold,
                )
                if model
                else clip.previous
            )
            segments = to_segments(estimate, duration)
            for item in segments:
                item["speaker"] = "ABC"[int(item["speaker"])]
            timing = {
                "duration_s": duration,
                "segments": segments,
                "method": "dictionary_verified",
            }
            (a.out / f"{path.name}_timing.json").write_text(
                json.dumps(timing, indent=2) + "\n"
            )
            if a.score:
                actual = masks(reference(path / "reference.rttm"), duration)[1].T
                if actual.shape[1] != 3:
                    raise ValueError(
                        "The evaluation requires three annotated identities."
                    )
                for name, sys in [
                    ("native", clip.native),
                    ("previous", clip.previous),
                    ("verified", estimate),
                ]:
                    raw = counts(replace(clip, actual=actual), sys)
                    pooled[name] = pooled.get(name, 0) + raw
                    scores["sessions"].setdefault(path.name, {})[name] = measures(raw)
        if a.score:
            scores["pooled"] = {n: measures(v) for n, v in pooled.items()}
            (a.out / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")


if __name__ == "__main__":
    main()
