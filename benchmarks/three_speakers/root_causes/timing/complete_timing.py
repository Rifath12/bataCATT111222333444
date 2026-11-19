"""Export anonymous A/B/C timing using an explicitly selected boundary profile."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from threadpoolctl import threadpool_limits

from calibrate_nemotron import to_segments
from diarize import masks, reference
from frozen_dictionary import load_frozen
from improve_overlap import Decoder
from multiview_score import Clip, align, counts, fit_length, measures
from new_recording import inputs
from verify_long import evidence, make_frames


def cached_inputs(audio, archive, base):
    info = sf.info(audio)
    if info.channels != 1 or info.samplerate != 16000 or info.frames < 1:
        raise ValueError(f"Expected nonempty 16 kHz mono audio: {audio}")
    length = (info.frames + 159) // 160
    with np.load(archive, allow_pickle=False) as data:
        if abs(float(data["duration_s"]) - info.duration) > 0.01:
            raise ValueError(f"Cache and audio durations disagree: {archive}")
        initial = fit_length(data["base"], length)
        alternate = fit_length(data["alternate"][int(data["offset"]) :], length)
        activity = np.zeros_like(initial, dtype=bool)
        centres = (np.arange(length) + 0.5) * 0.01
        for start, end, slot in data["native"]:
            left, right = np.searchsorted(centres, [start, end])
            activity[left:right, int(slot)] = True
    ids, first, second = align(initial, alternate, activity.sum(0))
    clip = Clip(
        audio.stem,
        audio.stem,
        None,
        activity[:, ids],
        Decoder(first).decode(base),
        None,
        None,
        None,
        0.0,
        0.0,
    )
    return clip, first, second, info.duration


def correct_turns(clip, first, second, audio, verifier, thresholds, profile):
    if len(first) >= 16:
        # The frozen verifier needs at least 16 samples for its spectral dictionary.
        proposed, features = evidence(clip, first, second, audio)
        confidence = verifier.predict_proba(features)[:, 1]
        verified = make_frames(clip, first, second, proposed, confidence, thresholds)
    else:
        verified = clip.previous
    narrowed = Decoder(first).decode(profile)
    # Retain the verifier's decision throughout every proposed three-voice event.
    # Apply the boundary profile elsewhere, preserving who spoke in each slot.
    verify_frames = clip.previous.all(1) | verified.all(1) | narrowed.all(1)
    adjusted = np.where(verify_frames[:, None], verified, narrowed)
    return verified, adjusted


def timing_metrics(raw):
    metrics = measures(raw)
    # The count vector contains matched speaker-time after the global permutation.
    ref, assigned, correct, error = map(int, raw[:4])
    miss = error - (assigned - correct)
    false = error - (ref - correct)
    confusion = error - miss - false
    for name, frames in (
        ("missed", miss),
        ("false_alarm", false),
        ("confused", confusion),
    ):
        metrics[f"{name}_speaker_s"] = frames / 100
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--input-dir",
        nargs="+",
        type=Path,
        help="One or more native and shifted input folders",
    )
    g.add_argument(
        "--cache",
        action="append",
        nargs=2,
        type=Path,
        metavar=("WAV", "NPZ"),
        help="Mono waveform and matching batch_probabilities cache. Repeat for a batch.",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--score",
        action="store_true",
        help="After export, score with directory/reference.rttm or WAV_STEM.rttm.",
    )
    a = p.parse_args()
    trained, thresholds, base = load_frozen(a.model)
    profile = json.loads(a.profile.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    pooled, per_meeting = {}, {}
    if a.input_dir:
        recordings = [
            (folder.name, folder / "mono.wav", folder / "reference.rttm", folder, None)
            for folder in a.input_dir
        ]
    else:
        recordings = [
            (audio.stem, audio, audio.with_suffix(".rttm"), None, cache)
            for audio, cache in a.cache
        ]
    if len({r[0] for r in recordings}) != len(recordings):
        p.error("Recording names must be unique within a batch.")
    for name, audio, ref, folder, cache in recordings:
        clip, first, second, duration = (
            inputs(folder, base)
            if folder is not None
            else cached_inputs(audio, cache, base)
        )
        verified, adjusted = correct_turns(
            clip, first, second, audio, trained, thresholds, profile
        )
        rows = to_segments(adjusted, duration)
        for row in rows:
            row["speaker"] = "ABC"[int(row["speaker"])]
        output = {
            "method": "dictionary_with_boundary_profile",
            "duration_s": duration,
            "segments": rows,
        }
        (a.out / (name + "_timing.json")).write_text(
            json.dumps(output, indent=2) + "\n"
        )
        if a.score:
            truth = masks(reference(ref), duration)[1].T
            if truth.shape[1] != 3:
                raise ValueError(f"Scoring needs three reference identities: {ref}")
            scored = replace(clip, actual=truth)
            per_meeting[name] = {}
            for label, mask in (
                ("baseline", clip.previous),
                ("dictionary", verified),
                ("combined", adjusted),
            ):
                raw = counts(scored, mask)
                pooled[label] = pooled.get(label, 0) + raw
                per_meeting[name][label] = timing_metrics(raw)
    if a.score:
        (a.out / "scores.json").write_text(
            json.dumps(
                {
                    "pooled": {k: timing_metrics(v) for k, v in pooled.items()},
                    "sessions": per_meeting,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    with threadpool_limits(2):
        main()
