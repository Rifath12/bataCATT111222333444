"""Evaluate a three-voice probability decoder on meetings held out from calibration."""

import argparse
import json
from itertools import product
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter
from scipy.optimize import linear_sum_assignment
import soundfile as sf

from calibrate_nemotron import to_segments
from diarize import masks, reference, score
from nemotron_benchmark import select_speakers


def get_input(audio, probabilities, predicted_rttm):
    duration = sf.info(audio).duration
    with np.load(probabilities) as data:
        if not np.isclose(data["step"], 0.01):
            raise ValueError("Expected a 10 ms probability timeline")
        raw = data["probabilities"]
        wall = float(data["wall_s"])
    segments, emitted = select_speakers(reference(predicted_rttm), duration)
    chosen = sorted(sorted(emitted, key=lambda key: (-emitted[key], key))[:3])
    if len(chosen) != 3:
        raise ValueError("The native decoder must expose three speaker identities")
    indices = [int(label.rsplit("_", 1)[1]) - 1 for label in chosen]
    n_frames = int(np.ceil(duration / 0.01))
    values = raw[:n_frames, indices]
    if len(values) < n_frames:
        values = np.pad(values, ((0, n_frames - len(values)), (0, 0)))
    return duration, values, segments, wall


class Decoder:
    def __init__(self, raw):
        self.probabilities = median_filter(raw, size=(5, 1))
        self.ranks = np.argsort(np.argsort(-self.probabilities, axis=1), axis=1)
        self.sorted = -np.sort(-self.probabilities, axis=1)
        self.third = self.ranks == 2

    def core(self, thresholds, expansion):
        active = self.probabilities >= np.asarray(thresholds)[self.ranks]
        return (
            maximum_filter1d(active, size=expansion, axis=0)
            if expansion > 1
            else active
        )

    def nearby_third(self, anchor, radius):
        confirmed = self.third & (self.probabilities >= anchor)
        return maximum_filter1d(confirmed, size=2 * radius + 1, axis=0)

    def promote(self, active, nearby, threshold, second):
        support = (self.sorted[:, 0] >= 0.55) & (self.sorted[:, 1] >= second)
        supported = nearby & self.third & (self.probabilities >= threshold)
        return active | (supported & support[:, None])

    def trim_triples(self, active, third, second):
        all_three = active.all(axis=1)
        unreliable = (self.sorted[:, 2] < third) | (self.sorted[:, 1] < second)
        return active & ~(self.third & (all_three & unreliable)[:, None])

    def decode(self, settings):
        core = self.core(settings["thresholds"], settings["expansion_frames"])
        if "promotion" not in settings:
            return core
        chance = settings["promotion"]
        before_or_after = self.nearby_third(chance["anchor"], chance["radius_frames"])
        active = self.promote(
            core,
            before_or_after,
            chance["minimum_probability"],
            chance["second_probability"],
        )
        return (
            self.trim_triples(active, *settings["trim_triples"])
            if "trim_triples" in settings
            else active
        )


def match(reference_mask, prediction):
    profile = reference_mask.T.astype(np.float32) @ prediction.astype(np.float32)
    ref, est = linear_sum_assignment(-profile)
    return reference_mask[:, ref[np.argsort(est)]]


def diagnostics(actual, prediction):
    reference_count, output_count = actual.sum(1), prediction.sum(1)
    correct = (actual & prediction).sum(1)
    triple = reference_count == 3
    output_triple = output_count == 3
    both_triple = (triple & output_triple).sum()
    multiple = reference_count >= 2
    reference_seconds = reference_count.sum()
    false_s = (output_count - correct).sum()
    miss_s = (reference_count - correct).sum()
    return dict(
        DER=float(
            (
                false_s
                + miss_s
                - (np.minimum(reference_count, output_count) - correct).sum()
            )
            / reference_seconds
        ),
        overall_precision=float(correct.sum() / max(1, output_count.sum())),
        overlap_recall=float(
            correct[multiple].sum() / max(1, reference_count[multiple].sum())
        ),
        triple_speaker_recall=float(
            correct[triple].sum() / max(1, reference_count[triple].sum())
        ),
        all_three_precision=float(both_triple / max(1, output_triple.sum())),
        all_three_recall=float(both_triple / max(1, triple.sum())),
        all_three_F1=float(
            2 * both_triple / max(1, output_triple.sum() + triple.sum())
        ),
        predicted_three_seconds=float(output_triple.sum()) * 0.01,
        reference_three_seconds=float(triple.sum()) * 0.01,
    )


def fit(decoder, actual, native):
    actual = match(actual, native)
    native_stats = diagnostics(actual, native)
    # Start with the stronger two-voice profile evaluated before this experiment.
    base = {"thresholds": [0.80, 0.38, 0.35], "expansion_frames": 15}
    core = decoder.decode(base)
    best = (-float("inf"), None, None)
    for anchor, radius in product((0.22, 0.36, 0.55), (8, 16, 28)):
        nearby = decoder.nearby_third(anchor, radius)
        for weak, second in product((0.008, 0.016, 0.03, 0.06, 0.12), (0.33, 0.48)):
            active = decoder.promote(core, nearby, weak, second)
            result = diagnostics(actual, active)
            if (
                result["overall_precision"] < native_stats["overall_precision"]
                or result["DER"] > native_stats["DER"]
            ):
                continue
            # Credit true three-way recovery and penalize false three-voice declarations.
            value = (
                result["triple_speaker_recall"]
                + 0.50 * result["all_three_F1"]
                - 1.25 * result["DER"]
                + 0.25 * result["overlap_recall"]
            )
            if value > best[0]:
                settings = dict(
                    base,
                    promotion={
                        "anchor": anchor,
                        "radius_frames": radius,
                        "minimum_probability": weak,
                        "second_probability": second,
                    },
                )
                best = (value, settings, result)
    if best[1] is None:
        return base, diagnostics(actual, core)
    # Recover confidence around clear triple events while rejecting low-confidence
    # three-speaker assignments. Both steps are selected on the development audio.
    untrimmed = decoder.decode(best[1])
    balanced = (-float("inf"), best[1], best[2])
    for third, second in product(
        (0.006, 0.02, 0.035, 0.06, 0.10, 0.20), (0.4, 0.5, 0.6, 0.7, 0.8)
    ):
        result = diagnostics(actual, decoder.trim_triples(untrimmed, third, second))
        if result["all_three_F1"] < native_stats["all_three_F1"]:
            continue
        value = (
            result["triple_speaker_recall"]
            + 0.5 * result["all_three_F1"]
            - 1.25 * result["DER"]
        )
        if value > balanced[0]:
            balanced = (value, dict(best[1], trim_triples=[third, second]), result)
    return balanced[1], balanced[2]


def run(args):
    duration, probabilities, input_baseline, model_wall = get_input(
        args.audio, args.probabilities, args.native_rttm
    )
    ref = reference(args.reference) if args.reference else None
    if ref is not None:
        _, actual = masks(ref, duration)
        _, baseline = masks(input_baseline, duration)
    start = time.perf_counter()
    decoder = Decoder(probabilities)
    preprocessing_s = time.perf_counter() - start
    if args.fit:
        settings, development = fit(decoder, actual.T, baseline.T)
        args.profile.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        print("Development profile", settings, development, flush=True)
    else:
        settings = json.loads(args.profile.read_text(encoding="utf-8"))
    start = time.perf_counter()
    candidate = decoder.decode(settings)
    wall = preprocessing_s + time.perf_counter() - start
    rows = to_segments(candidate, duration)
    complete = {
        "duration_s": duration,
        "model_wall_s": model_wall,
        "decoder_wall_s": wall,
        "parameters_fit_on_this_recording": args.fit,
        "profile": settings,
        "segments": rows,
    }
    if ref is not None:
        for name, predicted, segments in (
            ("native", baseline.T, input_baseline),
            ("candidate", candidate, rows),
        ):
            complete[name] = {
                "diagnostics": diagnostics(match(actual.T, predicted), predicted),
                "evaluation": score(ref, segments, duration),
            }
    args.out.write_text(json.dumps(complete, indent=2) + "\n", encoding="utf-8")
    for name in ("native", "candidate"):
        if name in complete:
            print(name, complete[name]["diagnostics"], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio", type=Path)
    p.add_argument("--probabilities", type=Path, required=True)
    p.add_argument("--native-rttm", type=Path, required=True)
    p.add_argument(
        "--reference", type=Path, help="Reference RTTM for optional evaluation"
    )
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--fit", action="store_true", help="Calibrate on a development recording only"
    )
    args = p.parse_args()
    if args.fit and args.reference is None:
        p.error("--fit requires --reference from a development recording")
    run(args)


if __name__ == "__main__":
    main()
