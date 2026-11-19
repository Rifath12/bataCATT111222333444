"""Apply independent confirmation to a previously published two-pass decoder."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from threadpoolctl import threadpool_limits

from calibrate_nemotron import to_segments
from multiview_score import (
    counts,
    decode,
    fit_model,
    load_clip,
    load_manifest,
    measures,
    predict,
)
from verify_count import design, posterior


def confirm(clip, model, verifier, feature_dir):
    prior = decode(clip, predict(model, clip, True), 0.125)
    p3 = posterior(verifier["parameters"], design(clip, feature_dir, True))
    veto = (prior.sum(axis=1) == 3) & (p3 < verifier["threshold"])
    refined = prior.copy()
    indices = np.flatnonzero(veto)
    refined[indices, np.argmin(clip.fused[indices], axis=1)] = False
    return prior, refined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["dev-manifest", "dev-probs", "profile", "frozen", "dz-features"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--test", nargs=3, action="append", default=[])
    parser.add_argument(
        "--predict", nargs=3, metavar=("AUDIO", "NEMOTRON_CACHE", "TIMING")
    )
    args = parser.parse_args()
    if not args.test and not args.predict:
        parser.error("Supply --test or --predict")
    if args.test and not args.output:
        parser.error("Supply --output for benchmark scoring")
    profile = json.loads(args.profile.read_text())
    training = load_manifest(args.dev_manifest, args.dev_probs, profile)
    model = fit_model(training, True)
    verifier = json.loads(args.frozen.read_text())["independent_verifier"]
    output = {}
    for name, manifest, probdir in args.test:
        clips = load_manifest(Path(manifest), Path(probdir), profile)
        rows = []
        for clip in clips:
            prior, refined = confirm(clip, model, verifier, args.dz_features)
            rows.append((counts(clip, prior), counts(clip, refined)))
        totals = np.sum(rows, axis=0)
        output[name] = {
            "two_pass": measures(totals[0]),
            "two_pass_confirmed": measures(totals[1]),
        }
    if args.output:
        args.output.write_text(json.dumps(output, indent=2) + "\n")
    if args.predict:
        wave, cache, timing = map(Path, args.predict)
        with np.load(cache) as archive:
            duration = float(archive["duration_s"])
            if abs(duration - sf.info(wave).duration) > 1 / 16000:
                raise ValueError("Source and posterior durations disagree")
            clip = load_clip(
                wave, archive, wave.stem, int(np.ceil(duration / 0.01)), None, profile
            )
        _, refined = confirm(clip, model, verifier, args.dz_features)
        segments = to_segments(refined, duration)
        for s in segments:
            s["speaker"] = "ABC"[int(s["speaker"])]
        timing.write_text(
            json.dumps(
                {
                    "duration_s": duration,
                    "experimental_decoder": "independent_confirmation",
                    "segments": segments,
                },
                indent=2,
            )
            + "\n"
        )
    print(output)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
