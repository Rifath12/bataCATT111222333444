"""Train a dedicated exactly-three detector on speaker-disjoint source folds."""

from argparse import ArgumentParser
from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.fft import rfft
from scipy.ndimage import uniform_filter1d, maximum_filter1d, binary_opening
import soundfile as sf
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from complete_timing import cached_inputs, correct_turns
from frozen_dictionary import load_frozen
from multiview_score import counts, measures
from portable import encode, NumericTrees, load
import structured as core

STEP = core.HOP


def distinct_pitch_features(audio, n_frames):
    """Describe the multiplicity and stability of plausible fundamental frequencies."""
    signal, rate = sf.read(audio, dtype="float32")
    if signal.ndim != 1 or rate != 16000:
        raise ValueError(f"Expected a 16 kHz mono waveform: {audio}")
    window = np.hanning(1440).astype("f")
    nfft = 2048
    fundamentals = 65 * 2 ** (np.arange(94, dtype="f") / 32)
    bins = np.arange(1, 9, dtype="f") * fundamentals[:, None] * (nfft / rate)
    indices = bins.astype("intp")
    shifts = bins - indices
    holes = (bins + fundamentals[:, None] * (nfft / (2 * rate))).astype("intp")
    weight = np.arange(1, 9, dtype="f") ** -0.45
    output = np.empty((n_frames, 11), dtype="f")
    padded = np.pad(signal, (640, n_frames * 320 + 1440 - len(signal)))
    frames = np.lib.stride_tricks.sliding_window_view(padded, len(window))[::320]
    for left in range(0, n_frames, 256):
        stop = min(left + 256, n_frames)
        power = np.abs(rfft(frames[left:stop] * window, n=nfft))
        floor = uniform_filter1d(power, size=31, axis=1)
        residual = np.log1p(power / np.maximum(floor, 1e-4))
        harmonic = (1 - shifts) * residual[:, indices] + shifts * residual[
            :, indices + 1
        ]
        scores = (harmonic - 0.5 * residual[:, holes]) @ weight
        row = np.arange(stop - left)
        strongest_scores, strongest_pitches = [], []
        available = scores.copy()
        for _ in range(3):
            peak = available.argmax(axis=1)
            strongest_scores.append(available[row, peak])
            strongest_pitches.append(peak / 32.0)
            # Distinct pitches must be separated by at least 2.6 semitones.
            available[np.abs(np.arange(len(fundamentals)) - peak[:, None]) <= 7] = -20
        p1, p2, p3 = strongest_scores
        output[left:stop] = np.column_stack(
            [
                p1,
                p2,
                p3,
                p1 - p2,
                p2 - p3,
                scores.mean(1),
                np.abs(strongest_pitches[0] - strongest_pitches[1]),
                np.abs(strongest_pitches[0] - strongest_pitches[2]),
                np.abs(strongest_pitches[1] - strongest_pitches[2]),
                np.sqrt(np.mean(frames[left:stop] ** 2, axis=1)),
                np.std(scores, axis=1),
            ]
        )
    return np.concatenate(
        [
            output,
            uniform_filter1d(output, 9, axis=0),
            uniform_filter1d(output, 27, axis=0),
        ],
        axis=1,
    )


@dataclass
class Case:
    example: core.Example
    dictionary: np.ndarray
    features: np.ndarray


def sorted_evidence(path, frames):
    """Pool local Sortformer speaker probabilities without reusing its slot identities."""
    with path.open("rb") as f:
        posterior = np.load(f, allow_pickle=False)
    if posterior.ndim != 2 or posterior.shape[1] != 4 or not len(posterior):
        raise ValueError(f"Invalid Sortformer cache: {path}")
    times = (np.arange(len(posterior)) + 0.5) * 0.08
    targets = (np.arange(frames) * STEP + 0.5) / 100
    ranked = np.sort(posterior, axis=1)[:, ::-1]
    pooled = np.column_stack(
        [
            ranked,
            np.prod(ranked[:, :3], axis=1),
            ranked[:, 0] - ranked[:, 1],
            ranked[:, 1] - ranked[:, 2],
            ranked[:, 2] - ranked[:, 3],
        ]
    )
    local = [
        pooled,
        *(uniform_filter1d(pooled, width, axis=0) for width in (3, 7, 15)),
    ]
    return np.column_stack(
        [
            np.interp(targets, times, features[:, i])
            for features in local
            for i in range(features.shape[1])
        ]
    ).astype("f")


def read_case(audio, cache, session, config, ref=None):
    independent = config.dz / f"{audio.stem}.npy"
    example = core.read_example(audio, cache, session, config.profile, ref, independent)
    clip, one, two, _ = cached_inputs(audio, cache, config.base)
    _, dictionary = correct_turns(
        clip, one, two, audio, config.verifier, config.thresholds, config.narrow
    )
    original = core.count_input(example, with_independent=True)
    d = dictionary[::STEP].sum(1).astype("f")
    p = np.minimum(one[::STEP], two[::STEP]).min(1)
    q = np.maximum(one[::STEP], two[::STEP]).min(1)
    decision = np.column_stack([d, d == 3, p, q, p * q, q - p])
    derived = [decision]
    for width in (5, 13, 27, 55):
        derived += [uniform_filter1d(decision, width, axis=0)]
    pitch = distinct_pitch_features(audio, len(original))
    columns = [original, *derived, pitch]
    if config.sortformer:
        path = config.sortformer / f"{audio.stem}.npy"
        columns.append(sorted_evidence(path, len(original)))
    features = np.column_stack(columns).astype("f")
    return Case(example, dictionary, features)


def read_panel(manifest, cache, config):
    entries = json.loads(manifest.read_text())
    entries = entries if isinstance(entries, list) else entries["clips"]
    return [
        read_case(
            manifest.parent / row["audio"],
            cache / (Path(row["audio"]).stem + ".npz"),
            row.get("source_session") or row.get("session") or row["meeting"],
            config,
            manifest.parent / (row.get("ref") or row["reference"]),
        )
        for row in entries
    ]


def fit(cases):
    x = np.concatenate([c.features[::2] for c in cases])
    truth = np.concatenate([c.example.actual[:: (STEP * 2)].all(1) for c in cases])
    model = HistGradientBoostingClassifier(
        max_iter=160,
        max_depth=5,
        max_leaf_nodes=16,
        min_samples_leaf=65,
        l2_regularization=8.0,
        early_stopping=False,
        random_state=763,
    )
    model.fit(x, truth)
    return model


def probs(model, case):
    return model.predict_proba(case.features)[:, 1]


def refine(case, new_prob, latest_activity, knobs):
    lower, upper, smooth, short = knobs
    confidence = uniform_filter1d(new_prob, smooth)
    active = latest_activity.copy()
    probabilities = np.repeat(confidence, STEP)[: len(active)]
    threshold = np.where(case.dictionary.all(1), lower, upper)
    proposed = probabilities >= threshold
    # Repair only near credible multi-voice activity and skip unsupported solo islands.
    credible = maximum_filter1d((active.sum(1) >= 2).astype("f"), 17) > 0.0
    proposed &= credible
    if short > 1:
        proposed = binary_opening(proposed, structure=np.ones(short, dtype=bool))
    ranks = (
        case.example.identity_features[:, :, 0]
        + case.example.identity_features[:, :, 7]
    )
    drop = ~proposed & active.all(1)
    active[np.flatnonzero(drop), ranks.argmin(axis=1)[drop]] = False
    active[proposed] = True
    return active


def vector(cases, activities):
    return sum(counts(c.example, h) for c, h in zip(cases, activities, strict=True))


def calibrate(cases, evidence, base):
    comparison = {key: measures(vector(cases, val)) for key, val in base.items()}

    def improves(score, refs):
        higher_is_better = (
            "speaker_recall",
            "speaker_precision",
            "three_way_speaker_recall",
            "exact_three_precision",
            "exact_three_recall",
            "exact_three_F1",
        )
        gaps = [score[key] - max(x[key] for x in refs) for key in higher_is_better]
        return min(*gaps, min(x["DER"] for x in refs) - score["DER"])

    grid = product(
        (0.20, 0.26, 0.32, 0.38, 0.44, 0.50, 0.56),
        (0.32, 0.38, 0.44, 0.50, 0.56, 0.62),
        (3, 7, 13),
        (1, 7),
    )
    ranked = []
    for knob in grid:
        proposal = [
            refine(c, p, h, knob)
            for c, p, h in zip(cases, evidence, base["structured"], strict=True)
        ]
        score = measures(vector(cases, proposal))
        gap = improves(score, comparison.values())
        ranked.append((gap, score["exact_three_F1"], knob, score))
    gap, _, knob, score = max(ranked, key=lambda row: row[:2])
    return dict(
        knobs=knob,
        development={**comparison, "candidate": score},
        minimum_gain=gap,
        search_size=len(ranked),
    )


def staged_comparison(categories, original, evidence, knobs):
    model, settings = original
    structured = core.tune_batch(
        [c.example for c in categories],
        [core.emissions(model, c.example) for c in categories],
        settings,
    )
    return {
        "dictionary_narrow": [c.dictionary for c in categories],
        "structured": structured,
        "candidate": [
            refine(c, p, h, knobs)
            for c, p, h in zip(categories, evidence, structured, strict=True)
        ],
    }


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--dictionary", type=Path, required=True)
    ap.add_argument("--narrow", type=Path, required=True)
    ap.add_argument("--development", type=Path, nargs=2, action="append", default=[])
    ap.add_argument("--holdout", type=Path, nargs=2, action="append", default=[])
    ap.add_argument("--predict", type=Path, nargs=3)
    ap.add_argument("--dz", type=Path, required=True)
    ap.add_argument(
        "--sortformer",
        type=Path,
        help="Optional independent Sortformer cache directory",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frozen", type=Path, required=True)
    ns = ap.parse_args()
    models, params, ns.profile = load(ns.model)
    ns.verifier, ns.thresholds, ns.base = load_frozen(ns.dictionary)
    ns.narrow = json.loads(ns.narrow.read_text())
    original_model = models["independent_diarizen"]
    settings = params["independent_diarizen"][0]
    if ns.development:
        cases = [c for panel in ns.development for c in read_panel(*panel, ns)]
        folds = sorted({c.example.session for c in cases})
        if len(folds) < 3:
            ap.error("Calibration requires at least three source sessions.")
        print("Development cases", len(cases), "sessions", folds, flush=True)
        records = {}
        for fold in range(3):
            heldout = set(folds[fold::3])
            train = [c for c in cases if c.example.session not in heldout]
            test = [c for c in cases if c.example.session in heldout]
            specialized = fit(train)
            (oof_model,) = core.fit(
                [c.example for c in train], [("structured", True)], train_skip=3
            ).values()
            structured = core.tune_batch(
                [c.example for c in test],
                [core.emissions(oof_model, c.example) for c in test],
                settings,
            )
            for c, h in zip(test, structured, strict=True):
                records[c.example.name] = (probs(specialized, c), h)
            print("Held-out calibration fold", fold, flush=True)
        aggregate = [records[c.example.name] for c in cases]
        published = {
            "dictionary_narrow": [c.dictionary for c in cases],
            "structured": [h for _, h in aggregate],
        }
        calibration = calibrate(cases, [p for p, _ in aggregate], published)
        print("Frozen", calibration, flush=True)
        model = fit(cases)
        ns.frozen.write_text(
            json.dumps(
                dict(
                    calibration=calibration,
                    model=encode(model),
                    requires_sortformer=bool(ns.sortformer),
                )
            )
        )
    else:
        payload = json.loads(ns.frozen.read_text())
        if bool(ns.sortformer) != payload.get("requires_sortformer", False):
            ap.error("Model feature mode must match its frozen Sortformer setting.")
        calibration, model = payload["calibration"], NumericTrees(payload["model"])
    result = {"calibration": calibration}
    if ns.holdout:
        result["holdout"] = {}
        for manifest, caches in ns.holdout:
            cases = read_panel(manifest, caches, ns)
            examples = [c.example for c in cases]
            phases = staged_comparison(
                cases,
                (original_model, settings),
                [probs(model, c) for c in cases],
                calibration["knobs"],
            )
            report = {k: core.evaluate(examples, h) for k, h in phases.items()}
            result["holdout"][str(manifest)] = report
            print("Holdout", {k: r["pooled"] for k, r in report.items()}, flush=True)
    if ns.predict:
        audio, cache, dest = ns.predict
        cases = [read_case(audio, cache, audio.stem, ns)]
        phases = staged_comparison(
            cases,
            (original_model, settings),
            [probs(model, cases[0])],
            calibration["knobs"],
        )
        duration = sf.info(audio).duration
        segments = core.to_segments(phases["candidate"][0], duration)
        for item in segments:
            item["speaker"] = "ABC"[int(item["speaker"])]
        dest.write_text(
            json.dumps({"duration_s": duration, "segments": segments}, indent=2) + "\n"
        )
    ns.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    with threadpool_limits(2):
        main()
