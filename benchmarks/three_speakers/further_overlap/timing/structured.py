"""Decode time-consistent anonymous A/B/C activity from mono acoustic evidence.

Development counts alone select all operating points. Predictions read only a
mono WAV and anonymous model posteriors, with an optional independent acoustic pass.
"""

import argparse
from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from calibrate_nemotron import to_segments
from diarize import masks, reference
from multiview_score import align, counts, fit_length, measures
from improve_overlap import Decoder
from portable import load as load_frozen, save as save_frozen

STATES = (np.arange(8)[:, None] & (1 << np.arange(3))) != 0
CARD = STATES.sum(axis=1)
CHANGES = (STATES[:, None] != STATES[None]).sum(axis=2).astype("f")
EXTRA_JUMP = np.maximum(0.0, CHANGES - 1.0)
HOP = 2


@dataclass
class Example:
    name: str
    session: str
    actual: np.ndarray | None
    previous: np.ndarray
    native: np.ndarray
    count_features: np.ndarray
    identity_features: np.ndarray
    independent: np.ndarray | None
    soft_reference: np.ndarray | None


def posterior_features(native, alternate):
    """Preserve each speaker's identity, then pool features across identities."""
    tracks = [
        native,
        alternate,
        np.maximum(native, alternate),
        np.minimum(native, alternate),
    ]
    identity = []
    sorted_context = []
    for value in tracks:
        local = [value] + [uniform_filter1d(value, w, axis=0) for w in (9, 23, 53, 121)]
        local += [maximum_filter1d(value, w, axis=0) for w in (19, 55)]
        sorted_context.extend(np.sort(v, axis=1) for v in local)
        identity.extend(local)
    for value in tracks[:2]:
        for dt in (7, 21, 43):
            left = np.maximum(0, np.arange(len(value)) - dt)
            right = np.minimum(len(value) - 1, np.arange(len(value)) + dt)
            for edge in (value - value[left], value[right] - value):
                sorted_context.append(np.sort(edge, axis=1))
                identity.append(edge)
    return np.concatenate(sorted_context, axis=1), np.stack(identity, axis=2)


def diarizen_features(path, posterior, acoustic):
    model = np.load(path)
    if not len(model):
        raise ValueError(f"Empty independent model features: {path}")
    model = model[np.minimum(np.arange(len(posterior)), len(model) - 1)]
    third = np.min(posterior, axis=1)
    pair_mass = model[:, 1]
    third_marginal = model[:, 3]
    seed = np.column_stack(
        [
            model,
            pair_mass * third,
            third_marginal * third,
            pair_mass - model[:, 7],
            third_marginal * acoustic[:, 0],
        ]
    )
    return np.concatenate(
        [seed, *(uniform_filter1d(seed, w, axis=0) for w in (13, 37, 101))], axis=1
    )


def acoustic_features(path, length):
    """Make the prior acoustic features a few seconds at a time."""
    block_frames = 512
    components = np.empty((length, 10), dtype=np.float32)
    window = np.hanning(640).astype(np.float32)
    bands = [(5, 19), (19, 51), (51, 102), (102, 205), (205, 410)]
    previous_spectrum = None
    with sf.SoundFile(path) as stream:
        if stream.channels != 1 or stream.samplerate != 16000:
            raise ValueError(f"Expected 16 kHz mono audio in {path}")
        for start in range(0, length, block_frames):
            stop = min(length, start + block_frames)
            left = start * 160 - 240
            right = (stop - 1) * 160 + 400
            stream.seek(max(0, left))
            wave = stream.read(
                max(0, min(right, len(stream)) - max(0, left)), dtype="float32"
            )
            wave = np.pad(wave, (max(0, -left), max(0, right - len(stream))))
            frames = np.lib.stride_tricks.sliding_window_view(wave, 640)[::160]
            log_level = np.log(np.maximum(1e-9, np.mean(frames**2, axis=1)))
            fft = np.fft.rfft(frames * window, n=1024, axis=1)
            power = np.abs(fft) ** 2
            normalized = power / np.maximum(1e-8, power.sum(axis=1, keepdims=True))
            band_levels = [
                np.log(np.maximum(1e-6, normalized[:, a:b].sum(axis=1)))
                for a, b in bands
            ]
            acf = np.fft.irfft(power, axis=1)
            peaks = acf[:, 32:224] / np.maximum(1e-7, acf[:, :1])
            centers = peaks[:, 1:-1]
            local = (centers > peaks[:, :-2]) & (centers >= peaks[:, 2:])
            periodic = np.sort(np.where(local, centers, 0.0), axis=1)[:, -3:]
            leading = normalized[:1] if previous_spectrum is None else previous_spectrum
            flux = np.abs(np.diff(normalized, axis=0, prepend=leading)).sum(axis=1)
            previous_spectrum = normalized[-1:].copy()
            components[start:stop] = np.column_stack(
                [log_level, *band_levels, periodic, flux]
            )
    components[:, 0] -= np.quantile(components[:, 0], 0.75)
    return np.column_stack([components, uniform_filter1d(components, 15, axis=0)])


def read_example(audio, cached, session, profile, ref=None, independent=None):
    with np.load(cached) as archive:
        info = sf.info(audio)
        if abs(float(archive["duration_s"]) - info.duration) > 0.01:
            raise ValueError(f"Cache duration does not match {audio}")
        length = (info.frames + 159) // 160
        base = fit_length(archive["base"], length)
        alt = fit_length(archive["alternate"][int(archive["offset"]) :], length)
        native = np.zeros_like(base, dtype=bool)
        frames = (np.arange(length) + 0.5) / 100.0
        for start, end, slot in archive["native"]:
            left, right = np.searchsorted(frames, (start, end))
            native[left:right, int(slot)] = True
    if length < HOP:
        raise ValueError(f"Audio requires at least {HOP / 100.0} seconds")
    ids, first, second = align(base, alt, native.sum(axis=0))
    native = native[:, ids]
    previous = Decoder(first).decode(profile)
    counts_input, identity_input = posterior_features(first, second)
    acoustic = acoustic_features(audio, length)
    counts_input = np.column_stack(
        [counts_input, acoustic, previous.sum(axis=1), native.sum(axis=1)]
    ).astype("f")
    identity_input = np.concatenate(
        [identity_input, previous[:, :, None], native[:, :, None]], axis=2
    ).astype("f")
    dz = (
        diarizen_features(independent, (first + second) * 0.5, acoustic)
        if independent is not None
        else None
    )
    actual = soft_ref = None
    if ref is not None:
        actual = masks(reference(ref), length / 100.0)[1].T[:length]
        if actual.shape != (length, 3):
            raise ValueError(f"Expected exactly three reference identities in {ref}")
        overlap = actual.astype("f").T @ (first + second).astype("f")
        i, j = linear_sum_assignment(-overlap)
        soft_ref = actual[:, i[np.argsort(j)]]
    return Example(
        audio.stem,
        session,
        actual,
        previous,
        native,
        counts_input,
        identity_input,
        dz,
        soft_ref,
    )


def load_panel(manifest, cache, profile, independent):
    data = json.loads(manifest.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data["clips"]
    examples = []
    for row in rows:
        audio = manifest.parent / row["audio"]
        model = independent / f"{audio.stem}.npy" if independent is not None else None
        ref = manifest.parent / (row.get("ref") or row["reference"])
        session = row.get("source_session") or row.get("session") or row["meeting"]
        examples.append(
            read_example(
                audio, cache / f"{audio.stem}.npz", session, profile, ref, model
            )
        )
    return examples


def classifier(max_iter):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=max_iter,
        max_depth=4,
        min_samples_leaf=70,
        l2_regularization=10.0,
        early_stopping=False,
        random_state=2461,
    )


def count_input(example, with_independent):
    values = example.count_features[::HOP]
    if with_independent:
        if example.independent is None:
            raise ValueError("The independent variant requires DiariZen features")
        values = np.column_stack([values, example.independent[::HOP]])
    return values


def identity_input(example, skip=1):
    stride = HOP * skip
    identity = example.identity_features[::stride]
    context = np.broadcast_to(
        example.count_features[::stride, None, :],
        (len(identity), 3, example.count_features.shape[1]),
    )
    matrix = np.concatenate([identity, context], axis=2)
    return matrix.reshape(-1, matrix.shape[-1])


def fit(examples, variants, train_skip=1):
    truth = np.concatenate(
        [c.actual[:: HOP * train_skip].sum(axis=1) for c in examples]
    )
    weights = np.sqrt(
        np.max(np.bincount(truth, minlength=4))
        / np.maximum(1, np.bincount(truth, minlength=4))
    )
    identity = classifier(33)
    identity.fit(
        np.concatenate([identity_input(c, train_skip) for c in examples]),
        np.concatenate(
            [c.soft_reference[:: HOP * train_skip].ravel() for c in examples]
        ),
    )
    return {
        name: (
            classifier(58).fit(
                np.concatenate([count_input(c, extra)[::train_skip] for c in examples]),
                truth,
                sample_weight=weights[truth],
            ),
            identity,
            extra,
        )
        for name, extra in variants
    }


def emissions(model, example):
    count_model, identity_model, independent = model
    count = np.full((len(example.native[::HOP]), 4), 1.0e-5, dtype="f")
    count[:, count_model.classes_] = count_model.predict_proba(
        count_input(example, independent)
    )
    slot = (
        identity_model.decision_function(identity_input(example))
        .reshape(-1, 3)
        .astype("f")
    )
    within_count = slot @ STATES.T.astype("f")
    for k in range(4):
        select = CARD == k
        within_count[:, select] -= logsumexp(within_count[:, select], axis=1)[:, None]
    return np.log(np.maximum(count[:, CARD], 1.0e-6)) + within_count


def tune_batch(examples, logits, knobs):
    """Padded batched dynamic program. Padding has no effect on individual traceback."""
    price, bias_pair, bias_triple = knobs
    lengths = np.array([len(x) for x in logits])
    size = len(examples)
    emissions_batch = np.full((max(lengths), size, 8), -80.0, dtype=np.float32)
    offsets = np.array([0.0, 0.0, bias_pair, bias_triple], dtype="f")[CARD]
    for index, values in enumerate(logits):
        emissions_batch[: len(values), index] = values + offsets
    if price == 0.0:
        return [
            STATES[np.repeat(np.argmax(v + offsets, axis=1), HOP)[: len(c.native)]]
            for c, v in zip(examples, logits, strict=True)
        ]
    transitions = -price * (CHANGES + EXTRA_JUMP * 0.75)
    trace = np.zeros_like(emissions_batch, dtype=np.int8)
    best = emissions_batch[0]
    ending_state = np.zeros(size, dtype=np.int8)
    for t in range(1, len(emissions_batch)):
        options = best[:, :, None] + transitions[None]
        trace[t] = options.argmax(axis=1)
        best = np.max(options, axis=1) + emissions_batch[t]
        ending = lengths == t + 1
        ending_state[ending] = best[ending].argmax(axis=1)
    output = []
    for index, (case, values) in enumerate(zip(examples, logits, strict=True)):
        n = len(values)
        state = np.empty(n, np.int8)
        if n == 1:
            state[0] = np.argmax(values[0] + offsets)
        else:
            state[-1] = ending_state[index]
            for t in range(n - 1, 0, -1):
                state[t - 1] = trace[t, index, state[t]]
        output.append(STATES[np.repeat(state, HOP)[: len(case.native)]])
    return output


def evaluate(examples, hypotheses):
    report, grouped = {}, {}
    for case, active in zip(examples, hypotheses, strict=True):
        c = counts(case, active)
        report[case.name] = c.tolist()
        grouped[case.session] = grouped.get(case.session, np.zeros_like(c)) + c
    return dict(
        pooled=measures(sum(grouped.values())),
        sessions={s: measures(c) for s, c in grouped.items()},
        per_clip_counts=report,
    )


def tuning(examples, probabilities, strict=False):
    prior = evaluate(examples, [c.previous for c in examples])["pooled"]
    grid = product((0.0, 2.0, 5.0, 10.0), (-0.4, 0.4), (-0.7, 0.0, 0.7, 1.4))
    rows = []
    for settings in grid:
        active = tune_batch(examples, probabilities, settings)
        val = measures(sum(counts(c, x) for c, x in zip(examples, active, strict=True)))
        objective = (
            val["exact_three_F1"]
            + 0.20 * val["three_way_speaker_recall"]
            - 0.60 * val["DER"]
        )
        rows.append((objective, settings, val))
    eligible = [
        r
        for r in rows
        if r[2]["DER"] <= prior["DER"] + 0.01
        and (
            not strict
            or r[2]["exact_three_precision"] >= prior["exact_three_precision"] - 0.01
        )
    ]
    best = max(eligible or rows, key=lambda x: x[0])
    no_timing = max((r for r in rows if r[1][0] == 0.0), key=lambda x: x[0])
    return best[1:], no_timing[1:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dev", action="append", nargs=2, type=Path, metavar=("MANIFEST", "CACHE")
    )
    parser.add_argument("--train-skip", type=int, default=1)
    parser.add_argument(
        "--match-development-precision",
        action="store_true",
        help="Prefer operating points that preserve calibration precision as well as DER",
    )
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--dz-dir", type=Path)
    parser.add_argument(
        "--test",
        action="append",
        nargs=3,
        default=[],
        metavar=("LABEL", "MANIFEST", "CACHE"),
    )
    parser.add_argument(
        "--test-mono",
        action="append",
        nargs=3,
        default=[],
        metavar=("LABEL", "MANIFEST", "CACHE"),
    )
    parser.add_argument("--save", type=Path)
    parser.add_argument("--from-saved", type=Path)
    parser.add_argument(
        "--per-frame",
        action="store_true",
        help="Use the frozen per-frame operating point for export",
    )
    parser.add_argument(
        "--predict", nargs=3, type=Path, metavar=("WAV", "POSTERIORS", "TIMING")
    )
    parser.add_argument("--out", type=Path, help="JSON report for training or scoring")
    args = parser.parse_args()
    if args.train_skip < 1:
        parser.error("--train-skip must be a positive number")
    if not args.out and (not args.from_saved or args.test or args.test_mono):
        parser.error("Training or benchmark scoring requires --out")
    if not args.out and not args.predict:
        parser.error("Specify --predict or --out")
    if args.test and not args.dz_dir:
        parser.error(
            "Use --dz-dir with --test, or --test-mono for acoustic-only scoring"
        )
    results = {"development": {}, "tests": {}}
    if args.from_saved:
        models, knobs, profile = load_frozen(args.from_saved)
    else:
        if args.dev is None or args.profile is None:
            parser.error("Provide --dev and --profile to train")
        profile = json.loads(args.profile.read_text())
        development = [
            c for group in args.dev for c in load_panel(*group, profile, args.dz_dir)
        ]
        sessions = sorted({c.session for c in development})
        variants = [("native_acoustic", False)] + (
            [("independent_diarizen", True)] if args.dz_dir else []
        )
        heldout = {v[0]: {} for v in variants}
        for i in range(3):
            held = set(sessions[i::3])
            fit_models = fit(
                [c for c in development if c.session not in held],
                variants,
                args.train_skip,
            )
            for c in development:
                if c.session in held:
                    for name, model in fit_models.items():
                        heldout[name][c.name] = emissions(model, c)
            print("OOF fold complete", i, flush=True)
        knobs = {}
        for name, frames in heldout.items():
            best, no_timing = tuning(
                development,
                [frames[c.name] for c in development],
                args.match_development_precision,
            )
            knobs[name] = [best[0], no_timing[0]]
            results["development"][name] = {
                "knobs": knobs[name],
                "structured": best[1],
                "per_frame": no_timing[1],
            }
            print("Frozen development operating point", name, knobs[name], flush=True)
        models = fit(development, variants, args.train_skip)
        if args.save:
            save_frozen(args.save, models, knobs, profile)
        args.out.write_text(json.dumps(results, indent=2) + "\n")
    for all_cues, panels in ((True, args.test), (False, args.test_mono)):
        for label, manifest, cache in panels:
            start = time.perf_counter()
            examples = load_panel(
                Path(manifest), Path(cache), profile, args.dz_dir if all_cues else None
            )
            table = {
                "previous": evaluate(examples, [c.previous for c in examples]),
                "native": evaluate(examples, [c.native for c in examples]),
            }
            for name, model in models.items():
                if not all_cues and model[2]:
                    continue
                prob = [emissions(model, c) for c in examples]
                for kind, settings in zip(
                    ("structured", "per_frame"), knobs[name], strict=True
                ):
                    table[name + "_" + kind] = evaluate(
                        examples, tune_batch(examples, prob, settings)
                    )
            results["tests"][label] = dict(
                results=table, elapsed_s=time.perf_counter() - start
            )
            args.out.write_text(json.dumps(results, indent=2) + "\n")
            print(label, {k: v["pooled"] for k, v in table.items()}, flush=True)
    if args.predict:
        audio, cache, timing = args.predict
        independent = args.dz_dir / f"{audio.stem}.npy" if args.dz_dir else None
        name = "independent_diarizen" if independent is not None else "native_acoustic"
        example = read_example(
            audio, cache, audio.stem, profile, independent=independent
        )
        (act,) = tune_batch(
            [example],
            [emissions(models[name], example)],
            knobs[name][int(args.per_frame)],
        )
        with np.load(cache) as archive:
            duration = float(archive["duration_s"])
        rows = to_segments(act, duration)
        for row in rows:
            row["speaker"] = "ABC"[int(row["speaker"])]
        timing.write_text(
            json.dumps({"duration_s": duration, "segments": rows}, indent=2) + "\n"
        )


if __name__ == "__main__":
    with threadpool_limits(1):
        main()
