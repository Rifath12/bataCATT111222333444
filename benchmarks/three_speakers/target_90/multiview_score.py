"""Fit a speaker-count decoder and measure it on speaker-disjoint meetings."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingClassifier
import soundfile as sf
from threadpoolctl import threadpool_limits

from calibrate_nemotron import to_segments
from diarize import masks, reference
from improve_overlap import Decoder


@dataclass
class Clip:
    name: str
    session: str
    actual: np.ndarray | None
    native: np.ndarray
    previous: np.ndarray
    features: np.ndarray
    fused: np.ndarray
    acoustic: np.ndarray
    native_s: float
    alternate_s: float


def fit_length(values, length):
    return np.pad(values[:length], ((0, max(0, length - len(values))), (0, 0)))


def align(base, alternate, durations):
    evidence = durations + 0.15 * (base >= 0.25).sum(axis=0)
    selected = np.sort(np.argsort(-evidence, kind="stable")[:3])
    anchor = base[:, selected]
    # Compare posterior trajectories without consulting speaker annotations.
    left = anchor - anchor.mean(axis=0)
    right = alternate - alternate.mean(axis=0)
    similarity = left.T @ right
    norm = np.linalg.norm(left, axis=0)[:, None] * np.linalg.norm(right, axis=0)
    similarity /= np.maximum(norm, 1e-6)
    mass = np.log1p((alternate >= 0.3).sum(axis=0))
    similarity += 0.005 * mass
    i, j = linear_sum_assignment(-similarity)
    return selected, anchor, alternate[:, j[np.argsort(i)]]


def feature_matrix(base, alternate):
    features = []
    for view in (
        base,
        alternate,
        (base + alternate) * 0.5,
        np.minimum(base, alternate),
    ):
        for width in (1, 9, 25, 59):
            values = uniform_filter1d(view, width, axis=0) if width > 1 else view
            features.append(-np.sort(-values, axis=1))
    return np.concatenate(features, axis=1).astype(np.float32)


def acoustic_features(path, length):
    wave, rate = sf.read(path, dtype="float32")
    if wave.ndim != 1 or rate != 16000:
        raise ValueError("Expected a mono 16 kHz waveform")
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(wave, (240, 400)), 640)[
        ::160
    ][:length]
    energy = np.mean(frames**2, axis=1)
    log_level = np.log(np.maximum(energy, 1e-9))
    middle_level = np.quantile(log_level, 0.75)
    fft = np.fft.rfft(frames * np.hanning(640).astype(np.float32), n=1024, axis=1)
    power = np.abs(fft) ** 2
    normalized = power / np.maximum(1e-8, power.sum(axis=1, keepdims=True))
    bands = [(5, 19), (19, 51), (51, 102), (102, 205), (205, 410)]
    shares = [
        np.log(np.maximum(1e-6, normalized[:, i:j].sum(axis=1))) for i, j in bands
    ]
    acf = np.fft.irfft(power, axis=1)
    peaks = acf[:, 32:224] / np.maximum(1e-7, acf[:, :1])
    local_peak = (peaks[:, 1:-1] > peaks[:, :-2]) & (peaks[:, 1:-1] >= peaks[:, 2:])
    peak_values = np.where(local_peak, peaks[:, 1:-1], 0.0)
    periodic = np.sort(peak_values, axis=1)[:, -3:]
    flux = np.abs(np.diff(normalized, axis=0, prepend=normalized[:1])).sum(axis=1)
    features = np.column_stack(
        [log_level - middle_level, *shares, periodic, flux]
    ).astype(np.float32)
    return np.concatenate([features, uniform_filter1d(features, 15, axis=0)], axis=1)


def load_clip(audio, archive, session, length, actual, old_profile):
    base = fit_length(archive["base"], length)
    alternate = fit_length(archive["alternate"][int(archive["offset"]) :], length)
    activity = np.zeros_like(base, dtype=bool)
    centers = (np.arange(length) + 0.5) * 0.01
    for start, end, channel in archive["native"]:
        i, j = np.searchsorted(centers, [start, end])
        activity[i:j, int(channel)] = True
    ids, first, second = align(base, alternate, activity.sum(axis=0))
    fused = 0.55 * first + 0.25 * second + 0.20 * np.maximum(first, second)
    return Clip(
        audio.stem,
        session,
        actual,
        activity[:, ids],
        Decoder(first).decode(old_profile),
        feature_matrix(first, second),
        fused,
        acoustic_features(audio, length),
        float(archive["base_wall_s"]),
        float(archive["alternate_wall_s"]),
    )


def load_manifest(path, probs, old_profile):
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data if isinstance(data, list) else data["clips"]
    result = []
    for entry in entries:
        audio = path.parent / entry["audio"]
        with np.load(probs / f"{audio.stem}.npz") as archive:
            ref = reference(path.parent / (entry.get("ref") or entry["reference"]))
            actual = masks(ref, float(archive["duration_s"]))[1].T
            if actual.shape[1] != 3:
                raise ValueError(f"Expected three reference speakers in {audio.stem}")
            session = entry.get("session") or entry["source_session"]
            result.append(
                load_clip(audio, archive, session, len(actual), actual, old_profile)
            )
    return result


def counts(clip, predicted):
    overlap = clip.actual.T.astype(np.float32) @ predicted.astype(np.float32)
    i, j = linear_sum_assignment(-overlap)
    correct = (clip.actual[:, i] & predicted[:, j]).sum(axis=1)
    ref = clip.actual.sum(axis=1)
    sys = predicted.sum(axis=1)
    exact = ref == 3
    marked = sys == 3
    return np.array(
        [
            ref.sum(),
            sys.sum(),
            correct.sum(),
            (np.maximum(ref, sys) - correct).sum(),
            exact.sum(),
            marked.sum(),
            (exact & marked).sum(),
            correct[exact].sum(),
            len(ref),
        ],
        np.int64,
    )


def measures(raw):
    (
        reference_time,
        estimated_time,
        correct,
        errors,
        actual,
        estimated,
        hits,
        triple_correct,
        frames,
    ) = map(int, raw)
    return {
        "speaker_recall": correct / max(1, reference_time),
        "speaker_precision": correct / max(1, estimated_time),
        "DER": errors / max(1, reference_time),
        "three_way_speaker_recall": triple_correct / max(1, actual * 3),
        "exact_three_precision": hits / max(1, estimated),
        "exact_three_recall": hits / max(1, actual),
        "exact_three_F1": 2 * hits / max(1, actual + estimated),
        "seconds": frames / 100,
        "reference_three_s": actual / 100,
        "predicted_three_s": estimated / 100,
        "false_three_s": (estimated - hits) / 100,
        "exact_three_true_positive_s": hits / 100,
    }


def decode(clip, probability, threshold):
    cardinality = np.argmax(probability[:, :3], axis=1)
    cardinality[probability[:, 3] >= threshold] = 3
    ranks = np.argsort(np.argsort(-clip.fused, axis=1), axis=1)
    return ranks < cardinality[:, None]


def choose_threshold(clips, probabilities):
    best = (-float("inf"), 0.5)
    recovery = None
    previous = measures(sum(counts(c, c.previous) for c in clips))
    for threshold in np.linspace(0.05, 0.85, 33):
        raw = sum(
            counts(c, decode(c, p, threshold))
            for c, p in zip(clips, probabilities, strict=True)
        )
        stats = measures(raw)
        value = (
            stats["exact_three_F1"]
            + 0.25 * stats["three_way_speaker_recall"]
            - 0.5 * stats["DER"]
        )
        if value > best[0]:
            best = (value, float(threshold))
        if (
            stats["three_way_speaker_recall"] >= 0.90
            and stats["DER"] <= previous["DER"]
        ):
            option = (stats["exact_three_F1"], float(threshold))
            if recovery is None or option > recovery:
                recovery = option
    return best[1], recovery[1] if recovery else None


def inputs(clip, with_acoustics):
    return (
        np.concatenate([clip.features, clip.acoustic], axis=1)
        if with_acoustics
        else clip.features
    )


def fit_model(clips, with_acoustics):
    features = np.concatenate([inputs(c, with_acoustics) for c in clips])
    actual = np.concatenate([c.actual.sum(axis=1) for c in clips])
    frequencies = np.bincount(actual, minlength=4)
    weights = np.sqrt(frequencies.max() / np.maximum(1, frequencies))
    model = HistGradientBoostingClassifier(
        max_depth=4,
        max_iter=135,
        min_samples_leaf=70,
        l2_regularization=10.0,
        early_stopping=False,
        random_state=19,
    )
    model.fit(features, actual, sample_weight=weights[actual])
    return model


def predict(model, clip, with_acoustics):
    probability = np.zeros((len(clip.fused), 4), np.float32)
    probability[:, model.classes_] = model.predict_proba(inputs(clip, with_acoustics))
    return probability


def evaluate(clips, models):
    results = []
    started = time.perf_counter()
    for clip in clips:
        common = {
            "native": counts(clip, clip.native),
            "previous_calibration": counts(clip, clip.previous),
        }
        for arm, model, threshold, use_audio in models:
            candidate = decode(clip, predict(model, clip, use_audio), threshold)
            common[arm] = counts(clip, candidate)
        results.append(
            dict(
                clip=clip.name,
                session=clip.session,
                **{k: v.tolist() for k, v in common.items()},
            )
        )
    info = {
        arm: measures(sum(np.asarray(row[arm]) for row in results))
        for arm in ("native", "previous_calibration", *(row[0] for row in models))
    }
    return {
        "pooled": info,
        "clips": results,
        "sessions": len({c.session for c in clips}),
        "runtime_s": {
            "native": sum(c.native_s for c in clips),
            "alternate": sum(c.alternate_s for c in clips),
            "scoring_and_decoder": time.perf_counter() - started,
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--train-probs", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument(
        "--test",
        action="append",
        nargs=3,
        metavar=("NAME", "MANIFEST", "PROBS"),
        default=[],
    )
    p.add_argument("--out", type=Path)
    p.add_argument(
        "--predict", nargs=3, metavar=("WAV", "PROBABILITIES", "TIMING_JSON")
    )
    args = p.parse_args()
    if args.test and args.out is None:
        p.error("--test requires --out")
    if not args.test and not args.predict:
        p.error("Supply --test or --predict")
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    development = load_manifest(args.train_manifest, args.train_probs, profile)
    sessions = sorted({c.session for c in development})
    if len(sessions) < 4:
        p.error("Use at least four development sessions")
    tuning_sessions = set(sessions[::4])
    train = [c for c in development if c.session not in tuning_sessions]
    tuning = [c for c in development if c.session in tuning_sessions]
    models = []
    with threadpool_limits(2):
        for name, use_audio in (
            ("two_view_count", False),
            ("two_view_acoustics", True),
        ):
            model = fit_model(train, use_audio)
            threshold, recovery = choose_threshold(
                tuning, [predict(model, c, use_audio) for c in tuning]
            )
            final_model = fit_model(development, use_audio)
            models.append((name, final_model, threshold, use_audio))
            if recovery is not None and recovery != threshold:
                models.append((name + "_recovery", final_model, recovery, use_audio))
    print(
        f"Frozen thresholds after fitting {len(development)} development clips",
        [(name, t) for name, _, t, _ in models],
        flush=True,
    )
    report = {
        "development_clips": len(development),
        "tuning_sessions": sorted(tuning_sessions),
        "thresholds": {name: t for name, _, t, _ in models},
        "benchmarks": {},
    }
    for name, manifest, folder in args.test:
        clips = load_manifest(Path(manifest), Path(folder), profile)
        out = evaluate(clips, models)
        report["benchmarks"][name] = out
        print(name, len(clips), out["pooled"], flush=True)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.predict:
        audio, probabilities, timing = map(Path, args.predict)
        with np.load(probabilities) as archive:
            duration = float(archive["duration_s"])
            if abs(sf.info(audio).duration - duration) > 1 / 16000:
                raise ValueError("WAV duration does not match the posterior cache")
            length = int(np.ceil(duration / 0.01))
            clip = load_clip(audio, archive, audio.stem, length, None, profile)
        name, model, threshold, use_audio = next(
            row for row in models if row[0] == "two_view_acoustics"
        )
        activity = decode(clip, predict(model, clip, use_audio), threshold)
        segments = to_segments(activity, duration)
        for segment in segments:
            segment["speaker"] = "ABC"[int(segment["speaker"])]
        timing.write_text(
            json.dumps(
                {
                    "duration_s": duration,
                    "experimental_decoder": name,
                    "threshold": threshold,
                    "segments": segments,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
