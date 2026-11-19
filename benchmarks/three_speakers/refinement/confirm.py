"""Verify candidate third voices using a classifier shared by the three speaker slots."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from calibrate_nemotron import to_segments
from diarize import masks, reference
from improve_overlap import Decoder
from multiview_score import align, counts, fit_length, measures

STRIDE = 4
LONG_DEV = "R8008_M8013"


@dataclass
class Sample:
    name: str
    session: str
    previous: np.ndarray
    actual: np.ndarray | None
    features: np.ndarray
    duration: float


def differences(view):
    result = []
    for slot in range(3):
        others = np.delete(view, slot, axis=1)
        result.append(
            np.stack([view[:, slot], others.max(axis=1), others.min(axis=1)], axis=-1)
        )
    return np.stack(result, axis=1)


def acoustic_evidence(first, second, activity, previous):
    # All three speaker slots use the same feature definitions and classifier.
    evidence = []
    for view in (first, second):
        for size in (1, 15, 51, 151):
            smooth = uniform_filter1d(view, size, axis=0) if size > 1 else view
            evidence.append(differences(smooth[::STRIDE]))
        peaks = maximum_filter1d(view, 41, axis=0)
        evidence.append(differences(peaks[::STRIDE]))
    for view in (activity, previous):
        evidence.append(differences(view[::STRIDE]))
        evidence.append(
            differences(uniform_filter1d(view.astype("f"), 51, axis=0)[::STRIDE])
        )
    # A speaker heard clearly elsewhere in this recording gives a local scale.
    record_features = []
    for view in (first, second):
        for threshold in (0.04, 0.35, 0.8):
            record_features.append(
                differences((view > threshold).mean(axis=0, keepdims=True))
            )
    evidence.extend(
        np.broadcast_to(x, (len(first[::STRIDE]), 3, 3)) for x in record_features
    )
    return np.concatenate(evidence, axis=2).astype("f")


def sample_from_views(
    name, session, duration, base, alternate, segments, profile, actual=None
):
    length = int(np.ceil(duration / 0.01))
    base, alternate = (fit_length(a, length) for a in (base, alternate))
    timing = np.zeros_like(base, dtype=bool)
    centers = (np.arange(length) + 0.5) * 0.01
    for start, end, slot in segments:
        i, j = np.searchsorted(centers, [start, end])
        timing[i:j, int(slot)] = True
    slots, first, second = align(base, alternate, timing.sum(axis=0))
    previous = Decoder(first).decode(profile)
    features = acoustic_evidence(first, second, timing[:, slots], previous)
    return Sample(name, session, previous, actual, features, duration)


def from_cache(path, profile, session=None, ref=None):
    with np.load(path) as a:
        duration = float(a["duration_s"])
        actual = masks(reference(ref), duration)[1].T if ref else None
        return sample_from_views(
            path.stem,
            session or path.stem,
            duration,
            a["base"],
            a["alternate"][int(a["offset"]) :],
            a["native"],
            profile,
            actual,
        )


def manifest_samples(manifest, posterior, profile):
    data = json.loads(manifest.read_text(encoding="utf-8"))
    for e in data if isinstance(data, list) else data["clips"]:
        stem = Path(e["audio"]).stem
        yield from_cache(
            posterior / (stem + ".npz"),
            profile,
            e.get("session") or e["source_session"],
            manifest.parent / (e.get("ref") or e["reference"]),
        )


def train_long(work, views, name, profile):
    import soundfile as sf

    if name == LONG_DEV:
        audio, prob, native, ref = (
            work / raw
            for raw in (
                "session/mono.wav",
                "nemotron_probabilities.npz",
                "nemotron_raw.rttm",
                name + ".rttm",
            )
        )
    else:
        part = work / "independent_3speaker"
        audio, prob, native = (
            part / name / f for f in ("mono.wav", "probabilities.npz", "native.rttm")
        )
        ref = part / "references/Test_Ali_far" / (name + ".rttm")
    duration = sf.info(audio).duration
    actual = masks(reference(ref), duration)[1].T
    native = [
        (s["start"], s["end"], int(s["speaker"].split("_")[-1]) - 1)
        for s in reference(native)
    ]
    with np.load(prob) as a, np.load(views / name / "view.npz") as b:
        return sample_from_views(
            name,
            name,
            duration,
            a["probabilities"],
            b["probabilities"][928:],
            native,
            profile,
            actual,
        )


def target(sample):
    affinity = sample.previous.T.astype("f") @ sample.actual.astype("f")
    rows, cols = linear_sum_assignment(-affinity)
    return sample.actual[:, cols[np.argsort(rows)]][::STRIDE]


def fit(records):
    features, labels, weights = [], [], []
    for sample, subset in records:
        x = sample.features[subset]
        y = target(sample)[subset]
        # Equalize corpus contributions and give rare true triples a training signal.
        w = 1 + 5 * np.all(y, axis=1)
        multiplier = 2.5 if sample.name == LONG_DEV else 1.0
        weights.append(np.repeat(w * multiplier, 3))
        features.append(x.reshape(-1, x.shape[-1]))
        labels.append(y.ravel())
    model = HistGradientBoostingClassifier(
        max_iter=90,
        max_leaf_nodes=15,
        min_samples_leaf=45,
        l2_regularization=8,
        early_stopping=False,
        random_state=137,
    )
    model.fit(
        np.concatenate(features),
        np.concatenate(labels),
        sample_weight=np.concatenate(weights),
    )
    return model


def evidence(model, sample):
    x = sample.features
    raw = model.predict_proba(x.reshape(-1, x.shape[-1]))[:, 1].reshape(-1, 3)
    centers = np.arange(len(raw)) * STRIDE
    return np.stack(
        [
            np.interp(np.arange(len(sample.previous)), centers, raw[:, i])
            for i in range(3)
        ],
        axis=1,
    )


def infer(sample, probability, thresholds):
    active = sample.previous.copy()
    minimum = probability.min(axis=1)
    count = active.sum(axis=1)
    veto = (count == 3) & (minimum < thresholds[0])
    active[np.flatnonzero(veto), probability[veto].argmin(axis=1)] = False
    active[(count >= 1) & (minimum >= thresholds[1])] = True
    return active


def development(records, scores):
    best = None
    for reject in (0.12, 0.22, 0.32, 0.42, 0.52, 0.62, 0.72):
        for promote in (0.55, 0.65, 0.75, 0.85, 0.95, 1.01):
            if promote < reject:
                continue
            groups = [[], []]
            for (sample, mask), prob in zip(records, scores):
                candidate = infer(sample, prob, (reject, promote))
                chosen = sample.actual[mask]
                selected = Sample(sample.name, sample.session, None, chosen, None, None)
                groups[int(sample.name == LONG_DEV)].append(
                    counts(selected, candidate[mask])
                )
            stats = [measures(np.sum(group, axis=0)) for group in groups]
            objective = np.mean(
                [
                    m["exact_three_F1"]
                    + 0.25 * m["three_way_speaker_recall"]
                    - 0.5 * m["DER"]
                    for m in stats
                ]
            )
            if best is None or objective > best[0]:
                best = (objective, (reject, promote))
    return best[1]


def train(args, profile):
    clips = list(manifest_samples(args.dev_manifest, args.dev_probs, profile))
    long = train_long(args.work, args.views, LONG_DEV, profile)
    tuning = set(sorted({s.session for s in clips})[::4])
    clips_train = [(s, slice(None)) for s in clips if s.session not in tuning]
    clips_val = [(s, slice(None)) for s in clips if s.session in tuning]
    # Reserve every third minute of the long development recording for threshold tuning.
    long_train = (np.arange(len(long.previous)) // 6000) % 3 != 2
    model = fit(clips_train + [(long, long_train[::STRIDE])])
    val = clips_val + [(long, ~long_train)]
    thresholds = development(val, [evidence(model, s) for s, _ in val])
    final = fit([(s, slice(None)) for s in [*clips, long]])
    joblib.dump(
        {"model": final, "thresholds": thresholds, "profile": profile}, args.bundle
    )
    print("Frozen thresholds", thresholds, flush=True)


def evaluate(samples, bundle):
    totals = {}
    details = []
    for s in samples:
        activity = infer(s, evidence(bundle["model"], s), bundle["thresholds"])
        stats = {}
        for key, pred in (("previous", s.previous), ("candidate", activity)):
            tally = counts(s, pred)
            totals[key] = totals.get(key, 0) + tally
            stats[key] = measures(tally)
        details.append({"recording": s.name, **stats})
    return {
        "pooled": {k: measures(v) for k, v in totals.items()},
        "recordings": details,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--profile", type=Path)
    p.add_argument("--dev-manifest", type=Path)
    p.add_argument("--dev-probs", type=Path)
    p.add_argument("--work", type=Path)
    p.add_argument("--views", type=Path)
    p.add_argument(
        "--test",
        action="append",
        nargs=3,
        default=[],
        metavar=("NAME", "MANIFEST", "PROBS"),
    )
    p.add_argument("--long", nargs="+")
    p.add_argument("--predict", nargs=2, metavar=("PROBABILITIES", "TIMING_JSON"))
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    with threadpool_limits(2):
        if args.dev_manifest:
            train(args, json.loads(args.profile.read_text()))
        bundle = joblib.load(args.bundle)
        profile = bundle["profile"]
        reports = {}
        for name, manifest, posterior in args.test:
            reports[name] = evaluate(
                manifest_samples(Path(manifest), Path(posterior), profile), bundle
            )
        if args.long:
            reports["long"] = evaluate(
                (train_long(args.work, args.views, n, profile) for n in args.long),
                bundle,
            )
        if args.predict:
            archive, out = map(Path, args.predict)
            sample = from_cache(archive, profile)
            active = infer(
                sample, evidence(bundle["model"], sample), bundle["thresholds"]
            )
            rows = to_segments(active, sample.duration)
            for row in rows:
                row["speaker"] = "ABC"[int(row["speaker"])]
            out.write_text(
                json.dumps({"duration_s": sample.duration, "segments": rows}, indent=2)
                + "\n"
            )
        if reports:
            args.report.write_text(
                json.dumps(
                    {"thresholds": list(bundle["thresholds"]), "benchmarks": reports},
                    indent=2,
                )
                + "\n"
            )
            print({k: v["pooled"] for k, v in reports.items()}, flush=True)


if __name__ == "__main__":
    main()
