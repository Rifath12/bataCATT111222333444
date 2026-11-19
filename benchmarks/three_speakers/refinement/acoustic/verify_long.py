"""Verify three-way speaker detections using recording-specific acoustics and separate posterior passes."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.fft import rfft
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier
import soundfile as sf
from threadpoolctl import threadpool_limits

from long_views import meeting
from multiview_score import counts, measures

DEV = "R8008_M8013"
TEST = ("R8008_M8014", "R8008_M8015")


def spectral_templates(audio, length, first, second, candidate):
    """Estimate each talker's spectral dictionary from confident single-speaker intervals.

    Compare sparse spectral reconstruction using all dictionaries with reconstruction
    omitting each talker. All dictionary enrollment is unsupervised and file-local.
    """
    audio, sr = sf.read(audio, dtype="float32")
    if sr != 16000 or audio.ndim != 1:
        raise ValueError("Need mono 16 kHz audio.")
    frame = np.lib.stride_tricks.sliding_window_view(np.pad(audio, (256, 512)), 512)[
        ::160
    ]
    frame = frame[:length]
    if len(frame) < length:
        frame = np.pad(frame, ((0, length - len(frame)), (0, 0)))
    window = np.hanning(512).astype("float32")
    # Mild auditory compression and logarithmically spaced bands from 94 Hz to 8 kHz.
    edges = np.unique(np.rint(np.geomspace(3, 257, 62)).astype(int))
    spectra = np.empty((length, len(edges) - 1), np.float32)
    for start in range(0, length, 2048):
        part = frame[start : start + 2048]
        magnitude = np.abs(rfft(part * window, axis=1))
        sums = np.add.reduceat(magnitude, edges[:-1], axis=1)
        spectra[start : start + len(part)] = np.sqrt(sums / np.diff(edges))
    spectra /= np.maximum(np.linalg.norm(spectra, axis=1, keepdims=True), 1e-6)
    certainty = (first + second) * 0.5
    dictionaries = []
    for i in range(3):
        others = np.max(certainty[:, np.arange(3) != i], axis=1)
        confidence = certainty[:, i] - others
        anchors = np.flatnonzero((certainty[:, i] > 0.90) & (others < 0.07))[::5]
        if len(anchors) < 128:
            anchors = np.argsort(confidence)[-256:]
        if len(anchors) > 6000:
            anchors = anchors[np.linspace(0, len(anchors) - 1, 6000).astype(int)]
        centers = (
            MiniBatchKMeans(n_clusters=16, batch_size=256, n_init=2, random_state=71)
            .fit(spectra[anchors])
            .cluster_centers_
        )
        centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-6)
        dictionaries.append(centers)
    # Match pursuit on the candidate frames. Held-out reference audio is never an input.
    s = spectra[candidate]
    dic = np.concatenate(dictionaries)
    scores = s @ dic.T
    result = []
    for omit in (-1, 0, 1, 2):
        atoms = np.repeat(np.arange(3), 16) != omit
        q = scores[:, atoms].copy()
        gram = dic[atoms] @ dic[atoms].T
        residual = np.ones(len(q), np.float32)
        for _ in range(4):
            best = q.argmax(axis=1)
            val = np.maximum(0, q[np.arange(len(q)), best])
            residual -= val**2
            q -= val[:, None] * gram[best]
        result.append(residual)
    required = np.stack(result[1:], axis=1) - result[0][:, None]
    weak = np.argmin(certainty[candidate], axis=1)
    return np.column_stack(
        [
            result[0],
            np.sort(required, axis=1),
            required[np.arange(len(required)), weak],
            np.sort(scores.reshape(-1, 3, 16).max(2), axis=1),
        ]
    )


def evidence(clip, first, second, file):
    combined = (first + second) * 0.5
    min_support = np.maximum(first.min(1), second.min(1))
    candidate = (clip.previous.sum(1) >= 2) & (
        (maximum_filter1d(min_support, size=61) >= 0.025) | clip.previous.all(1)
    )
    arrays = []
    for posterior in (first, second, combined):
        ordered = np.sort(posterior, axis=1)
        for width in (1, 13, 43):
            smoothed = (
                uniform_filter1d(ordered, width, axis=0) if width > 1 else ordered
            )
            arrays.append(smoothed[candidate])
        arrays.append(maximum_filter1d(ordered[:, 0], size=35)[candidate, None])
        # Time contrast distinguishes a voice that persists through a candidate from adjacent turns.
        arrays.append(
            (uniform_filter1d(ordered[:, 0], 11) - uniform_filter1d(ordered[:, 0], 83))[
                candidate, None
            ]
        )
    arrays.append(clip.previous[candidate].sum(1)[:, None])
    arrays.append(spectral_templates(file, len(first), first, second, candidate))
    return candidate, np.column_stack(arrays).astype("float32")


def learner(x, y):
    n = np.bincount(y.astype(int), minlength=2)
    balance = np.sqrt(n.max() / np.maximum(n, 1))
    m = HistGradientBoostingClassifier(
        max_iter=70,
        learning_rate=0.07,
        max_leaf_nodes=7,
        min_samples_leaf=60,
        l2_regularization=8.0,
        early_stopping=False,
        random_state=161,
    )
    m.fit(x, y, sample_weight=balance[y.astype(int)])
    return m


def make_frames(clip, first, second, candidates, confidence, threshold):
    frames = clip.previous.copy()
    verified = np.zeros(len(frames), bool)
    keep, promote = threshold
    existing = clip.previous[candidates].all(1)
    verified[candidates] = confidence >= np.where(existing, keep, promote)
    drop = frames.all(1) & ~verified
    weakest = np.argmin(0.65 * first + 0.35 * second, axis=1)
    frames[np.flatnonzero(drop), weakest[drop]] = False
    frames[verified] = True
    return frames


def optimize(clip, first, second, mask, x):
    # Fold entire 30 s intervals and purge one second near training–validation boundaries.
    # Every threshold is selected before either long test recording is opened.
    ids = np.flatnonzero(mask)
    groups = (ids // 3000) % 4
    dev_target = clip.actual[mask].all(1)
    probs = np.zeros(len(x))
    for fold in range(4):
        fit = (groups != fold) & ((ids % 3000) > 100) & ((ids % 3000) < 2900)
        if np.unique(dev_target[fit]).size < 2:
            raise ValueError("Too few distinct positive development events.")
        m = learner(x[fit], dev_target[fit])
        probs[groups == fold] = m.predict_proba(x[groups == fold])[:, 1]
    base = measures(counts(clip, clip.previous))
    # A model may overpredict third voices when applied to low-prior pairs.
    # Use distinct thresholds for retaining the old triple call and upgrading a pair.
    # Each grid candidate differs only on the proposal mask, so sparse counts suffice.
    from improve_overlap import match

    actual = match(clip.actual, clip.previous)
    pred = clip.previous[mask]
    target = actual[mask]
    base_raw = counts(clip, clip.previous)
    existing = pred.all(1)
    weak = np.argmin(0.65 * first[mask] + 0.35 * second[mask], axis=1)
    unverified = pred.copy()
    unverified[np.flatnonzero(existing), weak[existing]] = False

    def sparse_totals(frames):
        ref = target.sum(1)
        est = frames.sum(1)
        good = (target & frames).sum(1)
        trip = ref == 3
        called = est == 3
        return np.array(
            [
                ref.sum(),
                est.sum(),
                good.sum(),
                (np.maximum(ref, est) - good).sum(),
                trip.sum(),
                called.sum(),
                (trip & called).sum(),
                good[trip].sum(),
                len(est),
            ]
        )

    outside = base_raw - sparse_totals(pred)
    options = []
    for keep in np.r_[0.0, np.geomspace(0.002, 0.5, 16)]:
        for promote in np.r_[np.geomspace(max(0.03, keep), 0.97, 12), 1.0]:
            estimate = unverified.copy()
            estimate[probs >= np.where(existing, keep, promote)] = True
            score = measures(outside + sparse_totals(estimate))
            if (
                score["exact_three_precision"] >= base["exact_three_precision"]
                and score["exact_three_recall"] >= base["exact_three_recall"] * 0.82
                and score["DER"] <= base["DER"] + 0.002
                and score["exact_three_F1"] > base["exact_three_F1"]
            ):
                objective = (
                    score["exact_three_F1"]
                    + 0.30 * score["three_way_speaker_recall"]
                    - 0.25 * score["DER"]
                )
                options.append((objective, (float(keep), float(promote)), score))
    if not options:
        return None, None, base, base
    _, threshold, oof = max(options, key=lambda a: a[0])
    return learner(x, dev_target), threshold, oof, base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = json.loads(
        (args.work / "independent_3speaker/profile_balanced.json").read_text()
    )

    def get(session):
        data = meeting(args.work, args.views, session, profile)
        file = args.work / (
            "session/mono.wav"
            if session == DEV
            else f"independent_3speaker/{session}/mono.wav"
        )
        return (*data, *evidence(*data, file))

    with threadpool_limits(2):
        started = time.perf_counter()
        dev = get(DEV)
        model, threshold, oof, baseline = optimize(*dev)
        print("OOF", threshold, oof, flush=True)
        report = {
            "development_session": DEV,
            "oof_candidate": oof,
            "dev_baseline": baseline,
            "threshold": threshold,
            "holdout": {},
            "pooled": {},
        }
        pooled = {}
        for session in TEST:
            then = time.perf_counter()
            clip, first, second, mask, x = get(session)
            cand = (
                make_frames(
                    clip, first, second, mask, model.predict_proba(x)[:, 1], threshold
                )
                if model is not None
                else clip.previous
            )
            raw = {
                "native": counts(clip, clip.native),
                "previous": counts(clip, clip.previous),
                "verified": counts(clip, cand),
            }
            report["holdout"][session] = {name: measures(v) for name, v in raw.items()}
            report["holdout"][session]["postprocess_s"] = time.perf_counter() - then
            for name, val in raw.items():
                pooled[name] = pooled.get(name, 0) + val
        report["pooled"] = {name: measures(v) for name, v in pooled.items()}
        report["total_postprocess_and_fit_s"] = time.perf_counter() - started
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(report["pooled"], flush=True)


if __name__ == "__main__":
    main()
