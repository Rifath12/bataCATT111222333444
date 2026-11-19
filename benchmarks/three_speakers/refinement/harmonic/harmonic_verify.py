"""Verify three-speaker activity with harmonic pursuit, inferred solo segments and a count model."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier
import soundfile as sf
from threadpoolctl import threadpool_limits

from diarize import masks, reference
from improve_overlap import Decoder
from multiview_score import Clip, align, counts, feature_matrix, fit_length, measures

RATE = 16000
FFT = 2048
WINDOW = 1024
HOP = 320


def pitch_dictionary():
    """Unit-energy narrow harmonic combs at half-semitone intervals."""
    pitch = 67.0 * 2 ** (np.arange(132) / 48.0)
    hz = np.arange(FFT // 2 + 1) * RATE / FFT
    templates = np.zeros((len(hz), len(pitch)))
    for h in range(1, 10):
        center = h * pitch
        width = np.maximum(12.0, 0.010 * center)
        templates += np.exp(-0.5 * ((hz[:, None] - center) / width) ** 2) / np.sqrt(h)
    templates /= np.linalg.norm(templates, axis=0)
    return pitch, templates.astype(np.float32)


PITCH, DICTIONARY = pitch_dictionary()
GRAM = DICTIONARY.T @ DICTIONARY


def analyze_spectrum(wave):
    """Pursue three distinct harmonic templates in an envelope-whitened short-time spectrum."""
    framed = np.lib.stride_tricks.sliding_window_view(
        np.pad(wave, (WINDOW // 2, WINDOW // 2)), WINDOW
    )[::HOP]
    features = []
    pitches = []
    window = np.hanning(WINDOW).astype(np.float32)
    for begin in range(0, len(framed), 320):
        mag = np.abs(np.fft.rfft(framed[begin : begin + 320] * window, n=FFT)).astype(
            np.float32
        )
        floor = (np.max(mag, axis=1, keepdims=True) * 1e-3).clip(1e-7)
        log_mag = np.log(mag + floor)
        residual = (log_mag - gaussian_filter1d(log_mag, 6, axis=1)).clip(0, 3.0)
        residual /= np.maximum(np.sqrt((residual**2).sum(axis=1, keepdims=True)), 1e-7)
        corr = residual @ DICTIONARY
        energy = []
        selected = []
        for _ in range(3):
            idx = corr.argmax(axis=1)
            score = np.maximum(0, np.take_along_axis(corr, idx[:, None], axis=1)[:, 0])
            selected.append(idx)
            energy.append(score)
            corr -= 0.85 * score[:, None] * GRAM[idx]
            corr[np.abs(np.arange(len(PITCH)) - idx[:, None]) <= 4] = -1
        f = np.stack(selected, axis=1)
        e = np.stack(energy, axis=1)
        distances = np.sort(np.abs(f[:, [0, 0, 1]] - f[:, [1, 2, 2]]), axis=1) / 48.0
        features.append(
            np.column_stack(
                [
                    e,
                    e[:, 1:] / np.maximum(0.01, e[:, :1]),
                    distances,
                    np.sort(f, axis=1) / 48.0,
                ]
            )
        )
        pitches.append(f)
    return np.concatenate(features).astype(np.float32), np.concatenate(pitches)


def get_features(wave, base, second, previous):
    fused = 0.55 * base + 0.25 * second + 0.2 * np.maximum(base, second)
    acoustic, pitches = analyze_spectrum(wave)
    n = len(base)
    # Some audio durations do not land exactly on the 10 ms scoring grid.
    grid = np.minimum(np.arange(n) // 2, len(acoustic) - 1)
    on_pitch_grid = np.minimum(np.arange(len(acoustic)) * 2, n - 1)
    average_pitch = np.mean(acoustic[:, 0]) + 1e-7
    solo_fused = fused[on_pitch_grid]
    evidence = []
    for speaker in range(3):
        others = np.delete(solo_fused, speaker, axis=1).max(axis=1)
        weight = (solo_fused[:, speaker] - others - 0.15).clip(0) ** 2
        weight *= acoustic[:, 0] / average_pitch
        mass = np.bincount(pitches[:, 0], weights=weight, minlength=len(PITCH))
        prior = gaussian_filter1d(mass + 0.2, 5, mode="nearest")
        prior /= prior.max()
        evidence.append(prior[pitches] * acoustic[:, :3])
    evidence = np.stack(evidence, axis=1)
    # Match the three proposed pitches to the three speaker distributions.
    import itertools

    scores = [
        sum(evidence[:, s, i] for s, i in enumerate(perm))
        for perm in itertools.permutations(range(3))
    ]
    best = np.stack(scores, axis=1).max(axis=1)
    acoustic = np.column_stack([acoustic, best, np.sort(evidence.max(axis=2), axis=1)])[
        grid
    ]
    acoustic = np.concatenate(
        [acoustic, uniform_filter1d(acoustic, 19, axis=0)], axis=1
    )
    counts0 = previous.sum(axis=1).astype(np.float32)
    post = feature_matrix(base, second)
    extras = [
        uniform_filter1d((counts0 >= k).astype(np.float32), w)
        for w in (1, 19, 75)
        for k in (2, 3)
    ]
    return fused, np.column_stack([post, *extras, acoustic]).astype(np.float32)


@dataclass
class Case:
    clip: Clip
    features: np.ndarray
    corpus: str


def combine(stem, session, audio, gt, base, alt, native, profile, corpus):
    selected, base, alt = align(base, alt, native.sum(axis=0))
    previous = Decoder(base).decode(profile)
    wave, rate = sf.read(audio, dtype="float32")
    if rate != RATE or wave.ndim != 1:
        raise ValueError(audio)
    fused, features = get_features(wave, base, alt, previous)
    clip = Clip(
        stem, session, gt, native[:, selected], previous, None, fused, None, 0, 0
    )
    return Case(clip, features, corpus)


def training_rows(cases, pitch):
    features, targets, weights = [], [], []
    for c in cases:
        gt = c.clip.actual.sum(axis=1) == 3
        # Keep correlated positives at 30 ms, negatives at 70 ms; balance only by corpus.
        index = np.unique(np.r_[np.flatnonzero(gt)[::3], np.flatnonzero(~gt)[::7]])
        features.append(c.features[index, : None if pitch else 54])
        targets.append(gt[index])
        weights.append(np.full(len(index), 0.35 if c.corpus == "long" else 1.0))
    return np.concatenate(features), np.concatenate(targets), np.concatenate(weights)


def fit(cases, pitch):
    x, y, w = training_rows(cases, pitch)
    model = HistGradientBoostingClassifier(
        max_iter=95,
        learning_rate=0.065,
        max_leaf_nodes=9,
        min_samples_leaf=40,
        l2_regularization=7.0,
        early_stopping=False,
        random_state=31,
    )
    model.fit(x, y, sample_weight=w)
    return model


def probability(model, case, pitch):
    x = case.features[:, : None if pitch else 54]
    return uniform_filter1d(model.predict_proba(x)[:, 1], 13)


def gate(clip, prob, threshold):
    out = clip.previous.copy()
    mask = prob >= threshold
    drop = ~mask & out.all(axis=1)
    out[drop, clip.fused[drop].argmin(axis=1)] = False
    out[mask] = True
    return out


def choose(dev_predictions):
    best = (-np.inf, 1)
    for tau in np.geomspace(0.014, 0.7, 34):
        stats = measures(
            sum(counts(c.clip, gate(c.clip, p, tau)) for c, p in dev_predictions)
        )
        objective = (
            stats["exact_three_F1"]
            + 0.15 * stats["three_way_speaker_recall"]
            - 0.35 * stats["DER"]
        )
        best = max(best, (objective, tau))
    return best[1]


def guarded_mask(clip, probability, keep, add):
    active = clip.previous.sum(axis=1)
    return ((active == 3) & (probability >= keep)) | (
        (active == 2) & (probability >= add)
    )


def guarded_thresholds(real_oof, long_oof):
    # A single rule must respect the earlier decoder's precision independently
    # on overlap-enriched clips and unselected long-meeting development.
    groups = []
    for group in [real_oof, long_oof]:
        ref = np.concatenate([c.clip.actual.sum(axis=1) == 3 for c, _ in group])
        prev = np.concatenate([c.clip.previous.sum(axis=1) for c, _ in group])
        prob = np.concatenate([p for _, p in group])

        def stats(pred, ref=ref):
            tp = np.count_nonzero(pred & ref)
            return (
                tp / max(1, pred.sum()),
                tp / max(1, ref.sum()),
                2 * tp / max(1, pred.sum() + ref.sum()),
            )

        groups.append((ref, prev, prob, stats, stats(prev == 3)))
    decision = (-1.0, 0.0, 1.01)
    values = (
        0.0,
        0.01,
        0.025,
        0.05,
        0.075,
        0.10,
        0.15,
        0.20,
        0.30,
        0.45,
        0.65,
        0.85,
        1.01,
    )
    for keep in values:
        for add in values:
            if keep > add:
                continue
            improvements = []
            for ref, prev, prob, stats, baseline in groups:
                pred = ((prev == 3) & (prob >= keep)) | ((prev == 2) & (prob >= add))
                now = stats(pred)
                if now[0] + 1e-12 < baseline[0] or now[1] + 0.035 < baseline[1]:
                    break
                improvements.append(now[2] - baseline[2])
            if len(improvements) == len(groups):
                score = min(improvements) + 0.5 * np.mean(improvements)
                if score > decision[0]:
                    decision = (score, keep, add)
    return decision[1:]


def run(args):
    profile = json.loads(args.profile.read_text())
    real = args.realt
    prev = args.previous

    def from_manifest(manifest, probs, corpus):
        data = json.loads(manifest.read_text())
        rows = data if isinstance(data, list) else data["clips"]
        cases = []
        for r in rows:
            audio = manifest.parent / r["audio"]
            ref = manifest.parent / (r.get("ref") or r["reference"])
            with np.load(probs / (audio.stem + ".npz")) as z:
                actual = masks(reference(ref), float(z["duration_s"]))[1].T
                n = len(actual)
                base = fit_length(z["base"], n)
                second = fit_length(z["alternate"][int(z["offset"]) :], n)
                native = np.zeros_like(base, dtype=bool)
                centers = (np.arange(n) + 0.5) * 0.01
                for start, end, index in z["native"]:
                    lo, hi = np.searchsorted(centers, [start, end])
                    native[lo:hi, int(index)] = True
                cases.append(
                    combine(
                        audio.stem,
                        r.get("session") or r["source_session"],
                        audio,
                        actual,
                        base,
                        second,
                        native,
                        profile,
                        corpus,
                    )
                )
        return cases

    def long_case(session):
        if session == "R8008_M8013":
            audio = prev / "session/mono.wav"
            ref = prev / f"{session}.rttm"
            base = prev / "nemotron_probabilities.npz"
            native = prev / "nemotron_raw.rttm"
        else:
            folder = prev / "independent_3speaker" / session
            audio = folder / "mono.wav"
            ref = (
                prev
                / "independent_3speaker"
                / "references/Test_Ali_far"
                / f"{session}.rttm"
            )
            base = folder / "probabilities.npz"
            native = folder / "native.rttm"
        actual = masks(reference(ref), sf.info(audio).duration)[1].T
        n = len(actual)
        with np.load(base) as z:
            base = fit_length(z["probabilities"], n)
        with np.load(args.cache / session / "view.npz") as z:
            alt = fit_length(z["probabilities"][928:], n)
        activity = np.zeros_like(base, dtype=bool)
        centers = (np.arange(n) + 0.5) * 0.01
        for seg in reference(native):
            lo, hi = np.searchsorted(centers, [seg["start"], seg["end"]])
            activity[lo:hi, int(seg["speaker"].rsplit("_", 1)[1]) - 1] = True
        return combine(
            session, session, audio, actual, base, alt, activity, profile, "long"
        )

    train = from_manifest(
        real / "development_three_participant_sessions.json",
        args.cache / "realt_probs",
        "realt",
    )
    print("development REAL-T features ready", flush=True)
    long_dev = long_case("R8008_M8013")
    print("development long features ready", flush=True)
    dev_sessions = sorted({c.clip.session for c in train})
    models = {}
    for pitch in (False, True):
        oof = []
        for fold in range(3):
            val = set(dev_sessions[fold::3])
            m = fit([c for c in train if c.clip.session not in val] + [long_dev], pitch)
            oof.extend(
                (c, probability(m, c, pitch)) for c in train if c.clip.session in val
            )
        tau = choose(oof)
        guards = guarded_thresholds(
            oof, [(long_dev, probability(fit(train, pitch), long_dev, pitch))]
        )
        models[pitch] = fit(train + [long_dev], pitch), tau, guards
        print("frozen", pitch, tau, guards, flush=True)
    result = {
        "selection": {
            str(k): {"threshold": v[1], "guard_thresholds": v[2]}
            for k, v in models.items()
        },
        "scores": {},
    }

    def evaluate(name, cases):
        sums = {}
        per_session = {}
        for case in cases:
            c = case.clip
            arms = {"previous": c.previous, "native": c.native}
            for key, pitch in [("posterior", False), ("posterior_and_pitch", True)]:
                model, tau, guards = models[pitch]
                prob = probability(model, case, pitch)
                arms[key] = gate(c, prob, tau)
                mask = guarded_mask(c, prob, *guards)
                arms[key + "_guarded"] = gate(c, mask, 0.5)
            for arm, estimated in arms.items():
                values = counts(c, estimated)
                sums[arm] = sums.get(arm, 0) + values
                sess = per_session.setdefault(c.session, {})
                sess[arm] = sess.get(arm, 0) + values
        result["scores"][name] = {
            "pooled": {arm: measures(raw) for arm, raw in sums.items()},
            "session_counts": {
                name: {k: v.tolist() for k, v in a.items()}
                for name, a in per_session.items()
            },
        }
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(name, result["scores"][name]["pooled"]["posterior_and_pitch"], flush=True)

    evaluate(
        "real", from_manifest(real / "heldout.json", args.cache / "realt_probs", "real")
    )
    evaluate(
        "ali",
        from_manifest(args.ali / "manifest.json", args.cache / "ali_probs", "ali"),
    )
    evaluate("long", [long_case(s) for s in ["R8008_M8014", "R8008_M8015"]])
    if args.fresh_manifest:
        evaluate("fresh", from_manifest(args.fresh_manifest, args.fresh_probs, "fresh"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("profile", "realt", "cache", "ali", "previous", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--fresh-manifest", type=Path)
    p.add_argument("--fresh-probs", type=Path)
    with threadpool_limits(2):
        run(p.parse_args())
