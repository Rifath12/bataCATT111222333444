"""Learn what each room's speakers sound like together by remixing predicted solo speech."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import ExtraTreesClassifier
from threadpoolctl import threadpool_limits

from confirm import LONG_DEV, counts, measures
from long_views import meeting

HOP = 640


def band_energy(audio):
    wave, rate = sf.read(audio, dtype="float32")
    if rate != 16000 or wave.ndim != 1:
        raise ValueError("Expected 16 kHz mono WAV")
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(wave, (320, 320)), 640)[
        ::HOP
    ]
    edges = np.unique(np.round(np.geomspace(2, 300, 39)).astype(int))
    window = np.hanning(640).astype("f")
    power = np.empty((len(frames), len(edges) - 1), dtype="f")
    for lo in range(0, len(frames), 2048):
        spectrum = np.abs(np.fft.rfft(frames[lo : lo + 2048] * window)) ** 2
        power[lo : lo + len(spectrum)] = np.stack(
            [spectrum[:, a:b].mean(axis=1) for a, b in zip(edges, edges[1:])], axis=1
        )
    return power


def transform(power):
    normalized = np.log(np.maximum(1e-8, power))
    return normalized - normalized.mean(axis=1, keepdims=True)


def verifier(audio, previous, first, second):
    bands = band_energy(audio)
    # Local averages mimic the short fluctuations in activity and spectral shape.
    features = np.stack([bands, uniform_filter1d(bands, 5, axis=0)], axis=1)
    length = min(len(features), len(previous[::4]))
    features = features[:length]
    previous = previous[: 4 * length : 4]
    both = np.minimum(first[: 4 * length : 4], second[: 4 * length : 4])
    rng = np.random.default_rng(2039)
    banks = []
    for s in range(3):
        qualifying = previous[:, s] & (previous.sum(axis=1) == 1)
        eligible = np.flatnonzero(
            qualifying & (uniform_filter1d(qualifying.astype("f"), 9) > 0.90)
        )
        if len(eligible) < 16:
            return None
        # Strong examples come only from solo regions inferred from the mixture itself.
        cutoff = np.quantile(both[eligible, s], 0.30)
        sample = features[eligible[both[eligible, s] >= cutoff]]
        banks.append(sample / np.maximum(1e-7, sample[:, 1].sum(axis=1))[:, None, None])
    x = []
    y = []
    for count in (1, 2, 3):
        for active in itertools.combinations(range(3), count):
            mixture = np.zeros((1400, 2, features.shape[-1]), dtype="f")
            for index in active:
                source = banks[index][
                    rng.integers(len(banks[index]), size=len(mixture))
                ]
                gain = 10 ** (rng.uniform(-7.0, 3.0, size=len(mixture)) / 10.0)
                mixture += source * gain[:, None, None]
            x.append(np.concatenate([transform(mixture[:, j]) for j in (0, 1)], axis=1))
            y.append(np.tile(np.isin(np.arange(3), active), (len(mixture), 1)))
    forest = ExtraTreesClassifier(
        n_estimators=30, max_depth=14, min_samples_leaf=9, n_jobs=1, random_state=83
    )
    forest.fit(np.concatenate(x), np.concatenate(y))
    test = np.concatenate([transform(features[:, j]) for j in (0, 1)], axis=1)
    prob = np.column_stack([v[:, 1] for v in forest.predict_proba(test)])
    prob = uniform_filter1d(prob, 5, axis=0)
    return prob


def make_row(audio, clip, first, second):
    probability = verifier(audio, clip.previous, first, second)
    if probability is None:
        return (clip, None)
    xp = np.arange(len(probability)) * 4
    dense = np.stack(
        [np.interp(np.arange(len(first)), xp, probability[:, j]) for j in range(3)],
        axis=1,
    )
    return (clip, dense)


def decode(row, cut, keep):
    clip, dense = row
    predicted = clip.previous.copy()
    if dense is None:
        return predicted
    weak = dense.argmin(axis=1)
    minimum = dense.min(axis=1)
    reject = (predicted.sum(axis=1) == 3) & (minimum < cut)
    predicted[np.flatnonzero(reject), weak[reject]] = False
    # Restrict the promotion to frames for which the native diarizer already has two voices.
    add = (
        (clip.native.sum(axis=1) >= 2) & (predicted.sum(axis=1) == 2) & (minimum > keep)
    )
    predicted[add] = True
    return predicted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--parameters", type=Path, required=True)
    p.add_argument("--fit", action="store_true")
    p.add_argument("--test", nargs="+")
    p.add_argument("--out", type=Path)
    p.add_argument("--fresh", type=Path)
    a = p.parse_args()
    profile = json.loads(a.profile.read_text())

    def acquire(name):
        if a.fresh and (a.fresh / name).is_dir():
            return fresh_row(a.fresh / name, profile)
        clip, first, second = meeting(a.work, a.views, name, profile)
        audio = (
            a.work / "session"
            if name == LONG_DEV
            else a.work / "independent_3speaker" / name
        ) / "mono.wav"
        return make_row(audio, clip, first, second)

    if a.fit:
        row = acquire(LONG_DEV)
        baseline = measures(counts(row[0], row[0].previous))
        candidates = []
        for cut in (0, 0.12, 0.20, 0.28, 0.36, 0.44):
            for keep in (0.43, 0.49, 0.55, 0.61, 0.67, 1.01):
                stats = measures(counts(row[0], decode(row, cut, keep)))
                if (
                    stats["exact_three_recall"] + 0.02 >= baseline["exact_three_recall"]
                    and stats["DER"] <= baseline["DER"] + 0.002
                ):
                    candidates.append(
                        (
                            stats["exact_three_F1"],
                            stats["three_way_speaker_recall"],
                            cut,
                            keep,
                        )
                    )
        best = max(candidates)
        a.parameters.write_text(
            json.dumps(
                {"cut": best[2], "keep": best[3], "development_F1": best[0]}, indent=2
            )
            + "\n"
        )
    if a.test:
        params = json.loads(a.parameters.read_text())
        rows = {}
        total = {}
        for name in a.test:
            row = acquire(name)
            tally = {
                "previous": counts(row[0], row[0].previous),
                "remix": counts(row[0], decode(row, params["cut"], params["keep"])),
            }
            rows[name] = {k: measures(v) for k, v in tally.items()}
            for k, v in tally.items():
                total[k] = total.get(k, 0) + v
        a.out.write_text(
            json.dumps(
                {
                    "pooled": {k: measures(v) for k, v in total.items()},
                    "sessions": rows,
                },
                indent=2,
            )
            + "\n"
        )


def fresh_row(folder, profile):
    from multiview_score import Clip, align, fit_length
    from diarize import reference, masks
    from improve_overlap import Decoder

    audio = folder / "mono.wav"
    sec = sf.info(audio).duration
    actual = masks(reference(folder / "reference.rttm"), sec)[1].T
    with np.load(folder / "probabilities.npz") as a, np.load(folder / "view.npz") as b:
        base = fit_length(a["probabilities"], len(actual))
        alt = fit_length(b["probabilities"][928:], len(actual))
    timing = np.zeros_like(base, dtype=bool)
    centers = (np.arange(len(actual)) + 0.5) * 0.01
    for turn in reference(folder / "native.rttm"):
        left, right = np.searchsorted(centers, [turn["start"], turn["end"]])
        timing[left:right, int(turn["speaker"].split("_")[-1]) - 1] = True
    ids, first, second = align(base, alt, timing.sum(axis=0))
    clip = Clip(
        folder.name,
        folder.name,
        actual,
        timing[:, ids],
        Decoder(first).decode(profile),
        None,
        None,
        None,
        0.0,
        0.0,
    )
    return make_row(audio, clip, first, second)


if __name__ == "__main__":
    with threadpool_limits(2):
        main()
