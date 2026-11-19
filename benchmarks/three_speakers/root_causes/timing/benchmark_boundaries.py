"""Diagnose and correct boundary errors on fixed Mandarin speaker-timing benchmarks."""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment

from diarize import reference, masks
from improve_overlap import Decoder
from long_views import meeting
from multiview_score import align, counts, fit_length, measures
from turn_boundaries import runs, correct


@dataclass
class Recording:
    name: str
    actual: np.ndarray
    previous: np.ndarray


def short(manifest, posterior, profile):
    data = json.loads(manifest.read_text(encoding="utf-8"))
    for entry in data if isinstance(data, list) else data["clips"]:
        name = Path(entry["audio"]).stem
        with np.load(posterior / (name + ".npz")) as a:
            duration = float(a["duration_s"])
            actual = masks(
                reference(manifest.parent / (entry.get("ref") or entry["reference"])),
                duration,
            )[1].T
            first = fit_length(a["base"], len(actual))
            native = np.zeros_like(first, dtype=bool)
            centers = (np.arange(len(actual)) + 0.5) * 0.01
            for left, right, slot in a["native"]:
                i, j = np.searchsorted(centers, [left, right])
                native[i:j, int(slot)] = True
            _, first, _ = align(first, first, native.sum(axis=0))
            yield Recording(name, actual, Decoder(first).decode(profile))


def stats(sample):
    actual, system = sample.actual, sample.previous
    i, j = linear_sum_assignment(-(actual.T.astype("f") @ system.astype("f")))
    actual = actual[:, i[np.argsort(j)]]
    lags = [[], []]
    for a, b in zip(actual.T, system.T):
        truth = list(runs(a))
        guess = list(runs(b))
        if not guess:
            continue
        for edge in (0, 1):
            ref = np.array([r[edge] for r in truth])
            hyp = np.array([r[edge] for r in guess])
            for r in ref:
                dist = hyp[np.argmin(abs(hyp - r))] - r
                if abs(dist) <= 75:
                    lags[edge].append(int(dist))
    real_triple = actual.all(1)
    predicted_triple = system.all(1)
    false = predicted_triple & ~real_triple
    transitions = np.any(
        np.diff(actual.astype(np.int8), axis=0, prepend=actual[:1]), axis=1
    )
    distance = (
        distance_transform_edt(~transitions)
        if transitions.any()
        else np.full(len(actual), np.inf)
    )
    causes = Counter(
        false_frames=int(false.sum()),
        over_real_pair=int(((actual.sum(1) == 2) & false).sum()),
        near_transition_300ms=int((false & (distance <= 30)).sum()),
        near_transition_150ms=int((false & (distance <= 15)).sum()),
    )
    for left, right in runs(predicted_triple):
        causes["predicted_three_runs"] += 1
        if not real_triple[left:right].any():
            causes["pure_false_runs"] += 1
    return lags, causes


def diagnostics(samples):
    shifts = [[], []]
    events = Counter()
    for s in samples:
        a, b = stats(s)
        for i in (0, 1):
            shifts[i].extend(a[i])
        events.update(b)
    return {
        "median_onset_error_ms": int(round(np.median(shifts[0]) * 10)),
        "median_end_error_ms": int(round(np.median(shifts[1]) * 10)),
        "onsets_matched": len(shifts[0]),
        "offsets_matched": len(shifts[1]),
        **dict(events),
    }


def pooled(samples, config):
    raws = [counts(s, correct(s.previous, config)) for s in samples]
    return measures(sum(raws))


def evaluate(samples, chosen):
    arms = {
        "previous": dict(onset_frames=0, offset_frames=0, handoff_frames=0),
        "boundary_only": dict(chosen, handoff_frames=0),
        "corrected": chosen,
    }
    return {
        "counts": len(samples),
        "diagnostics": diagnostics(samples),
        "pooled": {k: pooled(samples, v) for k, v in arms.items()},
        "per_recording": [
            {
                "name": s.name,
                **{
                    k: counts(s, correct(s.previous, v)).tolist()
                    for k, v in arms.items()
                },
            }
            for s in samples
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--previous-profile", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--fit", action="store_true")
    p.add_argument(
        "--test",
        action="append",
        nargs=3,
        default=[],
        metavar=("name", "manifest", "posterior"),
    )
    p.add_argument("--long", nargs="*")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    old = json.loads(a.previous_profile.read_text())
    if a.fit:
        dev = [meeting(a.work, a.views, "R8008_M8013", old)[0]]
        baseline = measures(counts(dev[0], dev[0].previous))
        diag = diagnostics(dev)
        early = int(np.clip(round(-diag["median_onset_error_ms"] / 10), 0, 12))
        late = int(np.clip(round(diag["median_end_error_ms"] / 10), 0, 12))
        grid = []
        # Estimate the width only on the half-hour development recording.
        for factor in (0, 0.5, 1, 1.5):
            for handoff in (0, 3, 5, 8, 12, 18):
                config = dict(
                    onset_frames=round(early * factor),
                    offset_frames=round(late * factor),
                    handoff_frames=handoff,
                )
                m = pooled(dev, config)
                if (
                    m["three_way_speaker_recall"] + 0.025
                    >= baseline["three_way_speaker_recall"]
                    and m["exact_three_recall"] + 0.05 >= baseline["exact_three_recall"]
                ):
                    grid.append((m["exact_three_F1"] - 0.75 * m["DER"], config, m))
        best = max(grid, key=lambda row: row[0])
        a.profile.write_text(json.dumps(best[1], indent=2) + "\n")
        print("frozen", best[1], baseline["DER"], best[2]["DER"], flush=True)
    chosen = json.loads(a.profile.read_text())
    if a.out:
        groups = {"development": [meeting(a.work, a.views, "R8008_M8013", old)[0]]}
        if a.long:
            groups["repeated_long"] = [
                meeting(a.work, a.views, n, old)[0] for n in a.long
            ]
        for name, manifest, probs in a.test:
            groups[name] = list(short(Path(manifest), Path(probs), old))
        data = {n: evaluate(group, chosen) for n, group in groups.items()}
        a.out.write_text(json.dumps(data, indent=2) + "\n")
        print(
            {
                n: {
                    k: [
                        round(100 * m[p], 2)
                        for p in (
                            "exact_three_precision",
                            "exact_three_recall",
                            "three_way_speaker_recall",
                            "exact_three_F1",
                            "DER",
                        )
                    ]
                    for k, m in v["pooled"].items()
                }
                for n, v in data.items()
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
