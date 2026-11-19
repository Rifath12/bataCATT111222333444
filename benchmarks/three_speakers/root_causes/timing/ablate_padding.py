"""Isolate the contributor to turn-boundary bias by changing only the native decoder's dilation."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from diarize import masks, reference
from improve_overlap import Decoder
from long_views import meeting
from multiview_score import Clip, align, counts, fit_length, measures
from threadpoolctl import threadpool_limits


def arrays(path, probs, base):
    data = json.loads(path.read_text(encoding="utf8"))
    for row in data if isinstance(data, list) else data["clips"]:
        name = Path(row["audio"]).stem
        with np.load(probs / (name + ".npz")) as z:
            duration = float(z["duration_s"])
            actual = masks(
                reference(path.parent / (row.get("ref") or row["reference"])), duration
            )[1].T
            first = fit_length(z["base"], len(actual))
            native = np.zeros_like(first, dtype=bool)
            centers = (np.arange(len(actual)) + 0.5) * 0.01
            for left, right, slot in z["native"]:
                i, j = np.searchsorted(centers, [left, right])
                native[i:j, int(slot)] = True
            _, anchor, _ = align(first, first, native.sum(axis=0))
            current = Decoder(anchor).decode(base)
            yield Clip(name, "", actual, None, current, None, None, None, 0, 0), anchor


def totals(data, base, widths):
    sums = defaultdict(int)
    per = {}
    for clip, raw in data:
        decoder = Decoder(raw)
        per[clip.name] = {}
        for w in widths:
            time = decoder.decode(dict(base, expansion_frames=w))
            score = counts(clip, time)
            sums[w] += score
            per[clip.name][w] = score.tolist()
    return {w: measures(score) for w, score in sums.items()}, per


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--settings", type=Path, required=True)
    p.add_argument("--fit", action="store_true")
    p.add_argument("--test", nargs=3, action="append", default=[])
    p.add_argument("--long", nargs="*")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    base = json.loads(a.base.read_text())
    orig = base["expansion_frames"]

    def previous(name):
        row = meeting(a.work, a.views, name, base)
        return row[0], row[1]

    result = {}
    if a.fit:
        grid, _ = totals([previous("R8008_M8013")], base, range(1, orig + 1, 2))
        benchmark = grid[orig]
        usable = [
            (m["exact_three_F1"] - m["DER"], w)
            for w, m in grid.items()
            if m["three_way_speaker_recall"] + 0.025
            >= benchmark["three_way_speaker_recall"]
            and m["exact_three_recall"] + 0.05 >= benchmark["exact_three_recall"]
        ]
        w = max(usable)[1]
        a.settings.write_text(
            json.dumps(dict(base, expansion_frames=w), indent=2) + "\n"
        )
        result["development"] = {"all_widths": grid, "chosen": w}
    final = json.loads(a.settings.read_text())["expansion_frames"]
    for n, p, d in a.test:
        scores, by_recording = totals(
            arrays(Path(p), Path(d), base), base, [orig, final]
        )
        result[n] = {
            "previous": scores[orig],
            "corrected": scores[final],
            "raw_counts": by_recording,
        }
    if a.long:
        scores, pr = totals((previous(n) for n in a.long), base, [orig, final])
        result["repeated_long"] = {
            "previous": scores[orig],
            "corrected": scores[final],
            "raw_counts": pr,
        }
    a.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    with threadpool_limits(2):
        main()
