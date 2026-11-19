"""Count duplicated output audio on predicted simultaneous speaker pairs."""

import argparse
from itertools import combinations
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from diarize import masks


def count_copies(samples, rhythm):
    ticks = min(samples.shape[1] // 160, rhythm.shape[1])
    frames = samples[:, : ticks * 160].reshape(len(samples), ticks, 160).astype("d")
    possible = present = exact = 0
    for a, b in combinations(range(len(frames)), 2):
        active = rhythm[a, :ticks] & rhythm[b, :ticks]
        possible += int(active.sum())
        left = frames[a, active]
        right = frames[b, active]
        keep = (np.mean(left**2, axis=1) > 1e-8) & (np.mean(right**2, axis=1) > 1e-8)
        present += int(keep.sum())
        exact += int(((left == right).all(axis=1) & keep).sum())
    return {
        "active_pair_ticks": possible,
        "both_nonsilent_ticks": present,
        "identical_ticks": exact,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("cases", "rendered", "out"):
        p.add_argument("--" + arg, type=Path, required=True)
    a = p.parse_args()
    records = []
    for r in json.loads((a.cases / "cases.json").read_text()):
        key = r["source"]
        src = a.cases / key
        row = {"source": key}
        for method in ("routing", "dprnn"):
            d = a.rendered / key / method
            samples = np.stack(
                [
                    sf.read(w, dtype="float32")[0]
                    for w in sorted(d.glob("speaker_*.wav"))
                ]
            )
            _, activity = masks(
                json.loads((src / "timing.json").read_text())["segments"],
                samples.shape[1] / 16000.0,
            )
            row[method] = count_copies(samples, activity)
        records.append(row)
    sums = {
        m: {k: sum(r[m][k] for r in records) for k in records[0][m]}
        for m in ("routing", "dprnn")
    }
    a.out.write_text(json.dumps({"summary": sums, "cases": records}, indent=2) + "\n")
    print(sums)


if __name__ == "__main__":
    main()
