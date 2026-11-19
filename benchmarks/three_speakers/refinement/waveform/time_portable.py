"""Time the deployed NumPy filter and the same single-pass neural separator."""

import argparse
import json
from pathlib import Path
import time
import numpy as np
from refine_three import SpectralRefiner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    portable = SpectralRefiner(a.model)
    records = []
    for phase in ("test", "fresh"):
        for row in json.loads((a.cache / f"{phase}.json").read_text()):
            with np.load(a.cache / (row["name"] + ".npz")) as data:
                mix, tracks = data["mixture"], data["tracks"]
                started = time.perf_counter()
                enhanced = portable(mix, tracks)
                elapsed = time.perf_counter() - started
            if not np.isfinite(enhanced).all():
                raise ValueError("Nonfinite output")
            records.append(
                dict(name=row["name"], input_s=len(mix) / 8000, portable_s=elapsed)
            )
    from sepformer_three import SepFormer
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    neural = SepFormer(a.weights)
    fresh = json.loads((a.cache / "fresh.json").read_text())
    # Same process, immediately following each neural inference; both timings exclude init.
    matched = []
    for row in fresh[:4]:
        with np.load(a.cache / (row["name"] + ".npz")) as data:
            mix = data["mixture"]
        first = time.perf_counter()
        tracks = neural(mix)
        second = time.perf_counter()
        _ = portable(mix, tracks)
        end = time.perf_counter()
        matched.append(
            dict(
                name=row["name"],
                neural_s=second - first,
                portable_s=end - second,
                input_s=len(mix) / 8000,
            )
        )
    report = {
        "cache_only": records,
        "matched": matched,
        "matched_relative_cost": sum(r["portable_s"] for r in matched)
        / sum(r["neural_s"] for r in matched),
    }
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(report["matched_relative_cost"], matched, flush=True)


if __name__ == "__main__":
    main()
