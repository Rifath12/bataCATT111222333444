"""Cache each single-pass separator output and known synthetic reference for evaluation."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from benchmark_waveform import GROUPS, fixtures
from prepare_validation_cases import prepared_source
from sepformer_three import SepFormer

FRESH = [("0009", "0011", "0012"), ("0016", "0033", "0043"), ("0057", "0080", "0112")]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--fresh-sources", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model = SepFormer(a.weights)
    GROUPS["fresh"] = FRESH
    a.out.mkdir(parents=True, exist_ok=True)
    for phase in ("development", "test", "fresh"):
        rows = []
        for name, condition, refs, mix, provenance in fixtures(
            phase, a.fresh_sources if phase == "fresh" else a.sources, prepared_source
        ):
            path = a.out / (name + ".npz")
            if not path.exists():
                began = time.perf_counter()
                spec = model(mix)
                wall = time.perf_counter() - began
                np.savez(path, tracks=spec, mixture=mix, refs=refs, wall=wall)
            rows.append(dict(name=name, condition=condition, source_files=provenance))
            print(name, flush=True)
        (a.out / f"{phase}.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
