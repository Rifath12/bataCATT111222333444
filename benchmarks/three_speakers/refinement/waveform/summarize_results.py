"""Summarize paired isolation changes and observed CPU time by triplet."""

import json
from pathlib import Path
import argparse
import numpy as np


def summarize(record, baseline, variant):
    rows = record["cases"]
    rng = np.random.default_rng(197)
    n = len(rows) // 2
    boot = rng.integers(n, size=(3000, n))
    output = {}
    for condition in ("room", "clean", "pooled"):
        cases = (
            rows
            if condition == "pooled"
            else [r for r in rows if r["condition"] == condition]
        )
        scores = {}
        for key in ("si_sdri_db", "weakest_si_sdri_db", "whole_si_sdri_db"):

            def matrix(name):
                return np.stack(
                    [np.atleast_1d(r["methods"][name][key]) for r in cases]
                ).reshape(n, -1)

            a, b = matrix(baseline), matrix(variant)
            delta = np.median(b[boot], axis=(1, 2)) - np.median(a[boot], axis=(1, 2))
            scores[key] = {
                "baseline": float(np.median(a)),
                "candidate": float(np.median(b)),
                "delta": float(np.median(b) - np.median(a)),
                "triplet_bootstrap_95": np.quantile(delta, [0.025, 0.975]).tolist(),
            }
        base_s = sum(r["base_s"] for r in cases)
        scores["base_s"] = base_s
        scores["post_s"] = sum(r["methods"][variant]["post_s"] for r in cases)
        scores["reference_triple_s"] = sum(r["reference_triple_s"] for r in cases)
        scores["input_s"] = sum(r["input_s"] for r in cases)
        scores["clip_medians_improved"] = sum(
            bool(
                np.median(r["methods"][variant]["si_sdri_db"])
                > np.median(r["methods"][baseline]["si_sdri_db"])
            )
            for r in cases
        )
        output[condition] = scores
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    a = p.parse_args()
    summary = {}
    for phase in ("test", "fresh"):
        summary[phase] = {}
        for label, name, before, after in (
            (
                "frequency_fit",
                phase,
                "single_pass",
                json.loads((a.results / "selection.json").read_text())["method"],
            ),
            (
                "spectral_shrinkage",
                "shrinkage_" + phase,
                "0",
                str(
                    json.loads((a.results / "shrinkage_selection.json").read_text())[
                        "strength"
                    ]
                ),
            ),
        ):
            summary[phase][label] = summarize(
                json.loads((a.results / (name + ".json")).read_text()), before, after
            )
    (a.results / "paired_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
