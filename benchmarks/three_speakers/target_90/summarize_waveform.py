"""Summarize the frozen waveform comparison, keeping speaker-triplet pairs together."""

import argparse
import json
from pathlib import Path

import numpy as np


def estimate(rows, method, condition, metric):
    clips = [r for r in rows if condition == "pooled" or r["condition"] == condition]
    values = [np.atleast_1d(r["methods"][method][metric]) for r in clips]
    return float(np.median(np.concatenate(values)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    data = json.loads((args.results / "test.json").read_text())
    selected = json.loads((args.results / "selection.json").read_text())["method"]
    baseline = "libri_baseline"
    strata = {}
    for row in data["cases"]:
        key = row["name"].rsplit("_", 1)[0]
        strata.setdefault(key, []).append(row)
    speakers = list(strata.values())
    rng = np.random.default_rng(90)
    samples = rng.integers(0, len(speakers), size=(2000, len(speakers)))
    report = {}
    for condition in ("clean", "room", "pooled"):
        group = {}
        for metric in ("si_sdri_db", "weakest_si_sdri_db"):
            comparison = []
            for idx in samples:
                draw = [row for i in idx for row in speakers[i]]
                comparison.append(estimate(draw, selected, condition, metric) - estimate(draw, baseline, condition, metric))
            base_score = estimate(data["cases"], baseline, condition, metric)
            result_score = estimate(data["cases"], selected, condition, metric)
            group[metric] = {
                "baseline": base_score,
                "candidate": result_score,
                "improvement": result_score - base_score,
                "triplet_bootstrap_95_percent_interval": np.quantile(comparison, [0.025, 0.975]).tolist(),
            }
        report[condition] = group
    (args.results / "paired_summary.json").write_text(json.dumps({"method":selected,"triplets":len(speakers),"scores":report}, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
