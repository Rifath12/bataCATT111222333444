"""Summarize every timing metric and the all-metric promotion gate."""

import argparse
import json
from pathlib import Path

METRICS = (
    "exact_three_precision",
    "exact_three_recall",
    "exact_three_F1",
    "DER",
    "speaker_precision",
    "speaker_recall",
    "three_way_speaker_recall",
)


def passes(candidate, comparator):
    return all(
        candidate[k] <= comparator[k] if k == "DER" else candidate[k] >= comparator[k]
        for k in METRICS
    ) and any(candidate[k] != comparator[k] for k in METRICS)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("inputs", nargs="+", type=Path)
    a = p.parse_args()
    summary = {}
    for source in a.inputs:
        report = json.loads(source.read_text())
        for panel, results in report.get("holdout", {}).items():
            pool = {k: row["pooled"] for k, row in results.items()}
            gate = {
                k: passes(pool["candidate"], row)
                for k, row in pool.items()
                if k != "candidate"
            }
            summary[source.stem] = dict(
                manifest=panel,
                pooled=pool,
                candidate_dominates=gate,
                candidate_passes_all=all(gate.values()),
            )
            print(source.stem)
            for arm, row in pool.items():
                print(arm, *(round(row[k] * 100, 2) for k in METRICS))
            print("Strictly improves at least one metric and regresses none", gate)
    a.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
