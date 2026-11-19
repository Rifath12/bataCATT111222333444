"""Check paired metric changes without averaging away regressions."""

import argparse
import json
import math
from pathlib import Path


def evaluate(scenario):
    results = []
    seen = set()
    for row in scenario["metrics"]:
        key = row["metric"]
        if key in seen:
            raise ValueError(f"Duplicate metric {key}")
        seen.add(key)
        before, after = row["baseline"], row["candidate"]
        if row["better"] not in ("higher", "lower"):
            raise ValueError(f"Missing improvement direction for {key}")
        for score in (before, after):
            if score is not None and (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
            ):
                raise ValueError(f"Invalid score for {key}")
        delta = None if before is None or after is None else after - before
        if delta is None:
            result = "unverified"
        elif delta == 0:
            result = "unchanged"
        else:
            improves = delta > 0 if row["better"] == "higher" else delta < 0
            result = "improved" if improves else "regressed"
        results.append(dict(row, delta=delta, result=result))

    if not results:
        raise ValueError("Provide at least one paired metric")
    outcome = {r["result"] for r in results}
    return {
        **scenario,
        "metrics": results,
        "all_improved": outcome == {"improved"},
        "non_degrading_improvement": "improved" in outcome
        and outcome <= {"improved", "unchanged"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path, help="JSON list of paired evaluations")
    args = parser.parse_args()
    rows = json.loads(args.matrix.read_text(encoding="utf-8"))
    print(json.dumps([evaluate(r) for r in rows], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
