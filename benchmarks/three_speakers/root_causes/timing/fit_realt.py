"""Fit a separate REAL-T boundary profile on labeled development excerpts."""

import argparse
import json
from pathlib import Path

from threadpoolctl import threadpool_limits
from ablate_padding import arrays, totals


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--posteriors", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--settings", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    base = json.loads(a.base.read_text())
    grid, _ = totals(
        arrays(a.manifest, a.posteriors, base), base, [15, 21, 31, 41, 61, 81, 101]
    )
    old = grid[base["expansion_frames"]]
    # Fixed development objective and precision safeguard. Ties favor less padding.
    acceptable = [
        (m["exact_three_F1"] - m["DER"], -width)
        for width, m in grid.items()
        if m["exact_three_precision"] >= old["exact_three_precision"] - 0.02
    ]
    width = -max(acceptable)[1]
    a.settings.write_text(
        json.dumps(dict(base, expansion_frames=width), indent=2) + "\n"
    )
    a.out.write_text(json.dumps({"chosen": width, "all_widths": grid}, indent=2) + "\n")


if __name__ == "__main__":
    with threadpool_limits(2):
        main()
