"""Summarize matched anchor experiments without changing the original case results."""

import argparse
import json
from pathlib import Path

from rapidfuzz.distance import Levenshtein
from recognize import normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = {}
    for path in args.results:
        result = json.loads(path.read_text())
        report = {}
        for arm, row in result["summary"].items():
            report[arm] = dict(row)
            if arm not in ("direct", "residual", "average", "routed"):
                continue
            for backend in ("sense", "zip"):
                if f"{backend}_edits" not in row:
                    continue
                fixed = 0
                snippets = []
                for case in result["cases"]:
                    references = [normalized(s["text"]) for s in case["trio"]]
                    estimates = case["arms"][arm][backend]["hypotheses"]
                    direct_edits = [
                        Levenshtein.distance(ref, normalized(est["text"]))
                        for ref, est in zip(references, estimates, strict=True)
                    ]
                    fixed += sum(direct_edits)
                    snippets.append(direct_edits[case["quiet"]])
                report[arm][f"{backend}_fixed_identity_edits"] = fixed
                report[arm][f"{backend}_quiet_voice_edits_by_trio"] = snippets
        out[path.stem] = report
    args.out.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
