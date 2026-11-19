"""Create a compact diagnostic report from the completed controlled experiment."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import median


def summarize_panel(panel):
    result = {}
    for arm, summary in panel["summary"].items():
        output = {}
        for name in ["sensevoice", "zipformer"]:
            by_level = defaultdict(Counter)
            for case in panel["cases"]:
                for speaker in case["arms"][arm]["asr"][name]["per_source"]:
                    gain = case["source_dB"][int(speaker["reference"])]
                    by_level[gain]["chars"] += speaker["reference_chars"]
                    by_level[gain]["edits"] += sum(
                        speaker[op]
                        for op in ["insertions", "deletions", "substitutions"]
                    )
            a = summary[name]
            output[name] = {
                "reference_chars": a["reference_chars"],
                "edits": a["errors"],
                "exact_phrases": a["phrases_matched"],
                "reference_chars_matched": a["correct_reference_chars"],
                "relative_source_level": dict(by_level),
            }
        if "median_triple_SI_SDRi" in summary:
            output["median_three_way_SI_SDRi_dB"] = summary["median_triple_SI_SDRi"]
            db = [
                item["exact_overlap_improvement_dB"]
                for case in panel["cases"]
                for item in case["arms"][arm]["dry_reference_scores"]
            ]
            output["median_dry_target_three_way_SI_SDRi_dB"] = median(
                x for x in db if x is not None
            )
        result[arm] = output
    return {
        "seconds": sum(c["seconds"] for c in panel["cases"]),
        "three_active_seconds": sum(c["triple_s"] for c in panel["cases"]),
        "arms": result,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    data = {
        k: summarize_panel(json.loads((a.results / f"{k}.json").read_text()))
        for k in ["development", "repeat", "new"]
    }
    a.out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
