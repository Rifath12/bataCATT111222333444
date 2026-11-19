"""Compare the frozen text selector on an overlap-enriched external panel."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from cached_gate import apply_cached_gate, export
from contextual_asr import candidates, read_jsonl
from diarize import reference
from duplicate_leakage import apply, encounters
from overlap_characters import panel_score
from recognize import text_scores


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--copy-settings", type=Path, required=True)
    args = parser.parse_args()
    gate, duplicate = read(args.gate), read(args.copy_settings)
    clips = args.manifest.parent
    args.out.mkdir(exist_ok=True, parents=True)
    total, details = defaultdict(Counter), []
    for row in read(args.manifest):
        name = Path(row["audio"]).stem
        r = clips / name
        baseline = read_jsonl(r / "current" / "transcript.jsonl")
        estimates = read_jsonl(r / "stem" / "transcript.jsonl")
        timing = read(r / "timing.json")["segments"]
        data = read(r / "context_tokens.json")["tokens"]
        stems = r / "mask"
        frames = candidates(timing, stems, estimates, data)
        updated, changes = apply_cached_gate(frames, baseline, gate, stems, duplicate)
        speakers = sorted({str(r["speaker"]) for r in timing})
        export(updated, args.out / name / "Updated", speakers)
        export(baseline, args.out / name / "Previous", speakers)
        truth = read(clips / row["text_reference"])
        activity = reference(clips / row["ref"])
        result = {
            "id": name,
            "session": row["session"],
            "duration_s": row["duration_s"],
            "reference_triple_seconds": row["exact_three_s"],
            "changed_turns": changes,
        }
        fast = apply(estimates, encounters(estimates, stems), duplicate)
        export(fast, args.out / name / "Stem_only", speakers)
        for variant, prediction in (
            ("previous", baseline),
            ("refined", updated),
            ("stem_only", fast),
        ):
            text = text_scores(truth, prediction)
            local = panel_score(truth, prediction, activity, row["duration_s"], 1.0)
            result[variant] = {"overall": text, "timed": local}
            total[variant].update(text["totals"])
            total[variant + "_exact3"].update(local["midpoint"].get("3", {}))
            total[variant + "_strict_exact3"].update(local["strict_triple"])
        details.append(result)
    for v, n in total.items():
        if not "errors" in n:
            n["errors"] = sum(
                n.get(k, 0) for k in ("substitutions", "deletions", "insertions")
            )
        n["CER"] = (
            n["errors"] / n["reference_chars"] if n.get("reference_chars", 0) else None
        )
    (args.out / "scores.json").write_text(
        json.dumps(dict(pooled=total, clips=details), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print({k: (n["errors"], n["reference_chars"], n["CER"]) for k, n in total.items()})


if __name__ == "__main__":
    main()
