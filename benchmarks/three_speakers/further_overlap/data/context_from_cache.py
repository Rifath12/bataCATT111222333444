"""Apply the released transcript rules to the two previously decoded views."""

import argparse
import json
from pathlib import Path

from contextual_asr import (
    candidates,
    protect_long_turns,
    read_jsonl,
    restore_supported_text,
    select,
)
from duplicate_leakage import apply, encounters


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["bench", "case"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    loaded = lambda key: json.loads((a.bench / (key + "_settings.json")).read_text())
    context, restoration, protected, dedup = (
        loaded(k) for k in ("context", "restoration", "long_turn", "duplicate")
    )
    r = a.case
    timing = json.loads((r / "timing.json").read_text())["segments"]
    predicted = read_jsonl(r / "stem" / "transcript.jsonl")
    data = json.loads((r / "context_tokens.json").read_text())["tokens"]
    frames = candidates(timing, r / "mask", predicted, data)
    output = protect_long_turns(
        frames,
        restore_supported_text(frames, select(frames, context), restoration),
        protected,
    )
    output = apply(output, encounters(output, r / "mask"), dedup)
    dest = r / "current"
    dest.mkdir(exist_ok=True)
    (dest / "transcript.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output)
    )
    for name in ("all", "A", "B", "C"):
        matched = [row for row in output if name in ("all", row["speaker"])]
        file = "transcript.txt" if name == "all" else f"speaker_{name}.txt"
        (dest / file).write_text(
            "".join(
                f"[{r['start']:.2f} -- {r['end']:.2f}] {r['speaker']} {r['text']}\n"
                for r in matched
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
