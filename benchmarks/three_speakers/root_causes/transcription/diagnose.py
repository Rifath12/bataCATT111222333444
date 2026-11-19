"""Attribute aligned transcript insertions and substitutions to hypothesis spans."""

from collections import Counter, defaultdict
import argparse, json
from pathlib import Path
from rapidfuzz.distance import Levenshtein
from recognize import textgrid, text_scores, conversations, normalized


def run(ref, rows):
    result = text_scores(ref, rows)
    known = conversations(ref)
    grouped = defaultdict(Counter)
    seconds = defaultdict(float)
    for rec in result["speaker_scores"]:
        r = known[rec["reference"]]
        turns = [x for x in rows if x["speaker"] == rec["system"]]
        turns.sort(key=lambda x: x["start"])
        h = normalized("".join(x["text"] for x in turns))
        ps = []
        for row in turns:
            n = len(normalized(row["text"]))
            t = row["end"] - row["start"]
            bucket = next(
                (
                    b
                    for lim, b in [
                        (0.40, "0-.4s"),
                        (0.8, ".4-.8s"),
                        (1.6, ".8-1.6s"),
                        (3.2, "1.6-3.2s"),
                    ]
                    if t < lim
                ),
                "3.2s+",
            )
            grouped[bucket]["utterances"] += 1
            seconds[bucket] += t
            ps.extend([bucket] * n)
        assert len(h) == len(ps)
        for op, s, e, a, b in Levenshtein.opcodes(r, h):
            for k, v in Counter(ps[a:b]).items():
                grouped[k][op] += v
    return {
        "scores": result,
        "hypothesis_spans": {
            k: {**v, "seconds": seconds[k]} for k, v in grouped.items()
        },
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ref", type=Path)
    p.add_argument("rows", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    rows = [json.loads(line) for line in a.rows.read_text().splitlines()]
    a.out.write_text(json.dumps(run(textgrid(a.ref), rows), indent=2) + "\n")
