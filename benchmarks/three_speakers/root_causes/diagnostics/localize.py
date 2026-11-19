"""Assign transcript edit operations to reference utterance conditions at 10 ms resolution."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
from rapidfuzz.distance import Levenshtein
from recognize import textgrid, text_scores, normalized
from oracle_matrix import occupancy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--timing", type=Path, required=True)
    p.add_argument("--hyp", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--by-speaker", type=Path, help="Optional detail by matched output voice"
    )
    a = p.parse_args()
    ref = textgrid(a.reference)
    hyp = [json.loads(row) for row in a.hyp.read_text().splitlines() if row]
    pred = json.loads(a.timing.read_text())["segments"]
    src = sorted({r["speaker"] for r in ref})
    slots = sorted({r["speaker"] for r in pred})
    score = text_scores(ref, hyp)
    n = int(np.ceil(100 * max(r["end"] for r in ref + pred)))
    refmask = occupancy(ref, src, n)
    outmask = occupancy(pred, slots, n)
    centers = (np.arange(n) + 0.5) * 0.01
    groups = defaultdict(Counter)
    details = {}
    for row in score["speaker_scores"]:
        rname, hname = row["reference"], row["system"]
        speaker_counts = defaultdict(Counter)
        per = sorted(
            [r for r in ref if r["speaker"] == rname], key=lambda r: r["start"]
        )
        sref = []
        binid = []
        others = np.delete(refmask, src.index(rname), axis=0).any(axis=0)
        for turn in per:
            lo, hi = np.searchsorted(centers, [turn["start"], turn["end"]])
            hi = max(hi, lo + 1)
            overlapped = others[lo:hi].sum() >= 20
            coverage = outmask[slots.index(hname), lo:hi].mean()
            group = ("overlap" if overlapped else "solo") + (
                "_gated" if coverage < 0.75 else "_covered"
            )
            chars = normalized(turn["text"])
            binid.extend([group] * len(chars))
            sref.append(chars)
            speaker_counts[group]["turns"] += 1
            speaker_counts[group]["chars"] += len(chars)
        left = "".join(sref)
        right = normalized(
            "".join(
                x["text"]
                for x in sorted(
                    (h for h in hyp if h["speaker"] == hname), key=lambda r: r["start"]
                )
            )
        )
        for edit in Levenshtein.editops(left, right):
            speaker_counts[binid[min(len(binid) - 1, edit.src_pos)]][edit.tag] += 1
        for group, counts in speaker_counts.items():
            groups[group].update(counts)
        details[hname] = speaker_counts
    for v in [
        *groups.values(),
        *(counts for spk in details.values() for counts in spk.values()),
    ]:
        v["errors"] = v["insert"] + v["delete"] + v["replace"]
        v["error_rate"] = v["errors"] / v["chars"]
    assert sum(v["errors"] for v in groups.values()) == score["totals"]["errors"]
    if a.by_speaker:
        a.by_speaker.write_text(json.dumps(details, indent=2) + "\n")
    a.out.write_text(json.dumps(dict(groups), indent=2) + "\n")
    print(dict(groups))


if __name__ == "__main__":
    main()
