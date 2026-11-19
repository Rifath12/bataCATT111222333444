"""Partition time-constrained speaker-attributed character edits by speech overlap."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from diarize import masks, reference as read_rttm
from recognize import normalized, textgrid


def characters(turns):
    """Return timed characters and retain whether their timestamps were supplied."""
    output = defaultdict(list)
    for row in sorted(turns, key=lambda r: (r["start"], r["end"], str(r["speaker"]))):
        speaker = str(row["speaker"])
        text = normalized(row["text"])
        left, right = row["start"], row["end"]
        if right <= left or not text:
            continue
        timed = [
            (c, (q["start_ms"] + q["end_ms"]) / 2000)
            for q in row.get("characters", [])
            for c in normalized(q["text"])
        ]
        supplied = "".join(c for c, _ in timed) == text
        if supplied and len(text) == 1:
            timed = [(text, (left + right) / 2)]
        if not supplied:
            timed = [
                (c, left + (i + 0.5) * (right - left) / len(text))
                for i, c in enumerate(text)
            ]
        output[speaker].extend((c, t, supplied, left, right) for c, t in timed)
    return dict(output)


def edit_path(truth, hypothesis, collar):
    """Minimum edit alignment; permit diagonal steps only at plausible timestamps."""
    a, b = len(truth), len(hypothesis)
    cols = len(hypothesis) + 1
    steps = bytearray((a + 1) * cols)
    scores = list(range(cols))
    steps[:cols] = bytes([3]) * cols
    ht = [t for _, t, *_ in hypothesis]
    words = [c for c, *_ in hypothesis]
    for i, (word, time, *_) in enumerate(truth, 1):
        old = scores
        scores = [i] + [0] * b
        steps[i * cols] = 2
        for j, (char, clock) in enumerate(zip(words, ht), 1):
            delete, insert = old[j] + 1, scores[j - 1] + 1
            if abs(time - clock) <= collar:
                substitute = old[j - 1] + (word != char)
                if substitute <= delete and substitute <= insert:
                    scores[j], steps[i * cols + j] = substitute, 1
                    continue
            if delete <= insert:
                scores[j], steps[i * cols + j] = delete, 2
            else:
                scores[j], steps[i * cols + j] = insert, 3
    score = scores[-1]
    i, j = a, b
    data = []
    while i or j:
        step = steps[i * cols + j]
        if step == 1:
            data.append(
                (
                    "correct"
                    if truth[i - 1][0] == hypothesis[j - 1][0]
                    else "substitutions",
                    truth[i - 1],
                )
            )
            i -= 1
            j -= 1
        elif step == 2:
            data.append(("deletions", truth[i - 1]))
            i -= 1
        else:
            data.append(("insertions", hypothesis[j - 1]))
            j -= 1
    return score, data


def panel_score(ref, hypothesis, activity, duration, collar):
    ref_p, hyp_p = characters(ref), characters(hypothesis)
    if not ref_p:
        raise ValueError("Reference contains no characters")
    names, pred = sorted(ref_p), sorted(hyp_p)
    n = max(len(names), len(pred))
    ref_t = names + [None] * (n - len(names))
    hyp_t = pred + [None] * (n - len(pred))
    matrix, paths = np.empty((n, n), dtype="int64"), {}
    for i, a in enumerate(ref_t):
        for j, b in enumerate(hyp_t):
            matrix[i, j], paths[i, j] = edit_path(
                ref_p.get(a, []), hyp_p.get(b, []), collar
            )
    seq = masks(activity, duration)[1].sum(axis=0)
    counts = {str(n): Counter() for n in range(int(seq.max(initial=0)) + 1)}
    strict = Counter()
    i, j = linear_sum_assignment(matrix)
    for a, b in zip(i, j):
        for kind, (_, time, supplied, left, right) in paths[a, b]:
            frame = int(np.clip(np.floor(time * 100 + 1e-9), 0, len(seq) - 1))
            group = counts[str(seq[frame])]
            group[kind] += 1
            if kind != "insertions":
                group["reference_chars"] += 1
                group["supplied_reference_char_times"] += int(supplied)
            l = max(0, int(np.ceil(left * 100 - 0.5)))
            r = min(len(seq), int(np.ceil(right * 100 - 0.5)))
            if r > l and np.all(seq[l:r] == 3):
                strict[kind] += 1
                if kind != "insertions":
                    strict["reference_chars"] += 1
    return dict(
        mapping=[(ref_t[a], hyp_t[b]) for a, b in zip(i, j)],
        midpoint=dict(counts),
        strict_triple=dict(strict),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSON list with id, reference, hypothesis, rttm and duration_s for each clip",
    )
    p.add_argument("--collar", default=1.0, type=float)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    if args.collar < 0:
        p.error("--collar must be non-negative")
    normalized_rows, midpoint, strict = [], defaultdict(Counter), Counter()
    for row in json.loads(args.manifest.read_text()):
        ref = args.manifest.parent / row["reference"]
        h = args.manifest.parent / row["hypothesis"]
        rttm = args.manifest.parent / row["rttm"]
        truth = (
            textgrid(ref)
            if ref.suffix.lower() == ".textgrid"
            else json.loads(ref.read_text())
        )
        hyp = [json.loads(l) for l in h.read_text().splitlines() if l.strip()]
        result = panel_score(
            truth, hyp, read_rttm(rttm), row["duration_s"], args.collar
        )
        normalized_rows.append(dict(id=row["id"], **result))
        for condition, data in result["midpoint"].items():
            midpoint[condition].update(data)
        strict.update(result["strict_triple"])

    def measures(c):
        edits = sum(c[k] for k in ("insertions", "deletions", "substitutions"))
        reference_chars = c["reference_chars"]
        return dict(
            c, errors=edits, CER=edits / reference_chars if reference_chars else None
        )

    args.out.write_text(
        json.dumps(
            dict(
                time_collar_seconds=args.collar,
                convention="Character midpoints are uniformly interpolated within the utterance unless actual token times are present. Fixed global mapping per recording.",
                midpoint={k: measures(c) for k, c in midpoint.items()},
                strict_triple=measures(strict),
                files=normalized_rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
