"""Find reference-free, coincident cross-speaker text duplicated into a quieter track."""

import argparse
import json
from pathlib import Path
import itertools
import numpy as np
import soundfile as sf
from rapidfuzz.distance import Levenshtein
from recognize import conversations, normalized, textgrid, text_scores


def encounters(rows, tracks):
    streams = {
        p.stem.split("_")[-1]: sf.SoundFile(p) for p in tracks.glob("speaker_*.wav")
    }
    features = []
    try:
        for i, a in enumerate(rows):
            for j in range(i + 1, len(rows)):
                b = rows[j]
                if b["start"] >= a["end"]:
                    break
                if a["speaker"] == b["speaker"]:
                    continue
                left = max(a["start"], b["start"])
                right = min(a["end"], b["end"])
                if right - left < 0.12:
                    continue
                levels = []
                for r in [a, b]:
                    src = streams[r["speaker"]]
                    src.seek(round(left * 16000))
                    x = src.read(round((right - left) * 16000), dtype="float32")
                    levels.append(10 * np.log10(max(1e-12, float(np.mean(x**2)))))
                if levels[0] <= levels[1]:
                    weak, strong, iweak = a, b, i
                else:
                    weak, strong, iweak = b, a, j
                s = normalized(weak["text"])
                big = normalized(strong["text"])
                if not s or s not in big:
                    continue
                span = weak["end"] - weak["start"]
                frac = (right - left) / max(0.01, span)
                features.append((iweak, len(s), abs(levels[0] - levels[1]), frac))
    finally:
        for f in streams.values():
            f.close()
    return features


def apply(rows, feat, params):
    size, db, overlap = params
    drop = {
        i for i, n, ratio, frac in feat if n >= size and ratio >= db and frac >= overlap
    }
    return [r for i, r in enumerate(rows) if i not in drop]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "tracks", "params", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--fit", action="store_true")
    a = p.parse_args()
    rows = sorted(
        (json.loads(l) for l in a.baseline.read_text().splitlines()),
        key=lambda r: (r["start"], r["speaker"]),
    )
    if a.fit and not a.reference:
        p.error("--fit requires a reference")
    truth = None
    if a.reference:
        truth = (
            textgrid(a.reference)
            if a.reference.suffix.lower() == ".textgrid"
            else json.loads(a.reference.read_text())
        )
    feat = encounters(rows, a.tracks)
    base = text_scores(truth, rows) if truth is not None else None
    if a.fit:
        m = {r["system"]: r["reference"] for r in base["speaker_scores"]}
        refs = conversations(truth)
        best = (base["totals"]["errors"], 0, (9999, 0, 0))
        for sett in itertools.product(
            [1, 2, 3], [0.0, 3.0, 6.0, 9.0], [0.25, 0.50, 0.80]
        ):
            chosen = apply(rows, feat, sett)
            s = conversations(chosen)
            loss = sum(
                Levenshtein.distance(refs[r], s.get(h, "")) for h, r in m.items()
            )
            best = min(best, (loss, len(rows) - len(chosen), sett))
        a.params.write_text(json.dumps(best[-1]) + "\n")
    chosen = apply(rows, feat, json.loads(a.params.read_text()))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "transcript.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chosen),
        encoding="utf8",
    )
    for name in ["all"] + [
        p.stem.split("_")[-1] for p in sorted(a.tracks.glob("speaker_*.wav"))
    ]:
        items = chosen if name == "all" else [r for r in chosen if r["speaker"] == name]
        filename = "transcript.txt" if name == "all" else f"speaker_{name}.txt"
        (a.out / filename).write_text(
            "".join(
                f"[{r['start']:.2f}–{r['end']:.2f}] {r['speaker']}  {r['text']}\n"
                for r in items
            ),
            encoding="utf8",
        )
    if truth is None:
        return
    result = {
        "previous": base,
        "candidate": text_scores(truth, chosen),
        "removed": len(rows) - len(chosen),
        "candidate_duplicates": len(feat),
    }
    (a.out / "scores.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["removed"], base["cpCER"], result["candidate"]["cpCER"])


if __name__ == "__main__":
    main()
