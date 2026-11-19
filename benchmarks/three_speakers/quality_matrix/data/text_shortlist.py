"""Select three boundary-complete ASR excerpts inside frozen audio timing crops."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from diarize import masks, reference
from prepare_aishell import digest, write_json
from recognize import normalized, textgrid


def choose(row, root, references):
    a, b, duration = (row[key] for key in ("start", "end", "duration_s"))
    ticks = round(100 * duration)
    t = a + np.arange(ticks + 1) / 100
    turns = [
        x
        for x in textgrid(references / (row["session"] + ".TextGrid"))
        if x["speaker"] in row["speaker_ids"] and a < x["end"] and x["start"] < b
    ]
    safe = np.ones(len(t), dtype=bool)
    for r in turns:
        safe[(t > r["start"] + 1e-8) & (t < r["end"] - 1e-8)] = False
    left = np.flatnonzero(safe[: min(401, ticks // 3 + 1)])
    right = np.flatnonzero(safe[max(ticks - 400, 2 * ticks // 3) :]) + max(
        ticks - 400, 2 * ticks // 3
    )
    if not len(left) or not len(right):
        return
    lo, hi = int(left[0]), int(right[-1])
    if hi - lo < 500:
        return
    activity = reference(root / row["ref"])
    overlay = masks(activity, duration)[1][:, lo:hi].sum(axis=0)
    triple = int(np.sum(overlay == 3))
    speech = [
        dict(u, start=u["start"] - t[lo], end=u["end"] - t[lo])
        for u in turns
        if t[lo] - 1e-8 <= u["start"] < u["end"] <= t[hi] + 1e-8
    ]
    count = sum(len(normalized(u["text"])) for u in speech)
    if triple < 35 or count < 20 or len({u["speaker"] for u in speech}) != 3:
        return
    return (triple, count, hi - lo, row, lo, hi, speech)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--refs", type=Path, required=True)
    p.add_argument("--plan-only", action="store_true")
    a = p.parse_args()
    panel = a.root
    plan = panel / "top_selection.json"
    if not plan.exists():
        collection = json.loads((panel / "selection.json").read_text())
        rows = []
        for session in sorted({r["session"] for r in collection}):
            options = [
                x
                for r in collection
                if r["session"] == session
                and (x := choose(r, panel, a.refs)) is not None
            ]
            if not options:
                continue
            triple, count, ticks, source, start, end, content = max(
                options, key=lambda x: x[:3]
            )
            key = f"ASR_{source['id']}_{start:04d}_{end:04d}"
            ref = key + ".rttm"
            text = key + ".json"
            lines = []
            for span in reference(panel / source["ref"]):
                l = max(span["start"], start / 100) - start / 100
                r = min(span["end"], end / 100) - start / 100
                if r > l:
                    lines.append(
                        f"SPEAKER {key} 1 {l:.4f} {r - l:.4f} <NA> <NA> {span['speaker']} <NA> <NA>"
                    )
            (panel / ref).write_text("\n".join(lines) + "\n")
            write_json(panel / text, content)
            row = dict(
                source,
                id=key,
                source_crop_audio=source["audio"],
                source_crop_start_tick=start,
                source_crop_end_tick=end,
                start=round(source["start"] + start / 100, 2),
                end=round(source["start"] + end / 100, 2),
                duration_s=ticks / 100,
                exact_three_s=triple / 100,
                audio=key + ".wav",
                ref=ref,
                text_reference=text,
                reference_characters=count,
                clipped_reference_turns=0,
                cropped_activity_sha256=digest(panel / ref),
                cropped_text_sha256=digest(panel / text),
            )
            rows.append(row)
        write_json(plan, rows)
    rows = json.loads(plan.read_text())
    if a.plan_only:
        print(
            [
                (
                    r["id"],
                    r["duration_s"],
                    r["exact_three_s"],
                    r["reference_characters"],
                )
                for r in rows
            ]
        )
        return
    if len(rows) < 2:
        raise ValueError(f"Expected two boundary-complete ASR inputs, got {len(rows)}")
    for row in rows:
        y, rate = sf.read(panel / row["source_crop_audio"], dtype="int16")
        i, j = row["source_crop_start_tick"] * 160, row["source_crop_end_tick"] * 160
        target = panel / row["audio"]
        if rate != 16000 or y.ndim != 1 or not 0 <= i < j <= len(y):
            raise ValueError(f"Bad 16 kHz source crop in {row['source_crop_audio']}")
        sf.write(target, y[i:j], rate, subtype="PCM_16")
        row.update(wav_sha256=digest(target), frames=j - i)
    write_json(panel / "top_manifest.json", rows)
    (panel / "top_inputs.txt").write_text(
        "".join(str(panel / r["audio"]) + "\n" for r in rows)
    )


if __name__ == "__main__":
    main()
