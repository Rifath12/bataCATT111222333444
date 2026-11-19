"""Reuse full-file SenseVoice results for bit-identical PCM spans in a paired experiment."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from compare_recognizers import sensevoice
from recognize import inference_spans, text_scores, textgrid


def run(args):
    previous = {}
    for line in (
        (args.previous / "sensevoice" / "transcript.jsonl").read_text().splitlines()
    ):
        row = json.loads(line)
        previous[(row["speaker"], row["start"], row["end"])] = row
    settings = json.loads((args.previous / "sensevoice" / "scores.json").read_text())
    if (
        settings["backend"] != "sensevoice"
        or settings["target_rms"] != 0.035
        or settings["threads"] != 2
    ):
        raise ValueError(
            "Paired run requires identical backend, normalization and thread configuration"
        )
    timeline = json.loads((args.previous / "timing.json").read_text())["segments"]
    names = sorted({r["speaker"] for r in timeline})
    wavs = {}
    unchanged = []
    changed = []
    total = 0
    for name in names:
        wavs[name] = args.current / f"speaker_{name}.wav"
        with (
            sf.SoundFile(args.previous / f"speaker_{name}.wav") as baseline,
            sf.SoundFile(wavs[name]) as current,
        ):
            if (
                len(current) != len(baseline)
                or baseline.samplerate != 16000
                or current.samplerate != 16000
            ):
                raise ValueError("Track lengths or sample rate differ")
            length = len(current) / 16000
            spans = inference_spans(
                [r for r in timeline if r["speaker"] == name], length
            )
            available = {(a, b) for label, a, b in previous if label == name}
            if not available <= set(spans):
                raise ValueError(
                    "Previous transcript boundaries differ from the selected timing"
                )
            modified = []
            for start, end in spans:
                total += 1
                left = round(start * 16000)
                size = round(end * 16000) - left
                baseline.seek(left)
                current.seek(left)
                if np.array_equal(
                    baseline.read(size, dtype="float32"),
                    current.read(size, dtype="float32"),
                ):
                    if row := previous.get((name, start, end)):
                        unchanged.append(row)
                else:
                    modified.append(dict(speaker=name, start=start, end=end))
            # The recognizer must reconstruct exactly the expected spans from this reduced timeline.
            if inference_spans(modified, length) != [
                (r["start"], r["end"]) for r in modified
            ]:
                raise ValueError(
                    "Filtered timeline changes the recognizer's inference spans"
                )
            changed += modified
    decoded, elapsed = sensevoice(wavs, changed, args.model, 2, 0.035)
    merged = sorted(unchanged + decoded, key=lambda r: (r["start"], r["speaker"]))
    out = args.current / "sensevoice"
    out.mkdir(exist_ok=True)
    (out / "transcript.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged)
    )
    score = text_scores(textgrid(args.reference), merged)
    score.update(
        backend="sensevoice",
        target_rms=0.035,
        threads=2,
        inference_s=elapsed,
        compared_spans=total,
        redecoded_spans=len(changed),
        identical_spans_reused=total - len(changed),
    )
    (out / "scores.json").write_text(json.dumps(score, indent=2) + "\n")
    print(
        f"Reused {total - len(changed)}/{total} identical spans. cpCER {score['cpCER']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("previous", "current", "model", "reference"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
