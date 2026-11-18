"""Score three persistent speaker labels from NVIDIA's native diarizer."""

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import tempfile
import time

import soundfile as sf

from diarize import reference, score


def select_speakers(rows, duration):
    totals = Counter()
    clipped = []
    for row in rows:
        begin = max(0.0, row["start"])
        end = min(duration, row["end"])
        if end > begin:
            totals[row["speaker"]] += end - begin
            clipped.append(dict(row, start=begin, end=end))
    # The session count is given. The speech seconds come only from predictions.
    keep = sorted(totals, key=lambda key: (-totals[key], key))[:3]
    names = {name: str(i) for i, name in enumerate(sorted(keep))}
    spans = [
        dict(row, speaker=names[row["speaker"]])
        for row in clipped
        if row["speaker"] in names
    ]
    return spans, totals


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio", type=Path)
    p.add_argument("--binary", type=Path)
    p.add_argument("--model", type=Path)
    p.add_argument("--predicted-rttm", type=Path)
    p.add_argument("--measured-wall-seconds", type=float)
    p.add_argument("--rttm", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--preset", default="v3-offline", choices=["v3-offline", "v3-streaming"]
    )
    args = p.parse_args()
    info = sf.info(args.audio)
    if info.channels != 1 or info.samplerate != 16000:
        p.error("Provide a single-channel, 16 kHz WAV")
    if args.predicted_rttm:
        rows = reference(args.predicted_rttm)
        wall = args.measured_wall_seconds
    else:
        if not args.binary or not args.model:
            p.error("Specify --binary and --model, or supply --predicted-rttm")
        with tempfile.TemporaryDirectory() as tmp:
            hypothesis = Path(tmp) / "hypothesis.rttm"
            start = time.perf_counter()
            subprocess.run(
                [
                    str(args.binary.resolve()),
                    "diarize",
                    str(args.audio),
                    "--model",
                    str(args.model),
                    "--device",
                    "cpu",
                    "--preset",
                    args.preset,
                    "--format",
                    "rttm",
                    "--output",
                    str(hypothesis),
                ],
                check=True,
            )
            wall = time.perf_counter() - start
            rows = reference(hypothesis)
    segments, emitted = select_speakers(rows, info.duration)
    result = {
        "duration_s": info.duration,
        "inference_wall_s": wall,
        "known_session_speaker_count": 3,
        "native_output_speaker_seconds": emitted,
        "preset": args.preset,
        "segments": segments,
    }
    if args.rttm:
        result["evaluation"] = score(reference(args.rttm), segments, info.duration)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print({k: v for k, v in result.items() if k != "segments"}, flush=True)


if __name__ == "__main__":
    main()
