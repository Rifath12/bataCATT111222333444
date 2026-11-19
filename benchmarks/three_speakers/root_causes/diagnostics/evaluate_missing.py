"""Measure partial-enrollment separation against timing routing on matched Mandarin clips."""

import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys
import numpy as np
import soundfile as sf


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("render", "transcribe"))
    for arg in ("cases", "project", "models", "recognizer"):
        p.add_argument("--" + arg, type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args()
    cases = json.loads((a.cases / "cases.json").read_text())
    if a.phase == "render":
        import torch

        torch.set_num_threads(a.threads)
        sys.path.insert(0, str(a.project))
        import process_meeting

        # Initialize once for this benchmarking process; the underlying objects are reused
        # in the same serial order as independent CLI invocations.
        process_meeting.load_separator = lru_cache(maxsize=1)(
            process_meeting.load_separator
        )
        process_meeting.VoiceEmbeddings = lru_cache(maxsize=1)(
            process_meeting.VoiceEmbeddings
        )
        from render import output_tracks
        from diarize import masks

        a.workers = 1
        for r in cases:
            key = r["source"]
            d = a.cases / key
            wave, rate = sf.read(d / "mono.wav", dtype="float32")
            names, activity = masks(
                json.loads((d / "timing.json").read_text())["segments"],
                len(wave) / rate,
            )
            for method in ["routing", "dprnn"]:
                dest = a.out / key / method
                dest.mkdir(parents=True, exist_ok=True)
                tracks, stats = output_tracks(wave, activity, method, a)
                gain = min(1.0, 0.98 / max(1e-8, float(np.abs(tracks).max())))
                for label, track in zip(names, tracks):
                    sf.write(dest / f"speaker_{label}.wav", gain * track, rate)
                (dest / "render.json").write_text(json.dumps(stats, indent=2) + "\n")
            print(key, "rendered", flush=True)
    else:
        import sherpa_onnx as sx
        from compare_recognizers import sensevoice
        from recognize import text_scores

        sx.OfflineRecognizer.from_sense_voice = lru_cache(maxsize=1)(
            sx.OfflineRecognizer.from_sense_voice
        )
        rows = []
        for r in cases:
            name = r["source"]
            d = a.cases / name
            times = json.loads((d / "timing.json").read_text())["segments"]
            truth = json.loads((d / "reference.json").read_text())
            record = dict(r)
            for method in ["routing", "dprnn"]:
                root = a.out / name / method
                wavs = {w.stem.split("_")[-1]: w for w in root.glob("speaker_*.wav")}
                rows_out, elapsed = sensevoice(
                    wavs, times, a.recognizer, a.threads, 0.035
                )
                (root / "transcript.jsonl").write_text(
                    "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows_out),
                    encoding="utf8",
                )
                score = text_scores(truth, rows_out)
                record[method] = dict(score["totals"], cpCER=score["cpCER"])
            rows.append(record)
            print(
                name, record["routing"]["errors"], record["dprnn"]["errors"], flush=True
            )
            (a.out / "scores.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
