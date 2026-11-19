"""Transcribe three separated Mandarin tracks with a fixed ASR backend."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import sherpa_onnx as sx
import soundfile as sf

from recognize import inference_spans, text_scores, textgrid, transcribe


def sensevoice(wavs, times, model_dir, threads, target_rms):
    recognizer = sx.OfflineRecognizer.from_sense_voice(
        model=str(model_dir / "model.int8.onnx"),
        tokens=str(model_dir / "tokens.txt"),
        language="zh",
        use_itn=False,
        num_threads=threads,
    )
    rows = []
    start = time.perf_counter()
    for speaker, wav in sorted(wavs.items()):
        y, sr = sf.read(wav, dtype="float32")
        if sr != 16000 or y.ndim != 1:
            raise ValueError(f"Expected mono 16 kHz WAV: {wav}")
        turns = [t for t in times if str(t["speaker"]) == speaker]
        for a, b in inference_spans(turns, len(y) / sr):
            chunk = y[round(a * sr) : round(b * sr)]
            rms = max(float(np.sqrt(np.mean(chunk**2))), 1e-7)
            peak = max(float(np.max(np.abs(chunk))), 1e-7)
            gain = min(target_rms / rms, 0.98 / peak, 20.0)
            stream = recognizer.create_stream()
            stream.accept_waveform(sr, np.pad(chunk * gain, (800, 1280)))
            recognizer.decode_stream(stream)
            text = stream.result.text.strip()
            if text:
                rows.append(dict(speaker=speaker, start=a, end=b, text=text))
        print(f"Decoded speaker {speaker}", flush=True)
    return sorted(
        rows, key=lambda t: (t["start"], t["speaker"])
    ), time.perf_counter() - start


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracks", type=Path, required=True)
    p.add_argument("--timing", type=Path, required=True)
    p.add_argument(
        "--reference", type=Path, help="Optional TextGrid or JSON for scoring"
    )
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--backend", choices=("sensevoice", "zipformer"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    paths = {w.stem.split("_")[-1]: w for w in args.tracks.glob("speaker_*.wav")}
    if len(paths) != 3:
        p.error("The tracks folder must contain exactly three speaker WAVs")
    timing = json.loads(args.timing.read_text(encoding="utf-8"))["segments"]
    if args.backend == "sensevoice":
        rows, wall = sensevoice(paths, timing, args.model, args.threads, 0.035)
    else:
        rows, wall = transcribe(
            paths,
            timing,
            args.model / "model.int8.onnx",
            args.model / "tokens.txt",
            args.threads,
            target_rms=0.035,
        )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "transcript.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )

    def lines(items):
        return "".join(
            f"[{r['start']:.2f}–{r['end']:.2f}] {r['speaker']}  {r['text']}\n"
            for r in items
        )

    (args.out / "transcript.txt").write_text(lines(rows), encoding="utf-8")
    for speaker in paths:
        own = [r for r in rows if r["speaker"] == speaker]
        (args.out / f"speaker_{speaker}.txt").write_text(lines(own), encoding="utf-8")
    result = dict(
        backend=args.backend,
        target_rms=0.035,
        threads=args.threads,
        inference_s=wall,
    )
    if args.reference:
        truth = (
            textgrid(args.reference)
            if args.reference.suffix.lower() == ".textgrid"
            else json.loads(args.reference.read_text(encoding="utf-8"))
        )
        result.update(text_scores(truth, rows))
        print(f"{args.backend} cpCER {result['cpCER']:.4f}", flush=True)
    (args.out / "scores.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Decoded {len(rows)} utterances in {wall:.2f}s", flush=True)


if __name__ == "__main__":
    main()
