"""Benchmark reference-only timing and close-talk audio controls for a complete meeting."""

import argparse
import json
from pathlib import Path
from itertools import permutations
import numpy as np
import soundfile as sf
import sherpa_onnx as sx
import time
from recognize import inference_spans
from recognize import textgrid, text_scores


def occupancy(rows, labels, length):
    mat = np.zeros((len(labels), length), np.float32)
    centers = (np.arange(length) + 0.5) * 0.01
    for r in rows:
        lo, hi = np.searchsorted(centers, [r["start"], r["end"]])
        mat[labels.index(r["speaker"]), lo:hi] = 1
    return mat


def recognize_cached(waves, timing, model_dir, threads, out, arm, delays):
    state = out / (arm + ".progress.jsonl")
    existing = {}
    if state.exists():
        for line in state.read_text(encoding="utf8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            existing[(row["speaker"], row["start"], row["end"])] = row
    decoder = sx.OfflineRecognizer.from_sense_voice(
        model=str(model_dir / "model.int8.onnx"),
        tokens=str(model_dir / "tokens.txt"),
        language="zh",
        use_itn=False,
        num_threads=threads,
    )
    with state.open("a", encoding="utf8", buffering=1) as stream:
        for speaker, wav in sorted(waves.items()):
            with sf.SoundFile(wav) as audio:
                turns = [t for t in timing if str(t["speaker"]) == speaker]
                for left, right in inference_spans(
                    turns, len(audio) / audio.samplerate
                ):
                    key = (speaker, left, right)
                    if key in existing:
                        continue
                    start = time.perf_counter()
                    a, b = round(left * 16000), round(right * 16000)
                    offset = round(0.001 * 16000 * delays.get(speaker, 0))
                    lo = a - offset
                    hi = b - offset
                    audio.seek(max(0, lo))
                    chunk = audio.read(max(0, hi - max(0, lo)), dtype="float32")
                    if lo < 0:
                        chunk = np.pad(chunk, (-lo, 0))
                    if len(chunk) < b - a:
                        chunk = np.pad(chunk, (0, b - a - len(chunk)))
                    rms = max(float(np.sqrt(np.mean(chunk**2))), 1e-7)
                    peak = max(float(np.max(np.abs(chunk))), 1e-7)
                    gain = min(0.035 / rms, 0.98 / peak, 20.0)
                    sample = decoder.create_stream()
                    sample.accept_waveform(16000, np.pad(chunk * gain, (800, 1280)))
                    decoder.decode_stream(sample)
                    row = dict(
                        speaker=speaker,
                        start=left,
                        end=right,
                        text=sample.result.text.strip(),
                        wall_s=time.perf_counter() - start,
                    )
                    existing[key] = row
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print("Decoded", speaker, len(existing), flush=True)
    rows = sorted(
        [r for r in existing.values() if r["text"]],
        key=lambda r: (r["start"], r["speaker"]),
    )
    return rows, sum(r["wall_s"] for r in existing.values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--timing", type=Path, required=True)
    p.add_argument("--tracks", type=Path, required=True)
    p.add_argument("--headsets", type=Path)
    p.add_argument(
        "--headset-delays",
        type=Path,
        help="JSON from energy_audit.py; align near waveforms to the room recorder",
    )
    p.add_argument("--mixture", type=Path)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument(
        "--arm",
        choices=[
            "separated_oracle",
            "near_oracle",
            "mixture_oracle",
            "near_predicted",
            "continuous_mixture_predicted",
        ],
        required=True,
    )
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    ref = textgrid(a.reference)
    pred = json.loads(a.timing.read_text())["segments"]
    sources = sorted({r["speaker"] for r in ref})
    slots = sorted({r["speaker"] for r in pred})
    if len(slots) != 3 or len(sources) != 3:
        raise ValueError(
            "Expected three speakers in both the reference and the timing output"
        )
    far = {s: a.tracks / f"speaker_{s}.wav" for s in slots}
    seconds = sf.info(far[slots[0]]).duration
    n = int(np.ceil(seconds / 0.01))
    overlap = occupancy(ref, sources, n) @ occupancy(pred, slots, n).T
    order = max(
        permutations(range(3)),
        key=lambda q: sum(overlap[i, j] for i, j in enumerate(q)),
    )
    mapping = {s: slots[j] for s, j in zip(sources, order)}
    timing = (
        pred
        if a.arm.endswith("predicted")
        else [
            dict(start=r["start"], end=r["end"], speaker=mapping[r["speaker"]])
            for r in ref
        ]
    )
    if a.arm.startswith("near"):
        waves = {
            slot: next(a.headsets.glob(f"*_{source}.wav"))
            for source, slot in mapping.items()
        }
    elif "mixture" in a.arm:
        waves = dict.fromkeys(slots, a.mixture)
    else:
        waves = far
    for w in set(waves.values()):
        info = sf.info(w)
        if (info.samplerate, info.channels) != (16000, 1):
            raise ValueError(f"{w} must be a mono 16 kHz WAV")
        if abs(info.duration - seconds) > 1 / 16000:
            raise ValueError(f"Unaligned input {w}: {info.duration} vs {seconds} s")
    print(
        a.arm, mapping, {t: sf.info(w).duration for t, w in waves.items()}, flush=True
    )
    a.out.mkdir(parents=True, exist_ok=True)
    delays = {}
    if a.arm.startswith("near") and a.headset_delays:
        d = json.loads(a.headset_delays.read_text())["speakers"]
        delays = {
            slot: d[source]["solo_envelope_lag_ms"] for source, slot in mapping.items()
        }
    rows, wall = recognize_cached(
        waves, timing, a.model, a.threads, a.out, a.arm, delays
    )
    scores = text_scores(ref, rows)
    result = {
        "arm": a.arm,
        "duration_s": seconds,
        "inference_s": wall,
        "threads": a.threads,
        "ref_to_system_by_occupancy": mapping,
        "alignment_ms": delays,
        "original_turns": len(timing),
        "decoded_segments": len(rows),
        **scores,
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / (a.arm + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    (a.out / (a.arm + ".jsonl")).write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf8"
    )
    print("Result", a.arm, scores["cpCER"], flush=True)


if __name__ == "__main__":
    main()
