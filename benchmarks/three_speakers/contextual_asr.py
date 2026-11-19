"""Recover Mandarin context in a mono meeting while retaining per-speaker overlap hypotheses."""

import argparse
import os
from pathlib import Path
import json
import time
import numpy as np
from scipy.ndimage import uniform_filter1d
from rapidfuzz.distance import Levenshtein
import sherpa_onnx as sx
import soundfile as sf

from recognize import conversations, normalized, textgrid, text_scores, inference_spans


def decode_mixture(wav, model, threads=2):
    info = sf.info(wav)
    if info.channels != 1 or info.samplerate != 16000:
        raise ValueError("Expected mono 16 kHz WAV")
    if info.frames == 0:
        raise ValueError("Expected a nonempty audio file")
    rate = info.samplerate
    recognizer = sx.OfflineRecognizer.from_sense_voice(
        model=str(model / "model.int8.onnx"),
        tokens=str(model / "tokens.txt"),
        num_threads=threads,
        use_itn=False,
        language="zh",
    )
    # Compute the energy timeline in bounded blocks to support long recordings.
    length = info.frames
    hop = 160
    powers = []
    with sf.SoundFile(wav) as audio:
        for block in audio.blocks(blocksize=rate * 10, dtype="float32"):
            block = np.pad(block, (0, (-len(block)) % hop))
            powers.append(np.mean(block.reshape(-1, hop) ** 2, axis=1))
    smoothed = uniform_filter1d(np.concatenate(powers), size=13, mode="nearest")
    boundaries = [0]
    while boundaries[-1] + int(9.5 * rate) < length:
        left = (boundaries[-1] + 7 * rate) // hop
        right = (boundaries[-1] + 9 * rate) // hop
        cut = left + np.argmin(smoothed[left:right])
        boundaries.append(int(cut * hop))
    boundaries.append(length)
    tokens = []
    with sf.SoundFile(wav) as audio:
        for n, (a, b) in enumerate(zip(boundaries, boundaries[1:])):
            left = max(0, a - int(0.70 * rate))
            right = min(length, b + int(0.70 * rate))
            audio.seek(left)
            segment = audio.read(right - left, dtype="float32")
            rms = max(np.mean(segment**2) ** 0.5, 1e-7)
            peak = max(np.max(np.abs(segment)), 1e-7)
            segment *= min(0.035 / rms, 0.98 / peak, 20.0)
            stream = recognizer.create_stream()
            # Reuse the same model input dimensions across chunks to bound the ONNX arena.
            stream.accept_waveform(
                rate, np.pad(segment, (800, rate * 11 - len(segment) - 800))
            )
            recognizer.decode_stream(stream)
            result = stream.result
            if len(result.tokens) != len(result.timestamps):
                raise ValueError("The recognizer must provide timed tokens")
            for text, t in zip(result.tokens, result.timestamps):
                when = left / rate + t - 0.050
                if a / rate <= when < b / rate:
                    text = text.replace("▁", " ")
                    if normalized(text):
                        tokens.append(dict(time=float(when), text=text))
            if n % 20 == 0:
                print(f"Chunk {n + 1}/{len(boundaries) - 1}", flush=True)
    return tokens


def candidates(timing, tracks, old, tokens):
    stems = {p.stem.split("_")[-1]: p for p in tracks.glob("speaker_*.wav")}
    names = sorted(stems)
    if len(names) != 3:
        raise ValueError("Expected three speaker files")
    duration = sf.info(stems[names[0]]).duration
    frames = int(np.ceil(duration * 100)) + 1
    activity = np.zeros((frames, len(names)), np.float32)
    energies = np.empty_like(activity)
    rows = []
    base = {
        (str(r["speaker"]), round(r["start"], 3), round(r["end"], 3)): r["text"]
        for r in old
    }
    for j, name in enumerate(names):
        turns = [r for r in timing if str(r["speaker"]) == name]
        info = sf.info(stems[name])
        if (
            info.samplerate != 16000
            or info.channels != 1
            or abs(info.duration - duration) > 1 / 16000
        ):
            raise ValueError("Expected aligned 16 kHz tracks")
        parts = []
        with sf.SoundFile(stems[name]) as audio:
            for pcm in audio.blocks(blocksize=160000, dtype="float32"):
                pcm = np.pad(pcm, (0, (-len(pcm)) % 160))
                parts.append((pcm.reshape(-1, 160) ** 2).mean(1))
        power = np.concatenate(parts)
        energies[:, j] = np.maximum(
            1e-10,
            uniform_filter1d(
                np.pad(power, (0, frames - len(power))), size=13, mode="nearest"
            ),
        )
        for r in turns:
            activity[max(0, round(r["start"] * 100)) : round(r["end"] * 100), j] = 1
        for a, b in inference_spans(turns, duration):
            rows.append(
                dict(
                    speaker=name,
                    start=a,
                    end=b,
                    old=base.get((name, round(a, 3), round(b, 3)), ""),
                    mix=[],
                )
            )
    counts = activity.sum(1)
    maps = np.full((frames, len(names)), -1, np.int32)
    distances = np.full((frames, len(names)), 999, np.int32)
    for k, row in enumerate(rows):
        j = names.index(row["speaker"])
        a, b = round(row["start"] * 100), round(row["end"] * 100)
        own = activity[a:b, j] > 0
        row["overlap"] = float(np.mean(counts[a:b][own] > 1)) if np.any(own) else 1.0
        lo, hi = max(0, a - 40), min(frames, b + 40)
        t = np.arange(lo, hi)
        d = np.maximum(np.maximum(a - t, t - b), 0)
        take = d < distances[lo:hi, j]
        maps[t[take], j] = k
        distances[t[take], j] = d[take]
    support = uniform_filter1d(activity, size=19, axis=0, mode="nearest")
    for token in tokens:
        t = int(np.clip(round(token["time"] * 100), 0, frames - 1))
        nearby = maps[t] >= 0
        if not nearby.any():
            continue
        energy = energies[t]
        # Favor active and locally louder stems; a nearby inactive turn may recover a clipped edge.
        below_peak = np.clip(np.log10(energy.max() / np.maximum(energy, 1e-10)), 0, 4)
        conf = support[t] - 0.20 * below_peak - distances[t] * 0.012
        j = np.argmax(np.where(nearby, conf, -np.inf))
        rows[maps[t, j]]["mix"].append(token["text"])
    for r in rows:
        r["mix"] = "".join(r["mix"]).strip()
    return sorted(rows, key=lambda r: (r["start"], r["speaker"]))


def select(pairs, params):
    overlap_max, guard, empty_policy = params
    output = []
    for r in pairs:
        mixed, old = r["mix"], r["old"]
        isolated = r["overlap"] <= overlap_max
        # A short fragment with uncertain ownership is allowed to fall back to its stem.
        use = isolated and (
            bool(mixed) or (empty_policy and r["end"] - r["start"] < guard)
        )
        text = mixed if use else old
        if text:
            output.append(
                {k: r[k] for k in ("start", "end", "speaker")} | {"text": text}
            )
    return output


def restore_supported_text(pairs, rows, settings):
    """Restore the original rendering when the stem contains the whole contextual text."""
    anchor, max_extra, minimum_ratio, minimum_overlap = settings
    keyed = {(r["speaker"], r["start"], r["end"]): r for r in pairs}
    restored = []
    for row in rows:
        pair = keyed[row["speaker"], row["start"], row["end"]]
        mixed = normalized(pair["mix"])
        stem = normalized(pair["old"])
        if (
            row["text"] == pair["mix"]
            and len(mixed) >= anchor
            and 0 < len(stem) - len(mixed) <= max_extra
            and len(mixed) / len(stem) >= minimum_ratio
            and mixed in stem
            and pair["overlap"] >= minimum_overlap
        ):
            row = dict(row, text=pair["old"])
        restored.append(row)
    return restored


def protect_long_turns(pairs, rows, settings):
    """Keep a long stem transcript when the alternative adds insufficient characters."""
    selected = {(r["speaker"], r["start"], r["end"]): r["text"] for r in rows}
    result = []
    for r in pairs:
        chosen = selected.get((r["speaker"], r["start"], r["end"]), "")
        original = r["old"]
        n, m = len(normalized(chosen)), len(normalized(original))
        relative_gain = (n - m) / max(1, n, m)
        if (
            r["end"] - r["start"] > settings["minimum_turn_seconds"]
            and relative_gain <= settings["maximum_relative_character_gain"]
        ):
            chosen = original
        if chosen:
            result.append(
                {k: r[k] for k in ("start", "end", "speaker")} | {"text": chosen}
            )
    return result


def read_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def fit(pairs, ref):
    base = select(pairs, (-1, 0, False))
    baseline = text_scores(ref, base)
    match = {it["system"]: it["reference"] for it in baseline["speaker_scores"]}
    truth = conversations(ref)
    candidates = []
    for limit in (-1.0, 0.0, 0.05, 0.15, 0.35, 0.55, 0.8, 1.0):
        for guard in (0.0, 0.40, 0.8, 1.6):
            params = (limit, guard, bool(guard))
            streams = conversations(select(pairs, params))
            error = sum(
                Levenshtein.distance(truth[r], streams.get(h, ""))
                for h, r in match.items()
            )
            candidates.append((error, limit, guard, params))
    return min(candidates)[-1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mix", type=Path)
    p.add_argument("--model", type=Path)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--tokens", type=Path, required=True)
    p.add_argument("--tracks", type=Path)
    p.add_argument("--timing", type=Path)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--params", type=Path)
    p.add_argument("--reference", type=Path)
    p.add_argument("--fit", action="store_true")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    if a.mix:
        # Share a cache safely when benchmarking multiple speaker configurations on one mix.
        a.tokens.parent.mkdir(parents=True, exist_ok=True)
        inputs = [a.mix, a.model / "model.int8.onnx", a.model / "tokens.txt"]
        fingerprint = [
            [str(f.resolve()), f.stat().st_size, f.stat().st_mtime_ns] for f in inputs
        ]
        import fcntl  # CLI cache lock on Linux benchmark workers

        with a.tokens.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            cached = json.loads(a.tokens.read_text()) if a.tokens.exists() else {}
            if cached.get("recipe") != 1 or cached.get("inputs") != fingerprint:
                started = time.perf_counter()
                rows = decode_mixture(a.mix, a.model, a.threads)
                data = dict(
                    tokens=rows,
                    wall_s=time.perf_counter() - started,
                    recipe=1,
                    inputs=fingerprint,
                )
                temp = a.tokens.with_suffix(".tmp")
                temp.write_text(
                    json.dumps(data, ensure_ascii=False) + "\n", encoding="utf8"
                )
                os.replace(temp, a.tokens)
    if a.tracks:
        timing = json.loads(a.timing.read_text())["segments"]
        prior = read_jsonl(a.baseline)
        tokens = json.loads(a.tokens.read_text())["tokens"]
        pairs = candidates(timing, a.tracks, prior, tokens)
        ref = None
        if a.reference:
            ref = (
                textgrid(a.reference)
                if a.reference.suffix.lower() == ".textgrid"
                else json.loads(a.reference.read_text())
            )
        if a.fit:
            a.params.write_text(json.dumps(fit(pairs, ref)) + "\n")
        selected = select(pairs, json.loads(a.params.read_text()))
        a.out.mkdir(exist_ok=True, parents=True)
        (a.out / "transcript.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected),
            encoding="utf8",
        )
        for sp in ["all"] + sorted({r["speaker"] for r in timing}):
            use = (
                selected
                if sp == "all"
                else [r for r in selected if r["speaker"] == str(sp)]
            )
            file = "transcript.txt" if sp == "all" else f"speaker_{sp}.txt"
            (a.out / file).write_text(
                "".join(
                    f"[{r['start']:.2f}–{r['end']:.2f}] {r['speaker']}  {r['text']}\n"
                    for r in use
                ),
                encoding="utf8",
            )
        if ref:
            score = {
                "previous": text_scores(ref, prior),
                "candidate": text_scores(ref, selected),
            }
            (a.out / "scores.json").write_text(json.dumps(score, indent=2) + "\n")
            print({k: v["cpCER"] for k, v in score.items()}, flush=True)


if __name__ == "__main__":
    main()
