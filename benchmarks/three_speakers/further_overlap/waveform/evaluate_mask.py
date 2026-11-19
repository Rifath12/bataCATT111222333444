"""Compare recognition after spectral reconstruction with fixed speaker assignment inputs."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import sherpa_onnx
import soundfile as sf

from contextual_asr import (
    candidates,
    protect_long_turns,
    read_jsonl,
    restore_supported_text,
    select,
)
from diarize import masks
from duplicate_leakage import apply, encounters
from recognize import inference_spans, text_scores, textgrid
from spectral_mixture import PROFILES, separated_overlap


def encode(rows, path):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def render(args, segments, profiles):
    """Write PCM-identical solo estimates and bounded-memory overlap reconstructions."""
    from render_asr_mask import RATE, BLOCK_SIZE, ACTIVITY_HOP, PADDING
    import tempfile

    with (
        ExitStack() as stack,
        tempfile.TemporaryDirectory(dir=args.output) as temporary,
    ):
        original = stack.enter_context(sf.SoundFile(args.audio))
        names, activity = masks(segments, len(original) / RATE)
        stems = [
            stack.enter_context(sf.SoundFile(args.tracks / f"speaker_{s}.wav"))
            for s in names
        ]
        if len(names) != 3 or any(
            f.samplerate != RATE or f.channels != 1 or len(f) != len(original)
            for f in [original, *stems]
        ):
            raise ValueError(
                "Expected three aligned mono tracks and original 16 kHz mono audio"
            )
        work = Path(temporary)
        handles = {}
        peaks = dict.fromkeys(profiles, 0.0)
        for method in profiles:
            handles[method] = [
                stack.enter_context(
                    sf.SoundFile(
                        work / f"{method}_{s}.wav",
                        "w+",
                        samplerate=RATE,
                        channels=1,
                        subtype="FLOAT",
                    )
                )
                for s in names
            ]
        for start in range(0, len(original), BLOCK_SIZE):
            end = min(len(original), start + BLOCK_SIZE)
            left, right = max(0, start - PADDING), min(len(original), end + PADDING)
            for f in [original, *stems]:
                f.seek(left)
            wave = original.read(right - left, dtype="float32")
            estimates = np.stack([f.read(right - left, dtype="float32") for f in stems])
            flags = activity[
                :, left // ACTIVITY_HOP : (right + ACTIVITY_HOP - 1) // ACTIVITY_HOP
            ]
            for method in profiles:
                samples = separated_overlap(wave, estimates, flags, method)[
                    :, start - left : end - left
                ]
                peaks[method] = max(peaks[method], float(np.abs(samples).max()))
                for f, row in zip(handles[method], samples, strict=True):
                    f.write(row)
        for method in profiles:
            out = args.output / method
            out.mkdir(exist_ok=True)
            gain = min(1.0, 0.98 / max(1e-8, peaks[method]))
            for f, name in zip(handles[method], names, strict=True):
                f.seek(0)
                with sf.SoundFile(
                    out / f"speaker_{name}.wav",
                    "w",
                    samplerate=RATE,
                    channels=1,
                    subtype="PCM_16",
                ) as g:
                    while len(block := f.read(BLOCK_SIZE, dtype="float32")):
                        g.write(block * gain)


def old_scores(base):
    if not base:
        return {}
    return {
        (str(r["speaker"]), r["start"], r["end"]): r["text"] for r in read_jsonl(base)
    }


def decode(args, segments, variants):
    model = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(args.asr / "model.int8.onnx"),
        tokens=str(args.asr / "tokens.txt"),
        num_threads=args.threads,
        language="zh",
        use_itn=False,
    )
    baseline = old_scores(args.baseline)
    results = {k: [] for k in variants}
    reuse = dict.fromkeys(variants, 0)
    for name in sorted({str(s["speaker"]) for s in segments}):
        with ExitStack() as stack:
            files = {
                k: stack.enter_context(sf.SoundFile(path / f"speaker_{name}.wav"))
                for k, path in variants.items()
            }
            reader = next(iter(files.values()))
            for a, b in inference_spans(
                [r for r in segments if str(r["speaker"]) == name], len(reader) / 16000
            ):
                cache = {}
                for method, source in files.items():
                    source.seek(round(a * 16000))
                    data = source.read(
                        round(b * 16000) - round(a * 16000), dtype="float32"
                    )
                    key = data.tobytes()
                    if method == "previous" and args.baseline:
                        word = baseline.get((name, a, b), "")
                        cache[key] = word
                        reuse[method] += 1
                    elif key in cache:
                        word = cache[key]
                        reuse[method] += 1
                    else:
                        rms = max(float(np.sqrt(np.mean(data**2))), 1e-7)
                        peak = max(float(np.abs(data).max()), 1e-7)
                        stream = model.create_stream()
                        stream.accept_waveform(
                            16000,
                            np.pad(
                                data * min(0.035 / rms, 0.98 / peak, 20), (800, 1280)
                            ),
                        )
                        model.decode_stream(stream)
                        word = stream.result.text.strip()
                        cache[key] = word
                    if word:
                        results[method].append(
                            dict(speaker=name, start=a, end=b, text=word)
                        )
        print("Finished decoding", name, "reused", reuse, flush=True)
    return {
        k: sorted(rows, key=lambda r: (r["start"], r["speaker"]))
        for k, rows in results.items()
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["audio", "tracks", "timing", "asr", "output", "settings", "tokens"]:
        p.add_argument("--" + key, type=Path, required=True)
    for key in ["previous", "baseline", "reference"]:
        p.add_argument("--" + key, type=Path)
    p.add_argument("--profiles", nargs="+", choices=PROFILES, default=PROFILES)
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args()
    if a.baseline and not a.previous:
        p.error("Both previous tracks and transcript are needed to reuse hypotheses")
    a.output.mkdir(exist_ok=True, parents=True)
    segments = json.loads(a.timing.read_text())["segments"]
    start = perf_counter()
    profiles = [k for k in a.profiles if k != "previous" or not a.previous]
    if profiles:
        render(a, segments, profiles)
    variants = {
        k: (a.previous if k == "previous" and a.previous else a.output / k)
        for k in a.profiles
    }
    elapsed = perf_counter() - start
    print("Rendering elapsed", round(elapsed, 2), flush=True)
    raw = decode(a, segments, variants)
    ref = None
    if a.reference:
        ref = (
            textgrid(a.reference)
            if a.reference.suffix.lower() == ".textgrid"
            else json.loads(a.reference.read_text())
        )
    words = json.loads(a.tokens.read_text())["tokens"]
    params = {
        name: json.loads((a.settings / f"{name}_settings.json").read_text())
        for name in ("context", "restoration", "long_turn", "duplicate")
    }
    scores = {}
    for key, rows in raw.items():
        dest = a.output / key
        dest.mkdir(exist_ok=True)
        frames = candidates(segments, variants[key], rows, words)
        contextual = select(frames, params["context"])
        restored = restore_supported_text(frames, contextual, params["restoration"])
        protected = protect_long_turns(frames, restored, params["long_turn"])
        filtered = apply(
            protected, encounters(protected, variants[key]), params["duplicate"]
        )
        encode(rows, dest / "stem_asr.jsonl")
        encode(filtered, dest / "transcript.jsonl")
        if ref is not None:
            scores[key] = {
                "stem": text_scores(ref, rows),
                "full": text_scores(ref, filtered),
            }
            print(
                key,
                {k: v["totals"]["errors"] for k, v in scores[key].items()},
                flush=True,
            )
    if ref is not None:
        (a.output / "scores.json").write_text(
            json.dumps(
                dict(
                    transcription=scores,
                    render_seconds=elapsed,
                    total_seconds=perf_counter() - start,
                ),
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
