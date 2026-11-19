"""Compare waveform isolation and CPU inference on three-voice Mandarin mixtures."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.signal import fftconvolve
import torch
from torch import nn

from sepformer_three import SepFormer


def room_response(voice, index, rng):
    # Keep the direct wave and add modest broadband reflections unique to each voice.
    taper = np.exp(-np.arange(1450) / 315)
    response = (rng.standard_normal(1450) * taper * 0.008).astype(np.float32)
    response[0] = 1
    for delay, gain in (
        (95 + 23 * index, 0.24),
        (234 + 39 * index, 0.13),
        (369 + 21 * index, -0.10),
    ):
        response[delay] += gain
    return fftconvolve(voice, response)[: len(voice)].astype(np.float32)


def make_cases(fixtures, cache, source_loader):
    rng = np.random.default_rng(31719)
    for directory in sorted(fixtures.glob("evaluation*_3voices")):
        original = json.loads((directory / "reference.json").read_text())
        speakers = [utts[0]["recording_id"] for utts in original["speakers"].values()]
        source_files = [sorted(cache.glob(f"{sid}*.wav")) for sid in speakers]
        if any(len(files) < 3 for files in source_files):
            raise RuntimeError(
                f"Expected three independent utterances from each speaker in {directory.name}"
            )
        signals = [
            [source_loader(file) for file in files[:3]] for files in source_files
        ]
        for phrase in range(3):
            start = np.roll([1000, 2600, 4100], phrase)
            sources = [signals[spk][(phrase + spk) % 3] for spk in range(3)]
            frames = (
                max(offset + len(src) for offset, src in zip(start, sources)) + 1600
            )
            dry = np.zeros((3, frames), np.float32)
            for i, (offset, voice) in enumerate(zip(start, sources)):
                dry[i, offset : offset + len(voice)] = (
                    voice * np.roll([1.0, 0.69, 0.47], phrase)[i]
                )
            for condition in ("clean", "room"):
                refs = (
                    dry
                    if condition == "clean"
                    else np.stack([room_response(s, i, rng) for i, s in enumerate(dry)])
                )
                mix = refs.sum(axis=0)
                if condition == "room":
                    noise = rng.standard_normal(len(mix)).astype(np.float32)
                    mix += noise * (0.1 * np.sqrt(np.mean(mix**2)))
                gain = 0.92 / max(0.92, float(np.abs(mix).max()))
                yield (
                    f"{directory.name}.{phrase + 1}.{condition}",
                    condition,
                    refs * gain,
                    mix * gain,
                )


def dynamic_int8(network):
    compact = torch.ao.quantization.quantize_dynamic(
        network,
        {nn.LSTM, nn.Linear},
        dtype=torch.qint8,
    )
    # Asteroid calls flatten_parameters on the float RNN before every forward.
    # The quantized RNN already stores its weights in a packed representation.
    for layer in compact.modules():
        if isinstance(layer, torch.ao.nn.quantized.dynamic.LSTM):
            layer.flatten_parameters = lambda: None
    return compact


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--sepformer", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=3)
    args = p.parse_args()
    sys.path.insert(0, str(args.project))
    from process_meeting import Separation, load_separator
    from benchmark_examples import refmask, waveform_scores
    from prepare_validation_cases import prepared_source

    torch.set_num_threads(args.threads)
    torch.backends.quantized.engine = "fbgemm"
    original = load_separator(args.models, "tuned", "cpu")
    compact = dynamic_int8(original)
    baseline = Separation(original, "cpu")
    quantized = Separation(compact, "cpu")
    models = {
        "DPRNN_FP32": lambda wave: baseline.infer(wave, 3)[0],
        "DPRNN_INT8": lambda wave: quantized.infer(wave, 3)[0],
        "SepFormer_Libri3Mix": SepFormer(args.sepformer),
    }
    tests = list(make_cases(args.fixtures, args.cache, prepared_source))
    for model in models.values():
        model(tests[0][3])
    aggregate = defaultdict(list)
    recordings = []
    for i, (name, condition, references, mix) in enumerate(tests):
        exact = (
            float((np.stack([refmask(s) for s in references]).sum(0) == 3).sum()) * 0.02
        )
        if exact < 0.10:
            continue
        group = {
            "name": name,
            "condition": condition,
            "audio_seconds": len(mix) / 8000,
            "exact_three_seconds": exact,
            "models": {},
        }
        # Rotate processing order to distribute cache and load effects.
        choices = list(models)
        for label in choices[i % len(choices) :] + choices[: i % len(choices)]:
            started = time.perf_counter()
            output = models[label](mix)
            wall = time.perf_counter() - started
            scored = waveform_scores(references, dict(zip("ABC", output)), mix)
            improvements = [x["exact_overlap_improvement_dB"] for x in scored]
            group["models"][label] = {"elapsed_s": wall, "si_sdri_db": improvements}
            aggregate[condition, label].append((wall, improvements))
        recordings.append(group)
        print(
            name,
            round(exact, 2),
            {
                k: [
                    round(v["elapsed_s"], 2),
                    round(float(np.median(v["si_sdri_db"])), 2),
                ]
                for k, v in group["models"].items()
            },
            flush=True,
        )
    report = {
        condition: {
            label: {
                "seconds": sum(x[0] for x in aggregate[condition, label]),
                "median_exact_three_si_sdri_db": float(
                    np.median([v for _, x in aggregate[condition, label] for v in x])
                ),
            }
            for label in models
        }
        for condition in ("clean", "room")
    }
    args.out.write_text(
        json.dumps(
            {"cases": recordings, "summary": report, "threads": args.threads}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    print(report, flush=True)


if __name__ == "__main__":
    main()
