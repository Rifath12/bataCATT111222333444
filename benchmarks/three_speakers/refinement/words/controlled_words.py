"""Score complete Mandarin utterances through a three-source, simulated-room mixture."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.signal import resample_poly
import sherpa_onnx as sx
import soundfile as sf
import torch

from benchmark_examples import refmask
from benchmark_waveform import room_response
from recognize import text_scores
from refine_three import SpectralRefiner
from sepformer_three import SepFormer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recognize(model, streams):
    rows = []
    for channel, samples in enumerate(streams):
        speech = resample_poly(samples, 2, 1).astype("float32")
        rms = max(float(np.sqrt(np.mean(speech**2))), 1e-7)
        peak = max(float(np.abs(speech).max()), 1e-7)
        stream = model.create_stream()
        stream.accept_waveform(
            16000, np.pad(speech * min(0.035 / rms, 0.98 / peak, 20), (800, 1280))
        )
        model.decode_stream(stream)
        rows.append(
            dict(speaker=str(channel), start=0.0, text=stream.result.text.strip())
        )
    return rows


def make_mixture(paths, index):
    rate = 8000
    starts = np.roll(np.rint(np.array([0.0, 0.12, 0.24]) * rate).astype(int), index)
    offsets = np.roll([0.0, -4.0, -8.0], index)
    sources = []
    for path, start, offset in zip(paths, starts, offsets, strict=True):
        pcm, sr = sf.read(path, dtype="float32")
        if pcm.ndim != 1:
            raise ValueError(path)
        ratio = Fraction(rate, sr)
        x = resample_poly(pcm, ratio.numerator, ratio.denominator)
        # Retain every sample and use the active signal level to set relative gains.
        frame_len = len(x) // 160 * 160
        active = np.repeat(refmask(x), 160)
        rms = np.sqrt(np.mean(x[:frame_len][active] ** 2))
        sources.append(
            np.pad(
                (x * (0.05 / rms) * 10 ** (offset / 20)).astype("float32"), (start, 0)
            )
        )
    size = max(map(len, sources)) + 1760
    dry = np.stack([np.pad(x, (0, size - len(x))) for x in sources])
    rng = np.random.default_rng(19112026 + index)
    wet = np.stack([room_response(x, i, rng) for i, x in enumerate(dry)])
    clean_mix = wet.sum(axis=0)
    noise = rng.normal(size=size).astype(np.float32)
    mixed = clean_mix + noise * np.sqrt(
        np.mean(clean_mix**2) / (np.mean(noise**2) * 10**1.5)
    )
    gain = min(1.0, 0.92 / max(np.abs(mixed)))
    triple = np.all(np.stack([refmask(x) for x in dry]), axis=0).sum() * 0.02
    return (
        dry,
        (gain * mixed).astype(np.float32),
        float(triple),
        offsets.tolist(),
        (starts / 8000).tolist(),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in (
        "manifest",
        "sources",
        "content",
        "separator",
        "recognizer",
        "filter",
        "output",
    ):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument(
        "--zipformer",
        type=Path,
        help="Optional second ASR model on all identical inputs",
    )
    p = ap.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    table = {}
    for line in p.content.read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            key, body = line.split("\t", 1)
            table[key] = "".join(body.split()[::2])
    sep = SepFormer(p.separator)
    refiner = SpectralRefiner(p.filter)
    asr = sx.OfflineRecognizer.from_sense_voice(
        model=str(p.recognizer / "model.int8.onnx"),
        tokens=str(p.recognizer / "tokens.txt"),
        language="zh",
        use_itn=False,
        num_threads=2,
    )
    zipformer = None
    if p.zipformer:
        zipformer = sx.OfflineRecognizer.from_zipformer_ctc(
            model=str(p.zipformer / "model.int8.onnx"),
            tokens=str(p.zipformer / "tokens.txt"),
            num_threads=2,
        )
    data = [
        row for row in json.loads(p.manifest.read_text()) if row["condition"] == "room"
    ]
    cases = []
    for number, row in enumerate(data, start=1):
        inputs = [pair[0] for pair in row["source_files"]]
        paths = [p.sources / name for name, _ in inputs]
        for path, (_, expected) in zip(paths, inputs, strict=True):
            if digest(path) != expected:
                raise ValueError(f"Checksum mismatch: {path}")
        truth = [
            dict(speaker=str(i), start=0.0, text=table[path.name])
            for i, path in enumerate(paths)
        ]
        dry, mixed, triple, levels, starts = make_mixture(paths, number)
        began = time.perf_counter()
        tracks = sep(mixed)
        separator_seconds = time.perf_counter() - began
        began = time.perf_counter()
        refined = refiner(mixed, tracks)
        filter_seconds = time.perf_counter() - began
        arms = {}
        for key, est in (
            ("isolated_dry", dry),
            ("sepformer", tracks),
            ("refined", refined),
        ):
            hypotheses = recognize(asr, est)
            arms[key] = dict(hypotheses=hypotheses, **text_scores(truth, hypotheses))
            if zipformer is not None:
                other = recognize(zipformer, est)
                arms["zipformer_on_" + key] = dict(
                    hypotheses=other, **text_scores(truth, other)
                )
        cases.append(
            dict(
                name=row["name"],
                files=inputs,
                truth=truth,
                levels_dB=levels,
                delay_s=starts,
                audio_s=len(mixed) / 8000,
                exact_triple_s=triple,
                separator_s=separator_seconds,
                spectral_filter_s=filter_seconds,
                arms=arms,
            )
        )
        print(
            row["name"],
            len(mixed) / 8000,
            triple,
            {a: b["cpCER"] for a, b in arms.items()},
            flush=True,
        )
    summary = {}
    for key in cases[0]["arms"]:
        pooled = {
            k: sum(c["arms"][key]["totals"][k] for c in cases)
            for k in (
                "reference_chars",
                "substitutions",
                "deletions",
                "insertions",
                "errors",
                "correct_reference_chars",
            )
        }
        pooled["cpCER"] = pooled["errors"] / pooled["reference_chars"]
        summary[key] = pooled
    p.output.write_text(
        json.dumps(
            dict(
                aishell3_revision="f20d5db4a31fe779ef07bb1af4ea92da5c786622",
                data_split="test",
                text_sha256=digest(p.content),
                filter_sha256=digest(p.filter),
                summary=summary,
                cases=cases,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(summary)


if __name__ == "__main__":
    main()
