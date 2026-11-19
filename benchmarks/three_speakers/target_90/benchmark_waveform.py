"""Compare three-source waveform refinements on fixed Mandarin mixtures."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import fftconvolve, istft, stft
import torch


GROUPS = {
    "development": [("0287", "0288", "0299"), ("0315", "0316", "0323")],
    "test": [
        ("0339", "0341", "0342"), ("0366", "0379", "0382"),
        ("0394", "0395", "0407"), ("0434", "0470", "0482"),
        ("0307", "0338", "0354"), ("0502", "0534", "0544"),
    ],
}
CONDITIONS = ("clean", "room")
METHODS = {
    "libri_baseline": ("libri",),
    "libri_mask": ("libri",),
    "wsj": ("wsj",),
    "libri_polarity": ("libri", "polarity"),
    "libri_wsj": ("libri", "wsj"),
    "libri_wsj_mask": ("libri", "wsj"),
}
RATE = 8000
SAMPLES = 27200
SEED = 19011


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_track(files, source_loader, start):
    utterances = [source_loader(file) for file in files]
    signal = np.concatenate((utterances[0], np.zeros(400), utterances[1]))
    length = min(len(signal), SAMPLES - start)
    y = np.zeros(SAMPLES, np.float32)
    y[start:start + length] = signal[:length]
    y[start + length - 120:start + length] *= np.linspace(1.0, 0.0, 120)
    return y


def room_response(signal, index, rng):
    length = 1760
    response = (rng.normal(size=length) * np.exp(-np.arange(length) / 325) * 0.017).astype(np.float32)
    response[:22] = 0
    response[0] = 1
    response[72 + 13 * index] = 0.38
    response[148 - 11 * index] = -0.25
    response[281 + 37 * index] = 0.17
    return fftconvolve(signal, response)[:len(signal)]


def fixtures(split, cache, source_loader):
    rng = np.random.default_rng(SEED + int(split == "test"))
    for idx, speakers in enumerate(GROUPS[split], start=1):
        paths = [sorted(cache.glob(f"SSB{speaker}*.wav"))[:2] for speaker in speakers]
        if any(len(files) != 2 for files in paths):
            raise ValueError(f"Missing two source files per speaker in {speakers}")
        offsets = np.roll(np.array([160, 590, 1000]), idx)
        gains = np.roll(10 ** (np.array([0, -4, -8]) / 20), idx)
        underlying = np.stack([
            source_track(files, source_loader, start) * gain
            for files, start, gain in zip(paths, offsets, gains)
        ]).astype(np.float32)
        provenance = [[(file.name, sha(file)) for file in files] for files in paths]
        for condition in CONDITIONS:
            if condition == "room":
                refs = np.stack([room_response(s, i, rng) for i, s in enumerate(underlying)])
                noise = rng.normal(size=SAMPLES).astype(np.float32)
                mix = refs.sum(0)
                mix += noise * np.sqrt(np.mean(mix**2) / (np.mean(noise**2) * 10 ** 1.5))
            else:
                refs = underlying
                mix = refs.sum(0)
            gain = min(1.0, 0.92 / np.abs(mix).max())
            yield f"{split}_{idx}_{condition}", condition, (refs * gain).astype(np.float32), (mix * gain).astype(np.float32), provenance


def consensus(left, right):
    """Track permutation, sign, and scale in estimated waves, with no reference audio."""
    left_norm = np.linalg.norm(left, axis=1).clip(1e-6)
    right_norm = np.linalg.norm(right, axis=1).clip(1e-6)
    similarity = (left @ right.T) / np.outer(left_norm, right_norm)
    rows, cols = linear_sum_assignment(-np.abs(similarity))
    correlation = similarity[rows, cols]
    gain = correlation * left_norm[rows] / right_norm[cols]
    use = 0.5 * np.clip((np.abs(correlation) - 0.25) / 0.35, 0.0, 1.0)
    result = left.copy()
    result[rows] = (1.0 - use[:, None]) * left[rows] + use[:, None] * gain[:, None] * right[cols]
    return result


def contrast_mask(tracks):
    _, _, bins = stft(tracks, fs=RATE, nperseg=256, noverlap=192, axis=-1)
    magnitude = np.abs(bins)
    ratio = (magnitude + 1e-8) / (magnitude.sum(axis=0) + 3e-8)
    _, result = istft(bins * np.maximum(ratio, 0.04)**0.30, fs=RATE, nperseg=256, noverlap=192, time_axis=-1, freq_axis=-2)
    return result[:, :tracks.shape[-1]]


def scores(refs, tracks, mixture, scorer):
    sources = scorer(refs, dict(zip("ABC", tracks)), mixture)
    weakest = min(sources, key=lambda score: score["source_rms"])
    return {
        "si_sdri_db": [s["exact_overlap_improvement_dB"] for s in sources],
        "whole_si_sdri_db": [s["whole_improvement_dB"] for s in sources],
        "weakest_si_sdri_db": weakest["exact_overlap_improvement_dB"],
    }


def evaluate(method, outputs):
    if method.startswith("libri_wsj"):
        estimated = consensus(outputs["libri"], outputs["wsj"])
    elif method == "libri_polarity":
        estimated = consensus(outputs["libri"], outputs["polarity"])
    else:
        estimated = outputs[METHODS[method][0]]
    return contrast_mask(estimated) if method.endswith("_mask") else estimated


def run(args, phase, chosen=None):
    from benchmark_examples import refmask, waveform_scores
    from prepare_validation_cases import prepared_source
    from sepformer_three import SepFormer

    methods = tuple(METHODS) if phase == "development" else tuple(dict.fromkeys(("libri_baseline", chosen)))
    required = {part for method in methods for part in METHODS[method]}
    libri = SepFormer(args.libri)
    wsj = SepFormer(args.wsj) if "wsj" in required else None
    runners = {"libri": libri, "polarity": lambda x: -libri(-x), "wsj": wsj}
    rows = []
    for case, condition, refs, mixture, provenance in fixtures(phase, args.cache, prepared_source):
        exact = np.count_nonzero(np.stack([refmask(s) for s in refs]).all(axis=0)) * .02
        if exact < 0.10:
            raise ValueError(f"Insufficient exact three-voice overlap in the frozen fixture {case}: {exact}")
        outputs, elapsed, row = {}, {}, {}
        for part in sorted(required):
            started = time.perf_counter()
            outputs[part] = runners[part](mixture)
            elapsed[part] = time.perf_counter() - started
        for method in methods:
            started = time.perf_counter()
            pred = evaluate(method, outputs)
            wall = time.perf_counter() - started + sum(elapsed[k] for k in METHODS[method])
            row[method] = {**scores(refs, pred, mixture, waveform_scores), "wall_seconds": wall}
        print(case, f"{exact:.2f} s", {k: round(float(np.median(v["si_sdri_db"])), 2) for k, v in row.items()}, flush=True)
        rows.append({"name":case,"condition":condition,"exact_three_seconds":exact,"input_seconds":len(mixture)/RATE,"source_files":provenance,"methods":row})

    summary = {}
    for method in methods:
        reports = {}
        for condition in (*CONDITIONS, "pooled"):
            cases = rows if condition == "pooled" else [x for x in rows if x["condition"] == condition]
            stats = [x["methods"][method] for x in cases]
            reports[condition] = {
                "median_si_sdri_db": float(np.median([v for r in stats for v in r["si_sdri_db"]])),
                "median_whole_si_sdri_db": float(np.median([v for r in stats for v in r["whole_si_sdri_db"]])),
                "weakest_median_si_sdri_db": float(np.median([r["weakest_si_sdri_db"] for r in stats])),
                "inference_seconds": sum(r["wall_seconds"] for r in stats),
                "exact_three_seconds": sum(r["exact_three_seconds"] for r in cases),
                "input_seconds": sum(r["input_seconds"] for r in cases),
            }
        summary[method] = reports
    return {"phase":phase,"seed":SEED,"threads":2,"summary":summary,"cases":rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("development", "test"))
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--libri", type=Path, required=True)
    p.add_argument("--wsj", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    sys.path[:0] = [str(args.project), str(args.previous)]
    selection = args.results / "selection.json"
    args.results.mkdir(parents=True, exist_ok=True)
    fingerprint = sha(Path(__file__))
    chosen = None
    if args.phase == "test":
        fixed = json.loads(selection.read_text())
        if fixed["script_sha256"] != fingerprint:
            raise ValueError("The benchmark script changed after development")
        chosen = fixed["method"]
    report = run(args, args.phase, chosen)
    (args.results / f"{args.phase}.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.phase == "development":
        def criterion(item):
            scores_by_room = item[1]
            return sum((scores_by_room[c]["median_si_sdri_db"] + 0.25 * scores_by_room[c]["weakest_median_si_sdri_db"]) for c in CONDITIONS)

        chosen = max(report["summary"].items(), key=criterion)[0]
        selection.write_text(json.dumps({"method":chosen,"script_sha256":fingerprint}, indent=2) + "\n")
    print(json.dumps({"chosen":chosen,"summary":report["summary"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
