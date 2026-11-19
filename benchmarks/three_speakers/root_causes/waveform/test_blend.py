"""Evaluate an optional fixed two-pass ensemble on the already separated waveforms."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import linear_sum_assignment
import sherpa_onnx as sx

from test_wave import evaluate_cases, summarize


def aligned_blend(baseline, filtered):
    centered_a = baseline - baseline.mean(1, keepdims=True)
    centered_b = filtered - filtered.mean(1, keepdims=True)
    norm_a = np.linalg.norm(centered_a, axis=1)
    norm_b = np.linalg.norm(centered_b, axis=1)
    correlation = (centered_a @ centered_b.T) / np.maximum(
        norm_a[:, None] * norm_b[None, :], 1e-8
    )
    first, second = linear_sum_assignment(-np.abs(correlation))
    result = baseline.copy()
    for a, b in zip(first, second, strict=True):
        if abs(correlation[a, b]) < 0.25:
            continue
        gain = np.clip(norm_a[a] / max(1e-8, norm_b[b]), 0.25, 4.0) * np.sign(
            correlation[a, b]
        )
        result[a] = 0.5 * (baseline[a] + gain * filtered[b])
    return result


def blend_cases(data, cache, phase, rows, recognizer, backend):
    for idx in range(1, len(rows) + 1):
        path = cache / f"{phase}_{idx}_blend.npz"
        if path.is_file():
            continue
        with (
            np.load(cache / f"{phase}_{idx}_room_baseline.npz") as base,
            np.load(cache / f"{phase}_{idx}_wpe.npz") as filtered,
        ):
            start = perf_counter()
            blended = aligned_blend(base["tracks"], filtered["tracks"])
            seconds = (
                perf_counter()
                - start
                + float(base["seconds"])
                + float(filtered["seconds"])
            )
            np.savez(path, tracks=blended, seconds=seconds)
    cases = evaluate_cases(
        rows, phase, None, {backend: recognizer}, data / "sources", ["blend"], cache
    )
    (cache / f"{phase}_blend_{backend}.json").write_text(
        json.dumps(cases, ensure_ascii=False)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["data", "out", "sense", "zipformer"]:
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument(
        "--phase", required=True, nargs="+", choices=["development", "repeat", "new"]
    )
    p.add_argument(
        "--stage", required=True, choices=["sensevoice", "zipformer", "score"]
    )
    a = p.parse_args()
    rows = json.loads((a.data / "manifest.json").read_text())["groups"]
    cache = a.out / "cache"
    if a.stage == "score":
        for phase in a.phase:
            cases = json.loads((cache / f"{phase}_blend_sensevoice.json").read_text())
            others = json.loads((cache / f"{phase}_blend_zipformer.json").read_text())
            for case, other in zip(cases, others, strict=True):
                case["arms"]["blend"]["asr"]["zipformer"] = other["arms"]["blend"][
                    "asr"
                ]["zipformer"]
            (a.out / f"{phase}_blend.json").write_text(
                json.dumps(
                    {"summary": summarize(cases), "cases": cases},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )
        return
    if a.stage == "sensevoice":
        recognizer = sx.OfflineRecognizer.from_sense_voice(
            model=str(a.sense / "model.int8.onnx"),
            tokens=str(a.sense / "tokens.txt"),
            language="zh",
            use_itn=False,
            num_threads=1,
        )
    else:
        recognizer = sx.OfflineRecognizer.from_zipformer_ctc(
            model=str(a.zipformer / "model.int8.onnx"),
            tokens=str(a.zipformer / "tokens.txt"),
            num_threads=1,
        )
    for phase in a.phase:
        blend_cases(a.data, cache, phase, rows[phase], recognizer, a.stage)


if __name__ == "__main__":
    main()
