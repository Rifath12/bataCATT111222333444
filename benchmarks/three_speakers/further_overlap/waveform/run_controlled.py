"""Controlled tests of spectral reconstruction with separate waveform and character metrics."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import sherpa_onnx as sx

from benchmark_examples import waveform_scores
from recognize import text_scores
from spectral_mixture import PROFILES, separated_overlap
from test_wave import create_sources, recognize_native


def run(data, cases, backend, profiles):
    manifest = json.loads((data / "manifest.json").read_text())
    model = sx.OfflineRecognizer.from_sense_voice(
        model=str(backend / "model.int8.onnx"),
        tokens=str(backend / "tokens.txt"),
        language="zh",
        use_itn=False,
        num_threads=1,
    )
    for idx, group in enumerate(manifest["groups"][cases], 1):
        dry, wet, mix, _, overlap = create_sources(data / "sources", group, idx)
        f = lambda y: resample_poly(y, 2, 1, axis=-1).astype("f")
        wave, truth = f(mix), f(wet)
        cache = data / "results" / "cache"
        with np.load(cache / f"{cases}_{idx}_room_baseline.npz") as base:
            old = f(base["tracks"])
        active = np.ones((3, (len(wave) + 159) // 160), dtype=bool)
        references = [
            {"speaker": str(i), "start": 0.0, "text": r["text"]}
            for i, r in enumerate(group)
        ]
        result = {"room_baseline": old}
        result.update(
            {
                profile: separated_overlap(wave, old, active, profile)
                for profile in profiles
            }
        )
        for method, tracks in result.items():
            signal = waveform_scores(truth, dict(zip("ABC", tracks)), wave)
            score = text_scores(references, recognize_native(model, tracks))
            yield dict(
                case=idx,
                method=method,
                exact_three_seconds=overlap,
                separation=signal,
                text=score,
            )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for path in ("data", "model", "out"):
        cli.add_argument("--" + path, type=Path, required=True)
    cli.add_argument("--panel", choices=("development", "new", "repeat"), required=True)
    cli.add_argument("--profiles", choices=PROFILES, nargs="+", default=PROFILES)
    a = cli.parse_args()
    runs = list(run(a.data, a.panel, a.model, a.profiles))
    for profile in ["room_baseline", *a.profiles]:
        scores = [r["text"]["totals"] for r in runs if r["method"] == profile]
        print(
            profile,
            sum(n["errors"] for n in scores),
            sum(n["reference_chars"] for n in scores),
        )
    a.out.write_text(json.dumps(runs, indent=2) + "\n")


if __name__ == "__main__":
    main()
