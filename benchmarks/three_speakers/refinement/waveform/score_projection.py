"""Select and evaluate a mono mixture-projection refiner on fixed Mandarin triplets."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from benchmark_examples import refmask, waveform_scores
from benchmark_waveform import scores
from mixture_projection import refine_tracks

VARIANTS = {
    "residual_half": (False, 0.5),
    "frequency": (True, 0.0),
    "frequency_half": (True, 0.5),
    "frequency_full": (True, 1.0),
}


def summarize(rows):
    return {
        key: float(np.median([v for row in rows for v in np.atleast_1d(row[key])]))
        for key in ("si_sdri_db", "weakest_si_sdri_db", "whole_si_sdri_db")
    }


def evaluate(cache, phase, methods):
    rows = []
    for row in json.loads((cache / f"{phase}.json").read_text()):
        with np.load(cache / (row["name"] + ".npz")) as data:
            tracks, mix, refs = data["tracks"], data["mixture"], data["refs"]
            triple = np.stack([refmask(x) for x in refs]).all(axis=0).sum() * 0.02
            stats = {}
            for method in methods:
                began = time.perf_counter()
                if method == "single_pass":
                    result = tracks
                else:
                    f, g = VARIANTS[method]
                    result = refine_tracks(mix, tracks, frequency_fit=f, fraction=g)
                post_s = time.perf_counter() - began
                stats[method] = {
                    **scores(refs, result, mix, waveform_scores),
                    "post_s": post_s,
                }
            rows.append(
                {
                    **row,
                    "reference_triple_s": float(triple),
                    "input_s": len(mix) / 8000,
                    "base_s": float(data["wall"]),
                    "methods": stats,
                }
            )
    summary = {
        method: {
            condition: summarize(
                [
                    row["methods"][method]
                    for row in rows
                    if condition == "pooled" or row["condition"] == condition
                ]
            )
            for condition in ("clean", "room", "pooled")
        }
        for method in methods
    }
    return dict(summary=summary, cases=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("development", "test", "fresh"))
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    freeze = a.out / "selection.json"
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).with_name("mixture_projection.py")):
        digest.update(path.read_bytes())
    fingerprint = digest.hexdigest()
    if a.phase == "development":
        methods = ["single_pass", *VARIANTS]
    else:
        fixed = json.loads(freeze.read_text())
        if fixed["script_sha256"] != fingerprint:
            raise ValueError("Changed frozen scoring or inference")
        methods = list(dict.fromkeys(["single_pass", fixed["method"]]))
    result = evaluate(a.cache, a.phase, methods)
    (a.out / (a.phase + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    if a.phase == "development":

        def objective(key):
            return sum(
                s["si_sdri_db"]
                + 0.50 * s["weakest_si_sdri_db"]
                + 0.50 * s["whole_si_sdri_db"]
                for k, s in result["summary"][key].items()
                if k != "pooled"
            )

        chosen = max(methods, key=objective)
        freeze.write_text(
            json.dumps(
                dict(
                    method=chosen,
                    script_sha256=fingerprint,
                    objective={k: objective(k) for k in methods},
                ),
                indent=2,
            )
            + "\n"
        )
        print("chosen", chosen, flush=True)
    print(json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
