"""Train spectral gains on development only and score disjoint speakers."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from benchmark_waveform import scores
from benchmark_examples import refmask, waveform_scores
from score_projection import summarize
from spectral_shrinkage import fit, refine


def records(cache, phase):
    out = []
    for meta in json.loads((cache / f"{phase}.json").read_text()):
        with np.load(cache / (meta["name"] + ".npz")) as z:
            out.append((meta, z["mixture"], z["tracks"], z["refs"], float(z["wall"])))
    return out


def model(triples):
    return fit([(m, x, r) for _, m, x, r, _ in triples])


def evaluate(rows, reg, alphas):
    cases = []
    for meta, mix, tracks, refs, base_s in rows:
        stats = {}
        for alpha in alphas:
            began = time.perf_counter()
            y = refine(mix, tracks, reg, alpha) if alpha else tracks
            elapsed = time.perf_counter() - began
            stats[str(alpha)] = {
                **scores(refs, y, mix, waveform_scores),
                "post_s": elapsed,
            }
        cases.append(
            {
                **meta,
                "methods": stats,
                "base_s": base_s,
                "input_s": len(mix) / 8000,
                "reference_triple_s": float(
                    0.02 * np.stack([refmask(x) for x in refs]).all(axis=0).sum()
                ),
            }
        )
    return {
        "cases": cases,
        "summary": {
            str(a): {
                k: summarize(
                    [
                        r["methods"][str(a)]
                        for r in cases
                        if k == "pooled" or r["condition"] == k
                    ]
                )
                for k in ["clean", "room", "pooled"]
            }
            for a in alphas
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("development", "test", "fresh"))
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    dev = records(a.cache, "development")
    a.out.mkdir(exist_ok=True, parents=True)
    digest = hashlib.sha256()
    for f in (Path(__file__), Path(__file__).with_name("spectral_shrinkage.py")):
        digest.update(f.read_bytes())
    fingerprint = digest.hexdigest()
    frozen = a.out / "shrinkage_selection.json"
    if a.phase == "development":
        # Both settings and speaker ordering are selected without test examples.
        alphas = [0, 0.25, 0.5, 0.75, 1.0]
        halves = [dev[:2], dev[2:]]
        arms = [evaluate(halves[i], model(halves[1 - i]), alphas) for i in (0, 1)]
        criteria = {
            str(alpha): sum(
                s["si_sdri_db"]
                + 0.5 * s["weakest_si_sdri_db"]
                + 0.5 * s["whole_si_sdri_db"]
                for arm in arms
                for k, s in arm["summary"][str(alpha)].items()
                if k != "pooled"
            )
            for alpha in alphas
        }
        chosen = max(alphas, key=lambda x: criteria[str(x)])
        frozen.write_text(
            json.dumps(
                dict(strength=chosen, objectives=criteria, sha256=fingerprint), indent=2
            )
            + "\n"
        )
        print(frozen.read_text(), flush=True)
    else:
        decision = json.loads(frozen.read_text())
        if decision["sha256"] != fingerprint:
            raise ValueError("Refit development after changing this benchmark.")
        result = evaluate(
            records(a.cache, a.phase),
            model(dev),
            list(dict.fromkeys([0, decision["strength"]])),
        )
        (a.out / f"shrinkage_{a.phase}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
