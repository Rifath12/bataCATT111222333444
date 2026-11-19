"""Export and verify the regression trees fitted to the four development mixtures."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from refine_three import SpectralRefiner
from score_shrinkage import records, model
from spectral_shrinkage import observations, refine


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    fitted = model(records(a.cache, "development"))
    trees = []
    for (tree,) in fitted._predictors:
        nodes = []
        for node in tree.nodes:
            if node["is_leaf"]:
                nodes.append([float(node["value"])])
            else:
                nodes.append(
                    [
                        int(node["feature_idx"]),
                        float(node["num_threshold"]),
                        int(node["left"]),
                        int(node["right"]),
                    ]
                )
        trees.append(nodes)
    alpha = json.loads(a.selection.read_text())["strength"]
    data = {
        "format": "spectral_gains_v1",
        "features": 7,
        "strength": alpha,
        "baseline": float(fitted._baseline_prediction[0, 0]),
        "trees": trees,
    }
    a.out.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")
    portable = SpectralRefiner(a.out)
    rng = np.random.default_rng(567)
    x = rng.normal(size=(10000, 7)).astype(np.float32)
    np.testing.assert_allclose(
        portable.predict(x), fitted.predict(x), rtol=1e-12, atol=1e-12
    )
    for meta, mix, tracks, refs, _ in records(a.cache, "fresh")[:2]:
        _, features, _ = observations(mix, tracks)
        x = features.reshape(-1, features.shape[-1])
        np.testing.assert_allclose(
            portable.predict(x), fitted.predict(x), rtol=1e-12, atol=1e-12
        )
        began = time.perf_counter()
        new = portable(mix, tracks)
        seconds = time.perf_counter() - began
        expected = refine(mix, tracks, fitted, alpha)
        np.testing.assert_allclose(new, expected, rtol=1e-6, atol=1e-7)
        print(meta["name"], seconds, np.max(np.abs(new - expected)), flush=True)


if __name__ == "__main__":
    main()
