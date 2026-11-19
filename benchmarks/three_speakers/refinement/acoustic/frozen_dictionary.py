"""Read and write portable numeric trees for the mono speaker-activity verifier."""

import json
from pathlib import Path

import numpy as np
from scipy.special import expit


FORMAT = "mandarin-dictionary-verifier-1"


class FrozenVerifier:
    def __init__(self, data):
        if data.get("format") != FORMAT:
            raise ValueError("Unknown dictionary-verifier version")
        self.features = data["feature_count"]
        self.bias = data["bias"]
        self.trees = []
        for tree in data["trees"]:
            cols, cuts, left, right, missing_left, values = map(np.asarray, zip(*tree))
            ids = np.arange(len(tree))
            branch = cols >= 0
            for child in (left[branch], right[branch]):
                if np.any((child <= ids[branch]) | (child >= len(tree))):
                    raise ValueError("Invalid child index in verifier tree")
            if np.any(cols >= self.features):
                raise ValueError("Invalid feature in verifier tree")
            self.trees.append((cols, cuts, left, right, missing_left, values))

    def predict_proba(self, features):
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.features:
            raise ValueError(f"Expected {self.features} input features")
        raw = np.full(len(x), self.bias, dtype=np.float64)
        for cols, cuts, left, right, missing_left, values in self.trees:
            nodes = np.zeros(len(x), dtype=np.intp)
            unresolved = np.arange(len(x))
            while len(unresolved):
                current = nodes[unresolved]
                live = cols[current] >= 0
                unresolved = unresolved[live]
                current = current[live]
                measured = x[unresolved, cols[current]]
                choose_left = np.where(
                    np.isnan(measured), missing_left[current], measured <= cuts[current]
                )
                nodes[unresolved] = np.where(choose_left, left[current], right[current])
            raw += values[nodes]
        positive = expit(raw)
        return np.column_stack([1.0 - positive, positive])


def save_frozen(path, model, thresholds, profile):
    if model is None:
        raise ValueError("Development selected a fallback without a fitted verifier")
    trees = []
    for (predictor,) in model._predictors:
        nodes = []
        for n in predictor.nodes:
            nodes.append(
                [
                    -1 if n["is_leaf"] else int(n["feature_idx"]),
                    float(n["num_threshold"]),
                    int(n["left"]),
                    int(n["right"]),
                    bool(n["missing_go_to_left"]),
                    float(n["value"]),
                ]
            )
        trees.append(nodes)
    data = {
        "format": FORMAT,
        "training_session": "R8008_M8013",
        "feature_count": model.n_features_in_,
        "bias": float(model._baseline_prediction.item()),
        "thresholds": thresholds,
        "previous_profile": profile,
        "node_columns": [
            "feature_or_negative_for_leaf",
            "threshold",
            "left",
            "right",
            "missing_goes_left",
            "value",
        ],
        "trees": trees,
    }
    # One tree per line keeps the document compact while permitting line-oriented diffs.
    header = {k: v for k, v in data.items() if k != "trees"}
    encoded = json.dumps(header, indent=2)[:-2] + ',\n  "trees": [\n'
    encoded += ",\n".join(
        "    " + json.dumps(tree, separators=(",", ":")) for tree in trees
    )
    Path(path).write_text(encoded + "\n  ]\n}\n", encoding="utf-8")


def load_frozen(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return FrozenVerifier(data), data["thresholds"], data["previous_profile"]
