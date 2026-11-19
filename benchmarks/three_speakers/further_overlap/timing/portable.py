"""Serialize numeric prediction trees and load them without training or pickle."""

import json

import numpy as np
from scipy.special import expit, softmax

FORMAT = "mandarin-structured-timing-v1"


def encode(model):
    def nodes(tree):
        return [
            [
                -1 if n["is_leaf"] else int(n["feature_idx"]),
                float(n["num_threshold"]),
                int(n["left"]),
                int(n["right"]),
                bool(n["missing_go_to_left"]),
                float(n["value"]),
            ]
            for n in tree.nodes
        ]

    outputs = zip(*model._predictors, strict=True)
    return {
        "features": model.n_features_in_,
        "classes": model.classes_.tolist(),
        "bias": model._baseline_prediction.ravel().tolist(),
        "trees": [[nodes(tree) for tree in output] for output in outputs],
    }


class NumericTrees:
    def __init__(self, data):
        self.features = data["features"]
        self.classes_ = np.asarray(data["classes"])
        self.bias = np.asarray(data["bias"])
        self.trees = []
        for output in data["trees"]:
            parsed = []
            for tree in output:
                cols, cuts, left, right, missing_left, values = map(
                    np.asarray, zip(*tree, strict=True)
                )
                rows = np.arange(len(tree))
                branch = cols >= 0
                for child in (left[branch], right[branch]):
                    if np.any((child <= rows[branch]) | (child >= len(tree))):
                        raise ValueError("Invalid child index in frozen tree")
                if np.any(cols >= self.features):
                    raise ValueError("Invalid feature index in frozen tree")
                parsed.append((cols, cuts, left, right, missing_left, values))
            self.trees.append(parsed)

    def decision_function(self, features):
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.features:
            raise ValueError(f"Expected {self.features} frozen features")
        raw = np.broadcast_to(self.bias, (len(features), len(self.trees))).copy()
        for output, trees in enumerate(self.trees):
            for cols, cuts, left, right, missing_left, values in trees:
                nodes = np.zeros(len(features), dtype=np.intp)
                unresolved = np.arange(len(features))
                while unresolved.size:
                    current = nodes[unresolved]
                    live = cols[current] >= 0
                    unresolved = unresolved[live]
                    current = current[live]
                    measured = features[unresolved, cols[current]]
                    choose_left = np.where(
                        np.isnan(measured),
                        missing_left[current],
                        measured <= cuts[current],
                    )
                    nodes[unresolved] = np.where(
                        choose_left, left[current], right[current]
                    )
                raw[:, output] += values[nodes]
        return raw[:, 0] if raw.shape[1] == 1 else raw

    def predict_proba(self, features):
        raw = self.decision_function(features)
        if raw.ndim == 1:
            yes = expit(raw)
            return np.column_stack([1 - yes, yes])
        return softmax(raw, axis=1)


def save(path, models, knobs, profile):
    result = {
        "format": FORMAT,
        "knobs": knobs,
        "profile": profile,
        "identity": encode(next(iter(models.values()))[1]),
        "counts": {
            name: {"independent": model[2], "parameters": encode(model[0])}
            for name, model in models.items()
        },
    }
    path.write_text(json.dumps(result, separators=(",", ":")) + "\n")


def load(path):
    data = json.loads(path.read_text())
    if data["format"] != FORMAT:
        raise ValueError(f"Unknown decoder format in {path}")
    identity = NumericTrees(data["identity"])
    models = {
        name: (NumericTrees(row["parameters"]), identity, row["independent"])
        for name, row in data["counts"].items()
    }
    return models, data["knobs"], data["profile"]
