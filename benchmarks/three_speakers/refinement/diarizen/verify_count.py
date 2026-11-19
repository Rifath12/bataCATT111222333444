"""Fit and score an independent count verifier on mono DiariZen and Nemotron outputs."""

import argparse
import json
from pathlib import Path
from scipy.special import expit

import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from multiview_score import counts, load_manifest, measures


def design(clip, folder, with_diarizen):
    first = np.sort(clip.fused, axis=1)
    count_features = np.column_stack(
        [first, clip.native.sum(axis=1), clip.previous.sum(axis=1)]
    )
    if with_diarizen:
        z = np.load(folder / (clip.name + ".npy"))
        z = z[np.minimum(np.arange(len(first)), len(z) - 1)]
        independent = np.column_stack(
            [
                z,
                z[:, 1] * first[:, 0],
                z[:, 3] * first[:, 0],
                z[:, 1] - first[:, 1],
                z[:, 1] - z[:, -2],
            ]
        )
        count_features = np.column_stack([count_features, independent])
    return np.column_stack(
        [
            uniform_filter1d(count_features, width, axis=0, mode="nearest")
            for width in (1, 17, 67)
        ]
    )


def train(examples):
    X = np.concatenate([d[::3] for _, d in examples])
    y = np.concatenate([c.actual[::3].sum(axis=1) == 3 for c, _ in examples])
    model = make_pipeline(
        StandardScaler(), LogisticRegression(C=0.025, max_iter=500, random_state=0)
    )
    return model.fit(X, y)


def prediction(clip, p3, threshold):
    active = clip.previous.copy()
    add = p3 >= threshold
    remove = (active.sum(axis=1) == 3) & ~add
    indices = np.flatnonzero(remove)
    active[indices, np.argmin(clip.fused[indices], axis=1)] = False
    active[add] = True
    return active


def posterior(model, design_matrix):
    if isinstance(model, dict):
        x = (design_matrix - model["mean"]) / model["scale"]
        probabilities = expit(x @ np.asarray(model["weights"]) + model["bias"])
    else:
        probabilities = model.predict_proba(design_matrix)[:, 1]
    return uniform_filter1d(probabilities, 13, mode="nearest")


def parameters(model):
    scaler, reg = model.steps[0][1], model.steps[1][1]
    return {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "weights": reg.coef_[0].tolist(),
        "bias": float(reg.intercept_[0]),
    }


def freeze(development):
    by_session = sorted({c.session for c, _ in development})
    trials = []
    for fold in range(3):
        omitted = set(by_session[fold::3])
        train_set = [e for e in development if e[0].session not in omitted]
        model = train(train_set)
        for c, d in development:
            if c.session in omitted:
                trials.append((c, posterior(model, d)))
    # The complete operating point is fixed from out-of-session development predictions.
    thresholds = np.linspace(0.02, 0.88, 87)
    options = []
    for t in thresholds:
        raw = sum(counts(c, prediction(c, p, t)) for c, p in trials)
        m = measures(raw)
        score = (
            m["exact_three_F1"] + 0.2 * m["three_way_speaker_recall"] - 0.3 * m["DER"]
        )
        options.append((score, float(t), m))
    _, threshold, stats = max(options, key=lambda row: row[0])
    return train(development), threshold, stats


def evaluate(clips, models, folder):
    all_cases = []
    for c in clips:
        scores = {"previous": counts(c, c.previous)}
        for name, (model, t, with_dz) in models.items():
            features = design(c, folder, with_dz)
            p3 = posterior(model, features)
            preds = prediction(c, p3, t)
            scores[name] = counts(c, preds)
            if with_dz:
                # The independent powerset model directly distinguishes 0, 1, and 2.
                # Its third column below is total pair posterior, not a triple class.
                none = features[:, 5]
                pair = features[:, 6]
                small_counts = np.column_stack([none, 1 - none - pair, pair])
                size = np.argmax(uniform_filter1d(small_counts, 13, axis=0), axis=1)
                size[p3 >= t] = 3
                ranks = np.argsort(np.argsort(-c.fused, axis=1), axis=1)
                scores["independent_full_count"] = counts(c, ranks < size[:, None])
        all_cases.append((c.name, scores))
    pooled = {
        key: measures(sum(scores[key] for _, scores in all_cases))
        for key in all_cases[0][1]
    }
    return {
        "clips": len(clips),
        "pooled": pooled,
        "per_clip_counts": {
            n: {k: v.tolist() for k, v in score.items()} for n, score in all_cases
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dev-manifest", type=Path)
    p.add_argument("--dev-probs", type=Path)
    p.add_argument("--dz-features", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--save-frozen", type=Path)
    p.add_argument("--from-frozen", type=Path)
    p.add_argument(
        "--test",
        nargs=3,
        action="append",
        default=[],
        metavar=("NAME", "MANIFEST", "PROBS"),
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    old = json.loads(a.profile.read_text())
    models = {}
    report = {"development": {}, "tests": {}}
    if a.from_frozen:
        frozen = json.loads(a.from_frozen.read_text())
        models = {
            name: (row["parameters"], row["threshold"], row["independent"])
            for name, row in frozen.items()
        }
    else:
        if not a.dev_manifest or not a.dev_probs:
            p.error("Training requires both development inputs")
        clips = load_manifest(a.dev_manifest, a.dev_probs, old)
        frozen = {}
        for name, with_dz in [
            ("same_model_control", False),
            ("independent_verifier", True),
        ]:
            examples = [(c, design(c, a.dz_features, with_dz)) for c in clips]
            model, t, stats = freeze(examples)
            row = dict(parameters=parameters(model), threshold=t, independent=with_dz)
            frozen[name] = row
            models[name] = (row["parameters"], t, with_dz)
            report["development"][name] = {"threshold": t, "out_of_fold": stats}
        if a.save_frozen:
            a.save_frozen.write_text(json.dumps(frozen, indent=2) + "\n")
        print("Development", report["development"], flush=True)
    for name, manifest, probs in a.test:
        cases = load_manifest(Path(manifest), Path(probs), old)
        row = evaluate(cases, models, a.dz_features)
        report["tests"][name] = row
        print(name, row["clips"], row["pooled"], flush=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
