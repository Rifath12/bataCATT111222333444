"""Describe overlap both timing baselines miss on a known replay panel."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

from frozen_dictionary import load_frozen
from portable import load
from pareto_timing import read_panel
import structured as core


def runs(frame_values):
    edges = np.flatnonzero(np.diff(np.r_[False, frame_values, False]))
    return zip(edges[::2], edges[1::2], strict=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ("manifest", "cache", "model", "dictionary", "narrow", "dz", "out"):
        ap.add_argument("--" + arg, type=Path, required=True)
    a = ap.parse_args()
    options = SimpleNamespace(dz=a.dz, sortformer=None)
    models, knobs, options.profile = load(a.model)
    options.verifier, options.thresholds, options.base = load_frozen(a.dictionary)
    options.narrow = json.loads(a.narrow.read_text())
    cases = read_panel(a.manifest, a.cache, options)
    examples = [c.example for c in cases]
    name = "independent_diarizen"
    latest = core.tune_batch(
        examples, [core.emissions(models[name], e) for e in examples], knobs[name][0]
    )
    categories = [0, 0, 0, 0]
    false = [0, 0, 0, 0]
    durations = {}
    for case, recent in zip(cases, latest, strict=True):
        truth = case.example.actual.all(1)
        old = case.dictionary.all(1)
        new = recent.all(1)
        evidence = old.astype(np.int8) + 2 * new
        for index in range(4):
            categories[index] += int(np.sum(truth & (evidence == index)))
            false[index] += int(np.sum(~truth & (evidence == index)))
        for left, right in runs(truth):
            seconds = (right - left) / 100.0
            name = "under_300_ms" if seconds < 0.3 else "at_least_300_ms"
            bucket = durations.setdefault(
                name, dict(events=0, reference_frames=0, both_miss_frames=0)
            )
            bucket["events"] += 1
            bucket["reference_frames"] += int(right - left)
            bucket["both_miss_frames"] += int(np.sum(evidence[left:right] == 0))
    labels = ("both_missed", "dictionary_only", "structured_only", "both_found")
    result = {
        "reference_triple_seconds": dict(
            zip(labels, [f / 100 for f in categories], strict=True)
        ),
        "false_triple_seconds": dict(
            zip(labels, [f / 100 for f in false], strict=True)
        ),
        "reference_event_duration": durations,
        "oracle_union_recall_ceiling": sum(categories[1:]) / sum(categories),
    }
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    with threadpool_limits(2):
        main()
