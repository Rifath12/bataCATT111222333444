"""Check the portable probabilities against the original model on one complete mono file."""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from frozen_dictionary import load_frozen
from long_views import meeting
from new_recording import inputs
from verify_long import DEV, evidence, optimize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("work", "views", "input-dir", "frozen", "out"):
        p.add_argument("--" + flag, type=Path, required=True)
    a = p.parse_args()
    portable, thresholds, profile = load_frozen(a.frozen)
    with threadpool_limits(2):
        dev = meeting(a.work, a.views, DEV, profile)
        fitted, developed, _, _ = optimize(
            *dev, *evidence(*dev, a.work / "session/mono.wav")
        )
        assert list(developed) == thresholds
        clip, first, second, duration = inputs(a.input_dir, profile)
        _, x = evidence(clip, first, second, a.input_dir / "mono.wav")
        original = fitted.predict_proba(x)
        exported = portable.predict_proba(x)
    diff = float(np.max(np.abs(original - exported)))
    result = {
        "recording": a.input_dir.name,
        "duration_s": duration,
        "proposals": len(x),
        "numeric_features": x.shape[1],
        "number_of_trees": len(portable.trees),
        "max_abs_probability_difference": diff,
        "all_probabilities_identical": bool(np.array_equal(original, exported)),
    }
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    if not result["all_probabilities_identical"]:
        raise ValueError(result)


if __name__ == "__main__":
    main()
