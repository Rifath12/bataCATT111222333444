"""Describe annotated three-voice events and model support for fully missed events."""

import argparse
import json
from pathlib import Path

import numpy as np

from diarize import masks, reference
from new_recording import inputs
from turn_boundaries import runs


def audit(directory, timing_path, profile):
    _, first, second, duration = inputs(directory, profile)
    truth = masks(reference(directory / "reference.rttm"), duration)[1].all(0)
    timing = json.loads(timing_path.read_text())["segments"]
    system = masks(timing, duration)[1].sum(0) == 3
    events = []
    for left, right in runs(truth):
        hit = int(system[left:right].sum())
        row = {
            "start_s": int(left) / 100,
            "end_s": int(right) / 100,
            "duration_s": int(right - left) / 100,
            "detected_s": hit / 100,
        }
        if not hit:
            for name, scores in (("native", first), ("alternate", second)):
                weak = scores[left:right].min(1)
                row[name + "_weakest_posterior_median"] = float(np.median(weak))
                row[name + "_weakest_posterior_peak"] = float(weak.max())
        events.append(row)
    return events


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, nargs="+", required=True)
    p.add_argument("--output-timings", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    profile = json.loads(a.base.read_text())
    data = {
        d.name: audit(d, a.output_timings / (d.name + "_timing.json"), profile)
        for d in a.input_dir
    }
    a.out.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
