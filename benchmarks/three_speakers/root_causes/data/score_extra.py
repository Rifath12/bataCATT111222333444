"""Export baseline A/B/C timing for an exact-three excerpt panel without references."""

import argparse
import json
from pathlib import Path

import numpy as np

from calibrate_nemotron import to_segments
from improve_overlap import Decoder
from multiview_score import align, fit_length


def prepare(root, profile):
    for item in json.loads((root / "manifest.json").read_text()):
        name = Path(item["audio"]).stem
        duration = item["duration_s"]
        with np.load(root / "probabilities" / f"{name}.npz") as data:
            frames = int(np.ceil(duration / 0.01))
            raw = fit_length(data["base"], frames)
            native_ticks = np.zeros(raw.shape[1])
            for start, stop, slot in data["native"]:
                native_ticks[int(slot)] += round(100 * stop) - round(100 * start)
        _, ranked, _ = align(raw, raw, native_ticks)
        rows = to_segments(Decoder(ranked).decode(profile), duration)
        for row in rows:
            row["speaker"] = "ABC"[int(row["speaker"])]
        target = root / name
        target.mkdir(exist_ok=True)
        (target / "timing.json").write_text(
            json.dumps(dict(duration_s=duration, segments=rows), indent=2) + "\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    config = parser.parse_args()
    prepare(config.root, json.loads(config.profile.read_text()))
