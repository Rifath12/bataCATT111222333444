"""Render frozen native baseline activity and measure errors on the fixed panel."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from calibrate_nemotron import to_segments
from diarize import masks, reference
from improve_overlap import Decoder
from multiview_score import align, counts, fit_length, measures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["root", "manifest", "profile", "out"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    profile = json.loads(a.profile.read_text())
    results, pooled = [], None
    for item in json.loads(a.manifest.read_text()):
        key = Path(item["audio"]).stem
        dur = item["duration_s"]
        with np.load(a.root / "probabilities" / f"{key}.npz") as data:
            raw = fit_length(data["base"], int(np.ceil(dur / 0.01)))
            native_ticks = np.zeros(raw.shape[1])
            for left, right, slot in data["native"]:
                native_ticks[int(slot)] += round(100 * right) - round(100 * left)
            _, ranked, _ = align(raw, raw, native_ticks)
        hypothesis = Decoder(ranked).decode(profile)
        rows = to_segments(hypothesis, dur)
        for r in rows:
            r["speaker"] = "ABC"[int(r["speaker"])]
        # Include a zero-length placeholder for unobserved speaker IDs so that
        # renderer I/O keeps its three-channel shape without inventing activity.
        active = {r["speaker"] for r in rows}
        rows.extend(dict(speaker=s, start=0, end=0) for s in "ABC" if s not in active)
        out = a.root / key
        out.mkdir(exist_ok=True)
        (out / "timing.json").write_text(
            json.dumps(dict(duration_s=dur, segments=rows), indent=2) + "\n"
        )
        observed = masks(reference(a.root / item["ref"]), dur)[1].T
        raw_counts = counts(SimpleNamespace(actual=observed), hypothesis)
        pooled = raw_counts.copy() if pooled is None else pooled + raw_counts
        results.append(
            dict(
                id=key,
                native_labels_with_activity=sorted(active),
                **measures(raw_counts),
            )
        )
    report = dict(pooled=measures(pooled), clips=results)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(report["pooled"])


if __name__ == "__main__":
    main()
