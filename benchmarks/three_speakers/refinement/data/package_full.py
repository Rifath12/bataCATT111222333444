"""Score complete meetings without transcript or reference input to inference and join both mono views."""

import argparse
import json
from pathlib import Path

import numpy as np

from diarize import masks, reference
from improve_overlap import Decoder
from multiview_score import Clip, align, counts, fit_length, measures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--combine", action="store_true")
    args = p.parse_args()
    root = args.root
    rows = json.loads((root / "full_manifest.json").read_text(encoding="utf-8"))
    parameters = json.loads(args.profile.read_text(encoding="utf-8"))
    out = root / "full_probabilities"
    out.mkdir(exist_ok=True)
    audios = root / "full_audio"
    audios.mkdir(exist_ok=True)
    results = {}
    totals = {}
    for row in rows:
        session = row["session"]
        source = root / session
        unique = audios / (session + ".wav")
        if not unique.is_file():
            unique.hardlink_to(source / "mono.wav")
        row["audio"] = unique.relative_to(root).as_posix()
        native = source / "probabilities.npz"
        if not native.is_file():
            continue
        seconds = row["duration_s"]
        segments = np.array(
            [
                [s["start"], s["end"], int(s["speaker"].rsplit("_", 1)[-1]) - 1]
                for s in reference(source / "native.rttm")
            ],
            dtype=np.float64,
        ).reshape(-1, 3)
        with np.load(native) as base:
            first = base["probabilities"].copy()
            wall = float(base["wall_s"])
        if args.combine:
            with np.load(source / "view.npz") as archive:
                joined = out / (session + ".npz")
                np.savez(
                    joined,
                    base=first,
                    alternate=archive["probabilities"],
                    offset=928,
                    native=segments,
                    duration_s=seconds,
                    base_wall_s=wall,
                    alternate_wall_s=float(archive["wall_s"]),
                )
        actual = masks(reference(source / "reference.rttm"), seconds)[1].T
        first = fit_length(first, len(actual))
        activity = np.zeros_like(first, dtype=bool)
        centers = (np.arange(len(actual)) + 0.5) * 0.01
        for begin, end, speaker in segments:
            lo, hi = np.searchsorted(centers, [begin, end])
            activity[lo:hi, int(speaker)] = True
        chosen, calibrated_input, _ = align(first, first, activity.sum(axis=0))
        native_out = activity[:, chosen]
        previous = Decoder(calibrated_input).decode(parameters)
        clip = Clip(
            session, session, actual, native_out, previous, None, None, None, wall, 0
        )
        results[session] = {}
        for key, mask in [("native", native_out), ("default", previous)]:
            raw = counts(clip, mask)
            results[session][key] = measures(raw)
            totals[key] = totals.get(key, 0) + raw
    (root / "full_cache_manifest.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "complete_meeting_count": len(results),
        "sessions": results,
        "pooled": {k: measures(v) for k, v in totals.items()},
    }
    outfile = root / "full_baseline_scores.json"
    outfile.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{len(results)} scored", outfile, flush=True)


if __name__ == "__main__":
    main()
