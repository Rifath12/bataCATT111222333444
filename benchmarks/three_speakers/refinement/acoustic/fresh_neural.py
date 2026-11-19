"""Fit on the original development meeting, export a new mono recording and score it afterward."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.ndimage import maximum_filter1d
import soundfile as sf
from threadpoolctl import threadpool_limits

from calibrate_nemotron import to_segments
from diarize import masks, reference
from long_views import meeting
from multiview_score import counts, measures
from neural_probe import evidence, pack
from new_recording import inputs
from verify_long import DEV, make_frames, optimize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", required=True, type=Path)
    p.add_argument("--views", required=True, type=Path)
    p.add_argument(
        "--cache",
        required=True,
        type=Path,
        help="Directory with frozen development Sortformer outputs",
    )
    p.add_argument("--input-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--score", action="store_true")
    p.add_argument("--threads", default=1, type=int)
    a = p.parse_args()
    folder = a.input_dir
    profile = json.loads(
        (a.work / "independent_3speaker/profile_balanced.json").read_text()
    )
    a.out.mkdir(exist_ok=True, parents=True)
    with threadpool_limits(2):
        dev = meeting(a.work, a.views, DEV, profile)
        dev_mask, dev_features, _ = evidence(a, DEV, *dev, a.work / "session/mono.wav")
        model, threshold, _, _ = optimize(*dev, dev_mask, dev_features)
        del dev_mask, dev_features, dev
        started = time.perf_counter()
        c, f, s, duration = inputs(folder, profile)
        pcm = folder / "mono.wav"
        mask = (c.previous.sum(1) >= 2) & (
            (maximum_filter1d(np.maximum(f.min(1), s.min(1)), 31) >= 0.07)
            | c.previous.all(1)
        )
        destination = a.cache / (folder.name + "_sortformer.npy")
        reused = destination.exists()
        if not reused:
            mono, _ = sf.read(pcm, dtype="float32")
            compact, _ = pack(mono, f, s, mask)
            path = a.out / (folder.name + "_packed.npy")
            np.save(path, compact)
            del mono, compact
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("run_sortformer.py")),
                    str(path),
                    str(destination),
                    "--work",
                    str(a.work),
                    "--threads",
                    str(a.threads),
                ],
                check=True,
            )
        proposal, x, seconds = evidence(a, folder.name, c, f, s, pcm)
        estimated = (
            make_frames(c, f, s, proposal, model.predict_proba(x)[:, 1], threshold)
            if model
            else c.previous
        )
        segments = to_segments(estimated, duration)
        for seg in segments:
            seg["speaker"] = "ABC"[int(seg["speaker"])]
        name = folder.name
        (a.out / f"{name}_timing.json").write_text(
            json.dumps({"duration_s": duration, "segments": segments}, indent=2) + "\n"
        )
        report = {
            "session": name,
            "threshold": threshold,
            "probe_seconds": seconds,
            "postprocess_wall_s": time.perf_counter() - started,
            "probe_cache_reused": reused,
        }
        if a.score:
            actual = masks(reference(folder / "reference.rttm"), duration)[1].T
            raw = {
                k: counts(replace(c, actual=actual), pred)
                for k, pred in [
                    ("native", c.native),
                    ("previous", c.previous),
                    ("probe", estimated),
                ]
            }
            report["raw"] = {k: v.tolist() for k, v in raw.items()}
            report["scores"] = {k: measures(v) for k, v in raw.items()}
        (a.out / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
