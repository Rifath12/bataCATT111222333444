"""Evaluate the frozen long-meeting verifier on REAL-T clips with one speaker mapping per source session."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from diarize import masks, reference
from improve_overlap import Decoder
from long_views import meeting
from multiview_score import Clip, align, counts, fit_length, measures
from verify_long import DEV, evidence, make_frames, optimize


def inputs(cache):
    with np.load(cache) as raw:
        duration = float(raw["duration_s"])
        n = int(round(duration * 100))
        first = fit_length(raw["base"], n)
        second = fit_length(raw["alternate"][int(raw["offset"]) :], n)
        activity = np.zeros_like(first, dtype=bool)
        centers = (np.arange(n) + 0.5) * 0.01
        for start, stop, slot in raw["native"]:
            i, j = np.searchsorted(centers, [start, stop])
            activity[i:j, int(slot)] = True
    ids, first, second = align(first, second, activity.sum(0))
    return first, second, activity[:, ids]


def score_sources():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--context", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    profile = json.loads(
        (a.work / "independent_3speaker/profile_balanced.json").read_text()
    )
    entries = {
        Path(row["audio"]).stem: row for row in json.loads(a.manifest.read_text())
    }
    totals = {}
    report = {"sessions": {}}
    with threadpool_limits(2):
        dev = meeting(a.work, a.views, DEV, profile)
        model, threshold, _, _ = optimize(
            *dev, *evidence(*dev, a.work / "session/mono.wav")
        )
        report["threshold"] = threshold
        for row in json.loads((a.context / "catalog.json").read_text()):
            selected = [span for span in row["spans"] if span["clip"] in entries]
            if not selected:
                continue
            if len(selected) != len(row["spans"]):
                raise ValueError(
                    "This evaluation requires fully held-out source sessions."
                )
            file = a.context / Path(row["input"]).name
            source = row["session"]
            first, second, native = inputs(
                a.context / "probabilities" / (file.stem + ".npz")
            )
            clip = Clip(
                source,
                source,
                None,
                native,
                Decoder(first).decode(profile),
                None,
                None,
                None,
                0.0,
                0.0,
            )
            mask, x = evidence(clip, first, second, file)
            predicted = (
                make_frames(
                    clip, first, second, mask, model.predict_proba(x)[:, 1], threshold
                )
                if model
                else clip.previous
            )
            # Omit inter-clip packing silence. Align identical person labels across all clips.
            rows = []
            ids = []
            for span in selected:
                e = entries[span["clip"]]
                labels, active = masks(
                    reference(a.manifest.parent / e["ref"]), span["duration_s"]
                )
                size = min(active.shape[-1], span["length"])
                rows.append((labels, active[:, :size].T))
                ids.extend(range(span["offset"], span["offset"] + size))
            slots = sorted({s for labels, _ in rows for s in labels})
            if len(slots) != 3:
                raise ValueError(f"Expected exactly three identities in {source}")
            actual = np.concatenate(
                [active[:, [labels.index(s) for s in slots]] for labels, active in rows]
            )
            results = {}
            for method, frames in [
                ("native", native),
                ("previous", clip.previous),
                ("verified", predicted),
            ]:
                raw = counts(replace(clip, actual=actual), frames[ids])
                totals[method] = totals.get(method, 0) + raw
                results[method] = measures(raw)
            report["sessions"][source] = results
    report["pooled"] = {n: measures(v) for n, v in totals.items()}
    a.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    score_sources()
