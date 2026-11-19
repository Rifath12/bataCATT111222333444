"""Measure per-window stream labeling with independent enrollment and known clean sources.

The reference frame activity in this diagnostic is an upper-bound condition.
Natural meeting evaluation must instead use predicted frame activity.
"""

import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.optimize import linear_sum_assignment

from partial_enrollment import activity_affinity
from process_meeting import VoiceEmbeddings, resample, normalized_dot


def mapping(matrix):
    return linear_sum_assignment(-matrix)[1]


def ref_activity(x):
    frame = x[:, : x.shape[1] // 80 * 80].reshape(len(x), -1, 80)
    power = np.mean(frame**2, axis=2)
    return power > np.maximum(power.max(axis=1, keepdims=True) * 0.001, 1.0e-8)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "manifest", "audio", "embedder", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    voice = VoiceEmbeddings(a.embedder)
    embeddings = {}
    rows = []
    for spec in json.loads(a.manifest.read_text()):
        with np.load(a.cache / (spec["name"] + ".npz")) as dat:
            refs = dat["refs"]
            tracks = dat["tracks"]
        ground_truth = mapping(
            np.array([[abs(normalized_dot(r, s)) for s in tracks] for r in refs])
        )
        enrolled = []
        for used in spec["source_files"]:
            names = {r[0] for r in used}
            candidates = [
                f
                for f in sorted(a.audio.glob(used[0][0][:7] + "*.wav"))
                if f.name not in names
            ]
            if not candidates:
                raise FileNotFoundError(f"No independent enrollment for {used[0][0]}")
            anchor = candidates[0]
            if anchor not in embeddings:
                wave, sr = sf.read(anchor, dtype="float32")
                embeddings[anchor] = voice(resample(wave, sr, 8000))
            enrolled.append(embeddings[anchor])
        embeddings_out = [voice(x) for x in tracks]
        base_scores = np.array(
            [
                [
                    normalized_dot(e, cue) if e is not None and cue is not None else 0.0
                    for e in embeddings_out
                ]
                for cue in enrolled
            ]
        )
        timing = activity_affinity(tracks, ref_activity(refs), 0)
        conditions = []
        for n_missing in (0, 1, 2, 3):
            for missing in itertools.combinations(range(3), n_missing):
                scores = base_scores.copy()
                if missing:
                    scores[list(missing)] = 0.0
                voice_only = mapping(scores)
                with_timing = mapping(scores + timing) if missing else voice_only
                conditions.append(
                    dict(
                        missing=list(missing),
                        voice_only_correct=int((voice_only == ground_truth).sum()),
                        with_activity_correct=int((with_timing == ground_truth).sum()),
                    )
                )
        rows.append(
            dict(
                name=spec["name"],
                source_wav_if_reference_slot=ground_truth.tolist(),
                conditions=conditions,
            )
        )
    stats = {}
    for k in range(4):
        selected = [c for r in rows for c in r["conditions"] if len(c["missing"]) == k]
        stats[k] = {
            "speaker_slot_assignments": 3 * len(selected),
            "embeddings_only_correct": sum(c["voice_only_correct"] for c in selected),
            "with_oracle_activity_correct": sum(
                c["with_activity_correct"] for c in selected
            ),
        }
    a.out.write_text(json.dumps(dict(summary=stats, clips=rows), indent=2) + "\n")
    print(stats, flush=True)


if __name__ == "__main__":
    main()
