"""Select previous-unscored complete-utterance AliMeeting examples with missing long solo cues."""

import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from calibrate_nemotron import to_segments
from improve_overlap import Decoder
from multiview_score import align, fit_length
from recognize import textgrid


def eligibility(row, source, annotation, probs, profile):
    left, right = row["start_tick"] / 100, row["end_tick"] / 100
    start, end = left, right
    while any(r["start"] < start < r["end"] for r in annotation):
        start = max(r["end"] for r in annotation if r["start"] < start < r["end"])
    while any(r["start"] < end < r["end"] for r in annotation):
        end = min(r["start"] for r in annotation if r["start"] < end < r["end"])
    if end - start < 5:
        return
    truth = [
        dict(r, start=r["start"] - start, end=r["end"] - start)
        for r in annotation
        if r["start"] >= start and r["end"] <= end
    ]
    if len({r["speaker"] for r in truth}) != 3:
        return
    path = source / row["audio"]
    duration = sf.info(path).duration
    with np.load(probs / (path.stem + ".npz")) as data:
        features = fit_length(data["base"], int(np.ceil(duration / 0.01)))
        support = np.zeros(features.shape[1])
        for lo, hi, channel in data["native"]:
            support[int(channel)] += round(hi * 100) - round(lo * 100)
        _, primary, _ = align(features, features, support)
        states = Decoder(primary).decode(profile)
    offset = start - left
    segments = [
        dict(
            speaker=s["speaker"],
            start=max(0, s["start"] - offset),
            end=min(end - start, s["end"] - offset),
        )
        for s in to_segments(states, duration)
        if s["end"] > offset and s["start"] < end - left
    ]
    local = states[round((start - left) * 100) : round((end - left) * 100)]
    if min(local.any(axis=0)) == 0 or not np.any(local.sum(axis=1) > 1):
        return
    count = local.sum(axis=1)
    longest = []
    for state in local.T:
        edges = np.flatnonzero(
            np.diff(np.pad((state & (count == 1)).astype("i1"), (1, 1)))
        )
        longest.append(max(np.diff(edges)[::2], default=0))
    if min(longest) >= 75:
        return
    return path, start - left, end - left, segments, truth, list(map(int, longest))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("manifest", "labels", "prior", "profile", "probs", "out"):
        p.add_argument("--" + arg, type=Path, required=True)
    p.add_argument("--per-session", type=int, default=2)
    a = p.parse_args()
    rows = json.loads(a.manifest.read_text())["clips"]
    used = {x["source"] for x in json.loads(a.prior.read_text())}
    profile = json.loads(a.profile.read_text())
    summary = []
    for session in sorted({r["source_session"] for r in rows}):
        annotations = textgrid(a.labels / (session + ".TextGrid"))
        selected = 0
        for r in sorted(
            (r for r in rows if r["source_session"] == session and r["id"] not in used),
            key=lambda r: (-r["duration_s"], r["id"]),
        ):
            item = eligibility(r, a.manifest.parent, annotations, a.probs, profile)
            if not item:
                continue
            wav, left, right, segments, truth, longest = item
            dest = a.out / r["id"]
            dest.mkdir(parents=True, exist_ok=True)
            speech, rate = sf.read(wav, dtype="int16")
            sf.write(
                dest / "mono.wav",
                speech[round(left * rate) : round(right * rate)],
                rate,
            )
            (dest / "timing.json").write_text(
                json.dumps({"segments": segments}, indent=2) + "\n"
            )
            (dest / "reference.json").write_text(
                json.dumps(truth, indent=2, ensure_ascii=False) + "\n"
            )
            summary.append(
                dict(
                    source=r["id"],
                    session=session,
                    duration_s=right - left,
                    longest_solo_frames=longest,
                )
            )
            selected += 1
            if selected >= a.per_session:
                break
    (a.out / "cases.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(len(summary), "selected clips", sum(r["duration_s"] for r in summary))


if __name__ == "__main__":
    main()
