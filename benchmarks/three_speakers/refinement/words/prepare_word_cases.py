"""Prepare one long exactly-three-person excerpt from each eligible AliMeeting session."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from calibrate_nemotron import to_segments
from improve_overlap import Decoder
from recognize import textgrid
from multiview_score import align, fit_length


def prepare(a):
    source = json.loads(a.manifest.read_text())["clips"]
    profile = json.loads(a.profile.read_text())
    records = []
    for session in sorted({x["source_session"] for x in source}):
        row = max(
            (x for x in source if x["source_session"] == session),
            key=lambda x: x["duration_s"],
        )
        ref = textgrid(a.labels / f"{session}.TextGrid")
        # Use full-utterance outer boundaries to avoid scoring partial words.
        nominal_start = row["start_tick"] / 100
        nominal_end = row["end_tick"] / 100
        start, end = nominal_start, nominal_end
        while any(r["start"] < start < r["end"] for r in ref):
            start = max(r["end"] for r in ref if r["start"] < start < r["end"])
        while any(r["start"] < end < r["end"] for r in ref):
            end = min(r["start"] for r in ref if r["start"] < end < r["end"])
        if end <= start:
            raise ValueError(f"No complete-sentence interval in {session}")
        out = a.out / session
        out.mkdir(parents=True, exist_ok=True)
        clip_path = a.manifest.parent / row["audio"]
        wave, rate = sf.read(clip_path, dtype="int16")
        with np.load(a.probs / (clip_path.stem + ".npz")) as data:
            n = int(np.ceil(len(wave) / rate / 0.01))
            raw = fit_length(data["base"], n)
            dur = np.zeros(raw.shape[1])
            for lo, hi, chan in data["native"]:
                dur[int(chan)] += round(hi * 100) - round(lo * 100)
            _, primary, _ = align(raw, raw, dur)
            states = Decoder(primary).decode(profile)
        left = round((start - nominal_start) * rate)
        right = round((end - nominal_start) * rate)
        wav = wave[left:right]
        sf.write(out / "mono.wav", wav, rate, subtype="PCM_16")
        # Preserve the existing speaker order and clip partial hypothesis spans.
        segments = []
        offset = left / rate
        stop = right / rate
        for r in to_segments(states, len(wave) / rate):
            lo, hi = max(offset, r["start"]), min(stop, r["end"])
            if hi > lo:
                segments.append(
                    dict(speaker=r["speaker"], start=lo - offset, end=hi - offset)
                )
        labels = [
            dict(
                speaker=r["speaker"],
                start=r["start"] - start,
                end=r["end"] - start,
                text=r["text"],
            )
            for r in ref
            if r["start"] >= start and r["end"] <= end
        ]
        scores = dict(
            session=session,
            source=row["id"],
            crop_start_s=start,
            crop_end_s=end,
            duration_s=len(wav) / rate,
            reference_speakers=len({r["speaker"] for r in labels}),
            system_speakers=len({r["speaker"] for r in segments}),
        )
        (out / "timing.json").write_text(json.dumps({"segments": segments}, indent=2))
        (out / "reference.json").write_text(
            json.dumps(labels, ensure_ascii=False, indent=2)
        )
        records.append(scores)
    (a.out / "cases.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("manifest", "profile", "probs", "labels", "out"):
        p.add_argument("--" + field, type=Path, required=True)
    prepare(p.parse_args())
