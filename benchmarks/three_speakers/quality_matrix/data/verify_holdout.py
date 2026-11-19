"""Audit the frozen AISHELL-4 holdout and the separated audio offsets."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from diarize import masks, reference


def digest(file):
    return sha256(file.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    total, clips = 0, []
    rows = json.loads(a.manifest.read_text())
    for row in rows:
        for key, expected in [
            ("audio", "wav_sha256"),
            ("ref", "cropped_activity_sha256"),
            ("text_reference", "cropped_text_sha256"),
        ]:
            if digest(a.root / row[key]) != row[expected]:
                raise ValueError(f"Checksum failed: {row[key]}")
        info = sf.info(a.root / row["audio"])
        if (info.channels, info.samplerate, info.frames) != (1, 16000, row["frames"]):
            raise ValueError(f"Unexpected packed WAV parameters: {row['id']}")
        duration = row["duration_s"]
        people, activity = masks(reference(a.root / row["ref"]), duration)
        if sorted(people) != sorted(row["speaker_ids"]) or len(people) != 3:
            raise ValueError(f"Inconsistent reference IDs: {row['id']}")
        amount = int(np.count_nonzero(activity.sum(axis=0) == 3))
        if amount != round(row["exact_three_s"] * 100):
            raise ValueError(f"Reference offset/rounding mismatch: {row['id']}")
        total += amount
        clips.append(dict(id=row["id"], exact_three_frames=amount))
    expected_files = 3 * len(clips)
    a.out.write_text(
        json.dumps(
            dict(
                audited_files=expected_files,
                clips=clips,
                reference_three_seconds=total / 100,
            ),
            indent=2,
        )
        + "\n"
    )
    print(len(clips), expected_files, total / 100)


if __name__ == "__main__":
    main()
