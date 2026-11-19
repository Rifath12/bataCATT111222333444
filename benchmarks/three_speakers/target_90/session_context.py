"""Benchmark additional same-session context without using reference speech intervals."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def prepare(root, manifests, folder):
    folder.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for manifest in manifests:
        for entry in json.loads((root / manifest).read_text(encoding="utf-8")):
            groups[entry["session"]].append(entry)
    catalog = []
    for number, (session, entries) in enumerate(sorted(groups.items())):
        pieces = []
        spans = []
        cursor = 0
        for entry in sorted(entries, key=lambda x: x["start"]):
            audio, rate = sf.read(root / entry["audio"], dtype="int16")
            if rate != 16000 or audio.ndim != 1:
                raise ValueError(entry["audio"])
            spans.append(
                {
                    "clip": Path(entry["audio"]).stem,
                    "offset": cursor // 160,
                    "duration_s": len(audio) / 16000,
                    "length": (len(audio) + 159) // 160,
                }
            )
            segment = np.pad(audio, (0, (-len(audio)) % 160 + 16000))
            pieces.append(segment)
            cursor += len(segment)
        path = folder / f"session_{number:02d}.wav"
        sf.write(path, np.concatenate(pieces), 16000, subtype="PCM_16")
        catalog.append(
            {
                "input": str(path),
                "session": session,
                "total_s": cursor / 16000,
                "spans": spans,
            }
        )
    (folder / "catalog.json").write_text(
        json.dumps(catalog, indent=2) + "\n", encoding="utf-8"
    )
    (folder / "input.txt").write_text(
        "".join(row["input"] + "\n" for row in catalog), encoding="utf-8"
    )
    print("Prepared", len(catalog), "sessions")


def unpack(folder, output):
    output.mkdir(parents=True, exist_ok=True)
    for row in json.loads((folder / "catalog.json").read_text(encoding="utf-8")):
        with np.load(
            folder / "probabilities" / (Path(row["input"]).stem + ".npz")
        ) as archive:
            base = archive["base"]
            offset = int(archive["offset"])
            second = archive["alternate"][offset:]
            native = archive["native"]
            for span in row["spans"]:
                lo = span["offset"]
                hi = lo + span["length"]
                start = lo / 100
                stop = start + span["duration_s"]
                segments = native[(native[:, 0] < stop) & (native[:, 1] > start)].copy()
                segments[:, 0] = np.maximum(segments[:, 0], start) - start
                segments[:, 1] = np.minimum(segments[:, 1], stop) - start
                fraction = span["duration_s"] / row["total_s"]
                np.savez(
                    output / (span["clip"] + ".npz"),
                    base=base[lo:hi],
                    alternate=second[lo:hi],
                    native=segments,
                    offset=0,
                    duration_s=span["duration_s"],
                    base_wall_s=float(archive["base_wall_s"]) * fraction,
                    alternate_wall_s=float(archive["alternate_wall_s"]) * fraction,
                )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path)
    p.add_argument("--manifests", nargs="+")
    p.add_argument("--folder", type=Path, required=True)
    p.add_argument("--unpack", type=Path)
    a = p.parse_args()
    if a.unpack:
        unpack(a.folder, a.unpack)
    else:
        if not a.root or not a.manifests:
            p.error("Supply --root and --manifests when packing")
        prepare(a.root, a.manifests, a.folder)


if __name__ == "__main__":
    main()
