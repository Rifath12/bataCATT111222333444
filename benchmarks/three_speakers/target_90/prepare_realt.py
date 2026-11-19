"""Reconstruct the frozen mono REAL-T Mandarin train and test clips and labels."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import quote
from urllib.request import urlopen

import soundfile as sf

REVISION = "39ca3b88ffd4ce6078c1d045a3e8711fa2bd2f17"
BASE = f"https://huggingface.co/datasets/REAL-TSE/REAL-T/resolve/{REVISION}/"
LABELS = "REAL-T-eval2/EVAL2/json/unseen_CN/overlap_records.json"
MANIFESTS = ("development.json", "heldout.json")


def verify(path, expected):
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}")


def prepare(entry, rows, output):
    audio = output / entry["audio"]
    if not audio.exists():
        pending = audio.with_suffix(".download")
        with urlopen(BASE + quote(entry["source"]), timeout=60) as response:
            with pending.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        verify(pending, entry["wav_sha256"])
        pending.replace(audio)
    verify(audio, entry["wav_sha256"])
    metadata = sf.info(audio)
    if (metadata.channels, metadata.samplerate) != (1, 16_000):
        raise ValueError(f"Unexpected WAV format for {audio}")
    start = entry["start"]
    stop = entry["end"]
    matched = next(
        row["overlap_segments"]
        for row in rows[entry["meeting"]]
        if row["overlap_info"]["overlap_start_time"] == start
        and row["overlap_info"]["overlap_end_time"] == stop
    )
    lines = []
    for turn in matched:
        left = max(0.0, turn["start_time"] - start)
        right = min(metadata.duration, stop - start, turn["end_time"] - start)
        if right > left:
            lines.append(
                f"SPEAKER {audio.stem} 1 {left:.4f} {right - left:.4f}"
                f" <NA> <NA> {turn['speaker']} <NA> <NA>"
            )
    reference = output / entry["ref"]
    reference.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    verify(reference, entry["rttm_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--labels", type=Path, help="Use an existing public annotation JSON"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.labels:
        rows = json.loads(args.labels.read_text(encoding="utf-8"))
    else:
        with urlopen(BASE + quote(LABELS), timeout=60) as response:
            rows = json.load(response)
    for name in MANIFESTS:
        manifest = Path(__file__).with_name(name)
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda entry: prepare(entry, rows, args.output), entries))
        target = args.output / name
        if manifest.resolve() != target.resolve():
            shutil.copyfile(manifest, target)
    root = args.output.resolve()
    paths = sorted(
        root / row["audio"]
        for name in MANIFESTS
        for row in json.loads((root / name).read_text(encoding="utf-8"))
    )
    (root / "audio.txt").write_text(
        "".join(f"{path}\n" for path in paths), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
