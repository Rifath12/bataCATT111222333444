"""Rebuild the validation slice from checksum-pinned public recordings."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import soundfile as sf

from benchmark_examples import refmask
from process_meeting import RATE, resample


MANIFEST = Path(__file__).parent / "benchmarks" / "cases.json"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fetch(item, cache):
    path = cache / item["file"]
    if path.exists() and sha(path) == item["sha256"]:
        return path
    request = urllib.request.Request(item["url"], headers={"User-Agent": "Mandarin verification/1.0"})
    with urllib.request.urlopen(request, timeout=45) as remote:
        data = remote.read()
    if hashlib.sha256(data).hexdigest() != item["sha256"]:
        raise ValueError(f"Checksum mismatch for {item['file']}")
    path.write_bytes(data)
    return path


def prepared_source(path):
    y, rate = sf.read(path, dtype="float32")
    y = resample(y, rate, RATE)
    y = np.pad(y, (0, (-len(y)) % 160))
    mask = np.flatnonzero(refmask(y))
    y = y[max(0, mask[0] - 4) * 160:(mask[-1] + 5) * 160]
    return y * (0.06 / np.sqrt((y * y).mean()))


def restore(data, dest, cache):
    dest.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    inputs = {item["file"]: item for item in data["files"]}
    with ThreadPoolExecutor(max_workers=8) as workers:
        list(workers.map(lambda item: fetch(item, cache), inputs.values()))

    for item in data["cases"]:
        target = dest / "controlled" / item["name"]
        target.mkdir(parents=True, exist_ok=True)
        counts = item["source_count"]
        stems = np.zeros((counts, round(item["seconds"] * RATE)), np.float32)
        for i, turns in enumerate(item["speakers"].values()):
            gain = 10 ** (item["gain_db"][i] / 20)
            for turn in turns:
                x = prepared_source(cache / turn["source_url"].rsplit("/", 1)[1]) * gain
                start = round(turn["start_s"] * RATE)
                stems[i, start:start+len(x)] = x
        peak = float(np.max(np.abs(stems.sum(axis=0))))
        if peak > 0.92:
            stems *= 0.92 / peak
        sf.write(target / "mixed.wav", stems.sum(axis=0), RATE, subtype="FLOAT")
        for i, wave in enumerate(stems):
            sf.write(target / f"reference_{chr(65+i)}.wav", wave, RATE, subtype="FLOAT")
        (target / "reference.json").write_text(
            json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    natural = dest / "natural"
    natural.mkdir(exist_ok=True)
    for name in data["natural_files"]:
        (natural / name).write_bytes((cache / name).read_bytes())
    print(f"Prepared {len(data['cases'])} controlled clips and {len(data['natural_files'])} natural source files.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    restore(data, args.out, args.cache or args.out / "cache")


if __name__ == "__main__":
    main()
