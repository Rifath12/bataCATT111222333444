"""Select fresh public test speakers deterministically and record per-file hashes."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("content", "published", "previous", "out"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--extra-exclusions", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    previous = json.loads(args.previous.read_text())
    texts = {}
    for line in args.content.read_text().splitlines():
        if "\t" in line:
            name, pronunciation = line.split("\t", 1)
            texts[name] = "".join(pronunciation.split()[::2])
    grouped = {}
    excluded = {
        entry["id"]
        for trios in previous["groups"].values()
        for trio in trios
        for entry in trio
    }
    used_files = set(previous["sha256"])
    for directory in args.extra_exclusions:
        excluded.update(path.name[3:7] for path in directory.rglob("SSB*.wav"))
    for name in sorted(set(texts).intersection(json.loads(args.published.read_text()))):
        if name[3:7] not in excluded and name not in used_files:
            grouped.setdefault(name[3:7], []).append(name)
    candidates = sorted(
        key
        for key, files in grouped.items()
        if len(files) >= 2 and any(7 <= len(texts[name]) <= 24 for name in files)
    )
    random.Random(4131).shuffle(candidates)
    if len(candidates) < 9:
        raise ValueError(f"Found only {len(candidates)} eligible voices")
    records = []
    for i in range(0, 9, 3):
        trio = []
        for key in candidates[i : i + 3]:
            target = min(
                (name for name in grouped[key] if 7 <= len(texts[name]) <= 24),
                key=lambda name: (abs(len(texts[name]) - 14), name),
            )
            enrollment = min(
                (name for name in grouped[key] if name != target),
                key=lambda name: (abs(len(texts[name]) - 14), name),
            )
            trio.append(
                dict(id=key, target=target, enroll=enrollment, text=texts[target])
            )
        records.append(trio)
    destination = args.out / "sources"
    destination.mkdir(parents=True, exist_ok=True)

    def fetch(name):
        path = destination / name
        if not path.exists():
            url = f"https://huggingface.co/datasets/AISHELL/AISHELL-3/resolve/{previous['revision']}/test/wav/SSB{name[3:7]}/{name}"
            with urlopen(url, timeout=25) as response:
                data = response.read()
            if data[:4] != b"RIFF":
                raise ValueError(f"Not a WAV: {name}")
            path.write_bytes(data)
        return name, hashlib.sha256(path.read_bytes()).hexdigest()

    unique = sorted(
        {x[k] for trio in records for x in trio for k in ("target", "enroll")}
    )
    with ThreadPoolExecutor(4) as executor:
        hashes = dict(executor.map(fetch, unique))
    (args.out / "holdout.json").write_text(
        json.dumps(
            dict(
                revision=previous["revision"],
                sha256=hashes,
                groups=dict(heldout=records),
                speakers_disjoint_from_current_development=True,
                speakers_also_seen_in_previous_experiments=sorted(
                    set(candidates[:9]).intersection(
                        entry["id"]
                        for group in previous["groups"].values()
                        for trio in group
                        for entry in trio
                    )
                ),
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print([x["id"] for trio in records for x in trio], flush=True)


if __name__ == "__main__":
    main()
