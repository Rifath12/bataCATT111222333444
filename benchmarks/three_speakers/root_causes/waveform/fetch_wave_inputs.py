"""Download and validate the exact 54 public AISHELL-3 files used by the acoustic experiment."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, required=True, help="Directory containing manifest.json"
    )
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    rev = manifest["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", rev):
        raise ValueError("Expected an immutable dataset revision")
    dest = args.data / "sources"
    dest.mkdir(parents=True, exist_ok=True)

    def fetch(item):
        name, expected = item
        if not re.fullmatch(r"SSB[0-9]+\.wav", name):
            raise ValueError(name)
        path = dest / name
        if path.exists():
            data = path.read_bytes()
        else:
            url = f"https://huggingface.co/datasets/AISHELL/AISHELL-3/resolve/{rev}/test/wav/SSB{name[3:7]}/{name}"
            with urlopen(url, timeout=90) as stream:
                data = stream.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Audio checksum mismatch for {name}")
        if not path.exists():
            tmp = path.with_suffix(".download")
            tmp.write_bytes(data)
            tmp.replace(path)

    with ThreadPoolExecutor(3) as executor:
        list(executor.map(fetch, manifest["sha256"].items()))
    print(f"Verified {len(manifest['sha256'])} input files")


if __name__ == "__main__":
    main()
