"""Fetch the exact public AISHELL-3 utterances used in the waveform trials."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    manifest = json.loads(
        Path(__file__).with_name("source_manifest.json").read_text(encoding="utf-8")
    )
    a.out.mkdir(parents=True, exist_ok=True)
    for file in manifest["files"]:
        dest = a.out / Path(file["path"]).name
        if (
            dest.exists()
            and hashlib.sha256(dest.read_bytes()).hexdigest() == file["sha256"]
        ):
            continue
        base = f"https://huggingface.co/datasets/{manifest['dataset']}/resolve/{manifest['revision']}"
        pending = dest.with_suffix(".part")
        with (
            urlopen(base + "/" + file["path"], timeout=45) as response,
            pending.open("wb") as f,
        ):
            shutil.copyfileobj(response, f)
        if hashlib.sha256(pending.read_bytes()).hexdigest() != file["sha256"]:
            raise ValueError(f"Audio SHA-256 mismatch for {dest}")
        pending.replace(dest)


if __name__ == "__main__":
    main()
