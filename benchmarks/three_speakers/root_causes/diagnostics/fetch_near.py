"""Fetch and verify the three original close-microphone WAVs used only as a diagnosis."""

from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen

REPO = "ggfox00000/dia-alimeeting-test"
REVISION = "cef2837c4a1a0762074522c1bbbbb90eb4d1acf5"
MEETING = "R8008_M8016"
IDS = ("8060", "8061", "8062")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    endpoint = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/audio/near"
    with urlopen(endpoint, timeout=60) as s:
        objects = json.load(s)
    objects = {Path(o["path"]).name: o for o in objects}
    selected = [objects[f"{MEETING}_N_SPK{sid}.wav"] for sid in IDS]

    def install(item):
        name = Path(item["path"]).name
        dest = args.out / name

        def matches(path):
            with path.open("rb") as s:
                return (
                    hashlib.file_digest(s, "sha256").hexdigest() == item["lfs"]["oid"]
                )

        if not dest.exists() or not matches(dest):
            tmp = dest.with_suffix(".part")
            with (
                urlopen(
                    f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{item['path']}?download=true",
                    timeout=90,
                ) as s,
                tmp.open("wb") as o,
            ):
                shutil.copyfileobj(s, o, 2**20)
            if not matches(tmp):
                raise ValueError(f"SHA-256 mismatch for {name}")
            tmp.replace(dest)
        print(name, flush=True)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(install, selected))
    (args.out / "sources.json").write_text(
        json.dumps({"repo": REPO, "revision": REVISION, "files": selected}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
