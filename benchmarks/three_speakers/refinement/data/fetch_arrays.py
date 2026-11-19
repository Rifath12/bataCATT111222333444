"""Fetch verified public Mandarin meeting waveforms for the fresh benchmark."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import shutil
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent


def fetch(asset, dataset, revision):
    filename = Path(asset["path"]).name
    output = ROOT / filename
    if output.exists() and output.stat().st_size == asset["size"]:
        with output.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() == asset["lfs"]["oid"]:
                return
    temp = output.with_suffix(".part")
    for attempt in range(4):
        stamp = f"?download=true&attempt={attempt}"
        try:
            url = f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{asset['path']}{stamp}"
            with urlopen(url, timeout=50) as src, temp.open("wb") as dst:
                shutil.copyfileobj(src, dst, 2**20)
            with temp.open("rb") as stream:
                assert (
                    hashlib.file_digest(stream, "sha256").hexdigest()
                    == asset["lfs"]["oid"]
                )
            temp.replace(output)
            print("Verified", filename, flush=True)
            return
        except Exception as e:
            print(f"{filename} attempt {attempt + 1}: {e!r}", flush=True)
            time.sleep(2)
    raise RuntimeError(filename)


if __name__ == "__main__":
    data = json.loads((ROOT / "assets.json").read_text())
    with ThreadPoolExecutor(4) as pool:
        list(
            pool.map(
                lambda a: fetch(a, data["dataset"], data["revision"]), data["assets"]
            )
        )
