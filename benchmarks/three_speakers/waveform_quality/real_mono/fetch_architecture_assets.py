"""Fetch the exact model tensors and the optional upstream MossFormer2 code."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
REVISIONS = json.loads((HERE / "model_revisions.json").read_text())


def checksum(path, blob):
    with path.open("rb") as stream:
        if not blob:
            return hashlib.file_digest(stream, "sha256").hexdigest()
        digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode("ascii"))
        while data := stream.read(1024 * 1024):
            digest.update(data)
        return digest.hexdigest()


def fetch(url, path, expected, *, blob=False):
    if path.is_file() and checksum(path, blob) == expected:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        pending = Path(stream.name)
        try:
            with urlopen(url, timeout=120) as remote:
                while data := remote.read(1024 * 1024):
                    stream.write(data)
            stream.flush()
            if checksum(pending, blob) != expected:
                raise RuntimeError(f"Checksum mismatch for {path.name}")
        except BaseException:
            pending.unlink(missing_ok=True)
            raise
    pending.replace(path)


def prepare(directory, families):
    for name in families:
        metadata = REVISIONS[name]
        base = "https://huggingface.co/{source}/resolve/{revision}".format(**metadata)
        for filename, digest in metadata["files"].items():
            fetch(f"{base}/{filename}", directory / name / filename, digest)
        if "code" in metadata:
            code = metadata["code"]
            base = "https://raw.githubusercontent.com/{source}/{revision}".format(**code)
            for filename, digest in code["git_blobs"].items():
                fetch(f"{base}/{filename}", directory / "moss_official" / filename, digest, blob=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--models", type=Path, required=True)
    cli.add_argument("--family", action="append", choices=REVISIONS)
    args = cli.parse_args()
    prepare(args.models, args.family or list(REVISIONS))


if __name__ == "__main__":
    main()
