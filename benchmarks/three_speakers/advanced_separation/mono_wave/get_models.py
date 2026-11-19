"""Download and verify reproducible pretrained assets for the separator trial."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def download(url, destination, expected_hash):
    if destination.is_file() and sha256(destination) == expected_hash:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as pending:
        temporary = Path(pending.name)
        try:
            with urlopen(url, timeout=90) as response:
                shutil.copyfileobj(response, pending, length=1024 * 1024)
            pending.close()
            recorded = sha256(temporary)
            if recorded != expected_hash:
                raise ValueError(f"SHA-256 mismatch for {destination.name}: {recorded}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def source_tree(source, destination):
    github_path = source["repo"].removeprefix("https://github.com/")
    commit = source["commit"]
    archive = destination / "recorrnet.tar.gz"
    download(
        f"https://api.github.com/repos/{github_path}/tarball/{commit}",
        archive,
        source["source_archive_sha256"],
    )

    reference = destination / "reference"
    marker = reference / ".verified-commit"
    if marker.is_file() and marker.read_text().strip() == commit:
        return
    with tempfile.TemporaryDirectory(dir=destination) as pending:
        staging = Path(pending)
        with tarfile.open(archive, mode="r:gz") as tar:
            tar.extractall(staging, filter="data")
        entries = list(staging.iterdir())
        if len(entries) != 1 or not (entries[0] / "sr_corrnet").is_dir():
            raise ValueError("Unexpected CorrNet source tree")
        (entries[0] / ".verified-commit").write_text(commit + "\n")
        if reference.exists():
            shutil.copytree(entries[0], reference, dirs_exist_ok=True)
        else:
            entries[0].replace(reference)


def main():
    from separate import MODEL_NAMES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_NAMES), action="append")
    parser.add_argument("--output", type=Path, default=ROOT)
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((ROOT / "provenance.json").read_text())
    names = [MODEL_NAMES[tag] for tag in (options.model or MODEL_NAMES)]
    if any(name.startswith(("sr-corrnet", "recorrnet")) for name in names):
        source_tree(provenance["upstream_code"], options.output)

    pending = []
    for name in names:
        model = provenance["models"][name]
        for relative, expected_hash in model["files"].items():
            path = options.output / "models" / name / relative
            pending.append((
                f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{relative}?download=true",
                path,
                expected_hash,
            ))

    with ThreadPoolExecutor(max_workers=4) as pool:
        for job in [pool.submit(download, *item) for item in pending]:
            job.result()
    print(f"Verified {len(names)} model checkpoint(s) in {options.output}.")


if __name__ == "__main__":
    main()
