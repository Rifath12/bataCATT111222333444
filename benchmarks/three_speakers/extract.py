"""Select the original room and headset audio and annotation for one AliMeeting session."""

from pathlib import Path
import argparse
import json
import shutil
import tarfile
import soundfile as sf


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("--session", default="R8008_M8013")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    members = []
    with tarfile.open(args.archive, "r|gz") as arc:
        for entry in arc:
            if args.session not in entry.name or not entry.isfile():
                continue
            name = Path(entry.name).name
            if not name.lower().endswith((".wav", ".textgrid")):
                continue
            dest = args.out / name
            if "far/" in entry.name and name.lower().endswith(".wav"):
                dest = args.out / "array.wav"
            if "far/" in entry.name and name.lower().endswith(".textgrid"):
                dest = args.out / "far.TextGrid"
            with arc.extractfile(entry) as read, dest.open("wb") as write:
                shutil.copyfileobj(read, write, 1024 * 1024)
            assert dest.stat().st_size == entry.size
            members.append(
                {"file": entry.name, "local": dest.name, "bytes": entry.size}
            )
            print(entry.name, entry.size, flush=True)
            if dest.name == "array.wav":
                with (
                    sf.SoundFile(str(dest)) as original,
                    sf.SoundFile(
                        str(args.out / "mono.wav"),
                        "w",
                        samplerate=original.samplerate,
                        channels=1,
                        subtype="PCM_16",
                    ) as mono,
                ):
                    for frames in original.blocks(
                        blocksize=262144, dtype="int16", always_2d=True
                    ):
                        mono.write(frames[:, 0])
                print(
                    "Channel 1 WAV, sampled at",
                    sf.info(args.out / "mono.wav").samplerate,
                    flush=True,
                )
    if (
        not (args.out / "mono.wav").is_file()
        or not (args.out / "far.TextGrid").is_file()
    ):
        raise ValueError("Expected room microphone audio and a TextGrid")
    (args.out / "sources.json").write_text(
        json.dumps(members, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
