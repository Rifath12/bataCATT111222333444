"""Prepare full three-participant meetings and disjoint three-person excerpts on channel zero."""

import argparse
import json
import shutil
from pathlib import Path

import soundfile as sf

import prepare_alimeeting as protocol

CLIP_SESSIONS = ("R8004_M8006", "R8005_M8009")
COMPLETE_SESSIONS = ("R8008_M8016", "R8008_M8017")


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def materialize(args):
    base = args.root.resolve()
    clips = base / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    protocol.SESSIONS = CLIP_SESSIONS
    clip_plan = protocol.make_plan(args.references, base)
    clip_plan["corpus"] = (
        "AliMeeting Test far-field, natural excerpts of four-participant meetings"
    )
    complete = []
    all_outputs = {}
    for asset in json.loads((base / "assets.json").read_text())["assets"]:
        filename = Path(asset["path"]).name
        session = filename.rsplit("_", 1)[0]
        if args.group == "short" and session in COMPLETE_SESSIONS:
            continue
        if args.group == "full" and session in CLIP_SESSIONS:
            continue
        with sf.SoundFile(base / filename) as original:
            assert (
                original.samplerate == 16_000
                and original.channels == 8
                and original.subtype == "PCM_16"
            )
            if session in COMPLETE_SESSIONS:
                out = base / session
                out.mkdir(exist_ok=True)
                mono = out / "mono.wav"
                with sf.SoundFile(mono, "w", 16_000, 1, subtype="PCM_16") as dest:
                    for data in original.blocks(64_000, dtype="int16"):
                        dest.write(data[:, 0])
                ref = args.references / f"{session}.rttm"
                shutil.copyfile(ref, out / "reference.rttm")
                turns = protocol.read_rttm(ref)
                speakers, mask = protocol.speaker_masks(turns)
                assert len(speakers) == 3
                complete.append(
                    dict(
                        session=session,
                        source_session=session,
                        audio=str(mono.relative_to(base)),
                        ref=f"{session}/reference.rttm",
                        duration_s=len(original) / 16000,
                        exact_three_s=float((mask.sum(axis=0) == 3).sum()) / 100,
                        speaker_ids=speakers,
                        array_sha256=asset["lfs"]["oid"],
                        wav_sha256=protocol.file_hash(mono),
                        rttm_sha256=protocol.file_hash(ref),
                    )
                )
            else:
                for row in clip_plan["clips"]:
                    if row["source_session"] != session:
                        continue
                    original.seek(row["start_tick"] * 160)
                    expected = 160 * (row["end_tick"] - row["start_tick"])
                    array = original.read(expected, dtype="int16")
                    assert len(array) == expected
                    target = base / row["audio"]
                    sf.write(target, array[:, 0], 16_000, subtype="PCM_16")
                    all_outputs[row["id"]] = protocol.file_hash(target)
        print("Prepared", session, flush=True)
    if args.group == "short":
        save_json(base / "clips_manifest.json", clip_plan)
        save_json(base / "clip_wav_sha256.json", all_outputs)
        paths = [str(base / row["audio"]) for row in clip_plan["clips"]]
        (base / "clip_wavs.txt").write_text(
            "".join(p + "\n" for p in paths), encoding="utf-8"
        )
    else:
        save_json(base / "full_manifest.json", complete)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--group", choices=("short", "full"), required=True)
    materialize(parser.parse_args())
