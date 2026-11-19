"""Make a partly reference-scored, same-clock M8017 AliMeeting replay.

Supply three original headset WAVs named for their speaker IDs, the pinned RTTM and
the frozen scene manifest. The manifest supplies exact donor coordinates and gains.
None of the source recordings or controlled references need to be redistributed.
"""
from argparse import ArgumentParser
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.io import wavfile

RATE = 16000
HOP = 512
SPEAKERS = (8063, 8064, 8065)
CUTOFF = 120 * RATE
MARGIN = 1920
RAMP = 128


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def activity(x):
    padded = np.pad(x, (0, -len(x) % HOP))
    amp = np.sqrt((padded.reshape(-1, HOP) ** 2).mean(axis=1))
    return np.repeat(amp >= max(.0015, .16 * np.quantile(amp, .70)), HOP)[:len(x)]


def read_image(audio, start, end):
    return np.stack([track[start:end] for track in audio]).astype(np.float32) / 32768.


def read_sources(root, book, rttm):
    if sha(rttm) != book["rttm_sha256"]:
        raise ValueError("Unexpected RTTM content")
    tracks = []
    for source in book["sources"]:
        path = root / f"{source['speaker_id']}.wav"
        if sha(path) != source["sha256"]:
            raise ValueError(f"Unexpected headset WAV for {path.name}")
        rate, pcm = wavfile.read(path, mmap=True)
        if rate != RATE or pcm.dtype != np.int16 or pcm.ndim != 1 or len(pcm) != book["samples"]:
            raise ValueError("Unexpected recorder format")
        tracks.append(pcm)
    return tracks


def inputs_and_groups(args):
    manifest = json.loads(args.manifest.read_text())
    audio = read_sources(args.audio, manifest, args.rttm)
    return manifest, audio


def materialize(group, audio):
    members = group["members"]
    length = group["length"]
    mix = np.zeros((3, length), np.float32)
    ref = np.zeros_like(mix)
    active = np.zeros_like(mix, dtype=bool)
    for who, entry in enumerate(members):
        if entry is None:
            continue
        start = entry["start"]
        stop = entry["end"]
        off = entry["offset"]
        n = stop-start
        image = read_image(audio, start, stop)
        mark = activity(image[who])
        taper = np.ones(n, np.float32)
        taper[:RAMP] = np.linspace(0,1,RAMP)
        taper[-RAMP:] = taper[:RAMP][::-1]
        image *= taper * np.float32(entry["gain"])
        mix[:, off:off+n] += image
        ref[who, off:off+n] = image[who]
        active[who, off+RAMP:off+n-RAMP] = mark[RAMP:-RAMP]
    return mix, ref, active


def write_calibration(path, destination):
    """Restore the stored transfer fitted from the original meeting's first 120 seconds."""
    document = json.loads(path.read_text())
    transfer = np.tile(np.eye(3, dtype=np.float32), (513, 1, 1))
    for i, receiver in enumerate("ABC"):
        for j, source in enumerate("ABC"):
            if i != j:
                transfer[:, i, j] = document["paths"][f"{source}_to_{receiver}"]
    if not np.isfinite(transfer).all() or (transfer < 0).any():
        raise ValueError("Invalid numeric calibration")
    np.save(destination, transfer, allow_pickle=False)
    if sha(destination) != document["sha256"]:
        raise ValueError("Recovered power transfer differs from the frozen original")


def build(args):
    manifest, audio = inputs_and_groups(args)
    dest = args.out
    dest.mkdir(exist_ok=False, parents=True)
    if args.calibration:
        write_calibration(args.calibration, dest/"power_transfer.npy")
    refs = dest / "reference_windows"
    refs.mkdir()
    paths = [dest / f"{s}.wav" for s in "ABC"]
    counts = np.zeros(4, np.int64)
    replaced, vacated = 0, 0
    occupied = []
    with ExitStack() as stack:
        writers = [stack.enter_context(sf.SoundFile(p, mode="w", channels=1,
                     samplerate=RATE, subtype="FLOAT")) for p in paths]
        for start in range(0, manifest["samples"], RATE * 4):
            for x, writer in zip(audio, writers, strict=True):
                writer.write(x[start:start+RATE*4].astype(np.float32) / 32768.)
        for group in manifest["groups"]:
            mix, ref, active = materialize(group, audio)
            anchor = group["destination_start"]
            for entry in group["members"]:
                if not entry:
                    continue
                start, end = entry["start"], entry["end"]
                if start < CUTOFF + MARGIN:
                    raise ValueError("Calibration and evaluation overlap")
                occupied.append((start,end))
                if start == anchor:
                    continue
                for writer in writers:
                    writer.seek(start)
                    writer.write(np.zeros(end-start, np.float32))
                vacated += end-start
            for writer, channel in zip(writers, mix, strict=True):
                writer.seek(anchor)
                writer.write(channel)
            replaced += group["length"]
            counts += np.bincount(active.sum(axis=0), minlength=4)
            np.savez_compressed(refs / (group["id"] + ".npz"), references=ref, activity=active)
    occupied.sort()
    if any(a[1] > b[0] for a,b in zip(occupied, occupied[1:])):
        raise ValueError("A donor occurred twice")
    result = {"samples":manifest["samples"], "sample_rate":RATE, "duration_s":manifest["samples"]/RATE,
              "scored_windows":len(manifest["groups"]), "scored_window_s":replaced/RATE,
              "scored_activity_s_0_1_2_3":[int(x)/RATE for x in counts],
              "vacated_original_donor_slot_s":vacated/RATE,
              "untouched_original_s":(manifest["samples"]-replaced-vacated)/RATE,
              "sha256":{p.name:sha(p) for p in paths},"manifest_sha256":sha(args.manifest)}
    (dest / "scene.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--audio",type=Path,required=True)
    p.add_argument("--rttm",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--calibration",type=Path,help="Frozen numeric M8017 calibration sidecar")
    p.add_argument("--out",type=Path,required=True)
    build(p.parse_args())
