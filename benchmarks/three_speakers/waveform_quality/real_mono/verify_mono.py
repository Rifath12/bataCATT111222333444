"""Check immutable source spans, WAV samples and physical-receiver controls against raw PCM."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import wavfile

RATE = 16000
TAKE = 16384
FRAME = 512
ARMS = ("direct_sum", "array_channel0")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check(roomdir, manifest):
    fs, far = wavfile.read(roomdir / "array.wav", mmap=True)
    if fs != RATE or far.ndim != 2 or far.shape[1] != 8:
        raise ValueError("Invalid array PCM")
    rx = far[:, 0]
    near = [
        wavfile.read(roomdir / f"{s}.wav", mmap=True)[1]
        for s in manifest["speaker_ids"]
    ]
    labels = []
    for row in (roomdir / "room.rttm").read_text().splitlines():
        v = row.split()
        labels.append(
            (
                int(v[7].removeprefix("N_SPK")),
                round(float(v[3]) * RATE),
                round((float(v[3]) + float(v[4])) * RATE),
            )
        )
    span_list = [[], [], []]
    maxerr = 0
    clocks = manifest["clock"]
    members = []
    for li, s in enumerate("ABC"):
        for item in manifest["enrollment"][s]:
            members.append((li, item, True))
    for event in manifest["events"]:
        for li, s in enumerate("ABC"):
            if s in event["members"]:
                members.append((li, event["donors"][s], False))
    for li, info, enrollment in members:
        lo = info["array_start"]
        hi = lo + TAKE
        nl = info["near_start"]
        expected = round(lo * (1 + clocks["slope_s_per_s"]) + clocks["offset_s"] * RATE)
        if nl != expected:
            raise ValueError("Incorrect microphone clock offset")
        if enrollment and hi > len(rx) * 0.25:
            raise ValueError("Enrollment exceeds allowed recording prefix")
        if not enrollment and lo < len(rx) * 0.28 - 1:
            raise ValueError("Evaluation precedes 28% chronological boundary")
        speaker = manifest["speaker_ids"][li]
        if any(a < hi + 3184 and b > lo - 3184 and j != speaker for j, a, b in labels):
            raise ValueError("Another participant is annotated within the solo margin")
        if not any(a < lo and b > lo for j, a, b in labels if j == speaker):
            raise ValueError("Donor start is outside own annotation")
        span_list[li].append((lo, hi))
    spans = sorted(sum(span_list, []))
    if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError("Donor intervals are reused")
    with np.load(roomdir / ARMS[0] / "meeting.npz") as whole:
        total = whole["mixture"].size
    digest_arms = {}
    for arm in ARMS:
        # Reconstruct all contributions from the physical PCM and manifest; check true source channel.
        stack = np.zeros((3, total), np.float32)
        act = np.zeros((3, total), bool)
        for event in manifest["events"]:
            path = roomdir / arm / "held" / f"{event['id']}.npz"
            if sha(path) != event["sha256"][arm]:
                raise ValueError(f"Changed fixture {path}")
            with np.load(path, allow_pickle=False) as z:
                r = z["references"]
                obs = z["mixture"]
                a = z["activity"]
                y = np.zeros_like(r)
                for i, name in enumerate("ABC"):
                    if name not in event["members"]:
                        continue
                    info = event["donors"][name]
                    src = near[i] if arm == "direct_sum" else rx
                    lo = (
                        info["near_start"]
                        if arm == "direct_sum"
                        else info["array_start"]
                    )
                    rms = (
                        info["near_active_rms"]
                        if arm == "direct_sum"
                        else info["array_active_rms"]
                    )
                    samples = src[lo : lo + TAKE].astype("float32") / 32768
                    taper = np.ones(TAKE, np.float32)
                    taper[:80] = np.linspace(0, 1, 80)
                    taper[-80:] = taper[:80][::-1]
                    off = event["offsets"][name]
                    y[i, off : off + TAKE] = (
                        samples * taper * (0.05 / rms)
                    ) * np.float32(10.0 ** (event["nominal_gains_db"][i] / 20))
                err = float(np.max(abs(y - r)))
                maxerr = max(maxerr, err)
                if (
                    err > 1e-7
                    or not np.array_equal(obs, r.sum(0))
                    or not np.array_equal(z["exact_three"], a.all(0))
                ):
                    raise ValueError("Incorrect source mixture or active masks")
                stack[:, event["start"] : event["end"]] = r
                act[:, event["start"] : event["end"]] = a
        with np.load(roomdir / arm / "meeting.npz") as whole:
            assert np.array_equal(stack, whole["references"]) and np.array_equal(
                act, whole["activity"]
            )
            assert np.array_equal(
                wavfile.read(roomdir / arm / "meeting.wav")[1], whole["mixture"]
            )
            assert np.array_equal(whole["mixture"], stack.sum(0))
        digest_arms[arm] = {
            "pcm_mixture_sha256": hashlib.sha256(stack.sum(0).tobytes()).hexdigest(),
            "pcm_reference_sha256": hashlib.sha256(stack.tobytes()).hexdigest(),
        }
    return dict(
        room=roomdir.name,
        speakers=manifest["speaker_ids"],
        distinct_source_intervals=len(spans),
        enrollment_intervals=sum(x[-1] for x in members),
        evaluated_source_intervals=sum(not x[-1] for x in members),
        events=len(manifest["events"]),
        common_active_people_seconds=manifest["active_people_seconds"],
        maximum_reference_reconstruction_absolute_difference=maxerr,
        streams=digest_arms,
        manifest_sha256=sha(roomdir / "mono_manifest.json"),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    a = p.parse_args()
    files = json.loads((a.root / "sources.json").read_text())
    result = {
        "sources_sha256": sha(a.root / "sources.json"),
        "build_sha256": sha(Path(__file__).with_name("build_mono.py")),
        "sync_sha256": sha(Path(__file__).with_name("sync.py")),
        "rooms": [],
    }
    for room in files["rooms"]:
        d = a.root / room
        result["rooms"].append(
            check(d, json.loads((d / "mono_manifest.json").read_text()))
        )
    (a.root / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
