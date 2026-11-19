"""Build single-channel Mandarin mixtures from isolated turns at one real microphone."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import wavfile

from sync import calibrate

RATE = 16000
FRAME = 512
TAKE = 16384
EVENT = 24576
MARGIN = 3200
LEVELS = (0, -6, -12, -20)
ARMS = ("direct_sum", "array_channel0")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def annotation(path, room):
    rows = []
    for line in path.read_text().splitlines():
        z = line.split()
        if z[0] != "SPEAKER" or z[1] != room:
            raise ValueError("Unexpected annotation")
        rows.append(
            (
                int(z[7].removeprefix("N_SPK")),
                round(float(z[3]) * 100),
                round((float(z[3]) + float(z[4])) * 100),
            )
        )
    return rows


def intervals(rows, speakers, length):
    labels = sorted({x[0] for x in rows})
    act = np.zeros((len(labels), 1 + length // 160), bool)
    for s, a, b in rows:
        act[labels.index(s), a:b] = True
    result = []
    for who in speakers:
        solo = act[labels.index(who)] & (act.sum(axis=0) == 1)
        edges = np.flatnonzero(np.diff(np.r_[False, solo, False])) * 160
        result.append(
            [
                (int(a) + MARGIN, min(int(b), length) - MARGIN)
                for a, b in zip(edges[::2], edges[1::2])
            ]
        )
    return result


def wave(path):
    sr, x = wavfile.read(path, mmap=True)
    if sr != RATE or x.dtype != np.dtype("int16"):
        raise ValueError("Expected 16 kHz signed PCM")
    return x


def activity(x):
    amp = np.sqrt(np.mean(x.reshape(-1, FRAME) ** 2, axis=1))
    # Threshold relative to input-specific active-speech level; same frame clock for both arms.
    return np.repeat(amp >= max(0.00010, 0.16 * np.quantile(amp, 0.7)), FRAME)


def candidates(regions, array, near, clock, phase, who):
    result = []
    low, high = (
        (0, int(0.25 * len(array)))
        if phase == "enroll"
        else (int(0.28 * len(array)), len(array))
    )
    for a, b in regions:
        a = max(low, a)
        b = min(high, b)
        for start in range(a, b - TAKE + 1, TAKE):
            near_start = round(
                start * (1 + clock["slope_s_per_s"]) + clock["offset_s"] * RATE
            )
            if near_start < 0 or near_start + TAKE > len(near[who]):
                continue
            n = near[who][near_start : near_start + TAKE].astype(np.float32) / 32768
            y = array[start : start + TAKE].astype(np.float32) / 32768
            active = activity(n) & activity(y)
            if active.mean() < 0.6:
                continue
            nr = float(np.sqrt(np.mean(n[active] ** 2)))
            yr = float(np.sqrt(np.mean(y[active] ** 2)))
            if nr < 0.002 or yr < 0.00020:
                continue
            other = max(
                np.sqrt(
                    np.mean(
                        (x[near_start : near_start + TAKE].astype(np.float32) / 32768)
                        ** 2
                    )
                )
                for j, x in enumerate(near)
                if j != who
            )
            if other > 0.85 * nr:
                continue
            # Higher direct/room envelope correlation is used only as donor QA.
            n_env = np.sqrt((n.reshape(-1, FRAME) ** 2).mean(1))
            y_env = np.sqrt((y.reshape(-1, FRAME) ** 2).mean(1))
            corr = np.corrcoef(n_env, y_env)[0, 1]
            if not np.isfinite(corr) or corr < 0.12:
                continue
            fade = np.ones(TAKE, np.float32)
            fade[:80] = np.linspace(0, 1, 80)
            fade[-80:] = fade[:80][::-1]
            active[:80] = False
            active[-80:] = False
            result.append(
                (
                    dict(
                        array_start=start,
                        near_start=near_start,
                        correlation=float(corr),
                        near_active_rms=nr,
                        array_active_rms=yr,
                    ),
                    n * fade * (0.05 / nr),
                    y * fade * (0.05 / yr),
                    active,
                )
            )
    return result


def schedule():
    # Event order is identical across rooms. Each target is quiet at each triple speech level exactly once.
    index = {"s": 0, "d": 0, "t": 0}
    order = "stdttdtsdttdtdtstdttt"
    assert (order.count("s"), order.count("d"), order.count("t")) == (3, 6, 12)
    events = []
    for k in order:
        i = index[k]
        index[k] += 1
        active = {
            "s": [[0], [1], [2]],
            "d": [[0, 1], [1, 2], [2, 0]],
            "t": [[0, 1, 2]],
        }[k]
        who = active[i % len(active)]
        levels = [None] * 3
        for j in who:
            levels[j] = 0
        if k == "t":
            levels[(i // 4 + i % 4) % 3] = LEVELS[i % 4]
        if k == "d":
            levels[who[(i // 3) % 2]] = 0 if i < 3 else -12
        events.append((f"{k}{i + 1:02d}", who, levels))
    return events


def savewav(p, x):
    wavfile.write(p, RATE, x.astype(np.float32))


def build(root, room, speakers, entries):
    d = root / room
    for e in entries:
        if e["file"].startswith(room + "/") and digest(root / e["file"]) != e["sha256"]:
            raise ValueError(f"Bad source checksum {e['file']}")
    far = wave(d / "array.wav")
    if far.ndim != 2 or far.shape[1] != 8:
        raise ValueError("Expected eight physical array channels")
    array = far[:, 0]
    near = [wave(d / f"{s}.wav") for s in speakers]
    rows = annotation(d / "room.rttm", room)
    spans = intervals(rows, speakers, len(array))
    clock = calibrate(array, near)
    rng = np.random.default_rng(172311 + int(room[-4:]))
    bank = {
        k: [candidates(ss, array, near, clock, k, i) for i, ss in enumerate(spans)]
        for k in ("enroll", "test")
    }
    print(
        room,
        "clock",
        clock["offset_s"],
        clock["slope_s_per_s"],
        "donors",
        {k: list(map(len, v)) for k, v in bank.items()},
        flush=True,
    )
    if min(map(len, bank["test"])) < 17 or min(map(len, bank["enroll"])) < 3:
        raise ValueError("Insufficient disjoint known-source speech")
    for k, banks in bank.items():
        for b in banks:
            rng.shuffle(b)
    record = dict(
        room=room,
        speaker_ids=speakers,
        excluded_speakers=sorted({x[0] for x in rows} - set(speakers)),
        array_channel=0,
        sample_rate=RATE,
        clock=clock,
        eligible={k: list(map(len, v)) for k, v in bank.items()},
        enrollment={},
        events=[],
    )
    for arm in ARMS:
        (d / arm / "held").mkdir(parents=True, exist_ok=True)
    for i, name in enumerate("ABC"):
        examples = bank["enroll"][i][:8]
        record["enrollment"][name] = [x[0] for x in examples]
        for arm_idx, arm in enumerate(ARMS):
            x = np.concatenate([z[arm_idx + 1] for z in examples])
            savewav(d / arm / f"enroll_{name}.wav", x)
    data = [[], []]
    activity_parts = []
    head = 4096
    for arm in data:
        arm.append(np.zeros((3, head), np.float32))
    activity_parts.append(np.zeros((3, head), bool))
    start = head
    for ci, (label, members, levels) in enumerate(schedule()):
        curves = [np.zeros((3, EVENT), np.float32) for _ in ARMS]
        mask = np.zeros((3, EVENT), bool)
        donors = {}
        offsets = {}
        for j in members:
            info, close, room_image, activity_mask = bank["test"][j].pop()
            offset = (ci + 2 * j) % 4 * FRAME + 2048
            scale = np.float32(10.0 ** (levels[j] / 20))
            for values, arm in zip([close, room_image], curves, strict=True):
                arm[j, offset : offset + TAKE] = scale * values
            mask[j, offset : offset + TAKE] = activity_mask
            donors["ABC"[j]] = info
            offsets["ABC"[j]] = offset
        exact = mask.all(0)
        if len(members) == 3 and exact.sum() < 1024:
            raise ValueError("Insufficient triple overlap")
        names = {}
        for ai, arm in enumerate(ARMS):
            ref = curves[ai]
            mixture = ref.sum(0, dtype=np.float32)
            if np.max(abs(mixture)) > 0.99:
                raise ValueError("Unexpected clipping")
            filename = d / arm / "held" / f"{label}.npz"
            np.savez_compressed(
                filename,
                mixture=mixture,
                references=ref,
                activity=mask,
                exact_three=exact,
                sample_rate=RATE,
                nominal_gains_db=np.array(
                    [g if g is not None else -np.inf for g in levels]
                ),
            )
            names[arm] = digest(filename)
            data[ai].append(ref)
        activity_parts.append(mask)
        record["events"].append(
            dict(
                id=label,
                start=start,
                end=start + EVENT,
                members=["ABC"[j] for j in members],
                nominal_gains_db=levels,
                donors=donors,
                offsets=offsets,
                exact_three_samples=int(exact.sum()),
                sha256=names,
            )
        )
        start += EVENT
        if (ci + 1) % 5 == 0:
            gap = 2048
            for arm in data:
                arm.append(np.zeros((3, gap), np.float32))
            activity_parts.append(np.zeros((3, gap), bool))
            start += gap
    activity_parts.append(np.zeros((3, head), bool))
    act = np.concatenate(activity_parts, axis=1)
    record["duration_s"] = act.shape[1] / RATE
    record["active_people_seconds"] = (
        np.bincount(act.sum(0), minlength=4) / RATE
    ).tolist()
    for ai, arm in enumerate(ARMS):
        ref = np.concatenate(data[ai] + [np.zeros((3, head), np.float32)], axis=1)
        x = ref.sum(0, dtype=np.float32)
        savewav(d / arm / "meeting.wav", x)
        np.savez_compressed(
            d / arm / "meeting.npz",
            mixture=x,
            references=ref,
            activity=act,
            exact_three=act.all(0),
            sample_rate=RATE,
        )
    (d / "mono_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        "locked",
        room,
        record["duration_s"],
        record["active_people_seconds"],
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--room")
    a = parser.parse_args()
    s = json.loads((a.root / "sources.json").read_text())
    for room, people in s["rooms"].items():
        if a.room is None or a.room == room:
            build(a.root, room, people, s["sources"])
