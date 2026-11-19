"""Build an annotation-selected AISHELL-4 corpus of exactly-three excerpts."""

import argparse
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from itertools import combinations
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

from prepare_alimeeting import read_rttm, speaker_masks
from recognize import normalized, textgrid

REVISION = "aada72727856313b19d4a030383c426364931dbf"
UPSTREAM = f"https://huggingface.co/datasets/AISHELL/AISHELL-4/resolve/{REVISION}/test"
SESSIONS = ("L_R003S03C02", "M_R003S02C01", "S_R003S04C01", "L_R004S03C01")
RATE = 16000
TICKS = 100
GUARD = 20
WINDOW = 2200
MIN_DUR = 600
MIN_TRIPLE = 40
MAX_PER_SOURCE = 4


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(path, rows):
    path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )


def crop_candidates(speakers, masks):
    all_count = masks.sum(axis=0)
    triple = np.r_[0, np.cumsum(all_count == 3)]
    num_ticks = len(all_count)
    for indices in combinations(range(len(speakers)), 3):
        outsiders = masks[[i for i in range(len(speakers)) if i not in indices]].any(
            axis=0
        )
        valid_ends = np.flatnonzero(np.diff(np.r_[False, ~outsiders, False]))
        for begin, end in zip(valid_ends[::2], valid_ends[1::2], strict=True):
            left = int(begin) + (GUARD if begin > 0 else 0)
            right = int(end) - (GUARD if end < num_ticks else 0)
            if right - left < MIN_DUR or triple[right] - triple[left] < MIN_TRIPLE:
                continue
            regions = {(left, right)} if right - left <= WINDOW else set()
            hits = np.flatnonzero(all_count[left:right] == 3) + left
            if not regions:
                for anchor in np.unique(np.r_[hits[0], hits[::25], hits[-1]]):
                    a = max(left, min(right - WINDOW, anchor - (WINDOW * 2 // 5)))
                    regions.add((a, a + WINDOW))
            for a, b in regions:
                multi = int(triple[b] - triple[a])
                if multi >= MIN_TRIPLE and masks[indices, a:b].any(axis=1).all():
                    yield (a, b, multi, tuple(speakers[i] for i in indices))


def select(candidates):
    """Maximize disjoint exact-three duration; then prefer retained context."""
    candidates = sorted(set(candidates), key=lambda c: (c[1], c[0], c[3]))
    ends = [c[1] for c in candidates]
    none = ((0, 0), [])
    history = [[none] * (MAX_PER_SOURCE + 1)]
    for i, crop in enumerate(candidates):
        a, b, three, _ = crop
        before = bisect_right(ends, a, hi=i)
        row = [none]
        for limit in range(1, MAX_PER_SOURCE + 1):
            prior_quality, picks = history[before][limit - 1]
            choice = (
                (prior_quality[0] + three, prior_quality[1] + b - a),
                [*picks, crop],
            )
            reject = history[-1][limit]
            row.append(choice if choice[0] > reject[0] else reject)
        history.append(row)
    return history[-1][-1][1]


def plan(out, references, rttm_dir, sizes):
    for session in SESSIONS:
        rttm = rttm_dir / (session + ".rttm")
        grid = references / (session + ".TextGrid")
        timings = read_rttm(rttm)
        official = textgrid(grid)
        speakers, masks = speaker_masks(timings)
        for left, right, triple, trio in select(crop_candidates(speakers, masks)):
            a, b = left / TICKS, right / TICKS
            key = f"{session}_{left:06d}_{right:06d}"
            sidecar = out / (key + ".rttm")
            transcript = out / (key + ".json")
            intervals = []
            for turn in timings:
                lo, hi = max(a, turn.start), min(b, turn.end)
                if hi > lo and turn.speaker not in trio:
                    raise RuntimeError(f"Unexpected fourth speaker in {key}")
                if hi > lo:
                    intervals.append(
                        f"SPEAKER {key} 1 {lo - a:.4f} {hi - lo:.4f} <NA> <NA> {turn.speaker} <NA> <NA>"
                    )
            possible = [
                t
                for t in official
                if t["speaker"] in trio and a < t["end"] and t["start"] < b
            ]
            intact = [
                dict(t, start=t["start"] - a, end=t["end"] - a)
                for t in possible
                if a <= t["start"] and t["end"] <= b
            ]
            sidecar.write_text("\n".join(intervals) + "\n", encoding="utf8")
            write_json(transcript, intact)
            meta = sizes[session]
            yield dict(
                id=key,
                source_session=session,
                session=session,
                duration_s=(right - left) / TICKS,
                start=a,
                end=b,
                exact_three_s=triple / TICKS,
                speaker_ids=trio,
                full_session_speaker_count=len(speakers),
                audio=key + ".wav",
                ref=sidecar.name,
                text_reference=transcript.name,
                reference_characters=sum(len(normalized(t["text"])) for t in intact),
                clipped_reference_turns=len(possible) - len(intact),
                audio_channel_index=0,
                audio_url=f"{UPSTREAM}/wav/{session}.flac",
                source_bytes=meta["size"],
                source_lfs_sha256=meta["lfs"]["oid"],
                activity_reference_url=f"{UPSTREAM}/TextGrid/{session}.rttm",
                text_reference_url=f"{UPSTREAM}/TextGrid/{session}.TextGrid",
                activity_reference_sha256=digest(rttm),
                textgrid_sha256=digest(grid),
                cropped_activity_sha256=digest(sidecar),
                cropped_text_sha256=digest(transcript),
            )


class RangeReader(io.RawIOBase):
    """Bounded HTTP range I/O for a local seekable FLAC decoder."""

    block_size = 2 * 1024 * 1024

    def __init__(self, url, size):
        super().__init__()
        self.url = url
        self.size = size
        self.position = 0
        self.blocks = {}
        self.fetched_bytes = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[
            whence
        ]
        self.position = base + offset
        if self.position < 0:
            raise ValueError("Cannot seek before the beginning")
        return self.position

    def fetch_block(self, index):
        if index not in self.blocks:
            left = index * self.block_size
            right = min(left + self.block_size, self.size) - 1
            limit = f"{left}-{right}"
            request = Request(
                f"{self.url}?download=true&span={limit}",
                headers={"Range": f"bytes={limit}"},
            )
            with urlopen(request, timeout=40) as response:
                expected = f"bytes {limit}/{self.size}"
                if (
                    response.status != 206
                    or response.headers.get("Content-Range") != expected
                ):
                    raise IOError(
                        f"Range mismatch {response.status} {response.headers.get('Content-Range')} != {expected}"
                    )
                data = response.read()
            if len(data) != right - left + 1:
                raise IOError("Incomplete response")
            self.blocks[index] = data
            self.fetched_bytes += len(data)
            if len(self.blocks) > 24:
                del self.blocks[next(iter(self.blocks))]
        return self.blocks[index]

    def readinto(self, buffer):
        end = min(self.position + len(buffer), self.size)
        written = 0
        while self.position < end:
            index, offset = divmod(self.position, self.block_size)
            data = self.fetch_block(index)
            length = min(len(data) - offset, end - self.position)
            buffer[written : written + length] = data[offset : offset + length]
            self.position += length
            written += length
        return written


def fetch_session(out, group):
    first = group[0]
    remote = RangeReader(first["audio_url"], first["source_bytes"])
    try:
        with sf.SoundFile(remote) as source:
            rate = source.samplerate
            if source.channels != 8 or rate not in {16000, 48000}:
                raise ValueError(
                    f"Unexpected source clock/channels for {first['session']}"
                )
            for row in group:
                start = round(row["start"] * rate)
                count = round(row["duration_s"] * rate)
                if source.seek(start) != start:
                    raise ValueError(f"Decoder failed to find exact frame {start}")
                block = source.read(count, dtype="float32", always_2d=True)
                if len(block) != count:
                    raise ValueError(f"Incomplete audio window {row['id']}")
                mono = block[:, 0]
                if rate == 48000:
                    mono = resample_poly(mono, 1, 3)
                if len(mono) != round(row["duration_s"] * RATE):
                    raise ValueError("Bad 16 kHz crop size")
                target = out / row["audio"]
                sf.write(target, mono, RATE, subtype="PCM_16")
                check = sf.info(target)
                if (
                    check.samplerate != RATE
                    or check.channels != 1
                    or check.frames != len(mono)
                ):
                    raise ValueError(f"Incorrect packed crop {target}")
                row.update(
                    wav_sha256=digest(target),
                    original_sample_rate=rate,
                    frames=len(mono),
                )
                print(
                    row["id"],
                    row["duration_s"],
                    row["exact_three_s"],
                    "ready",
                    flush=True,
                )
    finally:
        print(
            first["session"],
            "compressed bytes fetched",
            remote.fetched_bytes,
            flush=True,
        )
    return group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--rttm-dir", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    frozen = args.out / "selection.json"
    if not frozen.exists():
        sizes = {
            Path(r["path"]).stem: r
            for r in json.loads(args.inventory.read_text())
            if r["path"].endswith(".flac")
        }
        rows = list(plan(args.out, args.references, args.rttm_dir, sizes))
        write_json(frozen, rows)
    else:
        rows = json.loads(frozen.read_text())
    for name in SESSIONS:
        group = [r for r in rows if r["session"] == name]
        print(
            name,
            len(group),
            sum(r["duration_s"] for r in group),
            sum(r["exact_three_s"] for r in group),
            flush=True,
        )
    if args.plan_only:
        return
    groups = [[r for r in rows if r["session"] == name] for name in SESSIONS]
    groups = [g for g in groups if g]
    with ThreadPoolExecutor(max_workers=min(4, len(groups))) as pool:
        done = list(pool.map(lambda group: fetch_session(args.out, group), groups))
    rows = sorted((r for group in done for r in group), key=lambda r: r["id"])
    write_json(args.out / "manifest.json", rows)
    (args.out / "inputs.txt").write_text(
        "".join(str(args.out / r["audio"]) + "\n" for r in rows)
    )


if __name__ == "__main__":
    main()
