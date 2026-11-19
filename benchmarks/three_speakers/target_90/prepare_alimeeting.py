"""Select and extract natural AliMeeting excerpts with three distinct speakers.

Selection uses only the published reference annotations. Excerpts retain the complete
clock interval, including silence and fewer-than-three speech. Inference receives WAVs.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import wave

import numpy as np


SESSIONS = ("R8001_M8004", "R8003_M8001", "R8007_M8010", "R8007_M8011")
TICKS_PER_SECOND = 100
RATE = 16_000
SAMPLES_PER_TICK = RATE // TICKS_PER_SECOND
ABSENCE_GUARD_TICKS = 25
MIN_DURATION_TICKS = 800
MIN_TRIPLE_TICKS = 40
REFERENCE_REVISION = "9527b7c64846fb38316a610f32e9d3466bd6d8b7"


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class Window:
    start: int
    end: int
    excluded_speaker: str
    triple_ticks: int


def read_rttm(path: Path) -> list[Turn]:
    turns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[0] == "SPEAKER":
            start = float(fields[3])
            turns.append(Turn(start, start + float(fields[4]), fields[7]))
    return turns


def speaker_masks(turns: list[Turn]) -> tuple[list[str], np.ndarray]:
    speakers = sorted({turn.speaker for turn in turns})
    count = int(np.ceil(max(turn.end for turn in turns) * TICKS_PER_SECOND))
    times = (np.arange(count, dtype=np.float64) + 0.5) / TICKS_PER_SECOND
    masks = np.zeros((len(speakers), count), bool)
    for turn in turns:
        start, end = np.searchsorted(times, [turn.start, turn.end])
        masks[speakers.index(turn.speaker), start:end] = True
    return speakers, masks


def inactive_windows(speakers: list[str], masks: np.ndarray) -> list[Window]:
    clock = masks.shape[1]
    triple = masks.sum(axis=0) == 3
    cumulative = np.r_[0, np.cumsum(triple)]
    candidates = []
    for speaker, active in zip(speakers, masks, strict=True):
        edges = np.flatnonzero(np.diff(np.r_[False, ~active, False]))
        for left, right in zip(edges[::2], edges[1::2], strict=True):
            start = int(left) + (ABSENCE_GUARD_TICKS if left else 0)
            end = int(right) - (ABSENCE_GUARD_TICKS if right < clock else 0)
            if end - start < MIN_DURATION_TICKS:
                continue
            triple_ticks = int(cumulative[end] - cumulative[start])
            if triple_ticks >= MIN_TRIPLE_TICKS:
                candidates.append(Window(start, end, speaker, triple_ticks))
    return candidates


def disjoint_windows(candidates: list[Window]) -> list[Window]:
    """Maximize true-three clock support, breaking ties by retained context."""
    order = sorted(candidates, key=lambda w: (w.end, w.start, w.excluded_speaker))
    ends = [window.end for window in order]
    best: list[tuple[tuple[int, int], list[Window]]] = [((0, 0), [])]
    for index, window in enumerate(order):
        before = bisect_right(ends, window.start, hi=index)
        (triple, context), chosen = best[before]
        include = (
            (triple + window.triple_ticks, context + window.end - window.start),
            [*chosen, window],
        )
        best.append(include if include[0] > best[-1][0] else best[-1])
    return best[-1][1]


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def make_plan(references: Path, out: Path) -> dict:
    clips = []
    checksums = {}
    for session in SESSIONS:
        path = references / f"{session}.rttm"
        checksums[session] = file_hash(path)
        turns = read_rttm(path)
        speakers, masks = speaker_masks(turns)
        if len(speakers) != 4:
            raise ValueError(f"Expected four annotated identities in source {session}")
        for window in disjoint_windows(inactive_windows(speakers, masks)):
            clip_id = f"{session}_{window.start:06d}_{window.end:06d}"
            left = window.start / TICKS_PER_SECOND
            right = window.end / TICKS_PER_SECOND
            clipped = [
                Turn(
                    max(left, turn.start) - left,
                    min(right, turn.end) - left,
                    turn.speaker,
                )
                for turn in turns
                if min(right, turn.end) > max(left, turn.start)
            ]
            if len({turn.speaker for turn in clipped}) != 3 or any(
                turn.speaker == window.excluded_speaker for turn in clipped
            ):
                raise ValueError(
                    f"Clip {clip_id} includes the excluded speaker or lacks a target"
                )
            rttm_path = out / "clips" / f"{clip_id}.rttm"
            rttm_path.write_text(
                "".join(
                    f"SPEAKER {clip_id} 1 {turn.start:.9f} {turn.end - turn.start:.9f} "
                    f"<NA> <NA> {turn.speaker} <NA> <NA>\n"
                    for turn in clipped
                ),
                encoding="utf-8",
            )
            clips.append(
                {
                    "id": clip_id,
                    "source_session": session,
                    "start_tick": window.start,
                    "end_tick": window.end,
                    "duration_s": (window.end - window.start) / TICKS_PER_SECOND,
                    "exact_three_s": window.triple_ticks / TICKS_PER_SECOND,
                    "speaker_ids": sorted({turn.speaker for turn in clipped}),
                    "audio": f"clips/{clip_id}.wav",
                    "reference": f"clips/{clip_id}.rttm",
                }
            )
    return {
        "corpus": "AliMeeting Eval far-field, natural excerpts of 4-participant meetings",
        "corpus_url": "https://www.openslr.org/119/",
        "reference_repo": "https://github.com/nttcslab-sp/diar-forced-alignment",
        "reference_revision": REFERENCE_REVISION,
        "reference_sha256": checksums,
        "selection": {
            "audio_channel": 0,
            "sample_rate": RATE,
            "ticks_per_second": TICKS_PER_SECOND,
            "absent_speaker_guard_s": ABSENCE_GUARD_TICKS / TICKS_PER_SECOND,
            "minimum_clip_s": MIN_DURATION_TICKS / TICKS_PER_SECOND,
            "minimum_exact_three_s": MIN_TRIPLE_TICKS / TICKS_PER_SECOND,
            "tie_breaker": "maximize retained context, given maximum disjoint triple support",
        },
        "clips": clips,
    }


class ForwardReader:
    def __init__(self, stream):
        self.stream = stream

    def read(self, size: int = -1) -> bytes:
        return self.stream.read(size)


def extract_session(stream, clips: list[dict], out: Path) -> None:
    """Read each source only forward and copy channel zero without gain changes."""
    with wave.open(ForwardReader(stream), "rb") as original:
        channels = original.getnchannels()
        if (
            original.getframerate() != RATE
            or original.getsampwidth() != 2
            or channels < 2
        ):
            raise ValueError(
                "Expected multichannel 16 kHz PCM16 input from the eval archive"
            )
        position = 0
        for clip in clips:
            start = clip["start_tick"] * SAMPLES_PER_TICK
            end = clip["end_tick"] * SAMPLES_PER_TICK
            if start < position or end > original.getnframes():
                raise ValueError(f"Invalid positions for clip {clip['id']}")
            while position < start:
                size = min(start - position, 64_000)
                if len(original.readframes(size)) != size * channels * 2:
                    raise EOFError("Source ended while advancing to a clip")
                position += size
            target = out / clip["audio"]
            partial = target.with_suffix(".wav.part")
            with wave.open(str(partial), "wb") as mono:
                mono.setparams((1, 2, RATE, end - start, "NONE", "not compressed"))
                while position < end:
                    size = min(end - position, 64_000)
                    raw = original.readframes(size)
                    if len(raw) != size * channels * 2:
                        raise EOFError("Source ended while reading a clip")
                    frames = np.frombuffer(raw, dtype="<i2").reshape(-1, channels)
                    mono.writeframesraw(frames[:, 0].tobytes())
                    position += size
            partial.replace(target)


def extract(archive: Path, plan: dict, out: Path) -> dict:
    by_session = {session: [] for session in plan["reference_sha256"]}
    for clip in plan["clips"]:
        by_session[clip["source_session"]].append(clip)
    archive_members = {}
    with tarfile.open(archive, mode="r|gz", bufsize=1 << 20) as tar:
        for entry in tar:
            path = PurePosixPath(entry.name)
            if not entry.isfile() or path.parent.parts[-2:] != (
                "Eval_Ali_far",
                "audio_dir",
            ):
                continue
            session = path.stem.rsplit("_", 1)[0]
            if path.suffix.lower() != ".wav" or session not in by_session:
                continue
            if session in archive_members:
                raise ValueError(
                    f"Source archive contains two far-field WAVs for {session}"
                )
            with tar.extractfile(entry) as stream:
                extract_session(stream, by_session[session], out)
            archive_members[session] = entry.name
            print(
                f"Extracted {len(by_session[session])} mono clips from {session}",
                flush=True,
            )
            if len(archive_members) == len(by_session):
                break
    if archive_members.keys() != by_session.keys():
        raise ValueError(
            f"Source archives missing {sorted(by_session.keys() - archive_members.keys())}"
        )
    return {
        "archive_members": archive_members,
        "audio_sha256": {
            clip["id"]: file_hash(out / clip["audio"]) for clip in plan["clips"]
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--references", type=Path)
    ap.add_argument("--archive", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    (args.out / "clips").mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest.json"
    if args.references:
        plan = make_plan(args.references, args.out)
        manifest.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    elif args.archive:
        plan = json.loads(manifest.read_text(encoding="utf-8"))
    else:
        ap.error("Supply --references to plan, --archive to extract, or both")
    if args.archive:
        checksums = extract(args.archive, plan, args.out)
        (args.out / "audio_checksums.json").write_text(
            json.dumps(checksums, indent=2) + "\n", encoding="utf-8"
        )
    print(
        f"{len(plan['clips'])} disjoint clips, "
        f"{sum(c['duration_s'] for c in plan['clips']):.2f} seconds context, "
        f"{sum(c['exact_three_s'] for c in plan['clips']):.2f} seconds exactly three",
        flush=True,
    )


if __name__ == "__main__":
    main()
