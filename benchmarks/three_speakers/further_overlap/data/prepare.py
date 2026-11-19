"""Create a fixed AliMeeting challenge panel with exactly three annotated active people."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
from struct import unpack_from
from urllib.request import Request, urlopen

import numpy as np
import soundfile as sf

from prepare_alimeeting import inactive_windows, read_rttm, speaker_masks
from recognize import normalized, textgrid

ALIGN_REV = "9527b7c64846fb38316a610f32e9d3466bd6d8b7"
AUDIO_REV = "68dd695283723a78d850895466b9a380f630b561"
WORD_REV = "d7188c179603db8566c50b6e43af5c8a1e7678ba"
AUDIO_ROOT = (
    "https://huggingface.co/datasets/AudioMarathon/AudioMarathon/resolve/"
    f"{AUDIO_REV}/AliMeeting/Test_Ali/Test_Ali_far"
)
SESSIONS = {"R8002_M8002": "MS802", "R8005_M8007": "MS806", "R8005_M8008": "MS806"}
CROPS_PER_SESSION = 6
MIN_TICKS = 800
MAX_TICKS = 7500
HOP = 100
RATE = 16000


@dataclass
class Crop:
    start: int
    end: int
    triple: int
    speaker_ids: tuple[str]
    utterances: list[dict]

    @property
    def duration(self):
        return self.end - self.start


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def fetch(url, path):
    if not path.exists():
        with urlopen(url, timeout=30) as source:
            path.write_bytes(source.read())


def byte_range(url, left, right):
    bounds = f"{left}-{right}"
    request = Request(
        f"{url}?download=true&bytes={bounds}", headers={"Range": f"bytes={bounds}"}
    )
    with urlopen(request, timeout=55) as source:
        if source.status != 206 or not source.headers.get(
            "Content-Range", ""
        ).startswith(f"bytes {bounds}/"):
            raise ValueError(f"Incomplete audio byte range for {url} ({bounds})")
        data = source.read()
    if len(data) != right - left + 1:
        raise ValueError("Wrong byte count from source")
    return data


def pcm_data_offset(header):
    if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise ValueError("PCM RIFF/WAVE input required")
    pos = 12
    is_pcm = False
    while pos + 8 <= len(header):
        label = header[pos : pos + 4]
        size = unpack_from("<I", header, pos + 4)[0]
        start = pos + 8
        if label == b"fmt " and start + size <= len(header):
            encoding, channels, rate, _, align, depth = unpack_from(
                "<HHIIHH", header, start
            )
            linear = encoding == 1 or (
                encoding == 65534
                and size >= 40
                and unpack_from("<I", header, start + 24)[0] == 1
            )
            is_pcm = linear and (channels, rate, align, depth) == (8, RATE, 16, 16)
        if label == b"data" and is_pcm:
            return start
        pos = start + size + (size % 2)
    raise ValueError("Expected 8-channel, 16 kHz PCM16 audio")


def crops(speakers, mask, utterances):
    triple_prefix = np.r_[0, np.cumsum(mask.sum(axis=0) == 3)]
    for window in inactive_windows(speakers, mask):
        start, end = window.start, window.end
        if end - start > MAX_TICKS:
            starts = np.arange(start, end - MAX_TICKS + 1)
            start = int(
                starts[
                    np.argmax(triple_prefix[starts + MAX_TICKS] - triple_prefix[starts])
                ]
            )
            end = start + MAX_TICKS
        begin, finish = start / HOP, end / HOP
        if end - start < MIN_TICKS or triple_prefix[end] - triple_prefix[start] < 40:
            continue
        people = sorted(t for i, t in enumerate(speakers) if (mask[i, start:end]).any())
        if len(people) != 3:
            continue
        included = [
            t
            for t in utterances
            if t["speaker"] in people and begin <= (t["start"] + t["end"]) / 2 < finish
        ]
        if sorted({r["speaker"] for r in included}) == people:
            yield Crop(
                start,
                end,
                int(triple_prefix[end] - triple_prefix[start]),
                tuple(people),
                included,
            )


def select(candidates):
    """Weighted interval scheduling with a fixed number of excerpts per meeting."""
    candidates = sorted(candidates, key=lambda w: (w.end, w.start, w.speaker_ids))
    ends = [w.end for w in candidates]
    levels = [[((0, 0), []) for _ in range(CROPS_PER_SESSION + 1)]]
    for i, pick in enumerate(candidates):
        before = bisect_right(ends, pick.start, hi=i)
        row = [((0, 0), [])]
        for budget in range(1, CROPS_PER_SESSION + 1):
            (triple, clock), earlier = levels[before][budget - 1]
            new = ((triple + pick.triple, clock + pick.duration), [*earlier, pick])
            old = levels[i][budget]
            row.append(new if new[0] > old[0] else old)
        levels.append(row)
    return levels[-1][-1][1]


def alignment(session, speaker, references):
    stem = f"{session}_{speaker}"
    url = (
        "https://huggingface.co/datasets/MultiTalk/MultiTalkBench/resolve/"
        f"{WORD_REV}/alignments/zh/{stem}.json"
    )
    path = references / f"{stem}.alignment.json"
    fetch(url, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["id"] == stem
    rows = [
        dict(t, speaker=speaker)
        for t in data["word_alignments"]
        if normalized(t["text"]) and float(t["start"]) < float(t["end"])
    ]
    return rows, dict(url=url, sha256=digest(path))


def timed_row(row, start, end):
    left, right = max(row["start"], start) - start, min(row["end"], end) - start
    return dict(
        row,
        start=left,
        end=right,
        characters=[
            dict(
                text=row["text"],
                start_ms=round(1000 * left),
                end_ms=round(1000 * right),
            )
        ],
    )


def selection(session, references, output):
    rttm = references / f"{session}.rttm"
    grid = references / f"{session}.TextGrid"
    alignment_url = (
        "https://raw.githubusercontent.com/nttcslab-sp/diar-forced-alignment/"
        f"{ALIGN_REV}/AliMeeting/Test_Ali_far/{session}.rttm"
    )
    textgrid_url = f"{AUDIO_ROOT}/textgrid_dir/{session}.TextGrid"
    wav_url = f"{AUDIO_ROOT}/audio_dir/{session}_{SESSIONS[session]}.wav"
    fetch(alignment_url, rttm)
    fetch(textgrid_url, grid)
    turns = read_rttm(rttm)
    speakers, mask = speaker_masks(turns)
    if len(speakers) != 4:
        raise ValueError(f"{session} needs four published speaker IDs")
    result = []
    words, word_sources = [], {}
    for name in speakers:
        entries, meta = alignment(session, name, references)
        words.extend(entries)
        word_sources[name] = meta
    words.sort(key=lambda row: (row["start"], row["speaker"], row["end"]))
    for item in select(crops(speakers, mask, words)):
        start, end = item.start / HOP, item.end / HOP
        key = f"{session}_{item.start:06d}_{item.end:06d}"
        ref, labels = output / f"{key}.rttm", output / f"{key}.json"
        with ref.open("w", encoding="utf-8") as target:
            for turn in turns:
                lo, hi = max(start, turn.start), min(end, turn.end)
                if lo < hi:
                    print(
                        f"SPEAKER {key} 1 {lo - start:.4f} {hi - lo:.4f} <NA> <NA> {turn.speaker} <NA> <NA>",
                        file=target,
                    )
        labels.write_text(
            json.dumps(
                [timed_row(r, start, end) for r in item.utterances],
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        target_activity = mask[:, item.start : item.end]
        result.append(
            dict(
                session=session,
                start=start,
                end=end,
                duration_s=item.duration / HOP,
                exact_three_s=item.triple / HOP,
                exact_two_s=float((target_activity.sum(axis=0) == 2).sum()) / HOP,
                speaker_ids=list(item.speaker_ids),
                session_speaker_ids=speakers,
                text_chars=sum(len(normalized(r["text"])) for r in item.utterances),
                midpoint_three_chars=sum(
                    len(normalized(r["text"]))
                    for r in item.utterances
                    if (mask[:, int((r["start"] + r["end"]) * HOP / 2)].sum() == 3)
                ),
                complete_textgrid_turns=sum(
                    start <= r["start"] < r["end"] <= end
                    and r["speaker"] in item.speaker_ids
                    for r in textgrid(grid)
                ),
                word_reference_sources={n: word_sources[n] for n in item.speaker_ids},
                word_reference_revision=WORD_REV,
                audio=f"{key}.wav",
                ref=ref.name,
                text_reference=labels.name,
                audio_url=wav_url,
                audio_channel_index=0,
                align_reference_url=alignment_url,
                textgrid_url=textgrid_url,
                align_reference_sha256=digest(rttm),
                textgrid_sha256=digest(grid),
                clipped_rttm_sha256=digest(ref),
                clipped_reference_sha256=digest(labels),
            )
        )
    return result


def download(row, output, offsets):
    dest = output / row["audio"]
    if not dest.exists():
        origin = offsets[row["session"]]
        left = origin + 16 * round(row["start"] * RATE)
        right = origin + 16 * round(row["end"] * RATE)
        pcm = np.frombuffer(
            byte_range(row["audio_url"], left, right - 1), dtype="<i2"
        ).reshape(-1, 8)
        sf.write(dest, pcm[:, 0], RATE, subtype="PCM_16")
    info = sf.info(dest)
    if (
        info.channels != 1
        or info.samplerate != RATE
        or info.frames != round(row["duration_s"] * RATE)
    ):
        raise ValueError(f"Wrong exported channel or duration for {dest}")
    return dict(row, wav_sha256=digest(dest))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--plan-only", action="store_true")
    a = p.parse_args()
    a.out.mkdir(exist_ok=True, parents=True)
    a.references.mkdir(exist_ok=True, parents=True)
    with ThreadPoolExecutor(3) as pool:
        groups = list(
            pool.map(
                partial(selection, references=a.references, output=a.out), SESSIONS
            )
        )
        plan = [row for group in groups for row in group]
        pools = [set(r["session_speaker_ids"]) for group in groups for r in group[:1]]
        assert sum(map(len, pools)) == len(set().union(*pools))
        (a.out / "selection.json").write_text(json.dumps(plan, indent=2) + "\n")
        for group in groups:
            assert group, "Expected at least one text-aligned crop per meeting"
            print(
                group[0]["session"],
                len(group),
                round(sum(r["duration_s"] for r in group), 2),
                round(sum(r["exact_three_s"] for r in group), 2),
                sum(r["text_chars"] for r in group),
                flush=True,
            )
        shortlist = [
            row
            for group in groups
            for row in sorted(
                group, key=lambda row: (-row["exact_three_s"], -row["duration_s"])
            )[:2]
        ]
        (a.out / "shortlist.json").write_text(json.dumps(shortlist, indent=2) + "\n")
        if not a.plan_only:
            offsets = {
                g[0]["session"]: pcm_data_offset(byte_range(g[0]["audio_url"], 0, 4095))
                for g in groups
            }
            prioritized = {row["audio"] for row in shortlist}
            plan = shortlist + [r for r in plan if r["audio"] not in prioritized]
            complete = []
            for row in pool.map(partial(download, output=a.out, offsets=offsets), plan):
                complete.append(row)
                names = [t["audio"] for t in complete]
                print(
                    "Downloaded",
                    row["audio"],
                    row["exact_three_s"],
                    row["midpoint_three_chars"],
                    flush=True,
                )
                if len(complete) == len(prioritized) and prioritized.issubset(names):
                    keep = [r for r in complete if r["audio"] in prioritized]
                    (a.out / "shortlist_ready.json").write_text(
                        json.dumps(keep, indent=2) + "\n"
                    )
                    (a.out / "shortlist_inputs.txt").write_text(
                        "".join(
                            str((a.out / r["audio"]).resolve()) + "\n" for r in keep
                        )
                    )
            (a.out / "manifest.json").write_text(json.dumps(complete, indent=2) + "\n")
            (a.out / "inputs.txt").write_text(
                "".join(str((a.out / r["audio"]).resolve()) + "\n" for r in complete)
            )


if __name__ == "__main__":
    main()
