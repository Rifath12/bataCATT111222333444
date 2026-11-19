"""Fetch three predeclared, speaker-disjoint AliMeeting Test excerpts and references."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
import struct
from urllib.request import Request, urlopen

import numpy as np
import soundfile as sf

from prepare_alimeeting import inactive_windows, read_rttm, speaker_masks
from recognize import textgrid


AUDIO_REVISION = "68dd695283723a78d850895466b9a380f630b561"
REFERENCE_REVISION = "9527b7c64846fb38316a610f32e9d3466bd6d8b7"
BASE = f"https://huggingface.co/datasets/AudioMarathon/AudioMarathon/resolve/{AUDIO_REVISION}/"
SOURCE_DIR = "AliMeeting/Test_Ali/Test_Ali_far"
MICROPHONES = {
    "R8002_M8003": "MS803",
    "R8004_M8005": "MS803",
    "R8005_M8007": "MS806",
    "R8005_M8008": "MS806",
    "R8006_M8012": "MS803",
}


def read_range(url, left, right):
    bounds = f"{left}-{right}"
    request = Request(
        f"{url}?download=true&bytes={bounds}", headers={"Range": f"bytes={bounds}"}
    )
    with urlopen(request, timeout=75) as response:
        if response.status != 206 or not response.headers.get(
            "Content-Range", ""
        ).startswith(f"bytes {bounds}/"):
            raise ValueError(f"Server failed to return requested byte range {bounds}")
        data = response.read()
    if len(data) != right - left + 1:
        raise ValueError("Incomplete audio response")
    return data


def pcm_wave_metadata(header):
    if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise ValueError("Expected a RIFF/WAVE source")
    pos = 12
    valid_format = False
    while pos + 8 <= len(header):
        chunk = header[pos : pos + 4]
        size = struct.unpack_from("<I", header, pos + 4)[0]
        start = pos + 8
        if chunk == b"fmt " and start + size <= len(header):
            enc, channels, rate, _, align, depth = struct.unpack_from(
                "<HHIIHH", header, start
            )
            pcm = enc == 1 or (
                enc == 65534
                and size >= 40
                and struct.unpack_from("<I", header, start + 24)[0] == 1
            )
            valid_format = pcm and (channels, rate, align, depth) == (8, 16000, 16, 16)
        if chunk == b"data" and valid_format:
            return start
        pos = start + size + (size % 2)
    raise ValueError("Expected 16 kHz, eight-channel, PCM16 source in the RIFF header")


def download_selected(row, output):
    session = row["session"]
    source = f"{SOURCE_DIR}/audio_dir/{session}_{MICROPHONES[session]}.wav"
    url = BASE + source
    pcm_offset = pcm_wave_metadata(read_range(url, 0, 4095))
    left, right = (round(row[key] * 16000) for key in ("start", "end"))
    raw = read_range(url, pcm_offset + 16 * left, pcm_offset + 16 * right - 1)
    samples = np.frombuffer(raw, dtype="<i2").reshape(-1, 8)[:, 0]
    target = output / row["audio"]
    sf.write(target, samples, 16000, subtype="PCM_16")
    row.update(
        source_audio_path=source,
        source_revision=AUDIO_REVISION,
        wav_sha256=sha256(target.read_bytes()).hexdigest(),
    )
    return row


def fetch_reference(url, path):
    if not path.exists():
        with urlopen(url, timeout=45) as response:
            path.write_bytes(response.read())


def candidate(session, output):
    reference_url = (
        "https://raw.githubusercontent.com/nttcslab-sp/diar-forced-alignment/"
        f"{REFERENCE_REVISION}/AliMeeting/Test_Ali_far/{session}.rttm"
    )
    rttm = output / f"{session}.full.rttm"
    grid = output / f"{session}.TextGrid"
    fetch_reference(reference_url, rttm)
    fetch_reference(BASE + f"{SOURCE_DIR}/textgrid_dir/{session}.TextGrid", grid)
    words = textgrid(grid)
    turns = read_rttm(rttm)
    speakers, mask = speaker_masks(turns)
    options = []
    for window in inactive_windows(speakers, mask):
        start, end = window.start / 100, window.end / 100
        while crossing := [w["end"] for w in words if w["start"] < start < w["end"]]:
            start = max(crossing)
        while crossing := [w["start"] for w in words if w["start"] < end < w["end"]]:
            end = min(crossing)
        if end - start < 30:
            continue
        ref = [
            dict(w, start=w["start"] - start, end=w["end"] - start)
            for w in words
            if start <= w["start"] < w["end"] <= end
        ]
        if len({w["speaker"] for w in ref}) != 3:
            continue
        activity = mask[:, round(100 * start) : round(100 * end)].sum(axis=0)
        triple = float((activity == 3).sum()) / 100
        options.append((triple, end - start, start, end, ref))
    if not options:
        return None
    triple, duration, start, end, ref = max(options, key=lambda v: v[:2])
    name = f"{session}_{round(100 * start):06d}_{round(100 * end):06d}"
    (output / f"{name}.json").write_text(json.dumps(ref, indent=2, ensure_ascii=False))
    with (output / f"{name}.rttm").open("w") as cropped:
        for turn in turns:
            left, right = max(turn.start, start), min(turn.end, end)
            if left < right:
                print(
                    f"SPEAKER {name} 1 {left - start:.4f} {right - left:.4f} <NA> <NA> {turn.speaker} <NA> <NA>",
                    file=cropped,
                )
    return dict(
        session=session,
        source_session=session,
        start=start,
        end=end,
        duration_s=duration,
        exact_three_s=triple,
        audio=f"{name}.wav",
        ref=f"{name}.rttm",
        text_reference=f"{name}.json",
        speaker_ids=sorted({w["speaker"] for w in ref}),
        aligned_reference_revision=REFERENCE_REVISION,
        full_textgrid_sha256=sha256(grid.read_bytes()).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("new_excerpts"))
    args = parser.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(2) as pool:
        rows = pool.map(partial(candidate, output=output), sorted(MICROPHONES))
        chosen = sorted(
            (r for r in rows if r),
            key=lambda r: (-r["exact_three_s"], -r["duration_s"]),
        )[:3]
        chosen = list(pool.map(partial(download_selected, output=output), chosen))
    (output / "manifest.json").write_text(
        json.dumps(chosen, indent=2, ensure_ascii=False) + "\n"
    )
    (output / "inputs.txt").write_text(
        "".join(f"{output / r['audio']}\n" for r in chosen)
    )
    print([(r["session"], r["duration_s"], r["exact_three_s"]) for r in chosen])


if __name__ == "__main__":
    main()
