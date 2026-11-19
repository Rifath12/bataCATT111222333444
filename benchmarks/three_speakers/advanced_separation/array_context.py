"""Decode the continuous array beams with bounded linguistic context and soft activity gating."""

from argparse import ArgumentParser
from contextlib import ExitStack
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
import sherpa_onnx as sx
import soundfile as sf

from mono_text import export_rows
from recognize import text_scores, textgrid


SPEAKERS = "ABC"
RATE = 16000
FRAME = 160


def valid_turns(rows, duration):
    """Keep valid portions of turns that intersect the analyzed sample clock."""
    for row in rows:
        speaker = row["speaker"]
        start = float(row["start"])
        end = float(row["end"])
        if speaker not in ("A", "B", "C") or not all(map(math.isfinite, (start, end))):
            raise ValueError("Timing requires finite start/end values and a speaker in A/B/C")
        if end < start:
            raise ValueError(f"Reversed timing interval for {speaker}: {start} to {end}")
        start, end = max(0.0, start), min(duration, end)
        if start < end:
            yield dict(speaker=speaker, start=start, end=end)


def activity(timing, samples):
    nframes = (samples + FRAME - 1) // FRAME
    active = np.zeros((nframes, len(SPEAKERS)), dtype=bool)
    for row in valid_turns(timing, samples / RATE):
        speaker = SPEAKERS.index(row["speaker"])
        start = int(np.floor(row["start"] * RATE / FRAME))
        end = int(np.ceil(row["end"] * RATE / FRAME))
        active[start:end, speaker] = True
    return active


def beam_power(source, samples):
    nframes = (samples + FRAME - 1) // FRAME
    powers = np.empty(nframes, dtype="float32")
    source.seek(0)
    for first in range(0, nframes, 1000):
        count = min(1000, nframes - first)
        requested = min(count * FRAME, samples - first * FRAME)
        waveform = source.read(requested, dtype="float32")
        if len(waveform) < count * FRAME:
            waveform = np.pad(waveform, (0, count * FRAME - len(waveform)))
        powers[first : first + count] = np.mean(waveform.reshape(count, FRAME) ** 2, axis=1)
    return uniform_filter1d(powers, 7)


def unmix_power(powers, active):
    """Fit the fixed leakage pattern from confidently solo frames."""
    silence = ~active.any(axis=1)
    noise_frames = powers[silence] if np.any(silence) else powers
    floor = np.quantile(noise_frames, 0.30 if np.any(silence) else 0.025, axis=0)
    observed = np.maximum(powers - floor, 0.0)
    total = observed.sum(axis=1)
    profiles = np.eye(len(SPEAKERS))
    for speaker in range(len(SPEAKERS)):
        solo = active[:, speaker] & (active.sum(axis=1) == 1)
        totals = total[solo]
        if len(totals):
            loud = solo & (total >= np.percentile(totals, 60))
            ratios = observed[loud] / np.maximum(total[loud, None], 1e-12)
            profile = np.median(ratios, axis=0)
            if profile.sum() > 0:
                profiles[speaker] = profile / profile.sum()
    mixing = profiles.T
    gram = mixing.T @ mixing + 0.015 * np.eye(len(SPEAKERS))
    projected = observed @ mixing
    step = 1.0 / np.linalg.eigvalsh(gram).max()
    estimates = np.maximum(np.linalg.solve(gram, projected.T).T, 0.0)
    for _ in range(18):
        estimates = np.maximum(estimates + step * (projected - estimates @ gram), 0.0)
    return estimates, mixing, floor


def spans(turns, duration, bridge, margin, limit):
    """Pad nearby turns, assigning any shared padded boundary to one decode window."""
    combined = []
    for row in sorted(turns, key=lambda item: item["start"]):
        start = max(0.0, row["start"] - margin)
        end = min(duration, row["end"] + margin)
        if end - start < 0.055:
            continue
        if (
            combined
            and start - combined[-1][1] < bridge
            and end - combined[-1][0] < limit
        ):
            combined[-1][1] = max(combined[-1][1], end)
        elif combined and end <= combined[-1][1]:
            # Entirely nested turns already occur in the previous decode window.
            continue
        else:
            if combined and start < combined[-1][1]:
                # The two spans could not be merged at the decoding duration cap.
                # Divide shared context rather than decoding both copies.
                cut = 0.5 * (start + combined[-1][1])
                combined[-1][1] = cut
                start = cut
            combined.append([start, end])
    for start, end in combined:
        while end - start > limit + 1:
            yield start, start + limit
            start += limit
        yield start, end


def decoder_context(mode, speaker, active, estimates):
    if mode != "extended":
        bridge, margin = (0.40, 0.14) if mode == "pad" else (0.30, 0.0)
        return bridge, margin, np.ones(len(active), dtype="float32")

    nearby = gaussian_filter1d(active[:, speaker].astype("float32"), 7, mode="constant")
    own = estimates[:, speaker]
    others = estimates.sum(axis=1) - own
    confidence = (own / np.maximum(own + others, 1e-12)) > 0.42
    retained = active[:, speaker] | ((nearby > 0.013) & (confidence | ~active.any(axis=1)))
    return 0.42, 0.18, gaussian_filter1d(retained.astype("float32"), 2.0, mode="constant")


def decode_span(source, start, end, weight, recognizer):
    first = round(start * RATE)
    last = round(end * RATE)
    frames = np.arange(first, last) / FRAME - 0.5
    left = np.clip(np.floor(frames).astype(int), 0, len(weight) - 1)
    fraction = np.maximum(frames - left, 0)
    right = np.minimum(left + 1, len(weight) - 1)
    gain = weight[left] * (1.0 - fraction) + weight[right] * fraction
    source.seek(first)
    speech = (source.read(last - first, dtype="float32") * gain).astype("float32")
    rms = max(float(np.sqrt(np.mean(speech ** 2))), 1e-7)
    peak = max(float(np.max(np.abs(speech))), 1e-7)
    stream = recognizer.create_stream()
    scale = min(0.035 / rms, 0.98 / peak, 20.0)
    stream.accept_waveform(RATE, np.pad(speech * scale, (800, 1280)))
    recognizer.decode_stream(stream)
    return stream.result.text.strip()


def recognize(sources, samples, timing, active, recognizer, mode, estimates, labels, limit=17.0):
    rows = []
    duration = samples / RATE
    timing = list(valid_turns(timing, duration))
    for label in labels:
        speaker = SPEAKERS.index(label)
        started = time.perf_counter()
        turns = [row for row in timing if row["speaker"] == label]
        bridge, margin, weight = decoder_context(mode, speaker, active, estimates)
        for start, end in spans(turns, duration, bridge, margin, limit):
            decoded = decode_span(sources[speaker], start, end, weight, recognizer)
            if decoded:
                rows.append(dict(speaker=label, start=start, end=end, text=decoded))
        print(mode, label, round(time.perf_counter() - started, 2), flush=True)
    return sorted(rows, key=lambda row: (row["start"], row["speaker"]))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Evaluation-only TextGrid")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=["extended", "pad", "bare"], default=["extended"])
    parser.add_argument("--pilot-seconds", type=float, default=float("inf"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--labels", default=SPEAKERS)
    parser.add_argument("--reference-labels")
    parser.add_argument("--limit", type=float, default=17.0)
    opts = parser.parse_args()
    if math.isnan(opts.pilot_seconds) or opts.pilot_seconds <= 0:
        parser.error("--pilot-seconds must be positive")
    if not math.isfinite(opts.limit) or opts.limit < 0.055:
        parser.error("--limit must be finite and at least 0.055 seconds")
    if opts.threads < 1:
        parser.error("--threads must be positive")
    opts.out.mkdir(parents=True, exist_ok=True)
    timing = json.loads(opts.timing.read_text(encoding="utf-8"))["segments"]
    files = [opts.tracks / f"beam_{speaker}.flac" for speaker in SPEAKERS]
    layouts = {(item.samplerate, item.channels, item.frames) for item in map(sf.info, files)}
    if len(layouts) != 1:
        parser.error("beam_A/B/C.flac must be sample-aligned and have identical layouts")
    rate, channels, count = layouts.pop()
    if (rate, channels) != (RATE, 1) or count < FRAME:
        parser.error("expected nonempty 16 kHz mono beams")
    if opts.pilot_seconds < float("inf"):
        count = min(count, int(opts.pilot_seconds * RATE))
    if count < FRAME:
        parser.error("the selected audio duration is shorter than one analysis frame")
    if not opts.labels or not set(opts.labels) <= set(SPEAKERS) or len(set(opts.labels)) != len(opts.labels):
        parser.error("--labels must contain distinct letters from ABC")
    reference = []
    if opts.reference:
        reference = [r for r in textgrid(opts.reference) if r["end"] <= count / RATE]
        if opts.reference_labels:
            reference = [r for r in reference if r["speaker"] in opts.reference_labels.split(",")]
    predicted = activity(timing, count)
    recognizer = sx.OfflineRecognizer.from_sense_voice(
        model=str(opts.model / "model.int8.onnx"), tokens=str(opts.model / "tokens.txt"),
        language="zh", use_itn=False, num_threads=opts.threads,
    )

    with ExitStack() as stack:
        sources = [stack.enter_context(sf.SoundFile(path)) for path in files]
        power = np.stack([beam_power(source, count) for source in sources], axis=1)
        estimates, mixing, floor = unmix_power(power, predicted)
        print("Leakage profile", mixing.tolist(), "noise", floor.tolist(), flush=True)
        for mode in opts.modes:
            started = time.perf_counter()
            root = opts.out / mode
            root.mkdir(exist_ok=True)
            rows = recognize(sources, count, timing, predicted, recognizer, mode, estimates, opts.labels, opts.limit)
            export_rows(rows, root)
            scores = {}
            if opts.reference:
                scores = (
                    text_scores(reference, rows)
                    if reference
                    else {"cpCER": None, "evaluation": "no_reference_speech_in_window"}
                )
            scores.update(asr_seconds=time.perf_counter() - started, duration_s=count / RATE)
            (root / "scores.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
            print(mode, scores.get("cpCER"), len(rows), flush=True)


if __name__ == "__main__":
    main()
