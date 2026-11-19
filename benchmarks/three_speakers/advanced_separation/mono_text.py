"""Refine single-microphone transcripts with two corroborating recognizers."""

import argparse
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from time import perf_counter

from rapidfuzz.distance import Levenshtein

from recognize import inference_spans, normalized

SPEAKERS = tuple("ABC")
RATE = 16000
SOURCE_CHARACTERS = re.compile(r"<[^>]*>|\[[^\]]*\]|[\s\S]")


def key(row):
    return row["speaker"], round(row["start"], 3), round(row["end"], 3)


def project(anchor, alternative):
    chars = list(anchor)
    insertions = [""] * (len(anchor) + 1)
    for op in Levenshtein.editops(anchor, alternative):
        index = op.src_pos
        if op.tag == "insert":
            insertions[index] += alternative[op.dest_pos]
        elif op.tag == "replace":
            chars[index] = alternative[op.dest_pos]
        else:
            chars[index] = ""
    return chars, insertions


def preserve_format(raw, anchor, revised):
    """Apply normalized edits at their source glyphs, preserving untouched formatting."""
    source = []
    for token in SOURCE_CHARACTERS.finditer(raw):
        chars = normalized(token.group())
        if chars:
            source.append((token.start(), token.end(), chars))
    if "".join(chars for _, _, chars in source) != anchor:
        # Composed Unicode across glyph boundaries has no unambiguous source offset.
        return raw

    output = []
    previous_end = position = 0
    for start, end, chars in source:
        replacement = "".join(revised[position:position + len(chars)])
        output.append(raw[previous_end:start])
        output.append(raw[start:end] if replacement == chars else replacement)
        previous_end = end
        position += len(chars)
    output.extend((revised[-1], raw[previous_end:]))
    rendered = "".join(output)
    return rendered if normalized(rendered) == "".join(revised) else raw


def updated_text(row, text):
    edited = dict(row, text=text)
    for field in ("characters", "tokens", "timestamps", "confidence"):
        edited.pop(field, None)
    return edited


def amend(base, alternate):
    out, n = [], 0
    for row in base:
        raw = row["text"]
        chars = normalized(raw)
        alternative = alternate.get(key(row))
        if len(chars) < 4 or not alternative:
            out.append(row)
            continue
        a, b = alternative
        if min(Levenshtein.normalized_similarity(chars, text) for text in (a, b)) < 0.65:
            out.append(row)
            continue
        t1, i1 = project(chars, a)
        t2, i2 = project(chars, b)
        keep = [u == v == orig for orig, u, v in zip(chars, t1, t2)]
        answer = list(chars) + [""]
        for j in range(1, len(chars) - 1):
            if keep[j - 1] and keep[j + 1] and t1[j] == t2[j]:
                if t1[j] or chars[j] == chars[j - 1] or chars[j] == chars[j + 1]:
                    answer[j] = t1[j]
            if keep[j - 1] and keep[j] and i1[j] == i2[j] and len(i1[j]) <= 2:
                answer[j] = i1[j] + answer[j]
        text = raw if "".join(answer) == chars else preserve_format(raw, chars, answer)
        if text == raw:
            out.append(row)
        else:
            out.append(updated_text(row, text))
            n += 1
    return out, n


def whole_turn_matches(rows, opinions):
    """Allow one extra edit when both available views agree on the entire turn."""
    revised = []
    edits = 0
    for row in rows:
        alternate = opinions.get(key(row))
        if alternate:
            text, corroboration = alternate
            anchor = normalized(row["text"])
            if (
                text == corroboration
                and max(len(text), len(anchor)) >= 5
                and Levenshtein.distance(anchor, text) == 1
            ):
                chars, insertions = project(anchor, text)
                planned = [addition + char for addition, char in zip(insertions, chars)]
                planned.append(insertions[-1])
                result = preserve_format(row["text"], anchor, planned)
                if result != row["text"]:
                    row = updated_text(row, result)
                    edits += 1
        revised.append(row)
    return revised, edits


def read_rows(path):
    rows, seen = [], set()
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                label, start, end = row["speaker"], row["start"], row["end"]
                if label not in SPEAKERS or not isinstance(row["text"], str):
                    raise ValueError("expected speaker A/B/C and text")
                if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
                    raise ValueError("expected finite, increasing turn timestamps")
                identity = key(row)
                if identity in seen:
                    raise ValueError(f"duplicate turn {identity}")
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error
            seen.add(identity)
            rows.append(row)
    return rows


def decode(recognizer, audio, spans):
    import numpy as np

    for start, end in spans:
        first, last = round(start * RATE), round(end * RATE)
        audio.seek(first)
        samples = audio.read(last - first, dtype="float32")
        if not len(samples):
            continue
        rms = max(float(np.sqrt(np.mean(samples**2))), 1e-7)
        peak = max(float(np.abs(samples).max()), 1e-7)
        samples = np.pad(samples * min(0.035 / rms, 0.98 / peak, 20.0), (800, 1280))
        stream = recognizer.create_stream()
        stream.accept_waveform(RATE, samples)
        recognizer.decode_stream(stream)
        yield start, end, stream.result.text.strip()


def aligned_tracks(tracks):
    import soundfile

    paths = {speaker: tracks / f"speaker_{speaker}.wav" for speaker in SPEAKERS}
    layouts = {(info.samplerate, info.channels, info.frames)
               for info in map(soundfile.info, paths.values())}
    if len(layouts) != 1:
        raise ValueError("Speaker tracks must have identical sample rates, channel counts and lengths")
    sample_rate, channel_count, samples = layouts.pop()
    if (sample_rate, channel_count) != (RATE, 1) or not samples:
        raise ValueError("Expected nonempty, aligned 16 kHz mono WAV tracks")
    return paths


def candidate_text(model, backend, paths, timing, selected, threads):
    import sherpa_onnx
    import soundfile

    options = dict(tokens=str(model / "tokens.txt"), num_threads=threads)
    weights = str(model / "model.int8.onnx")
    if backend == "zipformer":
        recognizer = sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(model=weights, **options)
    else:
        recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(paraformer=weights, **options)
    texts = {}
    for speaker, path in paths.items():
        with soundfile.SoundFile(path) as audio:
            turns = [row for row in timing if row["speaker"] == speaker]
            spans = inference_spans(turns, len(audio) / RATE)
            if selected is not None:
                spans = ((a, b) for a, b in spans if (speaker, round(a, 3), round(b, 3)) in selected)
            for start, end, speech in decode(recognizer, audio, spans):
                texts[(speaker, round(start, 3), round(end, 3))] = normalized(speech)
    return texts


def export_rows(rows, output):
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".transcript-", dir=output) as work:
        pending = Path(work)
        combined = pending / "transcript.jsonl"
        combined.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        (pending / "transcript.txt").write_text(
            "".join(
                f"[{row['start']:.3f}, {row['end']:.3f}] {row['speaker']}  {row['text']}\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        for label in SPEAKERS:
            text = "".join(
                f"[{row['start']:.3f}, {row['end']:.3f}] {row['text']}\n"
                for row in rows if row["speaker"] == label
            )
            (pending / f"speaker_{label}.txt").write_text(text, encoding="utf-8")
        for label in SPEAKERS:
            name = f"speaker_{label}.txt"
            (pending / name).replace(output / name)
        (pending / "transcript.txt").replace(output / "transcript.txt")
        combined.replace(output / combined.name)


def cached_views(path):
    return {key(row): normalized(row["text"]) for row in read_rows(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("baseline", "out"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    for option in ("tracks", "timing", "zipformer", "paraformer", "zipformer-jsonl", "paraformer-jsonl"):
        parser.add_argument(f"--{option}", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    started = perf_counter()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.zipformer_jsonl is None or args.paraformer_jsonl is None:
        if not args.tracks or not args.timing:
            parser.error("decoding an uncached recognizer requires --tracks and --timing")
        paths = aligned_tracks(args.tracks)
        timing = json.loads(args.timing.read_text(encoding="utf-8-sig"))["segments"]
    rows = read_rows(args.baseline)
    if args.zipformer_jsonl:
        zip_views = cached_views(args.zipformer_jsonl)
    elif args.zipformer:
        zip_views = candidate_text(args.zipformer, "zipformer", paths, timing, None, args.threads)
    else:
        parser.error("supply --zipformer or --zipformer-jsonl")
    if rows and not any(key(row) in zip_views for row in rows):
        raise ValueError("Zipformer opinions and baseline do not share any speaker/turn coordinates")

    if args.paraformer_jsonl:
        para_views = cached_views(args.paraformer_jsonl)
    elif args.paraformer:
        selected = {
            key(row) for row in rows
            if (view := zip_views.get(key(row))) and amend([row], {key(row): (view, view)})[1]
        }
        para_views = (
            candidate_text(args.paraformer, "paraformer", paths, timing, selected, args.threads)
            if selected else {}
        )
    else:
        parser.error("supply --paraformer or --paraformer-jsonl")
    opinions = {k: (zip_views[k], para_views[k]) for k in zip_views.keys() & para_views.keys()}
    revised, changes = amend(rows, opinions)
    revised, whole_turn_changes = whole_turn_matches(revised, opinions)
    export_rows(revised, args.out)
    metadata = {
        "updated_turns": changes,
        "whole_turn_edits": whole_turn_changes,
        "elapsed_s": perf_counter() - started,
        "used_cached_zipformer": args.zipformer_jsonl is not None,
        "used_cached_paraformer": args.paraformer_jsonl is not None,
    }
    (args.out / "mono_quality.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {changes} turns")


if __name__ == "__main__":
    main()
