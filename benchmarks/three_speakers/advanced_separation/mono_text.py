"""Refine single-microphone transcripts with two corroborating recognizers."""

import argparse
import json
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from recognize import inference_spans, normalized


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
        answer = list(chars)
        for j in range(1, len(chars) - 1):
            if keep[j - 1] and keep[j + 1] and t1[j] == t2[j]:
                if t1[j] or chars[j] == chars[j - 1] or chars[j] == chars[j + 1]:
                    answer[j] = t1[j]
            if keep[j - 1] and keep[j] and i1[j] == i2[j] and len(i1[j]) <= 2:
                answer[j] = i1[j] + answer[j]
        text = "".join(answer)
        if text == chars:
            out.append(row)
        else:
            edited = dict(row, text=text)
            edited.pop("characters", None)
            out.append(edited)
            n += 1
    return out, n


def read_rows(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def decode(recognizer, audio, spans):
    import numpy as np

    for start, end in spans:
        first, last = round(start * 16000), round(end * 16000)
        audio.seek(first)
        samples = audio.read(last - first, dtype="float32")
        if not len(samples):
            continue
        rms = max(float(np.sqrt(np.mean(samples**2))), 1e-7)
        peak = max(float(np.abs(samples).max()), 1e-7)
        samples = np.pad(samples * min(0.035 / rms, 0.98 / peak, 20.0), (800, 1280))
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, samples)
        recognizer.decode_stream(stream)
        yield start, end, stream.result.text.strip()


def candidate_text(model, backend, tracks, timing, selected, threads):
    import sherpa_onnx
    import soundfile

    options = dict(tokens=str(model / "tokens.txt"), num_threads=threads)
    weights = str(model / "model.int8.onnx")
    if backend == "zipformer":
        recognizer = sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(model=weights, **options)
    else:
        recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(paraformer=weights, **options)
    texts = {}
    for speaker in "ABC":
        with soundfile.SoundFile(tracks / f"speaker_{speaker}.wav") as audio:
            if (audio.samplerate, audio.channels) != (16000, 1):
                raise ValueError("Expected aligned 16 kHz mono WAV tracks")
            turns = [row for row in timing if row["speaker"] == speaker]
            spans = inference_spans(turns, len(audio) / 16000)
            if selected is not None:
                spans = ((a, b) for a, b in spans if (speaker, round(a, 3), round(b, 3)) in selected)
            for start, end, speech in decode(recognizer, audio, spans):
                texts[(speaker, round(start, 3), round(end, 3))] = normalized(speech)
    return texts


def export_rows(rows, output):
    output.mkdir(parents=True, exist_ok=True)
    partial = output / "transcript.jsonl.tmp"
    partial.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    partial.replace(output / "transcript.jsonl")
    for label in "ABC":
        text = "".join(
            f"[{row['start']:.3f}, {row['end']:.3f}] {row['text']}\n"
            for row in rows if row["speaker"] == label
        )
        (output / f"speaker_{label}.txt").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("baseline", "out"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    for option in ("tracks", "timing", "zipformer", "paraformer", "zipformer-jsonl", "paraformer-jsonl"):
        parser.add_argument(f"--{option}", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    rows = read_rows(args.baseline)
    if args.zipformer_jsonl and args.paraformer_jsonl:
        zip_views = {key(row): normalized(row["text"]) for row in read_rows(args.zipformer_jsonl)}
        para_views = {key(row): normalized(row["text"]) for row in read_rows(args.paraformer_jsonl)}
    else:
        if not all((args.tracks, args.timing, args.zipformer, args.paraformer)):
            parser.error("supply both cached opinions, or tracks, timing and both model directories")
        timing = json.loads(args.timing.read_text(encoding="utf-8"))["segments"]
        zip_views = candidate_text(args.zipformer, "zipformer", args.tracks, timing, None, args.threads)
        selected = {
            key(row) for row in rows
            if (view := zip_views.get(key(row))) and amend([row], {key(row): (view, view)})[1]
        }
        para_views = candidate_text(args.paraformer, "paraformer", args.tracks, timing, selected, args.threads)
    if rows and not any(key(row) in zip_views for row in rows):
        raise ValueError("Zipformer opinions and baseline do not share any speaker/turn coordinates")
    opinions = {k: (zip_views[k], para_views[k]) for k in zip_views.keys() & para_views.keys()}
    revised, changes = amend(rows, opinions)
    export_rows(revised, args.out)
    print(f"Updated {changes} turns")


if __name__ == "__main__":
    main()
