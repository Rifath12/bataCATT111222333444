"""Recognize Mandarin on aligned speaker tracks and score against the official TextGrid."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
import time
import unicodedata

import numpy as np
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment
import sherpa_onnx as sx
import soundfile as sf


def textgrid(path):
    text = path.read_text(encoding="utf-8-sig")
    result = []
    for item in re.split(r"(?m)^\s*item \[\d+\]:", text)[1:]:
        name = re.search(r'(?m)^\s*name = "([^"]+)"', item).group(1)
        for interval in re.split(r"(?m)^\s*intervals \[\d+\]:", item)[1:]:
            start = float(re.search(r"(?m)^\s*xmin = ([.\d]+)", interval).group(1))
            end = float(re.search(r"(?m)^\s*xmax = ([.\d]+)", interval).group(1))
            txt = (
                re.search(r'(?ms)^\s*text = "(.*)"\s*$', interval)
                .group(1)
                .replace('""', '"')
            )
            if normalized(txt):
                result.append(
                    {"speaker": name, "start": start, "end": end, "text": txt}
                )
    return sorted(result, key=lambda s: (s["start"], s["speaker"]))


def normalized(text):
    # Remove dataset event markup before punctuation/case normalization.
    text = re.sub(r"<[^>]*>|\[[^\]]*\]", "", text)
    return "".join(
        x.lower()
        for x in unicodedata.normalize("NFKC", text)
        if unicodedata.category(x)[0] in ("L", "N")
    )


def conversations(rows):
    names = sorted({str(x["speaker"]) for x in rows})
    return {
        n: normalized(
            "".join(
                x["text"]
                for x in sorted(rows, key=lambda t: t["start"])
                if str(x["speaker"]) == n
            )
        )
        for n in names
    }


def text_scores(ref, hyp):
    r, h = conversations(ref), conversations(hyp)
    rn, hn = list(r), list(h)
    k = max(len(r), len(h))
    a = list(r.values()) + [""] * (k - len(r))
    b = list(h.values()) + [""] * (k - len(h))
    matrix = np.asarray(
        [[Levenshtein.distance(x, y) for y in b] for x in a], dtype=np.int64
    )
    rr, hh = linear_sum_assignment(matrix)
    per = []
    total = Counter()
    for i, j in zip(rr, hh):
        ops = Counter(q.tag for q in Levenshtein.editops(a[i], b[j]))
        correct = len(a[i]) - ops["replace"] - ops["delete"]
        row = {
            "reference": rn[i] if i < len(rn) else None,
            "system": hn[j] if j < len(hn) else None,
            "reference_chars": len(a[i]),
            "recognized_chars": len(b[j]),
            "correct_reference_chars": correct,
            "substitutions": ops["replace"],
            "deletions": ops["delete"],
            "insertions": ops["insert"],
        }
        per.append(row)
        total.update(
            {key: value for key, value in row.items() if isinstance(value, int)}
        )
    denominator = total["reference_chars"]
    total["errors"] = total["substitutions"] + total["deletions"] + total["insertions"]
    return {
        "cpCER": total["errors"] / denominator,
        "correct_reference_characters": total["correct_reference_chars"] / denominator,
        "totals": dict(total),
        "speaker_scores": per,
    }


def inference_spans(intervals, duration):
    combined = []
    for t in sorted(intervals, key=lambda x: x["start"]):
        a = max(0.0, float(t["start"]))
        b = min(duration, float(t["end"]))
        if b - a < 0.055:
            continue
        if combined and a - combined[-1][1] < 0.30 and b - combined[-1][0] < 17.0:
            combined[-1][1] = max(combined[-1][1], b)
        else:
            combined.append([a, b])
    spans = []
    for a, b in combined:
        while b - a > 18.0:
            spans.append((a, a + 17.0))
            a += 17.0
        if b - a >= 0.055:
            spans.append((a, b))
    return spans


def transcribe(
    wav_paths, times, asr, tokenizer, threads=4, batch_size=1, character_fn=None
):
    decoder = sx.OfflineRecognizer.from_zipformer_ctc(
        model=str(asr), tokens=str(tokenizer), num_threads=threads
    )
    results = []
    started = time.perf_counter()
    for speaker, path in sorted(wav_paths.items()):
        y, sr = sf.read(path, dtype="float32")
        if sr != 16000 or y.ndim != 1:
            raise ValueError(f"Expected a mono 16 kHz input in {path}")
        spans = list(
            inference_spans(
                [s for s in times if str(s["speaker"]) == speaker], len(y) / sr
            )
        )
        for offset in range(0, len(spans), batch_size):
            section = spans[offset : offset + batch_size]
            streams = []
            for a, b in section:
                stream = decoder.create_stream()
                # Pad the ends without importing adjacent speakers' words.
                speech = y[round(a * sr) : round(b * sr)]
                stream.accept_waveform(sr, np.pad(speech, (800, 1280)))
                streams.append(stream)
            decoder.decode_streams(streams)
            for (a, b), stream in zip(section, streams):
                result = stream.result
                text = result.text.strip()
                if not text:
                    continue
                row = {"speaker": speaker, "start": a, "end": b, "text": text}
                if character_fn is not None:
                    chars = character_fn(result.tokens, result.timestamps, a - 0.05, b)
                    if "".join(x["text"] for x in chars) == "".join(text.split()):
                        for ch in chars:
                            ch["start_ms"] = max(round(a * 1000), ch["start_ms"])
                            ch["end_ms"] = max(ch["start_ms"], ch["end_ms"])
                        row["characters"] = chars
                results.append(row)
            if offset // batch_size % 30 == 0:
                print(
                    f"ASR {speaker} {min(offset + batch_size, len(spans))}/{len(spans)}",
                    flush=True,
                )
    return sorted(
        results, key=lambda t: (t["start"], t["speaker"])
    ), time.perf_counter() - started


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--timing", type=Path, required=True)
    p.add_argument(
        "--reference", type=Path, help="Optional TextGrid for character error scoring"
    )
    p.add_argument("--asr", type=Path, required=True)
    p.add_argument("--project", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--batch",
        type=int,
        default=1,
        help="The supplied int8 Zipformer expects one utterance per batch",
    )
    p.add_argument("--headsets", action="store_true")
    args = p.parse_args()
    if args.headsets and not args.reference:
        p.error("--headsets requires reference timing from --reference")
    refs = textgrid(args.reference) if args.reference else []
    timing = (
        refs
        if args.headsets
        else json.loads(args.timing.read_text(encoding="utf-8"))["segments"]
    )
    paths = (
        {x.name[:-4].split("_N_SPK")[-1]: x for x in args.audio.glob("*N_SPK*.wav")}
        if args.headsets
        else {x.stem.split("_")[-1]: x for x in args.audio.glob("speaker_*.wav")}
    )
    if args.headsets:
        paths = {"N_SPK" + name: val for name, val in paths.items()}
    if len(paths) != 3:
        p.error("The audio directory must contain three speaker files")
    export_characters = None
    if args.project:
        sys.path.insert(0, str(args.project))
        from process_meeting import characters

        export_characters = characters
    results, wall = transcribe(
        paths,
        timing,
        args.asr / "model.int8.onnx",
        args.asr / "tokens.txt",
        args.threads,
        args.batch,
        export_characters,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "transcript.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in results),
        encoding="utf-8",
    )
    (args.out / "transcript.txt").write_text(
        "".join(
            f"[{x['start']:.2f}–{x['end']:.2f}] {x['speaker']}  {x['text']}\n"
            for x in results
        ),
        encoding="utf-8",
    )
    evaluation = {
        "headset_oracle_timing": args.headsets,
        "asr_wall_s": wall,
        "speakers_decoded": len(paths),
    }
    if refs:
        evaluation.update(text_scores(refs, results))
    (args.out / "asr_scores.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(evaluation, flush=True)


if __name__ == "__main__":
    main()
