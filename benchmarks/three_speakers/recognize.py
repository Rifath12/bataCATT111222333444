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
    ).reshape((k, k))
    rr, hh = linear_sum_assignment(matrix)
    per = []
    total = Counter(dict.fromkeys((
        "reference_chars", "recognized_chars", "correct_reference_chars",
        "substitutions", "deletions", "insertions",
    ), 0))
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
        "cpCER": total["errors"] / denominator if denominator else None,
        "correct_reference_characters": (
            total["correct_reference_chars"] / denominator if denominator else None
        ),
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
    wav_paths,
    times,
    asr,
    tokenizer,
    threads=4,
    batch_size=1,
    character_fn=None,
    target_rms=None,
    backend="zipformer",
    selected_keys=None,
):
    options = dict(model=str(asr), tokens=str(tokenizer), num_threads=threads)
    if backend == "sensevoice":
        decoder = sx.OfflineRecognizer.from_sense_voice(
            language="zh", use_itn=False, **options
        )
    elif backend == "zipformer":
        decoder = sx.OfflineRecognizer.from_zipformer_ctc(**options)
    else:
        raise ValueError(f"Unknown ASR backend {backend}")
    results = []
    started = time.perf_counter()
    for speaker, path in sorted(wav_paths.items()):
        if selected_keys is not None and not any(
            key[0] == speaker for key in selected_keys
        ):
            continue
        with sf.SoundFile(path) as source:
            sr = source.samplerate
            if sr != 16000 or source.channels != 1:
                raise ValueError(f"Expected a mono 16 kHz input in {path}")
            spans = list(
                inference_spans(
                    [s for s in times if str(s["speaker"]) == speaker], len(source) / sr
                )
            )
            if selected_keys is not None:
                spans = [
                    (a, b)
                    for a, b in spans
                    if (speaker, round(a, 3), round(b, 3)) in selected_keys
                ]
            for offset in range(0, len(spans), batch_size):
                section = spans[offset : offset + batch_size]
                streams = []
                for a, b in section:
                    stream = decoder.create_stream()
                    # Pad the ends without importing adjacent speakers' words.
                    first, last = round(a * sr), round(b * sr)
                    source.seek(first)
                    speech = source.read(last - first, dtype="float32")
                    if target_rms:
                        rms = max(float(np.sqrt(np.mean(np.square(speech)))), 1e-7)
                        peak = max(float(np.abs(speech).max()), 1e-7)
                        speech = speech * min(target_rms / rms, 0.98 / peak, 20.0)
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
                        chars = character_fn(
                            result.tokens, result.timestamps, a - 0.05, b
                        )
                        known_text = re.sub(r"<(?:unk|blk|sos/eos)>", "", text)
                        if "".join(x["text"] for x in chars) == "".join(
                            known_text.split()
                        ):
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
    p.add_argument(
        "--backend", choices=("zipformer", "sensevoice"), default="zipformer"
    )
    p.add_argument(
        "--suppress-duplicates",
        action="store_true",
        help="Suppress quieter cross-speaker copies of overlapping text",
    )
    p.add_argument(
        "--protect-long-turns",
        action="store_true",
        help="Retain the stem on long turns if the alternative adds insufficient text",
    )
    p.add_argument(
        "--restore-supported-text",
        action="store_true",
        help="Retain longer stem text that contains the continuous decoded words",
    )
    p.add_argument(
        "--refine-context",
        action="store_true",
        help="Use cached cropped and continuous readings to recover missing text",
    )
    p.add_argument(
        "--context-mix",
        type=Path,
        help="Add a continuous mono SenseVoice pass to recover short fragments",
    )
    p.add_argument(
        "--consensus-zipformer",
        type=Path,
        help="Optional Chinese Zipformer directory for targeted character consensus",
    )
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
    p.add_argument(
        "--target-rms",
        type=float,
        help="Optional per-utterance gain with peak limiting",
    )
    args = p.parse_args()
    if args.consensus_zipformer:
        if args.backend != "sensevoice" or not args.context_mix or args.headsets:
            p.error(
                "--consensus-zipformer requires SenseVoice tracks and --context-mix"
            )
        model = args.consensus_zipformer
        if not all(
            (model / name).is_file() for name in ("model.int8.onnx", "tokens.txt")
        ):
            p.error("--consensus-zipformer needs model.int8.onnx and tokens.txt")
        args.refine_context = True
    if args.refine_context:
        if not args.context_mix:
            p.error("--refine-context requires --context-mix")
        # Apply the prior contextual configuration before evaluating alternate readings.
        args.restore_supported_text = True
        args.protect_long_turns = True
        args.suppress_duplicates = True
    if args.protect_long_turns and not (
        args.context_mix and args.restore_supported_text
    ):
        p.error(
            "--protect-long-turns requires --context-mix and --restore-supported-text"
        )
    if args.restore_supported_text and not args.context_mix:
        p.error("--restore-supported-text requires --context-mix")
    if args.suppress_duplicates and args.headsets:
        p.error("--suppress-duplicates requires the separated speaker tracks")
    if args.context_mix and (args.backend != "sensevoice" or args.headsets):
        p.error("--context-mix requires SenseVoice on the separated speaker tracks")
    if args.target_rms is not None and not 0 < args.target_rms < 1:
        p.error("--target-rms must be between 0 and 1")
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
    if args.project and args.backend == "zipformer":
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
        args.target_rms,
        backend=args.backend,
    )
    if args.context_mix:
        from contextual_asr import (
            candidates,
            decode_mixture,
            select,
            restore_supported_text,
            protect_long_turns,
        )

        audio_info = sf.info(args.context_mix)
        if audio_info.frames != sf.info(next(iter(paths.values()))).frames:
            p.error(
                "The context mix and separated tracks must have identical sample counts"
            )
        parameters = json.loads(
            Path(__file__).with_name("context_settings.json").read_text()
        )
        context_start = time.perf_counter()
        words = decode_mixture(args.context_mix, args.asr, args.threads)
        pairs = candidates(timing, args.audio, results, words)
        results = select(pairs, parameters)
        if args.restore_supported_text:
            settings = json.loads(
                Path(__file__).with_name("restoration_settings.json").read_text()
            )
            results = restore_supported_text(pairs, results, settings)
        if args.protect_long_turns:
            settings = json.loads(
                Path(__file__).with_name("long_turn_settings.json").read_text()
            )
            results = protect_long_turns(pairs, results, settings)
        wall += time.perf_counter() - context_start
    if args.suppress_duplicates:
        from duplicate_leakage import apply, encounters

        duplicate_start = time.perf_counter()
        settings = json.loads(
            Path(__file__).with_name("duplicate_settings.json").read_text()
        )
        results = apply(results, encounters(results, args.audio), settings)
        wall += time.perf_counter() - duplicate_start
    context_changes = None
    if args.refine_context:
        from cached_gate import apply_cached_gate

        gate_start = time.perf_counter()
        gate = json.loads(
            Path(__file__).with_name("cached_gate_config.json").read_text()
        )
        results, context_changes = apply_cached_gate(
            pairs, results, gate, args.audio, settings
        )
        wall += time.perf_counter() - gate_start
    consensus = None
    if args.consensus_zipformer:
        from character_consensus import apply as apply_consensus, decoder_keys, prepare

        started = time.perf_counter()
        entries = prepare(pairs, results)
        eligible_keys = decoder_keys(entries)
        extra_rows = []
        if eligible_keys:
            extra_rows, _ = transcribe(
                paths,
                timing,
                args.consensus_zipformer / "model.int8.onnx",
                args.consensus_zipformer / "tokens.txt",
                args.threads,
                target_rms=args.target_rms,
                backend="zipformer",
                selected_keys=eligible_keys,
            )
        results, changes = apply_consensus(entries, extra_rows)
        elapsed = time.perf_counter() - started
        wall += elapsed
        consensus = {
            "decoded_turns": len(eligible_keys),
            "all_turns": len(entries),
            "additional_seconds": elapsed,
            "changes": changes,
        }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "transcript.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in results),
        encoding="utf-8",
    )

    def transcript_lines(rows):
        return "".join(
            f"[{x['start']:.2f}–{x['end']:.2f}] {x['speaker']}  {x['text']}\n"
            for x in rows
        )

    (args.out / "transcript.txt").write_text(
        transcript_lines(results), encoding="utf-8"
    )
    for speaker in paths:
        rows = (x for x in results if x["speaker"] == speaker)
        (args.out / f"speaker_{speaker}.txt").write_text(
            transcript_lines(rows), encoding="utf-8"
        )
    evaluation = {
        "headset_oracle_timing": args.headsets,
        "asr_wall_s": wall,
        "backend": args.backend,
        **({"character_consensus": consensus} if consensus is not None else {}),
        "speakers_decoded": len(paths),
    }
    if args.context_mix:
        evaluation["continuous_context"] = True
    if args.protect_long_turns:
        evaluation["protect_long_turns"] = True
    if args.restore_supported_text:
        evaluation["restore_supported_text"] = True
    if args.suppress_duplicates:
        evaluation["suppress_duplicates"] = True
    if context_changes is not None:
        evaluation["context_refinements"] = context_changes
    if args.target_rms:
        evaluation["target_rms"] = args.target_rms
    if refs:
        evaluation.update(text_scores(refs, results))
    (args.out / "asr_scores.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(evaluation, flush=True)


if __name__ == "__main__":
    main()
