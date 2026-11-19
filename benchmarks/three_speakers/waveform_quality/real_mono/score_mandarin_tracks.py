"""Decode all exported speaker WAVs or rescore an existing full-excerpt decode."""
import argparse
from collections import Counter
from itertools import permutations
import json
import math
from pathlib import Path
import re
from time import perf_counter

from opencc import OpenCC
from rapidfuzz.distance import Levenshtein

MODEL = "Systran/faster-whisper-small"
REVISION = "536b0662742c02347bc0e980a01041f333bce120"
CONVERTER = OpenCC("t2s")
EXTRACTIONS = ("A", "B", "C")


def normalize(text):
    """Simplify Chinese and remove punctuation, whitespace and non-scored symbols."""
    return re.sub(r"[^\u3400-\u9fffA-Za-z0-9]", "", CONVERTER.convert(text).lower())


def edit_breakdown(reference, hypothesis):
    counts = Counter(op.tag for op in Levenshtein.editops(reference, hypothesis))
    return {
        "substitutions": counts["replace"],
        "deletions": counts["delete"],
        "insertions": counts["insert"],
    }


def score(references, hypotheses):
    """Keep a single global track permutation per excerpt and charge extra tracks."""
    if set(references) != set(EXTRACTIONS):
        raise ValueError("Expected reference_text.json to contain speaker labels A, B and C")
    reference = {key: normalize(value) for key, value in references.items()}
    hypothesis = {key: normalize(value) for key, value in hypotheses.items()}
    denominator = sum(map(len, reference.values()))
    if denominator == 0:
        raise ValueError("CER requires at least one scored reference character")
    labels = list(hypothesis)
    labels.extend([None] * max(0, len(reference) - len(labels)))
    distances = {
        (key, label): Levenshtein.distance(ref, hypothesis.get(label, ""))
        for key, ref in reference.items() for label in set(labels) | set(EXTRACTIONS)
    }

    def other_insertions(order):
        return sum(len(hyp) for key, hyp in hypothesis.items() if key not in order)

    def cost(order):
        return sum(distances[k, t] for k, t in zip(EXTRACTIONS, order)) + other_insertions(order)

    def breakdown(order):
        extras = other_insertions(order)
        by_speaker = {}
        totals = Counter(insertions=extras, substitutions=0, deletions=0)
        for speaker, track in zip(EXTRACTIONS, order):
            ref = reference[speaker]
            hyp = hypothesis.get(track, "")
            edits = edit_breakdown(ref, hyp)
            totals.update(edits)
            by_speaker[speaker] = {
                "track": track if track in hypothesis else None,
                "reference_characters": len(ref), "hypothesis_characters": len(hyp),
                "edits": sum(edits.values()), **edits,
            }
        total = sum(totals.values())
        return {
            "reference_characters": denominator, "edits": total,
            "character_error_rate": total / denominator, **totals,
            "identified_tracks": [t if t in hypothesis else None for t in order],
            "unassigned_track_insertions": extras, "by_speaker": by_speaker,
        }

    best = min(permutations(labels, len(reference)), key=cost)
    return {
        "fixed": breakdown(EXTRACTIONS),
        "best_one_permutation_per_excerpt": breakdown(best),
    }


def decode(args):
    import numpy as np
    import soundfile as sf
    from faster_whisper import WhisperModel
    from scipy.signal import resample_poly

    reference_doc = json.loads(args.reference.read_text())
    manifest = json.loads((args.tracks / "speakers.json").read_text())
    reference = reference_doc["fixed_text"]
    source = reference_doc.get("room") or args.reference.parent.name
    model = WhisperModel(str(args.model), device="cpu", compute_type="int8",
                         cpu_threads=2, local_files_only=True)
    segments_by_track = {}
    start = perf_counter()
    for key, spec in manifest["speakers"].items():
        track = key.removeprefix("spk_")
        if track in segments_by_track:
            raise ValueError(f"Duplicate output track {track!r}")
        samples, rate = sf.read(args.tracks / spec["file"], dtype="float32")
        if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
            raise ValueError(f"Invalid mono speaker WAV for {track}")
        if rate != 16000:
            div = math.gcd(rate, 16000)
            samples = resample_poly(samples, 16000 // div, rate // div)
        segments, _ = model.transcribe(samples, language="zh", beam_size=3,
                                       condition_on_previous_text=False,
                                       temperature=0.0, vad_filter=False)
        segments_by_track[track] = [
            {"start": s.start, "end": s.end, "text": s.text} for s in segments
        ]
    return {
        "source": source, "route": args.tracks.name, "fixed_text": reference,
        "asr_model": f"{MODEL}@{REVISION}",
        "compute": "int8 CPU, 2 threads, language zh, beam 3, no previous text, no VAD, no prompts",
        "decoded": segments_by_track, "decode_wall_seconds": perf_counter() - start,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    measure = commands.add_parser("decode", help="decode complete WAVs listed in speakers.json")
    measure.add_argument("--reference", type=Path, required=True)
    measure.add_argument("--tracks", type=Path, required=True)
    measure.add_argument("--model", type=Path, required=True, help="local pinned CTranslate2 model directory")
    rescore = commands.add_parser("rescore", help="compute CER from previously saved per-track transcripts")
    rescore.add_argument("input", type=Path)
    for command in (measure, rescore):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = decode(args) if args.command == "decode" else json.loads(args.input.read_text())
    hypotheses = {
        track: "".join(segment["text"] for segment in segments)
        for track, segments in result["decoded"].items()
    }
    result["scores"] = score(result["fixed_text"], hypotheses)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
