"""Evaluate the count margin on controlled and short natural Mandarin examples."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import time

import numpy as np
import torch

from benchmark_examples import (
    cperror, diarization_metrics, natural_activity, norm, text_by_track, waveform_scores,
)
from process_meeting import (
    DEFAULT_BLOCK, DEFAULT_HISTORY, DEFAULT_LOOKAHEAD, Mandarin, RATE,
    Separation, VoiceEmbeddings, load_separator, read_audio,
)


def score_controlled(path, y, tracks, texts):
    info = json.loads((path / "reference.json").read_text(encoding="utf-8"))
    references = [norm("".join(r["transcript"] for r in v)) for v in info["speakers"].values()]
    ref_audio = [read_audio(p)[0] for p in sorted(path.glob("reference_?.wav"))]
    return {"source_count":len(references),
            "full_overlap_seconds":info["exact_concurrency_seconds"][str(len(references))],
            "text":cperror(references,list(texts.values())),
            "waveform":waveform_scores(ref_audio, tracks, y)}


def run(args):
    torch.set_num_threads(args.threads)
    separator = load_separator(args.models, "tuned", "cpu")
    embeddings = VoiceEmbeddings(args.models / "voiceprint.onnx")
    asr = None if args.no_text else Mandarin(args.models, args.threads)
    states = [Separation(separator, "cpu", embeddings, margin) for margin in (0.0, args.margin)]
    results = []
    all_items = sorted(args.cases.glob(args.pattern + "/reference.json"))
    all_items = [(v.parent / "mixed.wav", v.parent) for v in all_items]
    all_items.extend((w, None) for w in args.natural)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for f, ref in all_items:
        y, _, _ = read_audio(f)
        block = round(args.block * RATE)
        samples = (block, round(DEFAULT_HISTORY * RATE), round(DEFAULT_LOOKAHEAD * RATE))
        trace = []
        for engine in states:
            t0 = time.perf_counter()
            tracks, commits = engine.separate(y, *samples, False)
            elapsed = time.perf_counter() - t0
            if asr:
                _, words = text_by_track(asr, tracks)
            else:
                words = {key:"" for key in tracks}
            metrics = {"margin":engine.overlap_margin, "track_count":len(tracks),
                       "audio_seconds":len(y)/RATE, "warm_audio_seconds":elapsed,
                       "model_counts":[x["decoded_source_count"] for x in commits],
                       "accepted_extra_sources":sum(x["count_review"] for x in commits)}
            if ref is None:
                session, start, spans, overlaps = natural_activity(f, args.rttm)
                metrics.update(session=session, start_s=start,
                               labeled_speakers=len({s for _,_,s in spans}),
                               rttm_overlap_seconds=overlaps,
                               **diarization_metrics(spans, tracks, len(y)))
            else:
                metrics.update(**score_controlled(ref, y, tracks, words))
                if asr is None:
                    metrics.pop("text")
            trace.append(metrics)
        result = {"clip":ref.name if ref else f.stem, "modes":trace}
        results.append(result)
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        print(result["clip"], [(m["track_count"], round(m.get("energy_mask_DER",m.get("text",{}).get("cpCER",0)),3))
                               for m in trace], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", required=True, type=Path)
    p.add_argument("--models", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--rttm", type=Path)
    p.add_argument("--natural", type=Path, nargs="*", default=[])
    p.add_argument("--pattern", default="evaluation*")
    p.add_argument("--block", type=float, default=DEFAULT_BLOCK)
    p.add_argument("--margin", type=float, default=1.4)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--no-text", action="store_true")
    run(p.parse_args())
