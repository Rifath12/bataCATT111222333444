"""Run three-speaker offline diarization and score speaker activity."""

from pathlib import Path
import argparse
import json
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
import sherpa_onnx as sx
import soundfile as sf


def reference(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = line.split()
        if len(cells) >= 8 and cells[0] == "SPEAKER":
            start = float(cells[3])
            end = start + float(cells[4])
            rows.append({"start": start, "end": end, "speaker": cells[7]})
    return rows


def masks(segments, duration, tick=0.01):
    keys = sorted({str(s["speaker"]) for s in segments})
    count = int(np.ceil(duration / tick))
    t = (np.arange(count, dtype=np.float64) + 0.5) * tick
    out = np.zeros((len(keys), count), bool)
    for seg in segments:
        start = int(np.searchsorted(t, seg["start"]))
        end = int(np.searchsorted(t, seg["end"]))
        out[keys.index(str(seg["speaker"])), start:end] = True
    return keys, out


def score(ref, hyp, duration):
    labels, a = masks(ref, duration)
    hypotheses, b = masks(hyp, duration)
    matrix = a.astype(np.float32) @ b.astype(np.float32).T
    i, j = linear_sum_assignment(-matrix)
    pair = {hypotheses[int(y)]: labels[int(x)] for x, y in zip(i, j)}
    correct = (a[i] & b[j]).sum(axis=0)
    refc = a.sum(axis=0)
    sysc = b.sum(axis=0)
    miss = np.maximum(refc - sysc, 0)
    false = np.maximum(sysc - refc, 0)
    confusion = np.minimum(refc, sysc) - correct
    stats = {}
    groups = [
        ("full", np.ones_like(refc, dtype=bool)),
        ("single", refc == 1),
        ("two_simultaneous", refc == 2),
        ("three_simultaneous", refc == 3),
        ("all_overlap", refc >= 2),
    ]
    for name, mask in groups:
        total = int(refc[mask].sum())
        stats[name] = {
            "clock_s": float(mask.sum()) * 0.01,
            "reference_speaker_s": total * 0.01,
            "speaker_recall": float(correct[mask].sum() / total) if total else None,
            "DER": float((miss + false + confusion)[mask].sum() / total)
            if total
            else None,
            "miss_s": float(miss[mask].sum()) * 0.01,
            "false_alarm_s": float(false[mask].sum()) * 0.01,
            "confusion_s": float(confusion[mask].sum()) * 0.01,
        }
    return {
        "ref_labels": labels,
        "system_labels": hypotheses,
        "mapping": pair,
        "scores": stats,
    }


def predict(y, seg, embed, threads=4, shift=0.2, known=3):
    segmentation = sx.OfflineSpeakerSegmentationModelConfig(
        pyannote=sx.OfflineSpeakerSegmentationPyannoteModelConfig(
            model=str(seg), window_shift_ratio=shift
        ),
        num_threads=threads,
        provider="cpu",
    )
    embedding = sx.SpeakerEmbeddingExtractorConfig(
        model=str(embed), num_threads=max(1, threads // 2), provider="cpu"
    )
    config = sx.OfflineSpeakerDiarizationConfig(
        segmentation=segmentation,
        embedding=embedding,
        clustering=sx.FastClusteringConfig(num_clusters=known, threshold=0.48),
        min_duration_on=0.16,
        min_duration_off=0.18,
    )
    if not config.validate():
        raise ValueError("Diarization model configuration is invalid")
    diarizer = sx.OfflineSpeakerDiarization(config)
    since = [0]

    def progress(current, total):
        if current >= since[0] + 150 or current == total:
            since[0] = current
            print(f"Diarization {current}/{total}", flush=True)
        return 0

    items = diarizer.process(y, progress).sort_by_start_time()
    return [
        {"start": float(t.start), "end": float(t.end), "speaker": str(t.speaker)}
        for t in items
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--seg", type=Path, required=True)
    p.add_argument("--embed", type=Path, required=True)
    p.add_argument("--rttm", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--shift", type=float, default=0.20)
    args = p.parse_args()
    wave, sr = sf.read(args.input, dtype="float32")
    if wave.ndim != 1 or sr != 16000:
        raise ValueError("Provide a mono, 16 kHz input WAV")
    started = time.perf_counter()
    segments = predict(wave, args.seg, args.embed, args.threads, args.shift)
    output = {
        "duration_s": len(wave) / sr,
        "inference_wall_s": time.perf_counter() - started,
        "window_shift_ratio": args.shift,
        "known_session_speaker_count": 3,
        "segments": segments,
    }
    if args.rttm:
        output["evaluation"] = score(reference(args.rttm), segments, len(wave) / sr)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print({k: v for k, v in output.items() if k != "segments"}, flush=True)


if __name__ == "__main__":
    main()
