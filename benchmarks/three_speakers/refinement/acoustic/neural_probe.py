"""Ask an independent Sortformer model to verify high-uncertainty intervals in each mono file."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d
import soundfile as sf
from threadpoolctl import threadpool_limits

from long_views import meeting
from multiview_score import counts, measures
from sortformer_benchmark import predict
from verify_long import DEV, TEST, make_frames, optimize


def boundaries(mask):
    edges = np.flatnonzero(np.diff(np.pad(mask.astype(np.int8), (1, 1))))
    return list(zip(edges[::2], edges[1::2], strict=True))


def pack(wave, first, second, candidate):
    # Collect automatically predicted clean samples to expose all three voices to the
    # independent model, then append suspected overlap regions including 0.96s of context.
    consensus = (first + second) / 2
    blocks = []
    for slot in range(3):
        good = (consensus[:, slot] > 0.85) & (
            consensus[:, np.arange(3) != slot].max(1) < 0.08
        )
        turns = boundaries(good)
        if turns:
            a, b = max(turns, key=lambda q: q[1] - q[0])
            length = min(240, b - a)
            a = (a + b - length) // 2
            blocks.append(wave[a * 160 : (a + length) * 160])
            blocks.append(np.zeros(2560, np.float32))
    prefix = np.concatenate(blocks) if blocks else np.zeros(0, np.float32)
    pieces = [np.pad(prefix, (0, (-len(prefix)) % 1280))]
    cursor = len(pieces[0]) // 160
    spans = []
    for a, b in boundaries(maximum_filter1d(candidate, 193)):
        a = (a // 8) * 8
        b = min(len(first), 8 * ((b + 7) // 8))
        spans.append((a, b, cursor))
        piece = wave[a * 160 : b * 160]
        piece = np.pad(piece, (0, (-len(piece)) % 1280 + 2560))
        pieces.append(piece)
        cursor += len(piece) // 160
    return np.concatenate(pieces), spans


def evidence(args, session, clip, first, second, audio):
    pmin = np.maximum(first.min(1), second.min(1))
    mask = (clip.previous.sum(1) >= 2) & (
        (maximum_filter1d(pmin, 31) >= 0.07) | clip.previous.all(1)
    )
    wave, sr = sf.read(audio, dtype="float32")
    if sr != 16000:
        raise ValueError("Expected mono 16 kHz input")
    compact, spans = pack(wave, first, second, mask)
    cache = args.cache / (session + "_sortformer.npy")
    if not cache.is_file():
        p = predict(
            SimpleNamespace(
                adapter=args.work / "sortformer-adapter",
                model=args.work / "sortformer.onnx",
                threads=2,
            ),
            compact,
        )
        np.save(cache, p)
    scores = np.repeat(np.sort(np.load(cache), axis=1)[:, ::-1], 8, axis=0)
    full = np.zeros((len(first), scores.shape[1]), np.float32)
    for a, b, offset in spans:
        full[a:b] = scores[offset : offset + b - a]
    feats = []
    for p in (full[:, :3], np.sort(0.5 * (first + second), axis=1)[:, ::-1]):
        for width in (1, 21, 65):
            feats.append(uniform_filter1d(p, width, axis=0)[mask])
    feats.append(clip.previous[mask].sum(1)[:, None])
    print(session, len(compact) / 16000, np.count_nonzero(mask) / 100, flush=True)
    return mask, np.column_stack(feats).astype(np.float32), len(compact) / 16000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.cache.mkdir(exist_ok=True, parents=True)
    profile = json.loads(
        (args.work / "independent_3speaker/profile_balanced.json").read_text()
    )

    def data(session):
        clip, first, second = meeting(args.work, args.views, session, profile)
        pcm = args.work / (
            "session/mono.wav"
            if session == DEV
            else f"independent_3speaker/{session}/mono.wav"
        )
        mask, x, seconds = evidence(args, session, clip, first, second, pcm)
        return (clip, first, second, mask, x), seconds

    with threadpool_limits(2):
        start = time.perf_counter()
        dev, seconds = data(DEV)
        model, threshold, oof, baseline = optimize(*dev)
        report = {
            "dev_session": DEV,
            "oof_score": oof,
            "dev_baseline": baseline,
            "threshold": threshold,
            "holdout": {},
            "dev_probe_duration": seconds,
        }
        print("OOF", threshold, oof, flush=True)
        totals = {}
        for session in TEST:
            began = time.perf_counter()
            (clip, first, second, mask, x), seconds = data(session)
            estimate = (
                make_frames(
                    clip, first, second, mask, model.predict_proba(x)[:, 1], threshold
                )
                if model
                else clip.previous
            )
            raw = {
                "native": counts(clip, clip.native),
                "previous": counts(clip, clip.previous),
                "probe": counts(clip, estimate),
            }
            report["holdout"][session] = {a: measures(v) for a, v in raw.items()}
            report["holdout"][session].update(
                probe_seconds=seconds, wall_s=time.perf_counter() - began
            )
            for a, v in raw.items():
                totals[a] = totals.get(a, 0) + v
        report["pooled"] = {a: measures(v) for a, v in totals.items()}
        report["wall_s"] = time.perf_counter() - start
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(report["pooled"], flush=True)


if __name__ == "__main__":
    main()
