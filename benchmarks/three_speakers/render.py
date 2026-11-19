"""Render three aligned tracks from a mono recording using session-wide speaker labels."""

from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
import soundfile as sf
import torch

from diarize import masks
from partial_enrollment import activity_affinity, assignment_is_tied, short_solo_cue


RATE = 16000
TICK = 0.01
HOP = 160


def runs(x):
    borders = np.flatnonzero(np.diff(np.pad(x.astype(np.int8), (1, 1))))
    return list(zip(borders[::2], borders[1::2]))


def enrollment(y, activity, count, length=4.0, allow_partial=False):
    """Use only regions predicted to be clean in the input recording."""
    anchors = []
    missing = []
    for slot, row in enumerate(activity):
        solo = row & (count == 1)
        fragments = []
        for a, b in sorted(runs(solo), key=lambda x: x[1] - x[0], reverse=True):
            if b - a < 75:
                continue
            available = min(
                int(length * RATE) - sum(len(x) for x in fragments),
                (b - a) * HOP - 3200,
            )
            if available <= 0:
                break
            center = (a + b) * HOP // 2
            start = center - available // 2
            fragments.append(y[start : start + available])
            if sum(len(x) for x in fragments) >= round(length * RATE):
                break
        if not fragments:
            if not allow_partial:
                raise ValueError("No usable solo anchor for one of the three speakers.")
            missing.append(slot)
            anchors.append(short_solo_cue(y, runs(solo), RATE, length))
            continue
        cue = np.concatenate(fragments)
        anchors.append(np.pad(cue, (0, max(0, round(length * RATE) - len(cue)))))
    if allow_partial:
        return anchors, missing
    return np.stack(anchors)


def merge_overlap(count):
    merged = []
    for a, b in runs(count > 1):
        if merged and a - merged[-1][1] <= 12:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return merged


def scale_sources(out, y):
    gram = (out @ out.T).astype(np.float64)
    ridge = max(1e-8, float(np.trace(gram)) * 0.003 / len(out))
    gains = np.linalg.solve(gram + ridge * np.eye(len(out)), out @ y)
    return out * np.clip(gains, -2.5, 2.5).astype(np.float32)[:, None]


def output_tracks(y, activity, method, args):
    outputs = np.zeros((len(activity), len(y)), np.float32)
    for i, row in enumerate(activity):
        for lo, hi in runs(row):
            a, b = lo * HOP, min(len(y), hi * HOP)
            outputs[i, a:b] = y[a:b]
    prefix = {"method": method}
    if method == "routing":
        return outputs, prefix
    count = activity.sum(axis=0)
    candidates = merge_overlap(count)
    if not candidates:
        prefix.update(overlap_regions=0, overlap_windows=0, processed_core_s=0)
        return outputs, prefix
    missing = []
    if method == "dprnn":
        cues, missing = enrollment(y, activity, count, allow_partial=True)
    else:
        cues = enrollment(y, activity, count)
    if missing:
        prefix["speakers_without_long_solo"] = missing
    overlap_chunks = [
        (lo, min(lo + 300, b)) for a, b in candidates for lo in range(a, b, 300)
    ]
    initialized = time.perf_counter()
    three_voice_separator = None
    refiner = None
    if method == "target":
        from target_extract import TargetExtractor

        separator = TargetExtractor(args.model_dir, args.toolkit, args.wespeaker)
        cue_emb = separator.anchors(cues)
    else:
        sys.path.insert(0, str(args.project))
        from process_meeting import (
            Separation,
            VoiceEmbeddings,
            load_separator,
            resample,
            normalized_dot,
        )

        separator = Separation(load_separator(args.models, "tuned", "cpu"), "cpu")
        voice = VoiceEmbeddings(args.models / "voiceprint.onnx")
        anchors = [
            voice(resample(s, RATE, 8000)) if s is not None else None for s in cues
        ]
        unavailable = [i for i, anchor in enumerate(anchors) if anchor is None]
        if unavailable:
            prefix["speakers_without_usable_embedding"] = unavailable
            missing = sorted(set(missing) | set(unavailable))
        sepformer_weights = getattr(args, "sepformer_three", None)
        if sepformer_weights:
            from sepformer_three import SepFormer

            three_voice_separator = SepFormer(sepformer_weights)
            if getattr(args, "refine_three", False):
                from refine_three import SpectralRefiner

                refiner = SpectralRefiner()
    started = time.perf_counter()
    lengths = []
    tied_identity_windows = 0

    def prepare(lo, hi):
        # An observed global identity must be active within the core interval.
        speakers = np.flatnonzero(activity[:, lo:hi].any(axis=1))
        a, b = lo * HOP, min(len(y), hi * HOP)
        left = max(0, a - 4800)
        right = min(len(y), b + 4800)
        use_three_voice_model = (
            three_voice_separator is not None and (count[lo:hi] == 3).sum() >= 5
        )
        return lo, hi, a, b, left, y[left:right], speakers, use_three_voice_model

    def infer(task):
        fragment, speakers, use_three_voice_model = task[-3:]
        if method == "target":
            return scale_sources(
                separator.extract(fragment, cue_emb[speakers]), fragment
            )
        mixture = resample(fragment, RATE, 8000)
        if use_three_voice_model:
            tracks = three_voice_separator(mixture)
            if refiner is not None:
                tracks = refiner(mixture, tracks)
            return scale_sources(tracks, mixture)
        return separator.infer(mixture, len(speakers))[0]

    jobs = [prepare(lo, hi) for lo, hi in overlap_chunks]
    workers = getattr(args, "workers", 1) if method == "dprnn" else 1
    with (
        ThreadPoolExecutor(max_workers=workers) if workers > 1 else nullcontext()
    ) as pool:
        separated_jobs = pool.map(infer, jobs) if pool else map(infer, jobs)
        for index, (job, separated) in enumerate(
            zip(jobs, separated_jobs, strict=True)
        ):
            lo, hi, a, b, left, fragment, speakers, _ = job
            if method == "dprnn":
                if len(separated) != len(speakers):
                    continue
                embeds = [voice(wave) for wave in separated]
                # Keep voice matching on the calling thread, in chronological order.
                score = np.asarray(
                    [
                        [
                            normalized_dot(e, anchors[i])
                            if e is not None and anchors[i] is not None
                            else 0.0
                            for e in embeds
                        ]
                        for i in speakers
                    ]
                )
                if missing:
                    # A stream that stays quiet during this speaker's pauses is more
                    # consistent with the diarization labels. Known anchors still vote.
                    score += activity_affinity(
                        separated, activity[speakers], left // HOP
                    )
                i, j = linear_sum_assignment(-score)
                if missing and assignment_is_tied(score, i, j):
                    tied_identity_windows += 1
                ordered = np.zeros((len(speakers), len(fragment)), np.float32)
                for rank, k in zip(i, j):
                    s = resample(separated[k], 8000, RATE)
                    ordered[rank, : min(len(s), len(fragment))] = s[: len(fragment)]
                separated = ordered
            # Merged regions can include short solo gaps. Keep their original
            # room signal and use separated samples only during actual overlap.
            overlap = count[lo:hi] > 1
            for i, idx in enumerate(speakers):
                take = separated[i, a - left : b - left]
                write = np.repeat(activity[idx, lo:hi] & overlap, HOP)[: b - a]
                np.copyto(outputs[idx, a:b], take, where=write)
            lengths.append(hi - lo)
            if (index + 1) % 32 == 0 or index + 1 == len(overlap_chunks):
                print(
                    f"{method} overlap windows {index + 1}/{len(overlap_chunks)}, {time.perf_counter() - started:.1f} s",
                    flush=True,
                )
    prefix.update(
        overlap_regions=len(candidates),
        overlap_windows=len(overlap_chunks),
        processed_core_s=sum(lengths) * TICK,
        initialize_s=started - initialized,
        overlap_inference_s=time.perf_counter() - started,
        workers=workers,
        sepformer_three_windows=sum(bool(job[-1]) for job in jobs),
    )
    if missing:
        prefix["tied_identity_windows"] = tied_identity_windows
    if refiner is not None:
        prefix["refined_three_voice_windows"] = prefix["sepformer_three_windows"]
    return outputs, prefix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--diarization", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--method", choices=["routing", "dprnn", "target"], default="dprnn")
    p.add_argument("--project", type=Path)
    p.add_argument("--models", type=Path)
    p.add_argument("--model-dir", type=Path, default=Path("spk_emb_100"))
    p.add_argument("--toolkit", type=Path, default=Path("wesep-real-tse"))
    p.add_argument("--wespeaker", type=Path, default=Path("wespeaker"))
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--workers", type=int, default=1, help="Independent CPU windows for DPRNN"
    )
    p.add_argument(
        "--sepformer-three", type=Path, help="Optional three-voice checkpoint directory"
    )
    p.add_argument(
        "--refine-three",
        action="store_true",
        help="Apply the experimental spectral gain filter to three-voice SepFormer windows",
    )
    args = p.parse_args()
    if args.refine_three and not args.sepformer_three:
        p.error("--refine-three requires --sepformer-three")
    if args.threads < 1 or args.workers < 1:
        p.error("Thread and worker counts must be positive")
    if args.sepformer_three and args.method != "dprnn":
        p.error("--sepformer-three requires --method dprnn")
    torch.set_num_threads(args.threads)
    wave, sr = sf.read(args.input, dtype="float32")
    if sr != RATE or wave.ndim != 1:
        raise ValueError("Expected one mono 16 kHz recording")
    diar = json.loads(args.diarization.read_text(encoding="utf-8"))
    names, activity = masks(diar["segments"], len(wave) / RATE)
    if len(names) != 3:
        raise ValueError("The session must contain three diarization labels")
    started = time.perf_counter()
    audio, metadata = output_tracks(wave, activity, args.method, args)
    args.out.mkdir(parents=True, exist_ok=True)
    gain = min(1.0, 0.98 / max(1e-8, float(np.abs(audio).max())))
    for name, wave in zip(names, audio):
        sf.write(args.out / f"speaker_{name}.wav", gain * wave, RATE, subtype="PCM_16")
    metadata.update(
        audio_wall_s=time.perf_counter() - started,
        speaker_ids=names,
        sample_rate=RATE,
        sample_count=audio.shape[1],
        detected_overlap_seconds=float((activity.sum(0) > 1).sum()) * TICK,
    )
    (args.out / "render.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(metadata, flush=True)


if __name__ == "__main__":
    main()
