"""Experimental past-only transfer updates for three synchronized 16 kHz headset tracks."""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from scipy.ndimage import gaussian_filter1d
import soundfile as sf

from close_waveform import MicrophoneSignals, FROZEN_PROFILE, calibrated_receivers, stored_transfer, suppress, validate_inputs

RATE = 16000
FFT = 1024
HOP = 512
BLOCK = 262144
CONTEXT = 2048
HISTORY_BLOCKS = 6
STARTUP_SECONDS = 120
LABELS = "ABC"
FREQUENCIES = np.fft.rfftfreq(FFT, 1 / RATE)
SPEECH_BAND = (FREQUENCIES >= 180) & (FREQUENCIES <= 3400)
WINDOW = np.hanning(FFT + 1)[:-1].astype(np.float32)


def dominant_speech(audio):
    """Retain interior speech spectra from runs dominated by one physical microphone."""
    if audio.shape[1] < FFT:
        return [[], [], []]
    frames = np.lib.stride_tricks.sliding_window_view(audio, FFT, axis=-1)[:, ::HOP]
    spectrum = np.fft.rfft(frames * WINDOW, axis=-1).astype(np.complex64)
    power = gaussian_filter1d(np.abs(spectrum)**2, 1.1, axis=-1, mode="nearest")
    broadband = power.sum(-1)
    speech = power[:, :, SPEECH_BAND].sum(-1)
    evidence = []
    for source in range(3):
        other_speech = np.max(np.delete(speech, source, axis=0), axis=0)
        other_all = np.max(np.delete(broadband, source, axis=0), axis=0)
        dominant = (
            (speech[source] > 10 * other_speech)
            & (broadband[source] > 6.31 * other_all)
            & (speech[source] > .38 * broadband[source])
            & (speech[source] > FFT * FFT * .1875 * .0045**2)
        )
        edges = np.flatnonzero(np.diff(np.r_[False, dominant, False])).reshape(-1, 2)
        runs = []
        for first, last in edges:
            if last - first >= 7:
                # Exclude the edges of a putative solo turn. Retain a spectrum every 64 ms.
                rows = np.arange(first + 2, last - 2, 2)
                runs.append(power[:, rows, :].astype(np.float32))
        evidence.append(runs)
    return evidence


def estimate_transfer(initial, previous, history):
    """Use completed blocks only. Fall back to the previous estimate for sparse speakers."""
    result = previous.copy()
    details = []
    for source, label in enumerate(LABELS):
        blocks = [item[source] for item in history if item[source]]
        runs = [run for block in blocks for run in block]
        rows = sum(run.shape[1] for run in runs)
        details.append(dict(source=label, blocks=len(blocks), runs=len(runs), rows=rows))
        if len(blocks) < 2 or len(runs) < 3 or rows < 32:
            continue
        values = np.concatenate(runs, axis=1)
        direct = values[source]
        supported = direct > np.maximum(np.quantile(direct, .5, axis=0), 1e-7)
        eligible = supported.sum(0) >= 12
        for receiver in range(3):
            if source == receiver:
                continue
            curve = np.zeros(direct.shape[1], np.float32)
            ratios = np.where(supported, values[receiver] / np.maximum(direct, 1e-10), np.nan)
            if eligible.any():
                curve[eligible] = np.nanquantile(ratios[:, eligible], .55, axis=0)
            curve[~eligible] = previous[~eligible, receiver, source]
            curve = gaussian_filter1d(curve, 1.8, mode="nearest")

            prior = np.maximum(initial[:, receiver, source], 1e-6)
            target = np.clip(curve, np.maximum(.25 * prior, 1e-6), np.minimum(.14, 12 * prior + .002))
            reliability = min(.85, rows / (rows + 45.))
            target = np.exp((1 - reliability) * np.log(prior) + reliability * np.log(target))
            old = np.maximum(previous[:, receiver, source], 1e-6)
            result[:, receiver, source] = np.exp(.58 * np.log(old) + .42 * np.log(target))
    return result, details


def render(paths, power_transfer, output):
    """Export three aligned tracks and an auditable history of past-only calibration updates."""
    samples = validate_inputs(paths)
    initial = stored_transfer(power_transfer)
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    previous = initial.copy()
    receivers = calibrated_receivers(initial)
    signals = MicrophoneSignals()
    history = deque(maxlen=HISTORY_BLOCKS)
    trace = []
    with TemporaryDirectory(prefix=".rolling-headsets-", dir=output.parent) as temporary:
        # Move the completed directory as one unit after all three writers have closed.
        staged = Path(temporary) / "complete"
        staged.mkdir()
        with ExitStack() as stack:
            readers = [stack.enter_context(sf.SoundFile(path)) for path in paths]
            writers = [
                stack.enter_context(sf.SoundFile(staged / f"speaker_{who}.wav", mode="w",
                                    samplerate=RATE, channels=1, subtype="FLOAT"))
                for who in LABELS
            ]
            for first in range(0, samples, BLOCK):
                last = min(first + BLOCK, samples)
                start = max(0, first - CONTEXT)
                stop = min(samples, last + CONTEXT)
                # Select the transfer before reading audio for the new block. The regular
                # offline STFT renderer separately retains a short amount of boundary context.
                if first >= STARTUP_SECONDS * RATE:
                    current, evidence = estimate_transfer(initial, previous, history)
                    # A receiver with zero measured incoming paths stays an exact passthrough.
                    current[:, ~receivers] = initial[:, ~receivers]
                else:
                    current, evidence = previous, None
                for reader in readers:
                    reader.seek(start)
                wave = np.stack([reader.read(stop - start, dtype="float32") for reader in readers])
                if not np.isfinite(wave).all():
                    raise ValueError("Nonfinite microphone samples")
                begin, end = first - start, last - start
                core = wave[:, begin:end]
                signals.observe(core)
                clean = suppress(wave, current, FROZEN_PROFILE)[:, begin:end].copy()
                if np.any(current != previous):
                    # Both matrices predate this block. Crossfade for 64 ms at each change.
                    width = min(1024, last - first)
                    old_wave = wave[:, :min(stop - start, begin + width + CONTEXT)]
                    old = suppress(old_wave, previous, FROZEN_PROFILE)[:, begin:begin + width]
                    amount = np.arange(width) / width
                    clean[:, :width] = old * (1 - amount) + clean[:, :width] * amount
                clean[~receivers] = core[~receivers]
                if clean.shape != core.shape or not np.isfinite(clean).all():
                    raise ValueError("Waveform correction produced invalid samples")
                for writer, track in zip(writers, clean, strict=True):
                    writer.write(track)
                path_db = {
                    f"{source}->{receiver}": float(10 * np.log10(max(1e-12, np.median(current[:, i, j]))))
                    for j, source in enumerate(LABELS)
                    for i, receiver in enumerate(LABELS) if i != j
                }
                trace.append({"sample": first, "history_end_exclusive": first, "evidence": evidence,
                              "median_path_db": path_db})
                history.append(dominant_speech(wave[:, begin:end]))
                previous = current
        signals.validate(samples)
        report = {"samples": samples, "seconds": samples / RATE, "processing_s": perf_counter() - started,
                  "updates": trace}
        (staged / "trace.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        staged.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("microphones", nargs=3, type=Path, metavar="MIC_WAV")
    parser.add_argument("--power-transfer", type=Path, required=True,
                        help="Initial physical power calibration from the same headset order and setup")
    parser.add_argument("--out", type=Path, required=True, help="New directory for A/B/C WAVs and trace.json")
    args = parser.parse_args()
    report = render(args.microphones, args.power_transfer, args.out)
    print(f"Wrote three {report['samples']}-sample tracks in {report['processing_s']:.2f} seconds")


if __name__ == "__main__":
    main()
