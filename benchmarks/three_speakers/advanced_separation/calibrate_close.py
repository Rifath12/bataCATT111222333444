"""Estimate close-microphone power transfer from separate A/B/C solo WAVs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.signal import get_window
import soundfile as sf

from close_waveform import ACTIVITY_HOP, FFT, FROZEN_PROFILE, NAMES, RATE, fit_spectral_paths, publish_new

MIN_ACTIVE_FRAMES = 60
MAX_FIT_FRAMES = 2800


def measure_frames(recording):
    """Measure all complete 64 ms windows, at a 32 ms hop, in bounded blocks."""
    if recording.samplerate != RATE or recording.channels != len(NAMES):
        raise ValueError("Each solo take must contain three 16 kHz microphone channels in A/B/C order")
    count = 1 + (recording.frames - FFT) // ACTIVITY_HOP
    if count < MIN_ACTIVE_FRAMES:
        raise ValueError("Each solo take needs at least 60 active windows (roughly 2 seconds of speech)")
    powers = np.empty((count, 3), dtype=np.float32)
    for first in range(0, count, 256):
        n = min(256, count - first)
        recording.seek(first * ACTIVITY_HOP)
        samples = recording.read((n - 1) * ACTIVITY_HOP + FFT, dtype="float32")
        if not np.isfinite(samples).all():
            raise ValueError("The solo take contains nonfinite samples")
        windows = np.lib.stride_tricks.sliding_window_view(samples, FFT, axis=0)[::ACTIVITY_HOP]
        powers[first:first+n] = np.var(windows, axis=2, dtype=np.float64)
    # The incomplete last window is excluded from fitting but still validate its samples.
    if not np.isfinite(recording.read(dtype="float32")).all() or not np.isfinite(powers).all():
        raise ValueError("The solo take contains nonfinite or out-of-range samples")
    return powers


def fit_take(recording, source, matrices):
    powers = measure_frames(recording)
    own = powers[:, source]
    threshold = max(1e-9, .04 * float(np.quantile(own, .85)))
    active = np.flatnonzero(own > threshold)
    label = NAMES[source]
    if len(active) < MIN_ACTIVE_FRAMES:
        raise ValueError(f"Solo {label} has too few active speech windows on its own microphone")
    average = np.mean(powers[active], axis=0, dtype=np.float64)
    competitor = np.argmax(np.where(np.arange(3) == source, -1, average))
    if average[competitor] >= average[source]:
        raise ValueError(f"Solo {label} is at least as strong on microphone {NAMES[competitor]}. Check channel order and repeat the solo take")
    indices = active[np.linspace(0, len(active)-1, min(len(active), MAX_FIT_FRAMES), dtype=int)]
    window = get_window("hann", FFT).astype(np.float32)
    frames = np.empty((3, len(indices), FFT), dtype=np.float32)
    for i, index in enumerate(indices):
        recording.seek(int(index) * ACTIVITY_HOP)
        frames[:, i] = recording.read(FFT, dtype="float32").T * window
    spectra = np.fft.rfft(frames, axis=2).astype(np.complex64).transpose(0,2,1)
    if not np.isfinite(spectra).all() or np.abs(spectra).max() >= np.sqrt(np.finfo(np.float32).max):
        raise ValueError(f"Solo {label} has out-of-range waveform levels")
    ratios = fit_spectral_paths(spectra, source, matrices)
    bleed = average / average[source]
    return {
        "source": label, "samples": recording.frames, "active_windows": len(active),
        "selected_windows": len(indices), "fitted_median_path_db": ratios,
        "measured_power_db": {mic: float(10*np.log10(max(x, 1e-15))) for mic,x in zip(NAMES,bleed,strict=True)},
        "above_model_bleed_limit": [s for i,s in enumerate(NAMES) if i != source and bleed[i] > .14],
    }


def calibrate(paths):
    if len(paths) != len(NAMES) or len({(s.st_dev, s.st_ino) for s in (Path(p).stat() for p in paths)}) != 3:
        raise ValueError("Use three distinct solo files, one per speaker in A/B/C order")
    percentile = FROZEN_PROFILE.spectral_transfer_percentile
    matrix = np.tile(np.eye(3, dtype=np.float32), (FFT // 2 + 1, 1, 1))
    details = []
    for source, path in enumerate(paths):
        with sf.SoundFile(path) as recording:
            details.append(fit_take(recording, source, {percentile: matrix}))
    return matrix, {
        "method": "three_labeled_solo_takes", "sample_rate": RATE, "microphone_order": list(NAMES),
        "matrix_axes": ["frequency", "microphone", "speaker"], "percentile": percentile,
        "fft": FFT, "hop": ACTIVITY_HOP, "takes": details,
    }


def save(matrix, report, output):
    if output.suffix.lower() != ".npy":
        raise ValueError("Select a .npy file for the calibration")
    sidecar = output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"Calibration or report already exists for {output}")
    with TemporaryDirectory(prefix=".close-calibration-", dir=output.parent) as staging:
        staged_matrix = Path(staging) / "transfer.npy"
        staged_report = Path(staging) / "report.json"
        with staged_matrix.open("wb") as stream:
            np.save(stream, matrix, allow_pickle=False)
        with staged_matrix.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        metadata = dict(report, transfer_sha256=digest)
        staged_report.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        # Hard links publish complete files, fail if a concurrent writer won,
        # and keep the matrix unavailable until its diagnostics exist.
        publish_new(staged_report, sidecar)
        identity = sidecar.stat()
        try:
            publish_new(staged_matrix, output)
        except BaseException:
            if sidecar.exists() and os.path.samestat(sidecar.stat(), identity):
                sidecar.unlink()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solo", nargs=3, type=Path, metavar="SOLO_WAV",
                        help="Speaker A, B and C solo takes, each containing all three microphone channels")
    parser.add_argument("--out", required=True, type=Path, help="New .npy file with accompanying .json diagnostics")
    args = parser.parse_args()
    matrix, report = calibrate(args.solo)
    save(matrix, report, args.out)
    print(f"Saved microphone calibration to {args.out}")
    for take in report["takes"]:
        if take["above_model_bleed_limit"]:
            print(f"Solo {take['source']} exceeds the model's 0.14 leakage power limit on mic(s) "
                  + ", ".join(take["above_model_bleed_limit"]))


if __name__ == "__main__":
    main()
