"""Reduce crosstalk in three synchronized close-microphone recordings."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
import soundfile as sf


RATE = 16000
FFT = 1024
HOP = FFT // 4
ACTIVITY_HOP = 512
NAMES = "ABC"


def spectral(wave):
    return signal.stft(
        wave, fs=RATE, nperseg=FFT, noverlap=FFT - HOP, boundary="zeros", padded=True
    )[2]


def waveform(spectrogram, length):
    restored = signal.istft(spectrogram, fs=RATE, nperseg=FFT, noverlap=FFT - HOP)[1]
    return restored[:, :length]


def frame_spectra(paths, time_steps):
    """Read isolated fitting windows in batches without storing the whole meeting."""
    window = signal.get_window("hann", FFT).astype(np.float32)
    frames = np.empty((3, len(time_steps), FFT), dtype=np.float32)
    with ExitStack() as stack:
        recordings = [stack.enter_context(sf.SoundFile(path)) for path in paths]
        for index, step in enumerate(time_steps):
            first = step * ACTIVITY_HOP + (ACTIVITY_HOP - FFT) // 2
            for channel, recording in enumerate(recordings):
                recording.seek(first)
                frames[channel, index] = recording.read(FFT, dtype="float32")
    return np.transpose(np.fft.rfft(frames * window).astype(np.complex64), (0, 2, 1))

@dataclass(frozen=True)
class MaskProfile:
    spectral_transfer_percentile: float
    exponent: float
    lower_gain: float
    time_smoothing: int
    cross_share: float = 1.0


def fit_power_transfer(paths, power, posterior, *, fit_fraction=0.45, percentiles=(0.40, 0.70)):
    """Predict cross-channel energy from each dominant source's own microphone."""
    if power.shape != posterior.shape or power.ndim != 2 or len(power) != 3:
        raise ValueError("Three aligned per-microphone power/posterior tracks are required")
    stop = int(power.shape[1] * fit_fraction)
    if stop < 3:
        matrices = {p: np.tile(np.eye(3, dtype=np.float32), (FFT // 2 + 1, 1, 1)) for p in percentiles}
        return matrices, [{"source": speaker, "selected_dominant": 0} for speaker in NAMES]
    scale = [
        max(1e-9, np.quantile(energy[prob >= 0.7], .85))
        if np.any(prob >= .7) else max(1e-9, np.quantile(energy, .90))
        for energy, prob in zip(power[:, :stop], posterior[:, :stop], strict=True)
    ]
    normalized = uniform_filter1d(power[:, :stop], 3, axis=1) / np.array(scale)[:, None]
    matrices = {p: np.tile(np.eye(3, dtype=np.float32), (FFT // 2 + 1, 1, 1)) for p in percentiles}
    evidence = []
    for source in range(3):
        primary = normalized[source]
        secondary = np.max(np.delete(normalized, source, axis=0), axis=0)
        dominant = (posterior[source, :stop] >= .85) & (primary > 0.08) & (secondary < .55 * primary)
        frames = np.flatnonzero(dominant)
        frames = frames[(frames > 0) & (frames < stop - 1)]
        if len(frames) < 60:
            evidence.append({"source": NAMES[source], "selected_dominant": int(len(frames))})
            continue
        fit_choices = np.linspace(0, len(frames) - 1, min(2800, len(frames)), dtype=int)
        spectra = frame_spectra(paths, frames[fit_choices])
        signal_power = np.abs(spectra) ** 2
        # Broad spectral bins by absolute frequency. Higher bins generally have fewer useful frames.
        signal_power = gaussian_filter1d(signal_power, 1.1, axis=1, mode="nearest")
        direct = signal_power[source]
        voice = direct > np.maximum(np.quantile(direct, .50, axis=1), 1e-8)[:, None]
        stats = {"source": NAMES[source], "selected_dominant": int(len(fit_choices)), "paths": {}}
        for output in range(3):
            if output == source:
                continue
            ratio = np.where(voice, signal_power[output] / np.maximum(direct, 1e-9), np.nan)
            valid = voice.any(axis=1)
            values = np.zeros((len(percentiles), FFT // 2 + 1))
            values[:, valid] = np.nanquantile(ratio[valid], percentiles, axis=1)
            for p, curve in zip(percentiles, values, strict=True):
                curve = np.nan_to_num(curve, nan=0.0, posinf=0.0, neginf=0.0)
                matrices[p][:, output, source] = np.clip(
                    gaussian_filter1d(curve, 1.8, mode="nearest"), 0.0, 0.14
                )
            stats["paths"][NAMES[output]] = {
                str(p): round(float(10 * np.log10(max(1e-15, np.median(matrices[p][:, output, source])))), 2)
                for p in percentiles
            }
        evidence.append(stats)
    return matrices, evidence


def suppress(audio, power_transfer, profile):
    """Render a fixed A/B/C order and keep physical input levels when there is no competing energy."""
    if audio.ndim != 2 or audio.shape[0] != 3:
        raise ValueError("Expected three aligned microphone tracks shaped [3, samples]")
    if audio.shape[1] == 0 or np.array_equal(power_transfer, np.broadcast_to(np.eye(3), power_transfer.shape)):
        return audio.copy()
    n = audio.shape[1]
    if n < FFT:
        audio = np.pad(audio, ((0, 0), (0, FFT - n)))
    X = spectral(audio)
    observed = np.abs(X) ** 2
    observed = gaussian_filter1d(observed, .8, axis=1, mode="nearest")
    if profile.time_smoothing > 1:
        observed = uniform_filter1d(observed, profile.time_smoothing, axis=2, mode="nearest")
    eye = np.eye(3, dtype=np.float32)
    mixing = eye[None] + profile.cross_share * (power_transfer - eye[None])
    direct = np.maximum(np.einsum("fij,jft->ift", np.linalg.inv(mixing), observed, optimize=True), 0.0)
    # Refine nonnegative source powers without reversing any source polarity.
    norm = np.einsum("fij,fik->fjk", mixing, mixing, optimize=True)
    rhs = np.einsum("fji,jft->ift", mixing, observed, optimize=True)
    steps = 1.0 / np.maximum(np.sum(norm, axis=-1).max(axis=-1), 1.0)
    for _ in range(3):
        grad = np.einsum("fij,jft->ift", norm, direct, optimize=True) - rhs
        direct = np.maximum(direct - steps[None, :, None] * grad, 0.0)
    cross = np.einsum("fij,jft->ift", mixing - eye[None], direct, optimize=True)
    fraction = direct / np.maximum(direct + cross, 1e-20)
    gain = np.maximum(profile.lower_gain, fraction ** profile.exponent)
    return waveform(X * gain, n)


FROZEN_PROFILE = MaskProfile(0.55, 0.90, 0.55, 5)
FITTING_FRACTION = 0.45
CORE_SAMPLES = 1 << 18
CONTEXT_SAMPLES = 8 * HOP


def validate_inputs(paths):
    identities = [(stat.st_dev, stat.st_ino) for stat in (path.stat() for path in paths)]
    if len(identities) != 3 or len(set(identities)) != 3:
        raise ValueError("Supply three distinct close-microphone recordings in A/B/C order")
    info = [sf.info(path) for path in paths]
    lengths = {track.frames for track in info}
    if len(lengths) != 1 or any(track.samplerate != RATE or track.channels != 1 or track.frames < 1 for track in info):
        raise ValueError("Three nonempty synchronized 16 kHz mono tracks of the same length are required")
    return lengths.pop()


def render(paths, output, power_transfer, profile=FROZEN_PROFILE):
    """Write aligned float WAVs in bounded blocks with enough transform context."""
    samples = validate_inputs(paths)
    output.mkdir(exist_ok=True, parents=True)
    output_paths = [output / f"speaker_{speaker}.wav" for speaker in NAMES]
    if any(path.exists() for path in output_paths):
        raise FileExistsError("The selected output directory already contains A/B/C waveform files")

    with TemporaryDirectory(prefix=".close-cleanup-", dir=output) as staging:
        temporary = Path(staging)
        with ExitStack() as stack:
            readers = [stack.enter_context(sf.SoundFile(path)) for path in paths]
            writers = [
                stack.enter_context(sf.SoundFile(
                    temporary / path.name, mode="w", samplerate=RATE, channels=1, subtype="FLOAT"
                ))
                for path in output_paths
            ]
            for first in range(0, samples, CORE_SAMPLES):
                last = min(first + CORE_SAMPLES, samples)
                start = max(0, first - CONTEXT_SAMPLES)
                end = min(samples, last + CONTEXT_SAMPLES)
                for reader in readers:
                    reader.seek(start)
                audio = np.array([reader.read(end - start, dtype="float32") for reader in readers])
                if not np.isfinite(audio).all():
                    raise ValueError("Microphone signal contains nonfinite samples")
                cleaned = suppress(audio, power_transfer, profile)
                for writer, row in zip(writers, cleaned, strict=True):
                    writer.write(row[first - start:last - start])
        for path in output_paths:
            (temporary / path.name).replace(path)


def main():
    from direct_microphones import calculate_features

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("microphones", nargs=3, type=Path, metavar="MIC_WAV")
    parser.add_argument("--vad", required=True, type=Path, help="Silero VAD ONNX model")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for the aligned WAVs")
    args = parser.parse_args()
    samples = validate_inputs(args.microphones)
    started = perf_counter()
    power, posterior, feature_samples = calculate_features(args.microphones, args.vad)
    after_features = perf_counter()
    if samples != feature_samples:
        raise ValueError("Audio changed while estimating speech activity")
    transfer, evidence = fit_power_transfer(
        args.microphones, power, posterior, fit_fraction=FITTING_FRACTION,
        percentiles=(FROZEN_PROFILE.spectral_transfer_percentile,)
    )
    after_fit = perf_counter()
    render(args.microphones, args.out, transfer[FROZEN_PROFILE.spectral_transfer_percentile])
    report = {
        "samples": samples,
        "sample_rate": RATE,
        "direct_mic_order": NAMES,
        "fitting_fraction": FITTING_FRACTION,
        "profile": asdict(FROZEN_PROFILE),
        "calibration": evidence,
        "speech_detection_s": after_features - started,
        "calibration_s": after_fit - after_features,
        "render_s": perf_counter() - after_fit,
        "processing_s": perf_counter() - started,
    }
    (args.out / "close_waveform.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote three {samples}-sample tracks in {report['processing_s']:.2f} seconds")


if __name__ == "__main__":
    main()
