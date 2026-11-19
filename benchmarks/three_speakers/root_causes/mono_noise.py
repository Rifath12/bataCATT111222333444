"""Estimate the room noise in detected pauses and apply a conservative common gain to speaker tracks."""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.fft import rfft, irfft
from scipy.ndimage import minimum_filter1d, uniform_filter1d
import soundfile as sf
from diarize import masks

HOP = 160
FFT = 640
WIN = np.sqrt(np.hanning(FFT + 1)[:-1]).astype("f")


def transform(audio):
    source = np.pad(audio, (FFT // 2, FFT // 2))
    return rfft(
        np.lib.stride_tricks.sliding_window_view(source, FFT)[::HOP] * WIN, axis=1
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mono", type=Path, required=True)
    p.add_argument("--timing", type=Path, required=True)
    p.add_argument("--tracks", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    info = sf.info(a.mono)
    size = info.frames
    rate = info.samplerate
    if rate != 16000 or info.channels != 1:
        raise ValueError("Expected 16 kHz mono WAV")
    names, mask = masks(json.loads(a.timing.read_text())["segments"], size / rate)
    quiet = minimum_filter1d((mask.sum(0) == 0).astype("B"), 31).astype(bool)
    rng = np.random.default_rng(131)

    def sampled_audio(file, ok, limit):
        starts = np.flatnonzero(ok) * HOP
        starts = starts[(starts >= FFT // 2) & (starts + FFT // 2 < size)]
        starts = np.sort(
            rng.choice(starts, size=min(limit, len(starts)), replace=False)
        )
        values = np.empty((len(starts), FFT), dtype="f")
        for i, t in enumerate(starts):
            file.seek(t - FFT // 2)
            values[i] = file.read(FFT, dtype="float32")
        return values

    with sf.SoundFile(a.mono) as f:
        samples = sampled_audio(f, quiet, 2400)
        if len(samples) < 10:
            raise ValueError("Less than 100 ms of usable noise")
        noise_db = float(10 * np.log10(np.mean(samples**2) + 1e-18))
        floor = (
            np.quantile(abs(rfft(samples * WIN, axis=1)) ** 2, 0.5, axis=0) / np.log(2)
            + 1e-13
        )
        solo_levels = {}
        for j, name in enumerate(names):
            solo = sampled_audio(f, mask[j] & (mask.sum(0) == 1), 800)
            solo_levels[name] = (
                float(10 * np.log10(np.median(np.mean(solo**2, axis=1)) + 1e-18))
                if len(solo)
                else None
            )
    a.out.mkdir(exist_ok=True, parents=True)
    for name in names:
        with (
            sf.SoundFile(a.tracks / f"speaker_{name}.wav") as inp,
            sf.SoundFile(
                a.out / f"speaker_{name}.wav",
                "w",
                samplerate=rate,
                channels=1,
                subtype="PCM_16",
            ) as dst,
        ):
            if len(inp) != size or inp.samplerate != rate or inp.channels != 1:
                raise ValueError("Misaligned input")
            for pos in range(0, size, 200 * HOP):
                end = min(pos + 200 * HOP, size)
                left = max(0, pos - 16 * HOP)
                right = min(size, end + 16 * HOP)
                inp.seek(left)
                y = inp.read(right - left, dtype="float32")
                spec = transform(y)
                avg = uniform_filter1d(abs(spec) ** 2, 7, axis=0)
                gain = np.clip(1.0 - floor[None, :] / np.maximum(avg, 1e-12), 0.12, 1)
                frames = irfft(spec * np.sqrt(gain), n=FFT, axis=1) * WIN
                reconstruct = np.zeros(len(y) + FFT, dtype="f")
                norm = np.zeros_like(reconstruct)
                for k, frame in enumerate(frames):
                    t = k * HOP
                    reconstruct[t : t + FFT] += frame
                    norm[t : t + FFT] += WIN**2
                lo = FFT // 2 + pos - left
                hi = lo + (end - pos)
                dst.write(reconstruct[lo:hi] / np.maximum(norm[lo:hi], 1e-8))
    (a.out / "noise.json").write_text(
        json.dumps(
            dict(
                noise_rms_dbfs=noise_db,
                solo_median_rms_dbfs=solo_levels,
                estimated_noise_frames=len(samples),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
