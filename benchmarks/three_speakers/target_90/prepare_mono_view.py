"""Create a second diarization view with fixed context shift and level correction."""

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
import soundfile as sf


def level_view(wave):
    hop = 1600
    frames = np.pad(wave, (0, -len(wave) % hop)).reshape(-1, hop)
    energy = np.sqrt(np.mean(frames**2, axis=1))
    level = gaussian_filter1d(np.maximum(energy, 0.003), 9)
    gain = np.clip(0.045 / level, 1.0, 8.0)
    output = np.empty_like(wave)
    for start in range(0, len(wave), 160000):
        stop = min(len(wave), start + 160000)
        values = np.interp(
            np.arange(start, stop) / hop, np.arange(len(gain)) + 0.5, gain
        )
        output[start:stop] = np.clip(wave[start:stop] * values, -0.97, 0.97)
    return output, gain


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--offset", type=float, default=9.28)
    args = p.parse_args()
    wave, rate = sf.read(args.input, dtype="float32")
    if rate != 16000 or wave.ndim != 1:
        p.error("Expected mono audio sampled at 16 kHz")
    output, gain = level_view(wave)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        args.out, np.pad(output, (round(rate * args.offset), 0)), rate, subtype="PCM_16"
    )
    print(
        f"{args.out.name} {len(wave) / rate:.2f} seconds, median gain {np.median(gain):.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
