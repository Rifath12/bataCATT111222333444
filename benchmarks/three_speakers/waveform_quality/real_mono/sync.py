"""Estimate the offset between headset and array sample clocks using speech envelopes."""

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
from scipy.io import wavfile


def envelope(data):
    hop = 160
    result = []
    for i in range(0, len(data) // hop * hop, hop * 1000):
        arr = np.array(
            data[i : min(i + hop * 1000, len(data) // hop * hop)], np.float32
        ).reshape(-1, hop)
        result.extend(np.sqrt(np.mean(arr**2, axis=1)))
    return np.array(result)


def calibrate(array, close):
    e = envelope(array)
    n = np.stack([envelope(x) for x in close])
    n = n / np.maximum(np.quantile(n, 0.85, axis=1)[:, None], 1.0)
    n = np.log(0.05 + n.max(axis=0))
    e = np.log(0.05 + e / max(np.quantile(e, 0.85), 1.0))

    def detail(x):
        return x - uniform_filter1d(x, 151)

    n = detail(n)
    e = detail(e)
    out = []
    for first in range(2000, min(len(n), len(e)) - 6000, 6000):
        ref = e[first : first + 4000]
        other = n[first - 1700 : first + 5700]
        a = signal.correlate(other, ref, mode="valid", method="fft")
        sums = signal.convolve(other**2, np.ones(len(ref)), mode="valid", method="fft")
        corr = a / np.sqrt(np.maximum(np.sum(ref**2) * sums, 1e-12))
        best = np.argmax(corr)
        out.append(
            [float((first + 2000) / 100), float((best - 1700) / 100), float(corr[best])]
        )
    out = np.asarray(out)
    good = out[:, 2] > 0.24
    if good.sum() < max(4, len(out) // 4):
        raise ValueError(f"Could not align independent clocks: {out.tolist()}")
    # Clock offsets are robustly fitted, rejecting individual windows dominated by a different talker.
    dif = out[good, 1]
    t = out[good, 0]
    slopes = (dif[:, None] - dif[None, :]) / np.where(
        t[:, None] != t[None, :], t[:, None] - t[None, :], np.nan
    )
    slope = float(np.nanmedian(slopes))
    offset = float(np.median(dif - slope * t))
    selected = good & (np.abs(out[:, 1] - out[:, 0] * slope - offset) < 0.06)
    if selected.sum() < 4:
        raise ValueError("Insufficient consensus about relative clocks")
    slope, offset = np.polyfit(out[selected, 0], out[selected, 1], 1)
    if np.max(np.abs(out[selected, 1] - (out[selected, 0] * slope + offset))) > 0.08:
        raise ValueError("Array clock fit exceeds fixed limit")
    return dict(
        offset_s=float(offset),
        slope_s_per_s=float(slope),
        window_rms_error_s=float(
            np.sqrt(
                np.mean((out[selected, 1] - out[selected, 0] * slope - offset) ** 2)
            )
        ),
        accepted_windows=int(selected.sum()),
        all_windows=out.tolist(),
    )


if __name__ == "__main__":
    import sys, json
    from pathlib import Path

    d = Path(sys.argv[1])
    hours = sorted(d.glob("[0-9]*.wav"))
    near = [wavfile.read(f, mmap=True)[1] for f in hours]
    arr = wavfile.read(d / "array.wav", mmap=True)[1][:, 0]
    print(json.dumps(calibrate(arr, near)), flush=True)
