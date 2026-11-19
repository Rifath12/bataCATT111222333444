"""Separate three meeting tracks from a synchronized multichannel WAV and mono-derived timing."""

import argparse
from contextlib import ExitStack
import tempfile
import json
from pathlib import Path
import time
import numpy as np
from scipy.fft import rfft, irfft
from scipy.ndimage import minimum_filter1d
import soundfile as sf

from diarize import masks

HOP = 160
FFT = 640
WINDOW = np.sqrt(np.hanning(FFT + 1)[:-1]).astype("f")


def spectra(f, start, stop):
    offset = start * HOP - FFT // 2
    n = (stop - start - 1) * HOP + FFT
    begin = max(0, offset)
    f.seek(begin)
    data = np.zeros((n, f.channels), dtype="f")
    block = f.read(
        max(0, min(offset + n, len(f)) - begin), dtype="float32", always_2d=True
    )
    data[begin - offset : begin - offset + len(block)] = block
    frames = np.lib.stride_tricks.sliding_window_view(data, FFT, axis=0)[::HOP]
    return rfft(frames * WINDOW, axis=-1).transpose(0, 2, 1)


def covariances(f, activity):
    sets = [
        minimum_filter1d(((activity.sum(0) == 1) & row).astype("B"), 17).astype(bool)
        for row in activity
    ]
    sets.append(minimum_filter1d((activity.sum(0) == 0).astype("B"), 17).astype(bool))
    rng = np.random.default_rng(2309)
    selections = []
    for j, mask in enumerate(sets):
        idx = np.flatnonzero(mask)
        if j < 3 and len(idx) < 30:
            raise ValueError(
                "Each speaker needs at least 0.3 s of interior solo speech for spatial enrollment"
            )
        picked = rng.choice(idx, size=min(4000, len(idx)), replace=False)
        select = np.zeros_like(mask)
        select[picked] = True
        selections.append(select)
    shape = (FFT // 2 + 1, f.channels, f.channels)
    cov = np.zeros((4, *shape), np.complex128)
    for start in range(0, activity.shape[1], 500):
        stop = min(start + 500, activity.shape[1])
        spectrum = spectra(f, start, stop)
        for j, mask in enumerate(selections):
            x = spectrum[mask[start:stop]].transpose(1, 0, 2)
            if len(x[0]):
                # Convention is E[x x^H] for a column-vector microphone signal.
                cov[j] += x.swapaxes(1, 2) @ x.conj()
    return cov / np.maximum(1, [s.sum() for s in selections])[:, None, None, None]


def filters(cov):
    num = cov.shape[-1]
    # Souden MVDR uses the first microphone as the target reference for all speakers.
    eye = np.eye(num)
    output = []
    for s in range(3):
        speech = cov[s] - cov[3]
        eig, basis = np.linalg.eigh(speech)
        principal = basis[:, :, -1]
        target = (
            principal[:, :, None]
            * principal.conj()[:, None, :]
            * np.maximum(1e-9, eig[:, -1])[:, None, None]
        )
        other = (
            np.sum([cov[i] for i in range(3) if i != s], axis=0) * 0.5 + 0.25 * cov[3]
        )
        ridge = 0.02 * np.trace(other, axis1=-2, axis2=-1).real / num + 1e-10
        inverse_target = np.linalg.solve(other + ridge[:, None, None] * eye, target)
        weights = inverse_target[:, :, 0] / np.maximum(
            inverse_target.trace(axis1=-2, axis2=-1).real[:, None], 1e-12
        )
        # Degenerate frequencies fall back to the reference channel.
        bad = (np.linalg.norm(weights, axis=1) > 8) | (eig[:, -1] < 1e-8)
        weights[bad] = 0
        weights[bad, 0] = 1
        output.append(weights)
    return np.stack(output).astype("F")


def write_continuous_beams(source, weights, frame_count, temp):
    """Carry the overlap-add tail across FFT blocks on the original sample clock."""
    tail_length = FFT - HOP
    channels = weights.shape[0]
    carry = np.zeros((channels, tail_length), dtype="float32")
    carry_norm = np.zeros(tail_length, dtype="float32")
    peak = 0.0
    with sf.SoundFile(
        temp, "w", samplerate=16000, channels=channels, subtype="FLOAT"
    ) as out:
        for start in range(0, frame_count, 350):
            stop = min(frame_count, start + 350)
            x = spectra(source, start, stop)
            y = np.einsum("tfc,sfc->tsf", x, weights.conj(), optimize=True)
            frames = irfft(y, n=FFT, axis=-1) * WINDOW
            advance = len(frames) * HOP
            samples = np.zeros((channels, advance + tail_length), dtype="float32")
            norm = np.zeros(advance + tail_length, dtype="float32")
            samples[:, :tail_length] = carry
            norm[:tail_length] = carry_norm
            for i, frame in enumerate(frames):
                offset = i * HOP
                samples[:, offset : offset + FFT] += frame
                norm[offset : offset + FFT] += WINDOW**2
            left = max(0, FFT // 2 - start * HOP)
            right = min(
                advance if stop < frame_count else len(norm),
                len(source) + FFT // 2 - start * HOP,
            )
            if left < right:
                block = samples[:, left:right] / np.maximum(1e-8, norm[left:right])
                out.write(block.T)
                peak = max(peak, float(np.abs(block).max()))
            carry, carry_norm = samples[:, advance:], norm[advance:]
        if out.tell() != len(source):
            raise RuntimeError("Output sample count does not match the input")
    return min(1.0, 0.98 / max(1e-8, peak))


def save_tracks(temp, dest, names, activity, gain, ungated):
    with ExitStack() as ctx:
        source = ctx.enter_context(sf.SoundFile(temp))

        def outputs(prefix, extension):
            return [
                ctx.enter_context(
                    sf.SoundFile(
                        dest / f"{prefix}_{name}.{extension}",
                        "w",
                        samplerate=16000,
                        channels=1,
                        subtype="PCM_16",
                    )
                )
                for name in names
            ]

        wavs = outputs("speaker", "wav")
        beams = outputs("beam", "flac") if ungated else []
        offset = 0
        for block in source.blocks(160000, dtype="float32", always_2d=True):
            frames = np.arange(offset, offset + len(block)) // HOP
            for slot, wav in enumerate(wavs):
                if beams:
                    beams[slot].write(block[:, slot] * gain)
                wav.write(block[:, slot] * activity[slot, frames] * gain)
            offset += len(block)


def separate(args):
    start_time = time.perf_counter()
    segments = json.loads(args.timing.read_text(encoding="utf-8"))["segments"]
    args.out.mkdir(parents=True, exist_ok=True)
    with (
        sf.SoundFile(args.audio) as f,
        tempfile.TemporaryDirectory(dir=args.out) as work,
    ):
        if f.samplerate != 16000 or f.channels < 2 or len(f) == 0:
            raise ValueError("Expected nonempty synchronized multichannel 16 kHz audio")
        size = len(f)
        names, activity = masks(segments, size / 16000)
        if len(names) != 3:
            raise ValueError("Expected exactly three timing labels")
        if any(Path(name).name != name or name in {"", ".", ".."} for name in names):
            raise ValueError("Invalid speaker ID")
        weights = filters(covariances(f, activity))
        temp = Path(work) / "continuous.wav"
        gain = write_continuous_beams(f, weights, activity.shape[1], temp)
        save_tracks(temp, args.out, names, activity, gain, args.ungated)
    (args.out / "render.json").write_text(
        json.dumps(
            dict(
                method="spatial_mvdr",
                speaker_ids=names,
                sample_count=size,
                array_channels=weights.shape[-1],
                gain=gain,
                render_wall_s=time.perf_counter() - start_time,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--timing", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--ungated",
        action="store_true",
        help="Also export continuous beams for leakage diagnosis",
    )
    separate(p.parse_args())
