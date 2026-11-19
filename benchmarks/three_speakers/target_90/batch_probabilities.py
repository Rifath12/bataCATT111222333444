"""Run paired mono diarization passes with one loaded native model."""

import argparse
import ctypes as c
from pathlib import Path
import time

import numpy as np
import soundfile as sf

from nemotron_probabilities import Config, Runtime, Segment
from prepare_mono_view import level_view


def run_view(rt, model, waveform, offline):
    stream = c.c_void_p()
    samples = waveform.ctypes.data_as(c.POINTER(c.c_float))
    began = time.perf_counter()
    try:
        if offline:
            rt.check(rt.offline(model, samples, len(waveform), 16000, c.byref(stream)))
        else:
            rt.check(rt.open(model, c.byref(stream)))
            rt.check(rt.push(stream, samples, len(waveform), 16000))
            rt.check(rt.finish(stream))
        if rt.start(stream) != 0:
            raise ValueError(
                "Probability history has been compacted, use a shorter excerpt"
            )
        result = np.empty((rt.end(stream), rt.speakers(model)), dtype=np.float32)
        rt.check(
            rt.copy(stream, result.ctypes.data_as(c.POINTER(c.c_float)), result.size)
        )
        n = c.c_size_t()
        rt.check(rt.segments(stream, None, None, 0, c.byref(n)))
        data = (Segment * n.value)()
        rt.check(rt.segments(stream, None, data, n.value, c.byref(n)))
        segments = np.asarray(
            [[s.start_time, s.end_time, s.speaker - 1] for s in data], np.float64
        ).reshape(-1, 3)
        return result, segments, time.perf_counter() - began
    finally:
        if stream.value:
            rt.close(stream)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio-list", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--alternate", choices=("offline", "shifted"), default="offline")
    args = p.parse_args()
    rt = Runtime(args.library)
    rt.offline = rt.bind(
        "offline_f32",
        c.c_int,
        c.c_void_p,
        c.POINTER(c.c_float),
        c.c_size_t,
        c.c_int32,
        c.POINTER(c.c_void_p),
    )
    model = c.c_void_p()
    config = Config(
        c.sizeof(Config),
        bytes(args.model.resolve()),
        -1,
        b"v3-offline",
        0,
        0,
        -1,
        0,
        0,
        0,
    )
    rt.check(rt.create(c.byref(config), c.byref(model)))
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        paths = [
            Path(line)
            for line in args.audio_list.read_text().splitlines()
            if line.strip()
        ]
        for i, path in enumerate(paths, start=1):
            target = args.output / f"{path.stem}.npz"
            if target.exists():
                continue
            wave, rate = sf.read(path, dtype="float32")
            if rate != 16000 or wave.ndim != 1:
                raise ValueError(f"Expected 16 kHz mono input at {path}")
            base, native, base_wall = run_view(rt, model, wave, False)
            alternate, _ = level_view(wave)
            if args.alternate == "offline":
                secondary, _, alt_wall = run_view(rt, model, alternate, True)
                offset_frames = 0
            else:
                offset_frames = 928
                secondary, _, alt_wall = run_view(
                    rt, model, np.pad(alternate, (160 * offset_frames, 0)), False
                )
            np.savez(
                target,
                base=base,
                alternate=secondary,
                native=native,
                offset=offset_frames,
                duration_s=len(wave) / rate,
                base_wall_s=base_wall,
                alternate_wall_s=alt_wall,
            )
            if i == 1 or i % 8 == 0 or i == len(paths):
                print(
                    i,
                    len(paths),
                    path.stem,
                    f"{base_wall:.2f}s + {alt_wall:.2f}s",
                    flush=True,
                )
    finally:
        rt.destroy(model)


if __name__ == "__main__":
    main()
