"""Export pre-threshold speaker probabilities using the public NeMo-Speech C API."""

import argparse
import ctypes as c
from pathlib import Path
import time

import numpy as np
import soundfile as sf


class Config(c.Structure):
    _fields_ = [
        ("size", c.c_size_t),
        ("model_path", c.c_char_p),
        ("gpu", c.c_int32),
        ("preset", c.c_char_p),
        ("chunk_frames", c.c_int32),
        ("right_context_frames", c.c_int32),
        ("left_context_frames", c.c_int32),
        ("fifo_frames", c.c_int32),
        ("spkcache_frames", c.c_int32),
        ("update_period_frames", c.c_int32),
    ]


class Runtime:
    def __init__(self, library):
        self.lib = c.CDLL(str(library.resolve()))
        handle, out = c.c_void_p, c.POINTER(c.c_void_p)
        floats = c.POINTER(c.c_float)
        self.create = self.bind("create", c.c_int, c.POINTER(Config), out)
        self.destroy = self.bind("destroy", None, handle)
        self.speakers = self.bind("num_speakers", c.c_int32, handle)
        self.interval = self.bind("seconds_per_frame", c.c_double, handle)
        self.open = self.bind("stream_open", c.c_int, handle, out)
        self.close = self.bind("stream_close", None, handle)
        self.push = self.bind(
            "stream_push_f32", c.c_int, handle, floats, c.c_size_t, c.c_int32
        )
        self.finish = self.bind("stream_finish", c.c_int, handle)
        self.start = self.bind("frame_probs_start", c.c_int64, handle)
        self.end = self.bind("frame_count", c.c_int64, handle)
        self.copy = self.bind("frame_probs", c.c_int, handle, floats, c.c_size_t)
        self.lib.nemo_speech_asr_last_error.restype = c.c_char_p

    def bind(self, name, returned, *args):
        function = getattr(self.lib, "nemo_speech_diar_" + name)
        function.argtypes = args
        function.restype = returned
        return function

    def check(self, status):
        if status:
            message = self.lib.nemo_speech_asr_last_error()
            raise RuntimeError(
                message.decode(errors="replace") if message else f"NeMo status {status}"
            )


def export(args):
    rt = Runtime(args.library)
    model = c.c_void_p()
    stream = c.c_void_p()
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
        rt.check(rt.open(model, c.byref(stream)))
        speakers, step = rt.speakers(model), rt.interval(model)
        parts = []
        end = 0

        def collect():
            nonlocal end
            start, current = rt.start(stream), rt.end(stream)
            if end < start:
                raise RuntimeError("The retained probability history has a gap")
            if current == end:
                return
            buff = np.empty((current - start, speakers), dtype=np.float32)
            rt.check(
                rt.copy(stream, buff.ctypes.data_as(c.POINTER(c.c_float)), buff.size)
            )
            parts.append(buff[end - start :].copy())
            end = current

        with sf.SoundFile(args.audio) as source:
            if source.channels != 1 or source.samplerate != 16000:
                raise ValueError("Expected 16 kHz mono audio")
            for index, chunk in enumerate(
                source.blocks(blocksize=60 * 16000, dtype="float32")
            ):
                rt.check(
                    rt.push(
                        stream,
                        chunk.ctypes.data_as(c.POINTER(c.c_float)),
                        len(chunk),
                        source.samplerate,
                    )
                )
                collect()
                if index % 5 == 0:
                    print(f"Native probability frames {end}", flush=True)
        rt.check(rt.finish(stream))
        collect()
        return np.concatenate(parts, axis=0), step
    finally:
        if stream.value:
            rt.close(stream)
        rt.destroy(model)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio", type=Path)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    started = time.perf_counter()
    probabilities, step = export(args)
    np.savez(
        args.output,
        probabilities=probabilities,
        step=step,
        wall_s=time.perf_counter() - started,
    )
    print(
        f"Saved {probabilities.shape[0]} probability frames, at {step:.3f} seconds per frame",
        flush=True,
    )


if __name__ == "__main__":
    main()
