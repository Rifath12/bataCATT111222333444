"""Verify that file-backed ASR spans exactly match the original loaded arrays."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from recognize import inference_spans


def main():
    mask_root = Path("next_root_fresh/guard_full")
    timings = {
        "R8008_M8016": Path("trial_next/listening/timing.json"),
        "R8008_M8017": Path("next_root_fresh/R8008_M8017/timing.json"),
    }
    total = 0
    for meeting, timing in timings.items():
        rows = json.loads(timing.read_text())["segments"]
        n = 0
        for s in "ABC":
            f = mask_root / meeting / f"speaker_{s}.wav"
            entire, sr = sf.read(f, dtype="float32")
            spans = inference_spans(
                [r for r in rows if str(r["speaker"]) == s], len(entire) / sr
            )
            with sf.SoundFile(f) as pcm:
                for a, b in spans:
                    left, right = round(a * sr), round(b * sr)
                    pcm.seek(left)
                    waveform = pcm.read(right - left, dtype="float32")
                    np.testing.assert_array_equal(waveform, entire[left:right])
            n += len(spans)
        print(meeting, n)
        total += n
    print(total, "exact sample-identical ASR input spans")


if __name__ == "__main__":
    main()
