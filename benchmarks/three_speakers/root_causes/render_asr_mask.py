"""Export ASR-only soft-overlap tracks with bounded waveform memory."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf

from asr_mask import overlap_asr_tracks
from diarize import masks


RATE = 16000
ACTIVITY_HOP = 160
# Both the 100-sample STFT hop and 160-sample activity hop divide this guard.
PADDING = 800
BLOCK_SIZE = 10 * RATE


def apply_mask_recording(source, tracks, timing, output):
    timeline = json.loads(timing.read_text())["segments"]
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as tmp, ExitStack() as stack:
        mix = stack.enter_context(sf.SoundFile(source))
        if mix.channels != 1 or mix.samplerate != RATE or len(mix) == 0:
            raise ValueError("Expected a nonempty mono input at 16000 Hz")
        names, activity = masks(timeline, len(mix) / RATE)
        if (
            any(not name or Path(name).name != name for name in names)
            or len(names) != 3
        ):
            raise ValueError("Expected exactly three valid track labels")
        inputs = [
            stack.enter_context(sf.SoundFile(tracks / f"speaker_{name}.wav"))
            for name in names
        ]
        if any(
            f.channels != 1 or f.samplerate != RATE or len(f) != len(mix)
            for f in inputs
        ):
            raise ValueError("All three input tracks must match the original recording")
        staged = [
            stack.enter_context(
                sf.SoundFile(
                    Path(tmp) / f"{name}.wav",
                    "w+",
                    samplerate=RATE,
                    channels=1,
                    subtype="FLOAT",
                )
            )
            for name in names
        ]
        total = len(mix)
        peak = 0.0
        for start in range(0, total, BLOCK_SIZE):
            end = min(start + BLOCK_SIZE, total)
            left = max(0, start - PADDING)
            right = min(total, end + PADDING)
            for handle in [mix, *inputs]:
                handle.seek(left)
            wave = mix.read(right - left, dtype="float32")
            estimates = np.stack(
                [handle.read(right - left, dtype="float32") for handle in inputs]
            )
            active = activity[
                :, left // ACTIVITY_HOP : (right + ACTIVITY_HOP - 1) // ACTIVITY_HOP
            ]
            guarded = overlap_asr_tracks(wave, estimates, active)[
                :, start - left : end - left
            ]
            peak = max(peak, float(np.abs(guarded).max()))
            for handle, samples in zip(staged, guarded, strict=True):
                handle.write(samples)
        gain = min(1.0, 0.98 / max(1e-8, peak))
        for name, handle in zip(names, staged, strict=True):
            handle.seek(0)
            # Publish complete WAVs after reading all inputs, including when directories coincide.
            target = Path(tmp) / f"speaker_{name}.wav"
            with sf.SoundFile(
                target, "w", samplerate=RATE, channels=1, subtype="PCM_16"
            ) as out:
                while len(block := handle.read(BLOCK_SIZE, dtype="float32")):
                    out.write(block * gain)
            target.replace(output / target.name)
    return {
        "speaker_ids": names,
        "sample_rate": RATE,
        "sample_count": total,
        "gain": gain,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = apply_mask_recording(args.mix, args.tracks, args.timing, args.out)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
