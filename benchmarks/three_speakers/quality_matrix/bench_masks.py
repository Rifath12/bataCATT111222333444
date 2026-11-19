"""Benchmark the original and sparse mask with exact PCM equality."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import tempfile
import time

import render_asr_mask


CASES = (
    ("M8016", "trial_next/listening"),
    ("M8017", "next_root_fresh/R8008_M8017"),
)


def pcm(folder):
    result = {}
    for path in sorted(folder.glob("*.wav")):
        with path.open("rb") as stream:
            result[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    functions = {
        "previous": importlib.import_module("asr_mask_original").overlap_asr_tracks,
        "sparse": importlib.import_module("asr_mask").overlap_asr_tracks,
    }
    results = {}
    for meeting, stem_path in CASES:
        folder = Path(stem_path)
        original = Path("trial_next_fresh") / f"R8008_{meeting}" / "mono.wav"
        wall = {k: [] for k in functions}
        expected = None
        for rep in range(args.repeat):
            order = list(functions) if rep % 2 == 0 else list(reversed(functions))
            for key in order:
                render_asr_mask.overlap_asr_tracks = functions[key]
                with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
                    started = time.perf_counter()
                    info = render_asr_mask.apply_mask_recording(
                        original, folder, folder / "timing.json", Path(tmp)
                    )
                    wall[key].append(time.perf_counter() - started)
                    scores = pcm(Path(tmp))
                if not expected:
                    expected = scores
                if scores != expected:
                    raise AssertionError(f"PCM equality failed on {meeting} {key}")
                print(meeting, key, f"{wall[key][-1]:.3f}s", flush=True)
        results[meeting] = {
            "timing_wall_seconds": wall,
            "medians_seconds": {k: statistics.median(a) for k, a in wall.items()},
            "exact_wav_sha256": expected,
            "audio_info": info,
        }
        args.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
