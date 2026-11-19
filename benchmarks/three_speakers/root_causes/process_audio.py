"""Separate an offline 16 kHz meeting and write aligned WAVs and Mandarin transcripts."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import soundfile as sf

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
ROOT = BENCH.parent.parent
WORDS = BENCH / "refinement" / "words"


def run(script, arguments, env):
    subprocess.run(
        [sys.executable, str(script), *map(str, arguments)], env=env, check=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audio", type=Path, help="16 kHz mono or synchronized microphone-array file"
    )
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--sensevoice", type=Path, required=True)
    parser.add_argument(
        "--native-library", type=Path, help="Nemotron native runtime library"
    )
    parser.add_argument("--diar-model", type=Path, help="Nemotron 3 model checkpoint")
    parser.add_argument(
        "--timing",
        type=Path,
        help="Optional existing three-speaker timing JSON, in seconds",
    )
    parser.add_argument(
        "--models", type=Path, help="DPRNN and voiceprint model directory for mono"
    )
    parser.add_argument(
        "--sepformer-three",
        type=Path,
        help="Optional three-source checkpoint directory for mono",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=ROOT,
        help="Directory containing process_meeting.py",
    )
    parser.add_argument(
        "--fast-transcript",
        action="store_true",
        help="Skip the continuous mono recognition pass",
    )
    parser.add_argument(
        "--mask-for-asr",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use soft overlap masks on temporary recognition inputs (enabled by default for mono)",
    )
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Thread count must be positive")
    if not args.timing and not (args.native_library and args.diar_model):
        parser.error(
            "Provide --native-library and --diar-model for automatic speaker timing"
        )
    info = sf.info(args.audio)
    if info.samplerate != 16000 or info.frames == 0:
        parser.error("Please supply a nonempty 16 kHz recording")
    if info.channels > 1 and args.mask_for_asr:
        parser.error("--mask-for-asr applies only to mono input")
    if info.channels == 1 and not args.models:
        parser.error("A single-channel recording requires the --models directory")
    target = args.out.resolve()
    if target.exists():
        parser.error("Choose a new output directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    paths = [str(ROOT), str(BENCH), str(WORDS), str(HERE), str(BENCH / "target_90")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        env[name] = str(args.threads)
    if args.native_library:
        runtime_dir = str(args.native_library.resolve().parent)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            filter(None, [runtime_dir, env.get("LD_LIBRARY_PATH")])
        )
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".meeting-") as tmp:
        work = Path(tmp)
        product = work / "result"
        product.mkdir()
        timing = product / "speaker_timing.json"
        if args.timing:
            data = json.loads(args.timing.read_text(encoding="utf-8"))
        else:
            mono = args.audio
            if info.channels > 1:
                mono = work / "first_channel.wav"
                with (
                    sf.SoundFile(args.audio) as f,
                    sf.SoundFile(
                        mono, "w", samplerate=16000, channels=1, subtype="FLOAT"
                    ) as out,
                ):
                    for block in f.blocks(160000, dtype="float32", always_2d=True):
                        out.write(block[:, 0])
            posterior, native = work / "probabilities.npz", work / "native.rttm"
            run(
                BENCH / "nemotron_probabilities.py",
                [
                    mono,
                    "--library",
                    args.native_library,
                    "--model",
                    args.diar_model,
                    "--output",
                    posterior,
                    "--default-rttm",
                    native,
                ],
                env,
            )
            run(
                BENCH / "improve_overlap.py",
                [
                    mono,
                    "--probabilities",
                    posterior,
                    "--native-rttm",
                    native,
                    "--profile",
                    BENCH / "target_90" / "previous_profile.json",
                    "--out",
                    timing,
                ],
                env,
            )
            data = json.loads(timing.read_text(encoding="utf-8"))
        segments = data["segments"]
        labels = sorted({str(row["speaker"]) for row in segments})
        if len(labels) != 3:
            raise ValueError(
                f"Expected three detected speaker IDs, found {len(labels)}"
            )
        names = dict(zip(labels, "ABC"))
        for row in segments:
            if not (0 <= row["start"] < row["end"] <= info.duration + 0.02):
                raise ValueError("Timing lies outside the recording")
            row["speaker"] = names[str(row["speaker"])]
        timing.write_text(
            json.dumps({"segments": segments, "input_to_output_ids": names}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        if info.channels == 1:
            opts = [
                args.audio,
                "--diarization",
                timing,
                "--models",
                args.models,
                "--project",
                args.project,
                "--threads",
                args.threads,
                "--workers",
                1,
                "--out",
                product,
            ]
            if args.sepformer_three:
                opts.extend(
                    ["--sepformer-three", args.sepformer_three, "--refine-three"]
                )
            run(BENCH / "render.py", opts, env)
        else:
            run(
                HERE / "spatial_render.py",
                ["--audio", args.audio, "--timing", timing, "--out", product],
                env,
            )
        decode_audio = product
        if info.channels == 1 and args.mask_for_asr is not False:
            decode_audio = work / "masked_audio"
            run(
                HERE / "render_asr_mask.py",
                [
                    "--mix",
                    args.audio,
                    "--tracks",
                    product,
                    "--timing",
                    timing,
                    "--out",
                    decode_audio,
                ],
                env,
            )
        if info.channels == 1:
            decode = [
                "--audio",
                decode_audio,
                "--timing",
                timing,
                "--asr",
                args.sensevoice,
                "--backend",
                "sensevoice",
                "--target-rms",
                0.035,
                "--out",
                product / "transcripts",
                "--threads",
                args.threads,
                "--suppress-duplicates",
            ]
            if not args.fast_transcript:
                decode.extend(
                    [
                        "--context-mix",
                        args.audio,
                        "--restore-supported-text",
                        "--protect-long-turns",
                    ]
                )
            run(BENCH / "recognize.py", decode, env)
        else:
            raw = work / "raw_text"
            run(
                WORDS / "compare_recognizers.py",
                [
                    "--tracks",
                    product,
                    "--timing",
                    timing,
                    "--model",
                    args.sensevoice,
                    "--backend",
                    "sensevoice",
                    "--out",
                    raw,
                    "--threads",
                    args.threads,
                ],
                env,
            )
            run(
                BENCH / "duplicate_leakage.py",
                [
                    "--baseline",
                    raw / "transcript.jsonl",
                    "--tracks",
                    product,
                    "--params",
                    BENCH / "duplicate_settings.json",
                    "--out",
                    product / "transcripts",
                ],
                env,
            )
        product.rename(target)
    print(target)


if __name__ == "__main__":
    main()
