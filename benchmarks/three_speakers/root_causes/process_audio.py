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
ADVANCED = BENCH / "advanced_separation"


def run(script, arguments, env):
    subprocess.run(
        [sys.executable, str(script), *map(str, arguments)], env=env, check=True
    )


def prepare_timing(data, duration, allow_silent):
    """Preserve declared speaker identities through clips with absent speakers."""
    segments = data["segments"]
    observed = set()
    for row in segments:
        if row["start"] == row["end"] == 0 and allow_silent:
            continue
        if not 0 <= row["start"] < row["end"] <= duration + 0.02:
            raise ValueError("Timing lies outside the recording")
        observed.add(str(row["speaker"]))

    present = {str(row["speaker"]) for row in segments}
    raw = data.get("speaker_ids") if allow_silent else None
    if raw is not None and not isinstance(raw, list):
        raise ValueError("Expected a list of speaker IDs")
    labels = sorted(present) if raw is None else [str(name) for name in raw]
    if len(labels) != 3 or len(set(labels)) != 3 or not present.issubset(labels):
        raise ValueError("Expected three distinct speaker IDs in the timing")
    names = dict(zip(labels, "ABC"))
    mapped = [
        dict(row, speaker=names[str(row["speaker"])])
        for row in segments
        if row["end"] > row["start"]
    ]
    mapped.extend(
        dict(speaker=names[label], start=0, end=0)
        for label in labels
        if label not in observed
    )
    return {
        "segments": mapped,
        "speaker_ids": list("ABC"),
        "input_to_output_ids": names,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audio", type=Path, help="16 kHz mono or synchronized microphone-array file"
    )
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--sensevoice", type=Path, required=True)
    parser.add_argument(
        "--consensus-zipformer",
        type=Path,
        help="Optional Zipformer directory for selected-turn text reconciliation",
    )
    parser.add_argument(
        "--quality-paraformer",
        type=Path,
        help="Optional second mono character-corroboration stage using Paraformer and Zipformer",
    )
    parser.add_argument(
        "--array-context", action="store_true",
        help="Decode contextual speech from the synchronized microphone array beams",
    )
    parser.add_argument(
        "--array-overlap-mask", action="store_true",
        help="Use the optional spatial overlap mask on eight physical microphones",
    )
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
    if args.consensus_zipformer and (info.channels > 1 or args.fast_transcript):
        parser.error("--consensus-zipformer requires contextual mono transcription")
    if args.quality_paraformer and not args.consensus_zipformer:
        parser.error("--quality-paraformer requires --consensus-zipformer")
    if args.quality_paraformer and not all(
        (args.quality_paraformer / file).is_file() for file in ("model.int8.onnx", "tokens.txt")
    ):
        parser.error("--quality-paraformer needs model.int8.onnx and tokens.txt")
    if args.array_context and info.channels < 2:
        parser.error("--array-context requires synchronized multichannel audio")
    if args.array_overlap_mask and info.channels != 8:
        parser.error("--array-overlap-mask requires eight synchronized physical channels")
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
        timing.write_text(
            json.dumps(
                prepare_timing(data, info.duration, allow_silent=info.channels == 1),
                indent=2,
            )
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
            spatial_options = ["--audio", args.audio, "--timing", timing, "--out", product]
            if args.array_context:
                spatial_options.append("--ungated")
            if args.array_overlap_mask:
                spatial_options.extend(["--profile", "overlap-mask"])
            run(HERE / "spatial_render.py", spatial_options, env)
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
                        "--refine-context",
                    ]
                )
            if args.consensus_zipformer:
                decode.extend(["--consensus-zipformer", args.consensus_zipformer])
            run(BENCH / "recognize.py", decode, env)
            if args.quality_paraformer:
                transcripts = product / "transcripts"
                run(
                    ADVANCED / "mono_text.py",
                    [
                        "--baseline", transcripts / "transcript.jsonl",
                        "--tracks", product, "--timing", timing,
                        "--zipformer", args.consensus_zipformer,
                        "--paraformer", args.quality_paraformer,
                        "--threads", args.threads,
                        "--out", transcripts,
                    ],
                    env,
                )
        else:
            raw = work / "raw_text"
            array_options = [
                "--tracks", product, "--timing", timing, "--model", args.sensevoice,
                "--out", raw, "--threads", args.threads,
            ]
            if args.array_context:
                run(ADVANCED / "array_context.py", array_options, env)
                raw = raw / "extended"
            else:
                run(WORDS / "compare_recognizers.py", array_options + ["--backend", "sensevoice"], env)
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
