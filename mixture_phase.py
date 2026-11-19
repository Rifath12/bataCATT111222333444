"""Restore speaker WAVs with the recorded mixture phase using bounded memory."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from scipy.signal import istft, stft
import soundfile as sf

WINDOW = 256
HOP = 64
RATE = 8000


def _restore(mixture, estimates):
    length = len(mixture)
    if length < WINDOW:
        mixture = np.pad(mixture, (0, WINDOW - length))
        estimates = np.pad(estimates, ((0, 0), (0, WINDOW - length)))
    spec = dict(nperseg=WINDOW, noverlap=WINDOW - HOP)
    measured = stft(mixture, **spec)[2]
    magnitudes = np.abs(stft(estimates, **spec)[2])
    weights = magnitudes / np.maximum(np.float32(1e-13), magnitudes.sum(axis=0))
    return istft(weights * measured, **spec)[1][:, :length]


def restore_wavs(source, destination, filenames, block=65536):
    """Read exported 8 kHz PCM and write aligned restored PCM to a separate folder.

    Include one FFT window of original samples on each side of a processing block.
    All block starts share the same hop grid. A shared gain prevents output clipping.
    """
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Restoration needs a separate output directory")
    names = list(filenames)
    if not names or len(set(names)) != len(names) or any(
        not name or Path(name).name != name or name == "mixed.wav" for name in names
    ):
        raise ValueError("Expected distinct speaker WAV filenames")
    if block < WINDOW or block % HOP:
        raise ValueError("The processing block must be a multiple of the hop and at least one FFT window")
    destination.mkdir(parents=True, exist_ok=True)
    keys = ["mixed.wav", *names]
    if any((destination / name).exists() for name in keys):
        raise FileExistsError("A destination WAV already exists")
    started = perf_counter()
    peak = 0.
    with TemporaryDirectory(dir=destination) as scratch:
        spool = Path(scratch) / "restored.wav"
        with ExitStack() as stack:
            inputs = [stack.enter_context(sf.SoundFile(source / name)) for name in keys]
            frames = len(inputs[0])
            if not frames or any(x.channels != 1 or x.samplerate != RATE or len(x) != frames for x in inputs):
                raise ValueError("Expected nonempty, aligned 8 kHz mono meeting exports")
            work = stack.enter_context(sf.SoundFile(spool, "w", samplerate=RATE, channels=len(names),
                                                   subtype="FLOAT", format="RF64"))
            for left in range(0, frames, block):
                right = min(frames, left + block)
                before, after = max(0, left - WINDOW), min(frames, right + WINDOW)
                for stream in inputs:
                    stream.seek(before)
                read = [s.read(after - before, dtype="float32") for s in inputs]
                if not all(np.isfinite(x).all() for x in read):
                    raise ValueError("Expected finite PCM samples in exported WAVs")
                restored = _restore(read[0], np.stack(read[1:]))[:, left-before:right-before]
                peak = max(peak, float(np.max(np.abs(restored))), float(np.max(np.abs(read[0]))))
                work.write(restored.T)
        gain = min(1., .999 / peak) if peak > 1. - 1./32768 else 1.
        with ExitStack() as stack:
            spool_stream = stack.enter_context(sf.SoundFile(spool))
            mixed = stack.enter_context(sf.SoundFile(source / "mixed.wav"))
            outputs = [stack.enter_context(sf.SoundFile(destination / name, "x", samplerate=RATE,
                        channels=1, format="WAV", subtype="PCM_16")) for name in keys]
            for start in range(0, frames, block):
                n = min(block, frames - start)
                outputs[0].write(mixed.read(n, dtype="float32") * gain)
                estimates = spool_stream.read(n, dtype="float32", always_2d=True)
                for out, wave in zip(outputs[1:], estimates.T, strict=True):
                    out.write(wave * gain)
    return {"method": "magnitude_share_mixture_phase", "window_samples": WINDOW, "hop_samples": HOP,
            "playback_gain": gain, "input": "exported PCM WAVs", "elapsed_s": perf_counter() - started,
            "activity_and_review": "from the separator before phase restoration"}


def restore_export(source, destination):
    """Create a fresh listening export from a finished meeting; transcripts need a new ASR pass."""
    from process_meeting import write_listening_report
    source, destination = Path(source), Path(destination)
    info = json.loads((source / "speakers.json").read_text(encoding="utf-8"))
    if info.get("mixture_phase"):
        raise ValueError("These WAVs already have mixture-phase restoration")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    filenames = [s["file"] for s in info["speakers"].values()]
    with TemporaryDirectory(dir=destination.parent) as scratch:
        stage = Path(scratch) / "export"
        report = restore_wavs(source, stage, filenames)
        info["mixture_phase"] = report
        info["wav_playback_gain"] = info.get("wav_playback_gain", 1.) * report["playback_gain"]
        if "audio_export_wall_s" in info:
            info["source_audio_export_wall_s"] = info["audio_export_wall_s"]
            info["audio_export_wall_s"] += report["elapsed_s"]
        info["asr"] = "disabled"
        info.pop("complete_wall_s", None)
        if (source / "commits.jsonl").is_file():
            shutil.copyfile(source / "commits.jsonl", stage / "commits.jsonl")
        (stage / "speakers.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_listening_report(stage, info, [])
        stage.rename(destination)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Existing process_meeting output directory")
    parser.add_argument("--out", type=Path, required=True, help="Fresh destination directory")
    args = parser.parse_args()
    print(json.dumps(restore_export(args.source, args.out), indent=2))
