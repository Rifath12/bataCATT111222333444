"""Reproduce the frozen 15% room-conditioned gain experiment on one locked headset panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from time import perf_counter
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter, uniform_filter

# This script may live in a subdirectory of the checked-out waveform benchmark.
for parent in Path(__file__).resolve().parents:
    if parent.name == "three_speakers" and (parent / "advanced_separation").is_dir():
        sys.path[:0] = [str(parent / "advanced_separation"), str(parent / "waveform_quality")]
        break

from boosted_close import BoostedMask
from close_waveform import FROZEN_PROFILE, stored_transfer, suppress
from waveform_acceptance import LABELS, evaluate_case

RATE = 16000
FFT = 512
HOP = 128
TAKE = 28672
MARGIN = 3200
DIMS = (3, 7, 19, 5, 5)
BLEND = 0.15
REMIXES = 128
SEED = 172547


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checked(path, expected):
    if checksum(path) != expected:
        raise ValueError(f"Checksum mismatch in {path.name}")
    return path


def read_manifest(directory):
    lock = load_json(directory.parent / "lock.json")
    matched = [r for r in lock["rooms"] if r["room_group"] == directory.name]
    if len(matched) != 1:
        raise ValueError("Exactly one matching room must appear in the panel lock")
    path = checked(directory / "manifest.json", matched[0]["manifest_sha256"])
    manifest = load_json(path)
    speakers = manifest["speaker_ids"]
    if (
        manifest["room_group"] != directory.name or manifest["sample_rate"] != RATE
        or len(set(speakers)) != 3 or any(type(s) is not int or s < 0 for s in speakers)
        or manifest["fit_fraction"] != .45
        or manifest["case_count"] != len(manifest["trials"])
    ):
        raise ValueError("Expected a locked three-headset room manifest")
    return manifest


def read_donors(directory, manifest):
    room = manifest["room_group"]
    speakers = manifest["speaker_ids"]
    original_hashes = manifest["source_sha256"]
    waveforms = []
    for speaker in speakers:
        name = f"audio/{speaker}.wav"
        path = checked(directory / name, original_hashes[f"{room}/{name}"])
        sample_rate, wave = wavfile.read(path, mmap=True)
        if sample_rate != RATE or wave.dtype != np.int16 or wave.shape != (manifest["recording_samples"],):
            raise ValueError("Expected synchronized mono PCM16 headset recordings")
        waveforms.append(wave)

    rttm = checked(directory / "room.rttm", original_hashes[f"{room}/room.rttm"])
    records = [line.split() for line in rttm.read_text().splitlines() if line.strip()]
    if any(len(row) < 8 or row[0] != "SPEAKER" or row[1] != room for row in records):
        raise ValueError("Unexpected room or speaker annotation")
    names = sorted({int(record[7].removeprefix("N_SPK")) for record in records})
    if names != sorted(manifest["all_annotated_speakers"]) or not set(speakers).issubset(names):
        raise ValueError("Annotation speaker identities differ from the locked manifest")
    length = len(waveforms[0])
    cutoff = int(manifest["fit_max_samples"] * .8)
    if manifest["fit_max_samples"] != int(length * .45):
        raise ValueError("Unexpected calibration interval")
    activity = np.zeros((len(names), (length + 159)//160), dtype=bool)
    for row in records:
        start, duration = float(row[3]), float(row[4])
        if not np.isfinite([start, duration]).all() or start < 0 or duration < 0:
            raise ValueError("Invalid annotation time")
        first, last = round(start*100), round((start+duration)*100)
        activity[names.index(int(row[7].removeprefix("N_SPK"))), first:last] = True
    count = activity.sum(axis=0)
    banks = []
    for target, speaker in enumerate(speakers):
        # Include every annotated participant when excluding competing speech.
        solo = activity[names.index(speaker)] & (count == 1)
        edges = np.flatnonzero(np.diff(np.r_[False, solo, False])) * 160
        portions = []
        for left, right in zip(edges[::2], edges[1::2], strict=True):
            stop = min(min(right, length)-MARGIN, cutoff)
            for first in range(left+MARGIN, stop-TAKE+1, TAKE):
                image = np.stack([w[first:first+TAKE] for w in waveforms]).astype(np.float32) / 32768
                direct = image[target]
                frame_rms = np.sqrt(np.mean(direct.reshape(-1, 512)**2, axis=1))
                active = frame_rms >= max(.002, .16*np.quantile(frame_rms, .7))
                if active.mean() < .64 or frame_rms[active].mean() <= .002:
                    continue
                relative = np.sqrt(np.mean(image**2, axis=1)) / max(1e-12, np.sqrt(np.mean(direct**2)))
                if np.max(np.delete(relative, target)) <= .70:
                    portions.append(image * (.045/frame_rms[active].mean()))
        if not portions:
            raise ValueError(f"No eligible early solo take for speaker {speaker}")
        banks.append(portions)
    return banks, cutoff


def spectrum(audio):
    return signal.stft(audio, fs=RATE, nperseg=FFT, noverlap=FFT-HOP)[2]


def ratio_db(numerator, denominator):
    return 10*np.log10((numerator+5e-13)/(denominator+5e-13))


def cell_indices(spectra, transfer):
    power = np.abs(spectra)**2
    local = uniform_filter(power, size=(1, 3, 5), mode="nearest")
    wide = uniform_filter(power, size=(1, 11, 21), mode="nearest")
    cross = transfer[::2] - np.eye(3, dtype=np.float32)[None]

    def foreign(energies):
        return np.einsum("fij,jft->ift", cross, energies, optimize=True)

    ratio = ratio_db(foreign(local), local)
    broad = ratio_db(foreign(wide), wide) - ratio
    fast = ratio_db(foreign(power), power) - ratio
    bands = np.searchsorted([10, 22, 41, 70, 108, 171], np.arange(spectra.shape[1]), side="right")
    channels = np.broadcast_to(np.arange(3)[:, None, None], spectra.shape)
    frequency = np.broadcast_to(bands[None, :, None], spectra.shape)
    coordinates = (
        channels, frequency, np.clip(np.floor((ratio+36)/3), 0, 18).astype(int),
        np.searchsorted([-8, -2, 2, 8], broad), np.searchsorted([-7, -2, 2, 7], fast)
    )
    return np.ravel_multi_index(coordinates, DIMS), local


def fit_table(banks, transfer):
    rng = np.random.default_rng(SEED)
    total = np.zeros(np.prod(DIMS))
    weight = np.zeros_like(total)
    for index in range(REMIXES):
        donors = np.stack([pool[rng.integers(len(pool))] for pool in banks])
        if index % 5 == 4:
            gains = rng.uniform(-24, 0, 3)
        else:
            gains = np.zeros(3)
            gains[rng.integers(3)] = (0., -7., -14., -22.)[index % 5]
        gains -= np.max(gains)
        donors *= np.asarray(10**(gains/20), np.float32)[:, None, None]
        mixture = spectrum(donors.sum(axis=0))
        target = spectrum(donors[np.arange(3), np.arange(3)])
        cells, local = cell_indices(mixture, transfer)
        power = np.abs(mixture)**2
        norm = np.maximum(np.quantile(local, .88, axis=(1, 2)), 2e-12)[:, None, None]
        energy_weight = np.minimum(power/norm, 4)
        clean_gain = np.clip((target*mixture.conj()).real/np.maximum(power+.008*local, 5e-14), .015, 1.18)
        total += np.bincount(cells.ravel(), weights=(clean_gain*energy_weight).ravel(), minlength=total.size)
        weight += np.bincount(cells.ravel(), weights=energy_weight.ravel(), minlength=weight.size)

    def smooth(cells):
        return gaussian_filter(cells.reshape(DIMS), sigma=(0, .72, 1., .80, .80), mode="nearest")

    table = np.clip((smooth(total)+5*.86)/(smooth(weight)+5), .12, 1.06).astype(np.float32)
    if not np.isfinite(table).all():
        raise ValueError("Room calibration produced nonfinite values")
    return table


def refine(audio, transfer, table, boosted):
    if not np.isfinite(audio).all() or not np.isfinite(boosted).all():
        raise ValueError("Nonfinite microphone or baseline waveform")
    if np.array_equal(transfer, np.broadcast_to(np.eye(3), transfer.shape)):
        predicted = audio
    else:
        padded = np.pad(audio, ((0, 0), (0, max(0, FFT-audio.shape[1]))))
        raw = spectrum(padded)
        cells, _ = cell_indices(raw, transfer)
        gain = uniform_filter(table.ravel()[cells], size=(1, 1, 3), mode="nearest")
        wave = signal.istft(raw*gain, fs=RATE, nperseg=FFT, noverlap=FFT-HOP)[1]
        predicted = wave[:, :audio.shape[1]].astype(np.float32)
    tracks = boosted + BLEND*(predicted-boosted)
    if not np.isfinite(tracks).all():
        raise ValueError("Room correction produced nonfinite waveforms")
    return tracks


def compact(score):
    voices = score["voices"]["exact_three"]
    return {
        "passes": score["passes_exact_three"],
        "alignment": [voices[label]["target_aligned_fraction"] for label in LABELS],
        "retention": [voices[label]["retained_active_fraction"] for label in LABELS],
        "quiet_alignment": voices[score["quietest_exact_three_speaker"]]["target_aligned_fraction"]
    }


def summarize(rows):
    result = {}
    for arm in ("boosted", "room_calibrated"):
        voices = np.array([row[arm]["alignment"] for row in rows])
        retention = np.array([row[arm]["retention"] for row in rows])
        result[arm] = {
            "joint96": sum(row[arm]["passes"]["96"] for row in rows),
            "joint98": sum(row[arm]["passes"]["98"] for row in rows),
            "weakest_voice": float(voices.min()),
            "quiet_voice_p10": float(np.quantile([row[arm]["quiet_alignment"] for row in rows], .10)),
            "mean_retention": float(retention.mean()),
            "retention_failures": int((retention < .96).sum()),
        }
    return result


def run(directory, output, case, save_wavs):
    manifest = read_manifest(directory)
    trials = [row for row in manifest["trials"] if case is None or row["id"] == case]
    if not trials or (case is not None and len(trials) != 1):
        raise ValueError("No unique matching fixture in the locked manifest")
    calibration = load_json(directory / "calibration.json")
    if not 0 < calibration["maximum_source_sample"] <= manifest["fit_max_samples"]:
        raise ValueError("Power calibration extends beyond the locked fitting interval")
    path = checked(directory / "published_transfer.npy", calibration["transfer_sha256"])
    transfer = stored_transfer(path)
    started = perf_counter()
    banks, cutoff = read_donors(directory, manifest)
    table = fit_table(banks, transfer)
    table_time = perf_counter() - started
    model = BoostedMask.load()
    rows = []
    for trial in trials:
        case_id = trial["id"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", case_id):
            raise ValueError("Invalid fixture ID in manifest")
        fixture = checked(directory / "held" / f"{case_id}.npz", trial["sha256"])
        with np.load(fixture, allow_pickle=False) as data:
            rate = int(data["sample_rate"])
            mix, refs, active, exact = [data[k] for k in ("mixture", "references", "activity", "exact_three")]
        if (
            rate != RATE or mix.shape != (3, trial["samples"]) or not np.isfinite(mix).all()
            or refs.shape != mix.shape or not np.isfinite(refs).all() or active.shape != mix.shape
            or exact.dtype.kind != "b" or active.dtype.kind != "b" or not np.array_equal(exact, active.all(axis=0))
            or not exact.any() or int(exact.sum()) != trial["exact_three_samples"]
            or len(trial["source_starts_samples"]) != 3
            or min(trial["source_starts_samples"]) < manifest["fit_max_samples"]
        ):
            raise ValueError(f"Unexpected sample clock, sources, values or mask in {case_id}")
        boosted = model.refine(mix, transfer, suppress(mix, transfer, FROZEN_PROFILE))
        corrected = refine(mix, transfer, table, boosted)
        row = {"id": case_id, "exact_three_s": int(exact.sum())/RATE}
        for arm, tracks in (("boosted", boosted), ("room_calibrated", corrected)):
            row[arm] = compact(evaluate_case(case_id, RATE, refs, tracks, active, exact_three=exact))
        rows.append(row)
        if save_wavs:
            for label, track in zip(LABELS, corrected, strict=True):
                wavfile.write(output / f"speaker_{label}.wav", RATE, track.astype(np.float32))
    return {
        "room": manifest["room_group"], "sample_rate": RATE, "speakers_ABC": manifest["speaker_ids"],
        "case_count": len(rows), "exact_three_s": sum(row["exact_three_s"] for row in rows),
        "early_donor_counts_ABC": list(map(len, banks)), "fit_cutoff_samples": cutoff,
        "table_preparation_s": table_time, "blend": BLEND, "remix_count": REMIXES, "seed": SEED,
        "table_bytes_sha256": hashlib.sha256(table.tobytes()).hexdigest(),
        "summary": summarize(rows), "trials": rows
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("room", type=Path, help="Room directory produced by the locked headset panel builder")
    parser.add_argument("--case", help="Score one fixture by its manifest ID (default: all fixtures)")
    parser.add_argument("--out", required=True, type=Path, help="New directory for the comparison report")
    parser.add_argument("--wavs", action="store_true", help="Also write three corrected float WAVs for --case")
    args = parser.parse_args()
    if args.wavs and not args.case:
        parser.error("--wavs requires --case")
    if args.out.exists():
        parser.error("--out must point to a new directory")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".room-gain-", dir=args.out.parent) as stage:
        output = Path(stage) / "result"
        output.mkdir()
        report = run(args.room.resolve(), output, args.case, args.wavs)
        (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        output.rename(args.out)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
