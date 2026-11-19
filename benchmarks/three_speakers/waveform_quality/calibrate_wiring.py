"""Measure a three-microphone setup by remixing separate, synchronized solo takes."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from waveform_acceptance import LABELS, evaluate_case


def voiced_frames(audio, sample_rate):
    audio = audio - audio.mean()
    width = round(sample_rate * 0.02)
    padded = np.pad(audio, (0, (-len(audio)) % width))
    power = np.mean(padded.reshape(-1, width)**2, axis=1)
    active = power > max(1e-8, 0.06**2 * float(power.max()))
    return np.repeat(active, width)[:len(audio)]


def prepare(takes, sample_rate):
    """Normalize each contribution by its own mic, preserving its leakage to the others."""
    if len(takes) != len(LABELS):
        raise ValueError("Provide one solo take for each of A, B and C")
    for take in takes:
        if take.ndim != 2 or take.shape[1] != len(LABELS) or not np.isfinite(take).all():
            raise ValueError("Each solo take must have three finite synchronized mic channels in A/B/C order")
    length = min(map(len, takes))
    if length < sample_rate:
        raise ValueError("Each solo take must have at least one second of audio")

    images = np.stack([take[:length].astype(np.float64) for take in takes])
    references = images[np.arange(len(LABELS)), :, np.arange(len(LABELS))]
    activity = np.stack([voiced_frames(source, sample_rate) for source in references])
    if any(int(active.sum()) < round(0.25 * sample_rate) for active in activity):
        raise ValueError("Each solo take needs distinct audible speech in its own mic channel")
    distinct = references - references.mean(axis=1, keepdims=True)
    distinct /= np.linalg.norm(distinct, axis=1, keepdims=True).clip(min=1e-12)
    if np.max(np.abs(np.triu(distinct @ distinct.T, k=1))) > 0.9:
        raise ValueError("The direct reference tracks appear duplicated. Use three distinct solo takes.")
    rms = np.array([np.sqrt(np.mean(source[active]**2)) for source, active in zip(references, activity)])
    # A fixed room measurement expressed in dB relative to the speaker's own channel.
    leakage = np.empty((len(LABELS), len(LABELS)))
    for j, active in enumerate(activity):
        matrix = images[j, active]
        matrix = matrix - matrix.mean(axis=0, keepdims=True)
        powers = np.mean(matrix**2, axis=0)
        leakage[j] = 10 * np.log10(np.maximum(powers, 1e-15) / max(powers[j], 1e-15))
    return images * (0.05 / rms[:, None, None]), activity, leakage


def remix(images, quiet=None, drop_db=0):
    gains = np.ones(len(LABELS), dtype=np.float64)
    if quiet is not None:
        gains[quiet] = 10.0 ** (-drop_db / 20.0)
    contributions = images * gains[:, None, None]
    mixture = contributions.sum(axis=0)
    master = min(1.0, 0.95 / max(1e-12, float(np.abs(mixture).max())))
    references = contributions[np.arange(len(LABELS)), :, np.arange(len(LABELS))]
    return mixture.T * master, references * master


def run(takes, sample_rate, quiet_db):
    if sample_rate != 16_000:
        raise ValueError("Use synchronized 16 kHz recordings")
    if any(not np.isfinite(db) or not 0 < db <= 40 for db in quiet_db):
        raise ValueError("Use finite quiet-speaker differences above 0 and at most 40 dB")
    images, activity, leakage = prepare(takes, sample_rate)
    triple = activity.all(axis=0)
    if triple.sum() < 0.25 * sample_rate:
        raise ValueError("Use solo takes with at least 0.25 seconds of common speech activity when aligned")
    cases = []
    settings = [(None, 0)] + [(i, db) for db in quiet_db for i in range(len(LABELS))]
    for quiet, drop in settings:
        title = "balanced" if quiet is None else f"{LABELS[quiet]}_quiet_{drop:g}dB"
        mixture, references = remix(images, quiet, drop)
        case = evaluate_case(title, sample_rate, references, mixture, activity, group="one_microphone_setup")
        case["deliberately_quiet"] = None if quiet is None else LABELS[quiet]
        case["level_difference_db"] = drop
        cases.append(case)
    return {
        "protocol": "Three recorded solo takes remixed on every physical channel",
        "input_speaker_order": list(LABELS),
        "solo_crosstalk_db": {
            speaker: {mic: float(leakage[i, j]) for j, mic in enumerate(LABELS)}
            for i, speaker in enumerate(LABELS)
        },
        "exact_three_seconds_per_remix": float(triple.sum()/sample_rate),
        "all_remixes_pass": {
            level: all(case["passes_exact_three"][level] for case in cases)
            for level in ("96","98")
        },
        "remixes": cases,
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("solo", nargs=3, type=Path, metavar="SOLO_WAV")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--quiet-db", type=float, nargs="+", default=(6, 12, 20))
    options = parser.parse_args()
    if len({path.resolve() for path in options.solo}) != len(LABELS):
        parser.error("provide separate solo recordings for A, B and C")
    waves, rates = zip(*(sf.read(file, dtype="float64", always_2d=True) for file in options.solo))
    if len(set(rates)) != 1:
        parser.error("the solo files must share their sample rate")
    report = run(waves, rates[0], options.quiet_db)
    options.out.parent.mkdir(parents=True, exist_ok=True)
    options.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(options.out)
    for case in report["remixes"]:
        score = case["voices"]["exact_three"]
        values = ", ".join(f"{label} {100*score[label]['target_aligned_fraction']:.2f}%"
                           for label in LABELS)
        print(case["id"], values)


if __name__ == "__main__":
    main()
