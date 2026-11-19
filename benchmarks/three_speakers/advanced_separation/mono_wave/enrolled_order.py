"""Map three 16-kHz mono stems to labeled speech examples from the same session."""
import argparse
import hashlib
from itertools import permutations
import json
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.fft import dct


SAMPLE_RATE = 16000
ENROLLMENT_SAMPLES = 8192
ORDERS = np.array(list(permutations(range(3))))


def _waveform(samples):
    wave = np.asarray(samples, dtype=np.float64)
    if wave.ndim != 1 or wave.size < 512 or not np.isfinite(wave).all():
        raise ValueError("Expected at least 512 finite mono PCM samples")
    return wave


def _envelope(wave):
    # Analyze only the passband produced by the checkpoint, even for full-band enrollment.
    low = signal.resample_poly(wave, 1, 2)
    bins = np.arange(129)
    edges = 700 * np.expm1(np.linspace(np.log1p(70 / 700), np.log1p(3850 / 700), 28)) * 256 / 8000
    rising = (bins - edges[:-2, None]) / (edges[1:-1, None] - edges[:-2, None])
    falling = (edges[2:, None] - bins) / (edges[2:, None] - edges[1:-1, None])
    filters = np.maximum(0, np.minimum(rising, falling))
    # The default STFT centers a window on sample zero and pads its last hop.
    # Evaluate the same frames in batches to bound working memory for long files.
    padded = np.pad(low, (128, 128))
    last_pad = (-(padded.size - 256)) % 96
    if last_pad:
        padded = np.pad(padded, (0, last_pad))
    count = 1 + (padded.size - 256) // 96
    power = np.empty((filters.shape[0], count))
    for start in range(0, count, 2048):
        stop = min(count, start + 2048)
        section = padded[start * 96:(stop - 1) * 96 + 256]
        spectrum = signal.stft(section, nperseg=256, noverlap=160, boundary=None, padded=False)[2]
        power[:, start:stop] = filters @ np.square(np.abs(spectrum))
    totals = power.sum(axis=0)
    if totals.max() <= 1e-14:
        raise ValueError("A silent track cannot be assigned a speaker label")
    active = totals >= 0.06 * totals.max()
    floor = np.max(power) * 1e-5 + 1e-13
    cepstra = dct(np.log(np.maximum(power[:, active], floor)), axis=0, norm="ortho")
    return cepstra[1:13].mean(axis=1)


def fit_templates(enrollment, sample_rate):
    """Build A/B/C templates from lists of labeled, physically isolated clips."""
    if sample_rate != SAMPLE_RATE or len(enrollment) != 3:
        raise ValueError("Expected three collections of 16-kHz A/B/C solo enrollment clips")
    profiles = []
    for clips in enrollment:
        features = []
        for clip in clips:
            wave = _waveform(clip)
            for left in range(0, wave.size - ENROLLMENT_SAMPLES + 1, ENROLLMENT_SAMPLES):
                excerpt = wave[left:left + ENROLLMENT_SAMPLES]
                rms = np.sqrt(np.mean(excerpt.reshape(16, -1) ** 2, axis=1))
                level = np.sqrt(np.mean(excerpt ** 2))
                if level > .004 and np.quantile(rms, .25) > .25 * level:
                    features.append(_envelope(excerpt * (.06 / level)))
                if len(features) == 32:
                    break
            if len(features) == 32:
                break
        if not features:
            raise ValueError("Each speaker needs at least one eligible 0.512-second solo excerpt")
        profiles.append(np.median(features, axis=0))
    return np.stack(profiles)


def assign_tracks(templates, tracks, sample_rate):
    """Return A/B/C indices into unordered tracks and the best-vs-second cost margin."""
    templates = np.asarray(templates, dtype=np.float64)
    tracks = np.asarray(tracks)
    if sample_rate != SAMPLE_RATE or templates.shape != (3, 12) or not np.isfinite(templates).all():
        raise ValueError("Expected three finite 12-dimensional templates and 16-kHz PCM")
    if tracks.ndim != 2 or tracks.shape[0] != 3:
        raise ValueError("Expected three sample-aligned mono tracks")
    observed = np.stack([_envelope(_waveform(track)) for track in tracks])
    return _resolve(templates, observed)


def _resolve(templates, observed):
    distances = np.linalg.norm(templates[:, None] - observed[None], axis=-1)
    costs = distances[np.arange(3), ORDERS].sum(axis=1)
    preferred = np.argsort(costs, kind="stable")
    margin = float(costs[preferred[1]] - costs[preferred[0]])
    if np.isclose(margin, 0.0, rtol=0.0, atol=1e-12 * max(1.0, costs[preferred[0]])):
        raise ValueError("Enrollment cannot distinguish a unique A/B/C assignment")
    return ORDERS[preferred[0]].copy(), margin


def _read_wav(path):
    import soundfile as sf

    with sf.SoundFile(path) as audio:
        if audio.format not in {"WAV", "WAVEX", "RF64"} or audio.channels != 1 or audio.samplerate != SAMPLE_RATE:
            raise ValueError(f"Expected a mono 16-kHz WAV file at {path}")
        return _waveform(audio.read(dtype="float64"))


def _identity(path):
    info = path.stat()
    return info.st_dev, info.st_ino


def _stamp(path):
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def export_labeled_wavs(track_paths, enrollment_paths, out):
    """Copy original WAV containers into a fresh directory in fixed A/B/C order."""
    if out.exists():
        raise FileExistsError(f"Output directory already exists: {out}")
    if len(track_paths) != 3 or len(enrollment_paths) != 3 or not all(enrollment_paths):
        raise ValueError("Supply three separated WAVs and at least one labeled enrollment take per speaker")
    files = [*track_paths, *(p for group in enrollment_paths for p in group)]
    if len({_identity(p) for p in files}) != len(files):
        raise ValueError("Supply distinct separated tracks and distinct labeled enrollment takes")
    # Strings, identical filenames in different directories and user-defined extensions do
    # not establish audio identity; use the actual WAV headers and file identities.
    enrollment = [(_read_wav(path) for path in paths) for paths in enrollment_paths]
    templates = fit_templates(enrollment, SAMPLE_RATE)
    observed, lengths, stamps = [], [], []
    for path in track_paths:
        stamp = _stamp(path)
        samples = _read_wav(path)
        observed.append(_envelope(samples))
        lengths.append(samples.size)
        if _stamp(path) != stamp:
            raise ValueError(f"Source WAV changed during analysis: {path}")
        stamps.append(stamp)
    if len(set(lengths)) != 1:
        raise ValueError("The separated track WAVs must have equal sample counts")
    order, margin = _resolve(templates, np.stack(observed))
    report = {"sample_rate": SAMPLE_RATE, "samples_per_track": lengths[0],
              "assignment_cost_margin": margin, "label_mapping": {}}
    out.mkdir(parents=False, exist_ok=False)
    created = []
    try:
        for name, index in zip("ABC", order, strict=True):
            source = track_paths[index]
            target = out / f"Speaker_{name}.wav"
            digest = hashlib.sha256()
            with source.open("rb") as reader, target.open("xb") as writer:
                created.append(target)
                while block := reader.read(1024 * 1024):
                    digest.update(block)
                    writer.write(block)
            if _stamp(source) != stamps[index]:
                raise ValueError(f"Source WAV changed before export completed: {source}")
            report["label_mapping"][name] = {
                "input_track_number": int(index + 1), "input_wav": str(source),
                "output_wav": target.name, "sha256": digest.hexdigest(),
                "enrollment_wavs": [str(p) for p in enrollment_paths["ABC".index(name)]],
            }
        manifest = out / "mapping.json"
        with manifest.open("x", encoding="utf-8") as fp:
            created.append(manifest)
            json.dump(report, fp, indent=2, ensure_ascii=False, allow_nan=False)
            fp.write("\n")
    except BaseException:
        for item in created:
            item.unlink(missing_ok=True)
        if not any(out.iterdir()):
            out.rmdir()
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, nargs=3, required=True, metavar=("SLOT1", "SLOT2", "SLOT3"),
                        help="Three time-aligned, already separated mono WAV tracks")
    for person in "abc":
        parser.add_argument(f"--enroll-{person}", type=Path, nargs="+", required=True, metavar="SOLO_WAV",
                            help=f"Isolated mono reference take(s) for Speaker {person.upper()}")
    parser.add_argument("--out", type=Path, required=True, help="New output directory under an existing parent")
    args = parser.parse_args()
    try:
        export_labeled_wavs(args.tracks, [args.enroll_a, args.enroll_b, args.enroll_c], args.out)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"Unable to label the tracks: {exc}\n")


if __name__ == "__main__":
    main()
