"""Score waveform isolation with fixed A/B/C labels and known source timelines."""

from argparse import ArgumentParser
from itertools import permutations
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

LABELS = ("A", "B", "C")
FIDELITY_TARGETS = (0.96, 0.98)
GAIN_LIMIT_DB = -12.0
MINIMUM_COVERAGE = 0.96
EPSILON = 1e-12
LABEL_PERMUTATIONS = tuple(permutations(range(len(LABELS))))


def centered(values):
    return values - values.mean(axis=-1, keepdims=True)


def boolean_mask(values):
    result = np.asarray(values)
    if result.dtype.kind != "b" and not np.all((result == 0) | (result == 1)):
        raise ValueError("Activity masks must contain only true/false values")
    return result.astype(bool, copy=False)


def fraction_db(fraction):
    """Convert an energy fraction to dB with a finite reporting floor."""
    return float(10.0 * np.log10(max(fraction, EPSILON) / max(1.0 - fraction, EPSILON)))


def signal_metrics(reference, track, speaker):
    """Decompose centered output into target, competing voices and residual."""
    sources = centered(reference)
    output = centered(track)
    target = sources[speaker]
    target_energy = float(target @ target)
    if target_energy <= EPSILON:
        return None
    gain = float((output @ target) / target_energy)
    estimate = gain * target
    output_energy = float(output @ output)
    residual = output - estimate
    others = np.delete(sources, speaker, axis=0)
    orthogonal_others = others - (others @ target / target_energy)[:, None] * target
    coefficients, *_ = np.linalg.lstsq(orthogonal_others.T, residual, rcond=1e-8)
    interference = coefficients @ orthogonal_others
    disturbance = residual - interference
    if output_energy > EPSILON:
        target_fraction = float(np.clip((estimate @ estimate) / output_energy, 0, 1))
        interference_fraction = float(np.clip((interference @ interference) / output_energy, 0, 1))
        residual_fraction = float(np.clip((disturbance @ disturbance) / output_energy, 0, 1))
    else:
        target_fraction, interference_fraction, residual_fraction = 0.0, 0.0, 0.0
    return {
        "target_aligned_fraction": target_fraction,
        "competing_voice_fraction": interference_fraction,
        "other_error_fraction": residual_fraction,
        "si_sdr_db": fraction_db(target_fraction),
        "issued_silence": output_energy <= EPSILON,
        "signed_target_gain": gain,
        "target_gain_db": float(20.0 * np.log10(max(abs(gain), EPSILON))),
        "reference_mean_square": target_energy / len(target),
    }


def gain_coverage(references, estimate, activity, width, minimum_active):
    """Fraction of reference-active sample time in blocks retaining target level."""
    eligible = retained = skipped = 0
    threshold = 10.0 ** (GAIN_LIMIT_DB / 20.0)
    for start in range(0, references.size, width):
        sl = slice(start, start + width)
        mask = activity[sl]
        duration = int(np.count_nonzero(mask))
        if duration < minimum_active:
            skipped += duration
            continue
        wanted = centered(references[sl][mask])
        actual = centered(estimate[sl][mask])
        squared = float(wanted @ wanted)
        if squared <= EPSILON:
            skipped += duration
            continue
        eligible += duration
        if abs(float(actual @ wanted) / squared) >= threshold:
            retained += duration
    return {
        "retained_active_fraction": retained / eligible if eligible else None,
        "retained_samples": retained,
        "eligible_samples": eligible,
        "skipped_active_samples": skipped,
    }


def label_diagnostic(reference, output):
    """Measure whether a different local mapping would look better without applying it."""
    r = centered(reference)
    y = centered(output)
    r_energy = np.sum(r**2, axis=1)
    y_energy = np.sum(y**2, axis=1)
    valid = np.all(r_energy > EPSILON) and np.all(y_energy > EPSILON)
    if not valid:
        return None
    fractions = np.clip((r @ y.T)**2 / (r_energy[:, None] * y_energy), 0.0, 1.0)
    scores = [sum(fractions[i, k] for i, k in enumerate(p)) / len(LABELS) for p in LABEL_PERMUTATIONS]
    best = int(np.argmax(scores))
    return {
        "different_assignment_improvement": float(scores[best] - scores[0]),
        "best_diagnostic_labels": [LABELS[k] for k in LABEL_PERMUTATIONS[best]],
    }


def threshold_pass(voices, target):
    """A trial passes only when all voices reach both fidelity and retention thresholds."""
    return all(
        voice is not None
        and voice["target_aligned_fraction"] >= target
        and voice["retained_active_fraction"] is not None
        and voice["retained_active_fraction"] >= MINIMUM_COVERAGE
        for voice in voices
    )


def evaluate_case(case_id, sample_rate, references, tracks, activity, *, mixture=None, exact_three=None, group=None):
    """Score same-clock source/estimate arrays ordered by fixed speaker identity."""
    references = np.asarray(references, dtype=np.float64)
    tracks = np.asarray(tracks, dtype=np.float64)
    activity = boolean_mask(activity)
    if (
        int(sample_rate) != sample_rate or sample_rate <= 0
        or references.ndim != 2 or references.shape[0] != len(LABELS)
        or references.shape != tracks.shape or references.shape != activity.shape
        or references.shape[1] < 2
        or not np.isfinite(references).all() or not np.isfinite(tracks).all()
    ):
        raise ValueError("Expected finite aligned [3, samples] reference, track and activity arrays")
    length = references.shape[1]
    if mixture is not None:
        mixture = np.asarray(mixture, dtype=np.float64)
        if mixture.ndim == 1:
            mixture = np.broadcast_to(mixture, references.shape)
        if mixture.shape != references.shape or not np.isfinite(mixture).all():
            raise ValueError("Mixture must be a finite mono or A/B/C-aligned array")
    known_three = np.all(activity, axis=0)
    if exact_three is None:
        triple = known_three
    else:
        triple = boolean_mask(exact_three)
        if triple.shape != (length,) or np.any(triple & ~known_three):
            raise ValueError("Exact-three mask must be sample-aligned and a subset of all-three activity")
    width = round(sample_rate * 0.1)
    minimum = round(sample_rate * 0.03)
    if minimum < 2:
        raise ValueError("Sample rate too small for the fixed activity windows")
    maybe_quietest = (
        LABELS[int(np.argmin(np.mean(references[:, triple] ** 2, axis=1)))]
        if np.any(triple) else None
    )
    speaker_scores = {}
    for phase, supported in (
        ("reference_active", activity),
        ("exact_three", np.broadcast_to(triple, activity.shape)),
    ):
        voices = {}
        for i, who in enumerate(LABELS):
            mask = supported[i]
            score = signal_metrics(references[:, mask], tracks[i, mask], i) if np.any(mask) else None
            if score is not None:
                score.update(gain_coverage(references[i], tracks[i], mask, width, minimum))
                score["seconds"] = int(np.count_nonzero(mask)) / sample_rate
                if phase == "reference_active":
                    others_only = np.any(activity, axis=0) & ~mask
                    score["other_active_while_target_inactive_s"] = int(np.count_nonzero(others_only)) / sample_rate
                    score["off_target_output_db_relative_to_reference"] = (
                        float(10 * np.log10(max(np.mean(centered(tracks[i, others_only]) ** 2), EPSILON)
                                            / score["reference_mean_square"]))
                        if np.any(others_only) else None
                    )
                if mixture is not None:
                    baseline = signal_metrics(references[:, mask], mixture[i, mask], i)
                    score["mixture_si_sdr_db"] = baseline["si_sdr_db"]
                    score["si_sdri_db"] = score["si_sdr_db"] - baseline["si_sdr_db"]
            voices[who] = score
        speaker_scores[phase] = voices

    local = []
    window_size = round(sample_rate * 0.25)
    for start in range(0, length, window_size):
        stop = min(length, start + window_size)
        masks = triple[start:stop]
        duration = int(np.count_nonzero(masks))
        if duration < minimum:
            continue
        targets = references[:, start:stop][:, masks]
        output = tracks[:, start:stop][:, masks]
        metrics = {who: signal_metrics(targets, output[i], i) for i, who in enumerate(LABELS)}
        gain_threshold = 10.0 ** (GAIN_LIMIT_DB / 20.0)
        local.append({
            "start_s": start / sample_rate,
            "end_s": stop / sample_rate,
            "exact_three_s": duration / sample_rate,
            "fixed_target_fraction": {
                who: None if metric is None else metric["target_aligned_fraction"]
                for who, metric in metrics.items()
            },
            "passes_all_three_with_level": {
                str(round(target * 100)): all(
                    metric is not None and metric["target_aligned_fraction"] >= target
                    and abs(metric["signed_target_gain"]) >= gain_threshold
                    for metric in metrics.values()
                )
                for target in FIDELITY_TARGETS
            },
            "label_diagnostic": label_diagnostic(targets, output),
        })
    exact = speaker_scores["exact_three"]
    both = {
        str(round(target * 100)): threshold_pass(exact.values(), target)
        for target in FIDELITY_TARGETS
    }
    return {
        "id": case_id,
        "group": str(case_id if group is None else group),
        "sample_rate": int(sample_rate),
        "duration_s": length / sample_rate,
        "exact_three_s": int(np.count_nonzero(triple)) / sample_rate,
        "quietest_exact_three_speaker": maybe_quietest,
        "passes_exact_three": both,
        "voices": speaker_scores,
        "exact_three_windows": local,
    }


def quantiles(values):
    if len(values) == 0:
        return None
    return dict(zip(("min", "p10", "median", "p90", "max"), map(float, np.quantile(values, (0, 0.1, 0.5, 0.9, 1)))))


def binomial_interval(successes, total, confidence=0.95):
    """Return exact two-sided Clopper–Pearson limits for independent groups."""
    if not total:
        return None
    alpha = (1 - confidence) / 2
    low = 0.0 if not successes else float(beta.ppf(alpha, successes, total - successes + 1))
    high = 1.0 if successes == total else float(beta.ppf(1 - alpha, successes + 1, total - successes))
    return [low, high]


def summarize(cases, *, independent_groups=False):
    """Report macro score distributions, quiet speakers and group-level pass rates."""
    known_cases = [x for x in cases if x["exact_three_s"] > 0]
    by_group = {}
    for case in known_cases:
        by_group.setdefault(case["group"], []).append(case)
    voice_rows = [
        row for case in known_cases for row in case["voices"]["exact_three"].values()
        if row is not None
    ]
    quiet_rows = [
        row for case in known_cases
        if (row := case["voices"]["exact_three"][case["quietest_exact_three_speaker"]]) is not None
    ]
    keys = [
        "target_aligned_fraction", "competing_voice_fraction", "other_error_fraction",
        "si_sdr_db", "si_sdri_db", "target_gain_db", "retained_active_fraction",
    ]
    passes = {}
    for target in FIDELITY_TARGETS:
        tag = str(round(target * 100))
        passing_cases = sum(x["passes_exact_three"][tag] for x in known_cases)
        passing_groups = sum(all(x["passes_exact_three"][tag] for x in group) for group in by_group.values())
        passing_voices = sum(
            v["target_aligned_fraction"] >= target
            and v["retained_active_fraction"] is not None
            and v["retained_active_fraction"] >= MINIMUM_COVERAGE
            for v in voice_rows
        )
        passes[tag] = {
            "voices_passed": passing_voices,
            "voices_eligible": len(known_cases) * len(LABELS),
            "cases_passed": passing_cases,
            "cases_eligible": len(known_cases),
            "groups_passed": passing_groups,
            "groups_eligible": len(by_group),
            "independent_group_interval_95": (
                binomial_interval(passing_groups, len(by_group)) if independent_groups else None
            ),
        }
    count_windows = sum(len(case["exact_three_windows"]) for case in known_cases)
    covered_window_time = sum(window["exact_three_s"] for case in known_cases for window in case["exact_three_windows"])
    pass_window_time = {
        tag: sum(
            window["exact_three_s"] for case in known_cases for window in case["exact_three_windows"]
            if window["passes_all_three_with_level"][tag]
        )
        for tag in passes
    }
    alternatives = [
        window["label_diagnostic"]["different_assignment_improvement"]
        for case in known_cases for window in case["exact_three_windows"]
        if window["label_diagnostic"] is not None
    ]
    return {
        "cases_no_exact_three": len(cases) - len(known_cases),
        "exact_three_s": sum(x["exact_three_s"] for x in known_cases),
        "scored_voice_rows": len(voice_rows),
        "score_by_voice": {key: quantiles([v[key] for v in voice_rows if v.get(key) is not None]) for key in keys},
        "score_by_quietest_voice": {key: quantiles([v[key] for v in quiet_rows if v.get(key) is not None]) for key in keys},
        "passes_at_fidelity_target_pct": passes,
        "exact_three_windows": count_windows,
        "exact_three_scored_window_seconds": covered_window_time,
        "exact_three_scored_window_fraction_meeting_all_targets": {
            tag: time / covered_window_time if covered_window_time else None
            for tag, time in pass_window_time.items()
        },
        "windows_local_permutation_improves_5pp": sum(x >= 0.05 for x in alternatives),
        "windows_with_non_silent_label_diagnostic": len(alternatives),
    }


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("fixtures", nargs="+", type=Path, help="Reference-aligned NPZs containing sample_rate, references, tracks, activity")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--independent-groups", action="store_true", help="Add binomial bounds only if the fixtures are independent random sampling groups")
    args = p.parse_args()
    cases = []
    for path in args.fixtures:
        with np.load(path, allow_pickle=False) as f:
            cases.append(evaluate_case(
                path.stem, int(f["sample_rate"]), f["references"], f["tracks"], f["activity"],
                mixture=f["mixture"] if "mixture" in f else None,
                exact_three=f["exact_three"] if "exact_three" in f else None,
                group=str(f["group"].item()) if "group" in f else None,
            ))
    result = {
        "acceptance_retained_active_fraction_min": MINIMUM_COVERAGE,
        "gain_min_db_for_retained_activity": GAIN_LIMIT_DB,
        "absolute_si_sdr_db_at_target": {str(round(q * 100)): fraction_db(q) for q in FIDELITY_TARGETS},
        "summary": summarize(cases, independent_groups=args.independent_groups),
        "cases": cases,
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
