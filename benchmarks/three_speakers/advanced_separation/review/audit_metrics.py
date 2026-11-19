"""Audit meeting activity and character edits on a shared clock and speaker map."""

from argparse import ArgumentParser
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
import re
import unicodedata

import numpy as np
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment

TICKS_PER_SECOND = 1_000_000
ERROR_FIELDS = ("insertions", "deletions", "substitutions")


def normalized(text):
    text = re.sub(r"<[^>]*>|\[[^\]]*\]", "", text)
    return "".join(
        char.lower()
        for char in unicodedata.normalize("NFKC", text)
        if unicodedata.category(char)[0] in {"L", "N"}
    )


def clock(seconds):
    seconds = float(seconds)
    if not isfinite(seconds):
        raise ValueError("All timestamps must be finite")
    return round(seconds * TICKS_PER_SECOND)


def textgrid(path):
    """Parse the long Praat TextGrid used for AliMeeting transcription."""
    source = path.read_text(encoding="utf-8-sig")
    quoted = r'"((?:""|[^"])*)"'
    numeric = r"([-+\d.eE]+)"
    rows = []
    for tier in re.split(r"(?m)^\s*item \[\d+\]:", source)[1:]:
        if not re.search(r'(?m)^\s*class = "IntervalTier"\s*$', tier):
            continue
        name = re.search(r"(?m)^\s*name = " + quoted, tier)[1].replace('""', '"')
        for item in re.split(r"(?m)^\s*intervals \[\d+\]:", tier)[1:]:
            begin = float(re.search(r"(?m)^\s*xmin = " + numeric, item)[1])
            end = float(re.search(r"(?m)^\s*xmax = " + numeric, item)[1])
            text = re.search(r"(?m)^\s*text = " + quoted, item)[1].replace('""', '"')
            if normalized(text):
                rows.append(dict(speaker=name, start=begin, end=end, text=text))
    if not rows:
        raise ValueError(f"No transcribed intervals found in {path}")
    return rows


def rttm(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts or parts[0] != "SPEAKER":
            continue
        if len(parts) < 8:
            raise ValueError(f"Invalid RTTM speaker interval in {path}")
        start, length = float(parts[3]), float(parts[4])
        rows.append(dict(speaker=parts[7], start=start, end=start + length))
    if not rows:
        raise ValueError(f"No speaking intervals found in {path}")
    return rows


def spans(reference, hypothesis, duration):
    """Sweep boundaries once and count each active speaker exactly once."""
    events = {0: [], duration: []}
    for index, segments in enumerate((reference, hypothesis)):
        for segment in segments:
            left = max(0, clock(segment["start"]))
            right = min(duration, clock(segment["end"]))
            if right <= left:
                continue
            name = str(segment["speaker"])
            events.setdefault(left, []).append((index, name, 1))
            events.setdefault(right, []).append((index, name, -1))
    active = [Counter(), Counter()]
    times = sorted(events)
    for left, right in zip(times, times[1:]):
        for index, name, delta in events[left]:
            levels = active[index]
            levels[name] += delta
            if levels[name] == 0:
                del levels[name]
        yield left, right, frozenset(active[0]), frozenset(active[1])


def maximize_mapping(intervals, ref_labels, hyp_labels):
    if not ref_labels or not hyp_labels:
        return {}
    ref_ids = {name: i for i, name in enumerate(ref_labels)}
    hyp_ids = {name: i for i, name in enumerate(hyp_labels)}
    weights = np.zeros((len(ref_labels), len(hyp_labels)), dtype=np.int64)
    for begin, end, ref, hyp in intervals:
        for hr in ref:
            for hs in hyp:
                weights[ref_ids[hr], hyp_ids[hs]] += end - begin
    i, j = linear_sum_assignment(weights, maximize=True)
    return {hyp_labels[b]: ref_labels[a] for a, b in zip(i, j)}


def activity_scores(intervals, mapping):
    totals = defaultdict(Counter)
    for left, right, actual, predicted in intervals:
        duration = right - left
        exact_count = len(actual)
        observed = {mapping[s] for s in predicted if s in mapping}
        right_speakers = len(observed & actual)
        groups = ("full", f"count_{exact_count}")
        if exact_count >= 2:
            groups += ("any_overlap",)
        for group in groups:
            counts = totals[group]
            counts["clock"] += duration
            counts["reference"] += exact_count * duration
            counts["hypothesis"] += len(predicted) * duration
            counts["correct"] += right_speakers * duration
            counts["miss"] += max(exact_count - len(predicted), 0) * duration
            counts["false"] += max(len(predicted) - exact_count, 0) * duration
            counts["confusion"] += (
                min(exact_count, len(predicted)) - right_speakers
            ) * duration
            if exact_count == len(predicted) == right_speakers:
                counts["exact_set"] += duration

    def convert(c):
        ct, ref, hyp = c["clock"], c["reference"], c["hypothesis"]
        return dict(
            clock_s=ct / TICKS_PER_SECOND,
            reference_speaker_s=ref / TICKS_PER_SECOND,
            predicted_speaker_s=hyp / TICKS_PER_SECOND,
            speaker_recall=c["correct"] / ref if ref else None,
            speaker_precision=c["correct"] / hyp if hyp else None,
            exact_speaker_set_accuracy=c["exact_set"] / ct if ct else None,
            DER=(c["miss"] + c["false"] + c["confusion"]) / ref if ref else None,
            missed_speaker_s=c["miss"] / TICKS_PER_SECOND,
            false_alarm_speaker_s=c["false"] / TICKS_PER_SECOND,
            confused_speaker_s=c["confusion"] / TICKS_PER_SECOND,
        )

    return {group: convert(counts) for group, counts in sorted(totals.items())}


def exact_three_windows(intervals, mapping):
    """Return maximal exact-three reference windows and their capture scores."""
    runs = []
    for begin, end, actual, predicted in intervals:
        if len(actual) != 3:
            continue
        right = len(actual.intersection(mapping.get(s) for s in predicted))
        exact = right == len(predicted) == 3
        elapsed = end - begin
        if runs and runs[-1][1] == begin and runs[-1][2] == actual:
            window = runs[-1]
            window[1] = end
            window[3] += elapsed * right
            window[4] += elapsed * exact
        else:
            runs.append([begin, end, actual, elapsed * right, elapsed * exact])
    return [
        dict(
            start_s=begin / TICKS_PER_SECOND,
            end_s=end / TICKS_PER_SECOND,
            reference_speakers=sorted(actual),
            speaker_recall=right / (3 * (end - begin)),
            exact_set_fraction=exact / (end - begin),
        )
        for begin, end, actual, right, exact in runs
    ]


def exact_three_detection(intervals, mapping):
    """Separate triple detection recall from false triple alarms elsewhere."""
    actual_three = predicted_three = correct = 0
    for begin, end, actual, predicted in intervals:
        duration = end - begin
        actual_three += duration * (len(actual) == 3)
        predicted_three += duration * (len(predicted) == 3)
        if len(actual) == len(predicted) == 3:
            if actual == {mapping.get(s) for s in predicted}:
                correct += duration
    combined = actual_three + predicted_three
    return dict(
        reference_three_s=actual_three / TICKS_PER_SECOND,
        predicted_three_s=predicted_three / TICKS_PER_SECOND,
        correctly_attributed_three_s=correct / TICKS_PER_SECOND,
        precision=correct / predicted_three if predicted_three else None,
        recall=correct / actual_three if actual_three else None,
        F1=2 * correct / combined if combined else None,
    )


def speakers(rows):
    return sorted({str(row["speaker"]) for row in rows})


def merged_text(rows):
    result = defaultdict(list)
    for row in sorted(rows, key=lambda x: (x["start"], x["end"])):
        result[str(row["speaker"])].append(normalized(row["text"]))
    return {speaker: "".join(items) for speaker, items in result.items()}


def speaker_pairs(ref_labels, hyp_labels, mapping):
    reverse = {ref: hyp for hyp, ref in mapping.items()}
    return (
        [(ref, reverse.get(ref)) for ref in ref_labels]
        + [(None, hyp) for hyp in hyp_labels if mapping.get(hyp) not in ref_labels]
    )


def score_characters(reference, hypothesis, mapping=None):
    ref, hyp = merged_text(reference), merged_text(hypothesis)
    if mapping is None:
        m = max(len(ref), len(hyp))
        r, h = sorted(ref), sorted(hyp)
        r += [None] * (m - len(r))
        h += [None] * (m - len(h))
        costs = np.asarray([
            [Levenshtein.distance(ref.get(a, ""), hyp.get(b, "")) for b in h]
            for a in r
        ])
        if m:
            ai, bi = linear_sum_assignment(costs)
            mapping = {h[j]: r[i] for i, j in zip(ai, bi) if r[i] is not None and h[j] is not None}
        else:
            mapping = {}
    totals = Counter()
    for a, b in speaker_pairs(sorted(ref), sorted(hyp), mapping):
        truth, guess = ref.get(a, ""), hyp.get(b, "")
        ops = Counter(q.tag for q in Levenshtein.editops(truth, guess))
        totals.update(dict(
            reference_chars=len(truth), recognized_chars=len(guess),
            insertions=ops["insert"], deletions=ops["delete"], substitutions=ops["replace"],
            correct=len(truth) - ops["replace"] - ops["delete"],
        ))
    return dict(mapping=mapping, **char_report(totals))


def char_report(totals):
    denominator = totals["reference_chars"]
    edits = sum(totals[name] for name in ERROR_FIELDS)
    return dict(
        **totals,
        edits=edits,
        CER=edits / denominator if denominator else None,
    )


@dataclass(slots=True, frozen=True)
class Token:
    char: str
    tick: int
    supplied: bool


def timed_characters(rows):
    """Use exact supplied character midpoints only when complete and valid."""
    people = defaultdict(list)
    for row in sorted(rows, key=lambda x: (x["start"], x["end"])):
        content = normalized(row["text"])
        if not content:
            continue
        left, right = clock(row["start"]), clock(row["end"])
        if right < left:
            raise ValueError("Transcript turn ends before its start")
        explicit = []
        for char in row.get("characters", []):
            begin, end = clock(char["start_ms"] / 1000), clock(char["end_ms"] / 1000)
            token = normalized(char["text"])
            if not left <= begin <= end <= right:
                explicit = []
                break
            explicit.extend(Token(letter, (begin + end) // 2, True) for letter in token)
        matches = (
            "".join(token.char for token in explicit) == content
            and all(a.tick <= b.tick for a, b in zip(explicit, explicit[1:]))
        )
        tokens = explicit if matches else [
            Token(c, left + (2 * i + 1) * (right - left) // (2 * len(content)), False)
            for i, c in enumerate(content)
        ]
        people[str(row["speaker"])].extend(tokens)
    return people


def constrained_matches(reference, hypothesis, tolerance):
    """Find a minimum-edit match within the collar via a sparse weighted chain.

    Relative to deleting every reference and inserting every hypothesis,
    pairing two equal characters saves two edits, and substitution saves one.
    The Fenwick tree finds the best chain of strictly increasing token indices.
    Secondary ties prefer more exact matches, then smaller total time offsets.
    """
    states = [(0, 0, 0, -1, -1, -1)]
    tree = [0] * (len(hypothesis) + 1)

    def better(a, b):
        return a if states[a][:3] > states[b][:3] else b

    def prefix(end):
        best = 0
        while end > 0:
            best = better(best, tree[end])
            end &= end - 1
        return best

    ordered = sorted((item.tick, j) for j, item in enumerate(hypothesis))
    ticks = [x[0] for x in ordered]
    for i, item in enumerate(reference):
        lo = bisect_left(ticks, item.tick - tolerance)
        hi = bisect_right(ticks, item.tick + tolerance)
        pending = []
        for moment, j in ordered[lo:hi]:
            previous = prefix(j)
            score, exact, distance = states[previous][:3]
            equality = int(item.char == hypothesis[j].char)
            states.append((score + 1 + equality, exact + equality,
                           distance - abs(moment - item.tick), previous, i, j))
            pending.append((j + 1, len(states) - 1))
        # Delayed updates stop a single reference token from being paired twice.
        for col, index in pending:
            while col < len(tree):
                tree[col] = better(tree[col], index)
                col += col & -col

    output = []
    state = prefix(len(hypothesis))
    while state:
        _, _, _, previous, i, j = states[state]
        output.append((i, j))
        state = previous
    return reversed(output)


def timed_scores(reference, hypothesis, mapping, reference_intervals, tolerance, duration):
    ref, hyp = timed_characters(reference), timed_characters(hypothesis)
    starts = [a for a, _, _, _ in reference_intervals]
    counts = [len(people) for _, _, people, _ in reference_intervals]
    totals = defaultdict(Counter)

    def add(kind, token, is_reference):
        tick = token.tick
        occupancy = str(counts[bisect_right(starts, tick) - 1]) if 0 <= tick < duration else "out_of_bounds"
        for group in ("full", f"count_{occupancy}"):
            totals[group][kind] += 1
            if is_reference:
                totals[group]["reference_chars"] += 1
                totals[group]["supplied_reference_char_timestamps"] += int(token.supplied)

    for p, q in speaker_pairs(sorted(ref), sorted(hyp), mapping):
        rr, hh = ref.get(p, []), hyp.get(q, [])
        deleted = inserted = 0
        for i, j in constrained_matches(rr, hh, tolerance):
            for token in rr[deleted:i]:
                add("deletions", token, True)
            for token in hh[inserted:j]:
                add("insertions", token, False)
            add("correct" if rr[i].char == hh[j].char else "substitutions", rr[i], True)
            deleted, inserted = i + 1, j + 1
        for token in rr[deleted:]:
            add("deletions", token, True)
        for token in hh[inserted:]:
            add("insertions", token, False)
    return {group: char_report(counts) for group, counts in sorted(totals.items())}


def audit(
    reference_text, reference_activity, hypothesis_text, hypothesis_activity,
    duration, collars, speaker_map=None,
):
    duration = clock(duration)
    if duration <= 0:
        raise ValueError("Recording duration must be positive")
    ref_labels = sorted(set(speakers(reference_text)) | set(speakers(reference_activity)))
    hyp_labels = sorted(set(speakers(hypothesis_text or [])) | set(speakers(hypothesis_activity)))
    intervals = list(spans(reference_activity, hypothesis_activity, duration))
    textgrid_intervals = list(spans(reference_text, hypothesis_activity, duration))
    if speaker_map is None:
        mapping = maximize_mapping(intervals, ref_labels, hyp_labels)
    else:
        mapping = {str(hyp): str(ref) for hyp, ref in speaker_map.items()}
        if len(set(mapping.values())) != len(mapping) or not set(mapping.values()) <= set(ref_labels):
            raise ValueError("Fixed speaker map must assign distinct known reference speakers")
    result = dict(
        mapping_policy="fixed" if speaker_map is not None else "maximize_activity_overlap",
        activity_mapping=mapping,
        RTTM_activity=activity_scores(intervals, mapping),
        alternate_TextGrid_activity=activity_scores(textgrid_intervals, mapping),
        exact_three_detection=dict(
            RTTM=exact_three_detection(intervals, mapping),
            TextGrid=exact_three_detection(textgrid_intervals, mapping),
        ),
        exact_three_windows=dict(
            RTTM=exact_three_windows(intervals, mapping),
            TextGrid=exact_three_windows(textgrid_intervals, mapping),
        ),
    )
    if hypothesis_text is not None:
        result.update(
            cpCER=score_characters(reference_text, hypothesis_text),
            text_fixed_activity_mapping=score_characters(reference_text, hypothesis_text, mapping),
            timed={
                str(collar): timed_scores(
                    reference_text, hypothesis_text, mapping, intervals, clock(collar), duration
                )
                for collar in collars
            },
        )
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--collars", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    args = parser.parse_args()
    if any(not isfinite(c) or c < 0 for c in args.collars):
        parser.error("Expected finite, nonnegative collars")
    cases = json.loads(args.manifest.read_text(encoding="utf-8"))
    results = {}
    for case in cases:
        text_reference = Path(case["text_reference"] if "text_reference" in case else case["textgrid"])
        if text_reference.suffix.lower() == ".textgrid":
            reference_text = textgrid(text_reference)
        else:
            reference_text = json.loads(text_reference.read_text(encoding="utf-8"))
        reference_activity = rttm(Path(case["rttm"]))
        outputs = {}
        for name, item in case["predictions"].items():
            hypothesis_text = None
            if source := item.get("text"):
                hypothesis_text = [
                    json.loads(line) for line in Path(source).read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            hypothesis_activity = json.loads(Path(item["activity"]).read_text(encoding="utf-8"))["segments"]
            outputs[name] = audit(
                reference_text, reference_activity, hypothesis_text, hypothesis_activity,
                case["seconds"], args.collars, speaker_map=item.get("speaker_map"),
            )
        results[case["id"]] = outputs
    args.output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
