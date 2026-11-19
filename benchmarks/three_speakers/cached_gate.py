"""Restore stem-supported words using existing cropped and contextual ASR views."""

import argparse
from collections import Counter
import json
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from contextual_asr import candidates, read_jsonl
from duplicate_leakage import apply, encounters
from recognize import normalized


OPTIONS = (
    "stem",
    "continuous",
    "continuous_with_stem_edges",
    "stem_with_continuous_edges",
)


def key(row):
    return str(row["speaker"]), round(row["start"], 5), round(row["end"], 5)


def extend_edges(text, supplement):
    """Prefix/suffix additions supported by a common anchor of at least two characters."""
    if len(text) < 2 or len(supplement) < 3:
        return text
    ops = Levenshtein.opcodes(text, supplement)
    if sum(x.src_end - x.src_start for x in ops if x.tag == "equal") < 2:
        return text
    add_start = ops[0].tag == "insert" and ops[0].dest_end <= 5
    add_end = ops[-1].tag == "insert" and ops[-1].dest_end - ops[-1].dest_start <= 5
    return (
        (supplement[: ops[0].dest_end] if add_start else "")
        + text
        + (supplement[ops[-1].dest_start :] if add_end else "")
    )


def build_options(row, released):
    a, b, c = normalized(row["old"]), normalized(row["mix"]), normalized(released)
    context_edges = extend_edges(b, a)
    stem_edges = extend_edges(a, b)
    options = {
        "stem": (a, row["old"]),
        "continuous": (b, row["mix"]),
        "continuous_with_stem_edges": (context_edges, context_edges),
        "stem_with_continuous_edges": (stem_edges, stem_edges),
    }
    seen = {c}
    for name in OPTIONS:
        clean, display = options[name]
        if clean in seen:
            continue
        seen.add(clean)
        yield name, clean, display


FEATURE_NAMES = (
    "duration",
    "fraction_overlap",
    "baseline_empty",
    "candidate_empty",
    "baseline_len",
    "candidate_len",
    "baseline_rate",
    "candidate_rate",
    "candidate_relative_gain",
    "difference_fraction",
    "replacement_fraction",
    "insertion_fraction",
    "deletion_fraction",
    "equal_fraction",
    "support_from_stem",
    "support_from_context",
    "option_stem",
    "option_continuous",
    "option_continuous_with_stem_edges",
    "option_stem_with_continuous_edges",
)


def feature_vector(row, base, alternative, name):
    stem, copied = normalized(row["old"]), normalized(row["mix"])
    sec = max(0.05, row["end"] - row["start"])
    n, m = len(base), len(alternative)
    scale = max(n, m, 1)
    ops = Counter(q.tag for q in Levenshtein.editops(base, alternative))
    replaced, inserted, deleted = (ops["replace"], ops["insert"], ops["delete"])
    return [
        sec,
        row["overlap"],
        not n,
        not m,
        n,
        m,
        n / sec,
        m / sec,
        (m - n) / scale,
        (replaced + inserted + deleted) / scale,
        replaced / scale,
        inserted / scale,
        deleted / scale,
        (n - replaced - deleted) / scale,
        1 - Levenshtein.normalized_distance(alternative, stem),
        1 - Levenshtein.normalized_distance(alternative, copied),
        *(name == choice for choice in OPTIONS),
    ]


def role(settings, data):
    while "feature" in settings:
        settings = (
            settings["left"]
            if data[settings["feature"]] <= settings["cut"]
            else settings["right"]
        )
    return settings["gain"]


def apply_cached_gate(pairs, prior, settings, tracks, duplicate_settings):
    """Apply the frozen selector to the already-suppressed baseline transcript."""
    base = {key(row): row["text"] for row in prior}
    outputs = []
    change_counts = Counter()
    for row in pairs:
        current = base.get(key(row), "")
        clean_base = normalized(current)
        choice, estimate = None, settings["margin"]
        for name, candidate, display in build_options(row, current):
            confidence = role(
                settings["tree"], feature_vector(row, clean_base, candidate, name)
            )
            if confidence > estimate:
                choice, estimate = (name, candidate, display), confidence
        if choice is not None:
            name, candidate, current = choice
            change_counts[name] += 1
        if current:
            outputs.append(
                dict(
                    speaker=row["speaker"],
                    start=row["start"],
                    end=row["end"],
                    text=current,
                )
            )
    # Do not rerun the release filter on unchanged rows. Only mark new quieter copies.
    findings = [
        r
        for r in encounters(outputs, tracks)
        if base.get(key(outputs[r[0]]), "") != outputs[r[0]]["text"]
    ]
    checked = apply(outputs, findings, duplicate_settings)
    change_counts["copies_rejected"] = len(outputs) - len(checked)
    return checked, dict(change_counts)


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def export(rows, output, speakers):
    output.mkdir(parents=True, exist_ok=True)
    (output / "transcript.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf8",
    )
    all_text = []
    per_speaker = {name: [] for name in speakers}
    for row in rows:
        speaker = row["speaker"]
        line = f"[{row['start']:.2f}–{row['end']:.2f}] {speaker}  {row['text']}\n"
        all_text.append(line)
        per_speaker[speaker].append(line)
    for name, lines in per_speaker.items():
        # The label originates in an audio filename and agrees with the diarization.
        (output / f"speaker_{name}.txt").write_text("".join(lines), encoding="utf8")
    (output / "transcript.txt").write_text("".join(all_text), encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("timing", "tracks", "cropped", "tokens", "baseline", "out"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("cached_gate_config.json"),
    )
    parser.add_argument("--duplicate-config", type=Path, required=True)
    args = parser.parse_args()
    timing = read_json(args.timing)["segments"]
    token_cache = read_json(args.tokens)
    tokens = token_cache["tokens"] if isinstance(token_cache, dict) else token_cache
    pairs = candidates(timing, args.tracks, read_jsonl(args.cropped), tokens)
    prior = read_jsonl(args.baseline)
    if not set(map(key, prior)).issubset(set(map(key, pairs))):
        parser.error("The baseline transcript must use the supplied speaker timing")
    rows, changes = apply_cached_gate(
        pairs,
        prior,
        read_json(args.config),
        args.tracks,
        read_json(args.duplicate_config),
    )
    export(rows, args.out, sorted({pair["speaker"] for pair in pairs}))
    print(changes)


if __name__ == "__main__":
    main()
