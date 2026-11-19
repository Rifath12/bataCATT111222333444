"""Apply the frozen local edit policy with an on-demand Zipformer vote."""

from collections import Counter
from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein

from recognize import normalized


@dataclass(frozen=True)
class Rule:
    similarity: float
    maximum_length: int
    needs_zipformer: bool = False
    repeat_only: bool = False


RULES = {
    "replace": Rule(0.45, 1),
    "insert": Rule(0.45, 3),
    "delete": Rule(0.85, 1, needs_zipformer=True, repeat_only=True),
}


def span_key(row):
    return str(row["speaker"]), round(row["start"], 3), round(row["end"], 3)


def align(anchor, view):
    """Project a hypothesized text into character slots and insertion boundaries."""
    mapped = list(anchor)
    additions = [""] * (len(anchor) + 1)
    for edit in Levenshtein.editops(anchor, view):
        position = edit.src_pos
        if edit.tag == "replace":
            mapped[position] = view[edit.dest_pos]
        elif edit.tag == "delete":
            mapped[position] = ""
        else:
            additions[position] += view[edit.dest_pos]
    return mapped, additions


def slot_votes(anchor, views):
    projected = {name: align(anchor, text) for name, text in views.items()}
    for position in range(len(anchor) + 1):
        yield (
            "",
            False,
            True,
            {name: tracks[1][position] for name, tracks in projected.items()},
        )
        if position < len(anchor):
            character = anchor[position]
            repeated = (position > 0 and anchor[position - 1] == character) or (
                position + 1 < len(anchor) and anchor[position + 1] == character
            )
            yield (
                character,
                repeated,
                False,
                {name: tracks[0][position] for name, tracks in projected.items()},
            )


def edit_rule(candidate, repeated, inserted):
    kind = "insert" if inserted else "replace" if candidate else "delete"
    rule = RULES[kind]
    if len(candidate) > rule.maximum_length or (rule.repeat_only and not repeated):
        return None
    return rule


def maximum_similarity(anchor_length, original, alternative):
    added = len(alternative) - len(original)
    changed = max(len(original), len(alternative))
    return 1 - changed / max(anchor_length, anchor_length + added, 1)


def potential_zipformer_edit(anchor, views):
    """Skip a turn only when even one additional ballot cannot change any slot."""
    if not anchor:
        # Any nonempty hypothesis has zero similarity to an empty anchor.
        return False
    for original, repeated, inserted, ballots in slot_votes(anchor, views):
        against = sum(word == original for word in ballots.values())
        alternatives = Counter(word for word in ballots.values() if word != original)
        for word, votes in alternatives.items():
            rule = edit_rule(word, repeated, inserted)
            if (
                rule
                and against < 1 + votes
                and maximum_similarity(len(anchor), original, word) >= rule.similarity
            ):
                return True
        if against == 0:
            # One independent ballot may offer a new character or delete this character.
            fresh_alternatives = ("字", "") if not inserted else ("字",)
            for word in fresh_alternatives:
                rule = edit_rule(word, repeated, inserted)
                if (
                    rule
                    and maximum_similarity(len(anchor), original, word)
                    >= rule.similarity
                ):
                    return True
    return False


@dataclass(frozen=True)
class Entry:
    record: dict
    anchor: str
    views: dict

    @property
    def key(self):
        return span_key(self.record)


def prepare(pairs, prior):
    indexed = {span_key(row): row for row in prior}
    entries = []
    for pair in pairs:
        row = indexed.get(span_key(pair))
        if row is None:
            row = {name: pair[name] for name in ("speaker", "start", "end")} | {
                "text": ""
            }
        views = {
            name: rendered
            for name, field in (("stem", "old"), ("context", "mix"))
            if (rendered := normalized(pair[field]))
        }
        entries.append(Entry(row, normalized(row["text"]), views))
    if not set(indexed).issubset({entry.key for entry in entries}):
        raise ValueError(
            "The refined transcript and recognizer use different turn boundaries"
        )
    return entries


def decoder_keys(entries):
    """Return turn keys eligible before running Zipformer. No annotations are read."""
    return {
        entry.key
        for entry in entries
        if potential_zipformer_edit(entry.anchor, entry.views)
    }


def slot_choice(original, repeated, inserted, ballots, similarity):
    against = sum(word == original for word in ballots.values())
    plans = []
    for candidate in sorted(set(ballots.values()) - {original}):
        supporters = [name for name, word in ballots.items() if word == candidate]
        kind = "insert" if inserted else "replace" if candidate else "delete"
        rule = RULES[kind]
        agreement = max(similarity[name] for name in supporters)
        if (
            len(supporters) > against
            and len(candidate) <= rule.maximum_length
            and agreement >= rule.similarity
            and (not rule.needs_zipformer or "zipformer" in supporters)
            and (not rule.repeat_only or repeated)
        ):
            plans.append((len(supporters), agreement, candidate, kind))
    if not plans:
        return original, None
    _, _, changed, kind = max(plans)
    return changed, kind


def apply(entries, zipformer):
    """Refine prior text. Unchanged turns retain original text and timestamps."""
    decoded = {span_key(row): normalized(row["text"]) for row in zipformer}
    rows, changes = [], Counter()
    for entry in entries:
        word = decoded.get(entry.key)
        views = entry.views | ({"zipformer": word} if word else {})
        similarities = {
            name: 1 - Levenshtein.normalized_distance(entry.anchor, text)
            for name, text in views.items()
        }
        revised = []
        changed = False
        for original, repeated, inserted, votes in slot_votes(entry.anchor, views):
            edit, kind = slot_choice(original, repeated, inserted, votes, similarities)
            revised.append(edit)
            if kind:
                changes[kind] += 1
                changed = True
        text = "".join(revised) if changed else entry.record["text"]
        if text:
            row = dict(entry.record, text=text)
            if changed:
                # Character times in a previous transcript do not describe newly emitted text.
                row.pop("characters", None)
            rows.append(row)
    return rows, dict(changes)
