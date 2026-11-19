"""Verify cross-panel AliMeeting speaker and session independence."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--previous", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    test = json.loads(a.test.read_text())
    people = {sp for row in test for sp in row["session_speaker_ids"]}
    sessions = {row["session"] for row in test}
    prior, sharing = [], []
    for path in a.previous:
        value = json.loads(path.read_text())
        rows = value.get("clips", []) if isinstance(value, dict) else value
        for row in rows:
            session = row.get("source_session", row.get("session"))
            names = set(
                row.get(
                    "session_speaker_ids",
                    row.get("speaker_ids", row.get("speakers", [])),
                )
            )
            if session in sessions or (names & people):
                sharing.append((str(path), session, sorted(names & people)))
        prior.append(str(path))
    a.out.write_text(
        json.dumps(
            dict(
                test_meetings=sorted(sessions),
                test_speakers=sorted(people),
                prior_manifests=prior,
                conflicts=sharing,
            ),
            indent=2,
        )
        + "\n"
    )
    assert not sharing, sharing


if __name__ == "__main__":
    main()
