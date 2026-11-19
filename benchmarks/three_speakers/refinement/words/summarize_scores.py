"""Collect matched natural Mandarin ASR errors from the full file and excerpt trials."""

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def scores(row):
    d = row["totals"]
    return {
        k: d[k]
        for k in (
            "reference_chars",
            "substitutions",
            "deletions",
            "insertions",
            "errors",
        )
    } | {
        "cpCER": row["cpCER"],
        "inference_s": row.get("inference_s", row.get("asr_wall_s")),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--full", type=Path, required=True)
    p.add_argument("--excerpts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    all_scores = {
        "recognizer": "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
        "revision": "2365baeacb507f821a0c8120fcee3d484dba7a07",
        "model_sha256": "c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51",
        "tokens_sha256": "f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc",
        "complete_meeting": {
            "session": "R8008_M8013",
            "zipformer": scores(read(a.baseline)),
            "sensevoice": scores(read(a.full)),
        },
    }
    cases = []
    for case in read(a.excerpts / "cases.json"):
        folder = a.excerpts / case["session"]
        case["render"] = read(folder / "render" / "render.json")["method"]
        for name in ("zipformer", "sensevoice"):
            case[name] = scores(read(folder / name / "scores.json"))
        cases.append(case)
    all_scores["excerpts"] = cases
    all_scores["excerpt_totals"] = {}
    for name in ("zipformer", "sensevoice"):
        pooled = {
            k: sum(c[name][k] for c in cases)
            for k in (
                "reference_chars",
                "substitutions",
                "deletions",
                "insertions",
                "errors",
                "inference_s",
            )
        }
        pooled["cpCER"] = pooled["errors"] / pooled["reference_chars"]
        all_scores["excerpt_totals"][name] = pooled
    folder = a.excerpts / "R8003_M8001"
    if (folder / "refined" / "asr" / "scores.json").exists():
        all_scores["spectral_filter_exploratory"] = {
            "session": folder.name,
            "input_sec": read(folder / "refined" / "render.json")["sample_count"]
            / 16_000,
            "evaluated_with": "sensevoice",
            **{
                k: scores(read(folder / k / "asr" / "scores.json"))
                for k in ("sepformer", "refined")
            },
            "sepformer_windows": read(folder / "sepformer" / "render.json")[
                "sepformer_three_windows"
            ],
        }
    a.out.write_text(json.dumps(all_scores, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
