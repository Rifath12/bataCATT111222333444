"""Time the optional additional CTC and check full-meeting parity on the archived test."""

import argparse
from pathlib import Path
from time import perf_counter
import json

from character_consensus import apply, decoder_keys, prepare
from verify_consensus import read, load
from contextual_asr import read_jsonl
from recognize import transcribe


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="M8017")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--full-then-selective", action="store_true")
    a = p.parse_args()
    case, old, pairs, _ = next(
        load(c for c in read("further_recognition/cases.json") if c["id"] == a.case)
    )
    paths = {
        f.stem.split("_")[-1]: f for f in Path(case["tracks"]).glob("speaker_*.wav")
    }
    timing = read(case["timing"])["segments"]
    started = perf_counter()
    entries = prepare(pairs, old)
    selected = decoder_keys(entries)
    select_time = perf_counter() - started
    modes = (False, True) if a.full_then_selective else (True,)
    reference = read_jsonl(Path(f"pareto_text/production/{a.case}/transcript.jsonl"))
    reports = {}
    for selective in modes:
        started = perf_counter()
        rows, model_time = transcribe(
            paths,
            timing,
            a.model / "model.int8.onnx",
            a.model / "tokens.txt",
            threads=1,
            target_rms=0.035,
            backend="zipformer",
            selected_keys=selected if selective else None,
        )
        prediction, changes = apply(entries, rows)
        assert prediction == reference
        info = dict(
            elapsed_seconds=perf_counter() - started,
            decoder_seconds=model_time,
            selected_turns=len(selected) if selective else len(entries),
            all_turns=len(entries),
            preprocessing_seconds=select_time,
            changes=changes,
        )
        reports["selective" if selective else "full"] = info
        a.out.write_text(json.dumps(reports, indent=2) + "\n")
        print(info, flush=True)


if __name__ == "__main__":
    main()
