"""Compare waveform recovery in controlled three-speaker mixtures."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--enrollment", type=Path, required=True)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--target-model", type=Path, default=Path("spk_emb_100"))
    p.add_argument("--wesep", type=Path, default=Path("wesep-real-tse"))
    p.add_argument("--wespeaker", type=Path, default=Path("wespeaker"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=3)
    args = p.parse_args()
    sys.path.insert(0, str(args.project))
    from benchmark_examples import waveform_scores, refmask
    from process_meeting import Separation, load_separator, resample
    from target_extract import TargetExtractor

    torch.set_num_threads(args.threads)
    blinded = Separation(load_separator(args.models, "tuned", "cpu"), "cpu")
    enrolled = TargetExtractor(args.target_model, args.wesep, args.wespeaker)
    cases = []
    for directory in sorted(args.fixtures.glob("evaluation*_3voices")):
        fixture = json.loads((directory / "reference.json").read_text())
        mix, rate = sf.read(directory / "mixed.wav", dtype="float32")
        references = [
            sf.read(directory / f"reference_{letter}.wav", dtype="float32")[0]
            for letter in "ABC"
        ]
        embeddings = []
        for utterances in fixture["speakers"].values():
            name = utterances[0]["recording_id"]
            y, sr = sf.read(args.enrollment / f"{name}.wav", dtype="float32")
            voice = resample(y, sr, 16000)
            embeddings.append(np.pad(voice[:64000], (0, max(0, 64000 - len(voice)))))
        separate, _, _ = blinded.infer(mix, 3)
        extract = enrolled.extract(
            resample(mix, rate, 16000), enrolled.anchors(embeddings)
        )
        extract = resample(extract, 16000, rate)[:, : len(mix)]
        row = {
            "case": directory.name,
            "activity_all_three_s": float(
                (np.stack([refmask(s) for s in references]).sum(0) == 3).sum()
            )
            * 0.02,
        }
        for name, estimates in (("dprnn", separate), ("enrolled_bsrnn", extract)):
            row[name] = waveform_scores(references, dict(zip("ABC", estimates)), mix)
        print(row, flush=True)
        cases.append(row)
    totals = {
        name: float(
            np.median(
                [s["exact_overlap_improvement_dB"] for c in cases for s in c[name]]
            )
        )
        for name in ("dprnn", "enrolled_bsrnn")
    }
    result = {
        "diagnostic_with_known_count_and_separate_enrollment": True,
        "case_count": len(cases),
        "activity_all_three_s": sum(c["activity_all_three_s"] for c in cases),
        "median_three_way_overlap_si_sdr_improvement_db": totals,
        "cases": cases,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(totals, flush=True)


if __name__ == "__main__":
    main()
