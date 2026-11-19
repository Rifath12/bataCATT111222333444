"""Evaluate previously released timing choices on a locked panel."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from complete_timing import cached_inputs, correct_turns
from diarize import masks, reference
from frozen_dictionary import load_frozen
from improve_overlap import Decoder
from multiview_score import counts, measures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--probs", type=Path, required=True)
    p.add_argument("--dictionary", type=Path, required=True)
    p.add_argument("--wide", type=Path, required=True)
    p.add_argument("--narrow", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    source = json.loads(a.manifest.read_text())
    entries = source if isinstance(source, list) else source["clips"]
    model, thresholds, profile = load_frozen(a.dictionary)
    widen = json.loads(a.wide.read_text())
    narrow = json.loads(a.narrow.read_text())
    summary, detail = {}, {}
    for row in entries:
        audio = a.manifest.parent / row["audio"]
        ref = a.manifest.parent / (row.get("ref") or row["reference"])
        clip, first, alternate, duration = cached_inputs(
            audio, a.probs / f"{audio.stem}.npz", profile
        )
        dictionary, fixed = correct_turns(
            clip, first, alternate, audio, model, thresholds, narrow
        )
        wide = Decoder(first).decode(widen)
        truth = masks(reference(ref), duration)[1].T[: len(first)]
        clip = replace(clip, actual=truth)
        detail[clip.name] = {}
        for method, settings in [
            ("previous", clip.previous),
            ("dictionary", dictionary),
            ("dictionary_narrow", fixed),
            ("wide_real_t", wide),
        ]:
            score = counts(clip, settings)
            summary[method] = summary.get(method, np.zeros_like(score)) + score
            detail[clip.name][method] = score.tolist()
    out = {
        "pooled": {k: measures(s) for k, s in summary.items()},
        "counts_per_clip": detail,
    }
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(out["pooled"], flush=True)


if __name__ == "__main__":
    with threadpool_limits(1):
        main()
