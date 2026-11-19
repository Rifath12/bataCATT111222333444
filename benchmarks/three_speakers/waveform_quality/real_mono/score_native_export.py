"""Reproduce the product-recognizer character score from exported speaker WAVs."""
from argparse import ArgumentParser
from itertools import permutations
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
from rapidfuzz.distance import Levenshtein

from score_full_text import canonical

# Make the repository's normal Mandarin recognizer available from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from process_meeting import Mandarin, RATE


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path)
    parser.add_argument("export", type=Path)
    parser.add_argument("reference", type=Path, help="reference_text.json for the same excerpt")
    parser.add_argument("out", type=Path)
    args=parser.parse_args()
    decoder=Mandarin(args.models,1)
    info=json.loads((args.export/"speakers.json").read_text(encoding="utf-8"))
    ref=json.loads(args.reference.read_text(encoding="utf-8"))["fixed_text"]
    labels=tuple("ABC")
    reference={k:canonical(ref[k]) for k in labels}
    hypotheses={}
    for key,record in info["speakers"].items():
        samples,rate=sf.read(args.export/record["file"], dtype="float32")
        if samples.ndim!=1 or rate!=RATE or not np.isfinite(samples).all():
            raise ValueError(f"Invalid exported native waveform for {key}")
        rows=decoder.transcribe(key,record["display_name"],samples)
        hypotheses[key.removeprefix("spk_")]="".join(canonical(row["text"]) for row in rows)
    all_outputs=list(hypotheses)
    while len(all_outputs)<len(labels):
        all_outputs.append(f"missing_{len(all_outputs)}")
    def count(order):
        edits=sum(Levenshtein.distance(reference[k],hypotheses.get(p,"")) for k,p in zip(labels,order,strict=True))
        return edits+sum(len(text) for k,text in hypotheses.items() if k not in order)
    best=min(permutations(all_outputs,len(labels)),key=count)
    denominator=sum(map(len,reference.values()))
    if denominator==0:
        raise ValueError("Reference has no scorable characters")
    result={"reference_characters":denominator,"fixed_edits":count(labels),
            "best_global_edits":count(best),"best_global_order":best,"normalized_hypotheses":hypotheses}
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
