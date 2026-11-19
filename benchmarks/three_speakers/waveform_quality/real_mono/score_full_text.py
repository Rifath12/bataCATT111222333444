"""Score a whole meeting's transcript after a fixed enrollment cutoff."""
import argparse
from itertools import permutations
import json
from pathlib import Path
import re

from opencc import OpenCC
from rapidfuzz.distance import Levenshtein

NORMALIZE=OpenCC("t2s")


def canonical(text):
    return re.sub(r"[^\u3400-\u9fffA-Za-z0-9]", "", NORMALIZE.convert(text).lower())


def evaluate(reference, predictions):
    target=json.loads(reference.read_text(encoding="utf-8"))
    cut=round(1000*target["scoring_start_s"])
    wanted={k:canonical(v) for k,v in target["fixed_text"].items()}
    actual={k:"" for k in wanted}
    boundary=[]
    for line in predictions.read_text(encoding="utf-8").splitlines():
        row=json.loads(line)
        if row["end_ms"]<=cut:
            continue
        key=row["speaker_id"].removeprefix("spk_")
        if row["start_ms"]>=cut:
            value=row["text"]
        elif row.get("characters"):
            value="".join(c["text"] for c in row["characters"] if (c["start_ms"]+c["end_ms"])/2>=cut)
            boundary.append({"label":key,"method":"character midpoint","kept":canonical(value)})
        else:
            value=row["text"] if (row["start_ms"]+row["end_ms"])/2>=cut else ""
            boundary.append({"label":key,"method":"segment midpoint","kept":canonical(value)})
        actual[key]+=canonical(value)
    def score(order):
        per={}
        for key,other in zip(wanted,order,strict=True):
            edits=Levenshtein.editops(wanted[key],actual[other])
            counts={op:0 for op in ("delete","insert","replace")}
            for edit in edits:counts[edit.tag]+=1
            per[key]={"reference_characters":len(wanted[key]),"edits":len(edits),**counts}
        total=sum(v["edits"] for v in per.values()); n=sum(map(len,wanted.values()))
        return {"edits":total,"reference_characters":n,"CER":total/n,"track_order":"".join(order),"per_speaker":per}
    fixed=score(wanted)
    best=min((score(p) for p in permutations(actual)),key=lambda d:d["edits"])
    return {"corpus":target["room"],"excluded_start_s":target["scoring_start_s"],
            "protocol":"Whole recording decoded once, then both measurement streams restricted to times after the fixed enrollment quarter. The per-speaker primary measure keeps enrolled A/B/C. Characters in straddling decoded segments use their CTC time midpoint when available.",
            "fixed":fixed,"one_global_permutation":best,"hypothesis_boundary_segments":boundary}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference",type=Path)
    parser.add_argument("transcript",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    result=evaluate(args.reference,args.transcript)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(result["fixed"]["edits"],result["fixed"]["reference_characters"],result["fixed"]["CER"])
