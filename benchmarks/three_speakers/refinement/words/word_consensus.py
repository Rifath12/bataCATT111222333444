"""Choose between aligned outputs from two Mandarin recognizers using development-only rules."""

import argparse
from itertools import product
import json
from pathlib import Path

from rapidfuzz.distance import LCSseq, Levenshtein
from recognize import conversations, normalized, textgrid, text_scores


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def pair(strong, other):
    def keyed(rows):
        return {(str(r['speaker']), round(r['start'],3), round(r['end'],3)): r for r in rows}
    a,b=keyed(strong),keyed(other)
    result=[]
    for key in sorted(a.keys() | b.keys(), key=lambda k:(k[1],k[0])):
        left=a.get(key)
        right=b.get(key)
        x=normalized(left['text']) if left else ''
        y=normalized(right['text']) if right else ''
        common=LCSseq.similarity(x,y)
        seconds=max(.05,key[2]-key[1])
        result.append((left,right,len(x),len(y),common,seconds))
    return result


def choose(pairs,settings):
    rescue, max_rate, truncate_ratio=settings
    chosen=[]
    for left,right,n,m,agreement,seconds in pairs:
        if m and (not n and m>=rescue and m/seconds<12
                  or n/seconds>max_rate and m<n and agreement>=.6*m
                  or n and m>=1.5*n and agreement>=.8*n and m/seconds<truncate_ratio):
            chosen.append(right)
        elif left:
            chosen.append(left)
    return chosen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--primary',type=Path,required=True)
    p.add_argument('--other',type=Path,required=True)
    p.add_argument('--reference',type=Path)
    p.add_argument('--settings',type=Path,required=True)
    p.add_argument('--fit',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    primary=read_lines(args.primary)
    pairs=pair(primary,read_lines(args.other))
    reference=None
    if args.reference:
        reference=textgrid(args.reference) if args.reference.suffix.lower()=='.textgrid' else json.loads(args.reference.read_text())
    if args.fit:
        baseline=text_scores(reference,primary)
        # Keep the evaluation speaker mapping fixed throughout development selection.
        mapping={r['system']:r['reference'] for r in baseline['speaker_scores']}
        truth=conversations(reference)
        best=(baseline['totals']['errors'],0,[9999,9999,0])
        for params in product([1,2,3,9999],[4,6,8,12,9999],[0,4,6,9]):
            hyp=choose(pairs,params)
            by_person=conversations(hyp)
            edits=sum(Levenshtein.distance(by_person.get(system,''),truth[real]) for system,real in mapping.items())
            changed=sum(a is not b for a,b in zip(hyp,primary))+abs(len(hyp)-len(primary))
            if (edits,changed)<best[:2]:
                best=(edits,changed,list(params))
        args.settings.write_text(json.dumps(best[2])+'\n')
    selection=json.loads(args.settings.read_text())
    rows=choose(pairs,selection)
    args.output.mkdir(exist_ok=True,parents=True)
    (args.output/'transcript.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
    if reference is not None:
        stats=text_scores(reference,rows)
        (args.output/'scores.json').write_text(json.dumps(stats,indent=2)+'\n')
        print(selection,stats['cpCER'],flush=True)


if __name__=='__main__':
    main()
