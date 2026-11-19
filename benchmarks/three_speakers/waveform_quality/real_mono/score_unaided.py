"""Evaluate anonymous separated tracks using one global renaming and track-count diagnostics."""
from argparse import ArgumentParser
from itertools import permutations
import json
from pathlib import Path

import numpy as np

from score_natural import read_levels


def evaluate(input_dir, output_dir):
    levels, names, duration, gain, probes = read_levels(output_dir,input_dir/'mixture.wav',0)
    mixed = levels[0]
    outputs = levels[1:]
    ticks = np.load(input_dir/'score_activity_ticks.npy',allow_pickle=False)
    truth = ticks[:,:10*len(mixed)].reshape(3,len(mixed),10)
    every = truth.all(2)
    stable = (every | (~truth).all(2)).all(0)
    eligible = stable & (mixed/gain >= .00178)
    cardinality = every.sum(0)
    each = [eligible & (cardinality == count) for count in range(4)]
    winner = np.argmax(outputs,axis=0)
    unique = (outputs == outputs.max(0)).sum(0)==1
    solo_hits = np.array([[(each[1] & every[i] & unique & (winner==slot)).sum()
                            for slot in range(len(outputs))] for i in range(3)])
    high = outputs >= np.maximum(mixed,1e-15)[None,:]*10**(-18/20)
    predicted_count = high.sum(0)

    def rows(order):
        hits = sum(int(solo_hits[i,slot]) for i,slot in enumerate(order))
        active = high[list(order)]
        # Separate active/inactive terms for each speaker so talk time is not an implicit weight.
        # Exclude triples: those frames carry no information about identity swaps.
        contrasts = []
        for i in range(3):
            relevant = each[1] | each[2]
            on = relevant & every[i]
            off = relevant & ~every[i]
            if on.any():
                contrasts.append(active[i,on].mean())
            if off.any():
                contrasts.append((~active[i,off]).mean())
        balance = float(np.mean(contrasts)) if contrasts else 0.0
        return (hits,balance)

    candidates = [(order,rows(order)) for order in permutations(range(len(outputs)),3)]
    # Lexicographic enumeration breaks any remaining exact tie deterministically.
    best_order,best_values = max(candidates,key=lambda pair:pair[1])
    best = high[list(best_order)]
    case = dict(duration_s=duration, labels=names, aligned_waveform_probes=probes,
        fixed_global_order_using_scoring_labels=[names[i] for i in best_order],
        eligible_solo_per_physical_speaker=[int((each[1] & row).sum()) for row in every],
        global_permutations_tied_on_primary_solo_hits=sum(v[0]==best_values[0] for _,v in candidates),
        global_permutations_tied_after_activity_tiebreak=sum(v==best_values for _,v in candidates),
        solo_correct_loudest=best_values[0], eligible_solo=int(each[1].sum()),
        double_both_named_outputs_audible=int(((best | ~every).all(0) & each[2]).sum()), eligible_double=int(each[2].sum()),
        triple_three_named_outputs_audible=int((best.all(0) & each[3]).sum()), eligible_triple=int(each[3].sum()),
        count_of_eligible_blocks_by_actual_then_number_of_audible_outputs={
            str(n):np.bincount(predicted_count[mask],minlength=len(outputs)+1).tolist() for n,mask in enumerate(each)})
    return case


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('input_dir',type=Path)
    parser.add_argument('output_dir',type=Path)
    parser.add_argument('report',type=Path)
    args = parser.parse_args()
    args.report.write_text(json.dumps(evaluate(args.input_dir,args.output_dir),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
