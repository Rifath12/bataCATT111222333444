"""Evaluate the released close-microphone options on the three locked room manifests."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from boosted_close import BoostedMask
from close_refinement import load_parameters, refine
from close_waveform import FROZEN_PROFILE, stored_transfer, suppress
from waveform_acceptance import evaluate_case


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def condense(case):
    exact = case['voices']['exact_three']
    return {'pass96':case['passes_exact_three']['96'],'pass98':case['passes_exact_three']['98'],
            'quietest':case['quietest_exact_three_speaker'],
            'quiet_alignment':exact[case['quietest_exact_three_speaker']]['target_aligned_fraction'],
            'alignment':[item['target_aligned_fraction'] for item in exact.values()],
            'retention':[item['retained_active_fraction'] for item in exact.values()]}


def room_score(root, lock, cache):
    room=lock['room_group']; directory=root/room
    if checksum(directory/'manifest.json') != lock['manifest_sha256']:
        raise ValueError('Manifest checksum mismatch')
    manifest=json.loads((directory/'manifest.json').read_text()); calibration=stored_transfer(directory/'published_transfer.npy')
    params=load_parameters(); model=BoostedMask.load(); rows=[]
    if cache:
        (directory/'boosted').mkdir(exist_ok=True)
    for trial in manifest['trials']:
        path=directory/'held'/f"{trial['id']}.npz"
        if checksum(path) != trial['sha256']:
            raise ValueError(f'Fixture checksum mismatch in {trial["id"]}')
        with np.load(path,allow_pickle=False) as data:
            mix, reference, active, exact = [data[k] for k in ('mixture','references','activity','exact_three')]
        standard=suppress(mix,calibration,FROZEN_PROFILE)
        boosted=model.refine(mix,calibration,standard)
        if cache:
            np.save(directory/'boosted'/f"{trial['id']}.npy",boosted,allow_pickle=False)
        outputs={'raw':mix,'standard':standard,'gentle':refine(mix,standard,params),'boosted':boosted}
        row={'id':trial['id'],'level_db':min(trial['nominal_gains_db']),'room':room,'exact_three_s':float(exact.sum()/16000)}
        for name, output in outputs.items():
            row[name]=condense(evaluate_case(trial['id'],16000,reference,output,active,exact_three=exact,group=room))
        rows.append(row)
    (directory/'baseline_by_case.json').write_text(json.dumps(rows,indent=2)+'\n')
    return rows


def aggregate(rows):
    groups=defaultdict(list)
    for row in rows:
        for tag in ('all',row['room'],str(row['level_db']),f"{row['room']}_{row['level_db']}"):
            groups[tag].append(row)
    report={}
    for tag, group in groups.items():
        summary={'clips':len(group),'exact_three_s':sum(row['exact_three_s'] for row in group)}
        for arm in ('raw','standard','gentle','boosted'):
            rr=[row[arm] for row in group]; aligned=np.array([row['alignment'] for row in rr]); quiet=[row['quiet_alignment'] for row in rr]
            summary[arm]={'pass96':sum(row['pass96'] for row in rr),'pass98':sum(row['pass98'] for row in rr),
                          'worst_voice':float(aligned.min()),'median_quiet':float(np.median(quiet)),
                          'p10_quiet':float(np.quantile(quiet,.1)),'retention_fail_voices':int((np.array([row['retention'] for row in rr]) < .96).sum())}
        report[tag]=summary
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--room');p.add_argument('--summarize',action='store_true');p.add_argument('--cache-boosted',action='store_true');a=p.parse_args()
    rooms=json.loads((a.root/'lock.json').read_text())['rooms']
    if a.summarize:
        rows=[row for room in rooms for row in json.loads((a.root/room['room_group']/'baseline_by_case.json').read_text())]
        (a.root/'baseline_summary.json').write_text(json.dumps(aggregate(rows),indent=2)+'\n')
        print('Scored',len(rows),'fixed-label trials')
    else:
        for room in rooms:
            if a.room is None or a.room==room['room_group']:
                t=time.perf_counter();rows=room_score(a.root,room,a.cache_boosted)
                print('Saved',len(rows),'trials in',room['room_group'],round(time.perf_counter()-t,2),'s',flush=True)

if __name__=='__main__':main()
