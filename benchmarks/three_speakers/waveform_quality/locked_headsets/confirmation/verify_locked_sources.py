"""Reconstruct the marked source images directly from PCM, independently of generation."""
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io.wavfile import read


def read_labels(path):
    labels = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        start = round(float(fields[3]) * 16000)
        stop = round((float(fields[3]) + float(fields[4])) * 16000)
        labels.setdefault(int(fields[7].removeprefix('N_SPK')), []).append((start, stop))
    return labels


def verify(root, locked):
    room = root / locked['room_group']
    manifest = room / 'manifest.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != locked['manifest_sha256']:
        raise ValueError(f'Manifest changed in {room.name}')
    meta=json.loads(manifest.read_text()); ids=meta['speaker_ids']
    pcm = [read(room/'audio'/f'{speaker}.wav',mmap=True)[1] for speaker in ids]
    labels = read_labels(room/'room.rttm'); spans=[]
    for trial in meta['trials']:
        path=room/'held'/f"{trial['id']}.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != trial['sha256']:
            raise ValueError(f'Fixture changed in {trial["id"]}')
        with np.load(path,allow_pickle=False) as data:
            images = data['source_images']; gains=data['nominal_gains_db']; activity=data['activity']
            if not (np.array_equal(images.sum(axis=0),data['mixture'])
                    and np.array_equal(images[np.arange(3),np.arange(3)],data['references'])
                    and np.array_equal(activity.all(axis=0),data['exact_three'])):
                raise ValueError(f'Incorrect channel sums, target mapping or activity in {trial["id"]}')
            for mic, start in enumerate(trial['source_starts_samples']):
                stop=start+trial['samples']; spans.append((start,stop))
                if start < meta['fit_max_samples'] + 3200 or any(
                    a < stop + 3199 and b > start - 3199 for sid, regions in labels.items()
                    if sid != ids[mic] for a,b in regions):
                    raise ValueError(f'Calibration or annotated speaker contamination in {trial["id"]}')
                actual=np.stack([x[start:stop] for x in pcm]) / 32768.
                rms=np.sqrt(np.mean(actual[mic,activity[mic]]**2))
                expected=actual * (.06*10**(gains[mic]/20) / rms)
                if not np.allclose(expected,images[mic],rtol=1e-6,atol=5e-8):
                    raise ValueError(f'PCM/source image discrepancy in {trial["id"]}')
    spans.sort()
    if any(end>start for (_,end),(start,_) in zip(spans[:-1],spans[1:],strict=True)):
        raise ValueError(f'Reused time intervals in {room.name}')
    return {'room':room.name,'cases':len(meta['trials']),'verified_nonreused_source_images':len(spans)}


def main():
    p=ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);root=p.parse_args().root
    result=[verify(root,entry) for entry in json.loads((root/'lock.json').read_text())['rooms']]
    (root/'verified_pcm.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)

if __name__=='__main__': main()
