"""Build disjoint, fixed-identity three-speaker remixes from pinned AliMeeting headsets.

Place sources.json and its downloaded files under --root. Install NumPy and SciPy.
Optional --prepare-power-transfer also needs advanced_separation on PYTHONPATH.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

RATE = 16000
TICK = 160
FRAME = 512
TAKE = 32 * FRAME
MARGIN = 3200
LEVELS_DB = (0, -6, -12, -20)
MAX_TRIALS = 84


def checksum(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_activity(path, group, samples):
    rows = [line.split() for line in path.read_text().splitlines()]
    names = sorted({int(row[7].removeprefix('N_SPK')) for row in rows})
    activity = np.zeros((len(names), (samples + TICK - 1) // TICK), dtype=bool)
    for row in rows:
        if row[0] != 'SPEAKER' or row[1] != group:
            raise ValueError(f'Unexpected label line: {row}')
        start = round(float(row[3]) * 100)
        stop = round((float(row[3]) + float(row[4])) * 100)
        activity[names.index(int(row[7].removeprefix('N_SPK'))), max(start, 0):stop] = True
    return activity, names


def single_speaker_intervals(activity, index, samples):
    alone = activity[index] & (activity.sum(axis=0) == 1)
    ends = np.flatnonzero(np.diff(np.r_[False, alone, False])) * TICK
    return [(left + MARGIN, min(right, samples) - MARGIN)
            for left, right in zip(ends[::2], ends[1::2], strict=True) if right - left > 2 * MARGIN]


def donor_at(audio, start, target):
    image = np.stack([channel[start:start + TAKE] for channel in audio]).astype(np.float32) / 32768
    rms = np.sqrt(np.mean(image[target].reshape(-1, FRAME)**2, axis=1))
    active = np.repeat(rms >= max(.0015, .16 * np.quantile(rms, .70)), FRAME)
    if active.mean() < .70:
        return None
    direct = np.sqrt(np.mean(image[target, active]**2))
    channel_rms = np.sqrt(np.mean(image**2, axis=1))
    if direct < .002 or np.max(np.delete(channel_rms, target)) > .70 * direct:
        return None
    return image * (.06 / direct), active


def prepare_power_transfer(paths, audio, intervals, cutoff, directory):
    # The reference annotation serves only as a selector for early single-speaker calibration.
    from close_waveform import FROZEN_PROFILE, fit_power_transfer
    steps = cutoff // FRAME
    power = np.stack([np.mean((channel[:steps * FRAME].reshape(steps, FRAME).astype(np.float32) / 32768)**2,
                              axis=1) for channel in audio])
    solo = np.zeros_like(power)
    for target, regions in enumerate(intervals):
        for start, stop in regions:
            first = max(1, (start + FRAME - 1) // FRAME)
            last = min(steps, stop // FRAME)
            if first < last:
                solo[target, first:last] = 1
    percentile = FROZEN_PROFILE.spectral_transfer_percentile
    fitted, counts = fit_power_transfer(paths, power, solo, fit_fraction=1., percentiles=(percentile,))
    if any(row['selected_dominant'] == 0 for row in counts):
        raise ValueError('Missing calibration voice')
    np.save(directory / 'published_transfer.npy', fitted[percentile], allow_pickle=False)
    (directory / 'calibration.json').write_text(json.dumps({'maximum_source_sample': int(steps*FRAME),
        'method':'early annotated single-speaker intervals, 200 ms boundary trim, frozen power estimator',
        'details':counts,'transfer_sha256':checksum(directory/'published_transfer.npy')}, indent=2) + '\n')


def generate(directory, room, speaker_ids, entries, prepare_transfer):
    files = [directory / 'audio' / f'{speaker}.wav' for speaker in speaker_ids]
    annotations = directory / 'room.rttm'
    audio = []
    for path in files:
        rate, samples = wavfile.read(path, mmap=True)
        if rate != RATE or samples.ndim != 1 or samples.dtype != np.int16:
            raise ValueError(f'Expected 16 kHz mono PCM16 in {path}')
        audio.append(samples)
    if len({len(stream) for stream in audio}) != 1:
        raise ValueError(f'Track lengths differ in {room}')
    length = len(audio[0]); cutoff = int(length * .45)
    activity, names = read_activity(annotations, room, length)
    if len(names) != 4 or not set(speaker_ids).issubset(names):
        raise ValueError('Unexpected meeting participants')
    intervals = [single_speaker_intervals(activity, names.index(s), length) for s in speaker_ids]
    if prepare_transfer:
        prepare_power_transfer(files, audio, intervals, cutoff, directory)
    banks = []
    for target, regions in enumerate(intervals):
        bank = []
        for lo, hi in regions:
            for start in range(max(lo,cutoff + MARGIN), hi - TAKE + 1, TAKE):
                if donor_at(audio, start, target) is not None:
                    bank.append(start)
        banks.append(bank)
    rng = np.random.default_rng(61000 + int(room[-4:]))
    for bank in banks:
        rng.shuffle(bank)
    trials = []; all_spans = []
    held = directory / 'held'; held.mkdir(exist_ok=True)
    for i in range(min(MAX_TRIALS, min(map(len, banks)))):
        starts = [bank[i] for bank in banks]
        quiet = (i // 4 + i % 4) % 3; gap = LEVELS_DB[i % 4]
        gains = np.zeros(3); gains[quiet] = gap
        sources, masks = zip(*(donor_at(audio, start, k) for k, start in enumerate(starts)), strict=True)
        active = np.stack(masks); exact = active.all(axis=0)
        images = (np.stack(sources) * 10**(gains[:,None,None] / 20)).astype(np.float32)
        mixture = images.sum(axis=0); references = images[np.arange(3), np.arange(3)]
        if exact.sum() < .1 * RATE or np.max(np.abs(mixture)) >= 1:
            raise ValueError(f'Failed overlap/headroom expectation in {room} case {i}')
        case = f'{room.lower()}_{i+1:03d}'
        path = held / f'{case}.npz'
        np.savez_compressed(path,sample_rate=RATE,mixture=mixture,source_images=images,references=references,
                            activity=active,exact_three=exact,nominal_gains_db=gains)
        # Deterministic fingerprint independent of ZIP compression or metadata.
        values = (mixture,images,references,active,exact,gains)
        sha = hashlib.sha256()
        for value in values:
            sha.update(np.ascontiguousarray(value).tobytes())
        trials.append({'id':case,'source_starts_samples':starts,'samples':TAKE,'exact_three_samples':int(exact.sum()),
                       'nominal_gains_db':gains.tolist(),'quiet_speaker':'ABC'[quiet] if gap else None,
                       'payload_sha256':sha.hexdigest(),'sha256':checksum(path)})
        all_spans.extend((start,start + TAKE) for start in starts)
    spans = sorted(all_spans)
    if not trials or any(b > c for (_, b), (c, _) in zip(spans[:-1],spans[1:],strict=True)):
        raise ValueError('Expected nonempty disjoint benchmark')
    manifest = {'room_group':room,'speaker_ids':speaker_ids,'sample_rate':RATE,'all_annotated_speakers':names,
                'fit_max_samples':cutoff,'fit_fraction':.45,'recording_samples':length,
                'source_sha256':{item['file']:item['sha256'] for item in entries if item['file'].startswith(room+'/')},
                'donor_eligible_counts':list(map(len,banks)),'case_count':len(trials),'duration_exact_three_s':
                sum(case['exact_three_samples'] for case in trials)/RATE,'seed':61000 + int(room[-4:]),'trials':trials}
    (directory / 'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return {k:manifest[k] for k in ('room_group','case_count','donor_eligible_counts','duration_exact_three_s')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--prepare-power-transfer',action='store_true')
    args=p.parse_args(); source = json.loads((args.root/'sources.json').read_text()); locks=[]
    for entry in source['sources']:
        if checksum(args.root / entry['file']) != entry['sha256']:
            raise ValueError(f"Unexpected bytes in {entry['file']}")
    for room, speaker_ids in source['rooms'].items():
        data = generate(args.root/room,room,speaker_ids,source['sources'],args.prepare_power_transfer)
        data['manifest_sha256'] = checksum(args.root/room/'manifest.json'); locks.append(data); print(data,flush=True)
    (args.root/'lock.json').write_text(json.dumps({'generator_sha256':checksum(Path(__file__)),
        'source_list_sha256':checksum(args.root/'sources.json'),'rooms':locks},indent=2)+'\n')

if __name__ == '__main__':
    main()
