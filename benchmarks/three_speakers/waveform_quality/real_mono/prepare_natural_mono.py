"""Rebuild fixed physical mono meeting inputs and independent A/B/C examples from public audio."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import numpy as np
import soundfile as sf


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def verified_source(filename, source, cache):
    local = cache / filename
    if local.exists():
        if sha256(local) != source['sha256']:
            raise ValueError(f'Incorrect cache checksum for {filename}')
        return local
    with NamedTemporaryFile(dir=cache, delete=False) as stage:
        pending = Path(stage.name)
        try:
            with urlopen(source['url'], timeout=150) as response:
                shutil.copyfileobj(response, stage, length=1024*1024)
            stage.flush()
            if sha256(pending) != source['sha256']:
                raise ValueError(f'Unexpected source bytes for {filename}')
            pending.replace(local)
        finally:
            pending.unlink(missing_ok=True)
    return local


def read_clip(path, channel, start, samples, expected_channels):
    with sf.SoundFile(path) as source:
        if (source.samplerate, source.subtype, source.channels) != (16000, 'PCM_16', expected_channels):
            raise ValueError(f'Unexpected PCM format in {path.name}')
        if not 0 <= channel < source.channels or start < 0 or start+samples > source.frames:
            raise ValueError('Invalid requested source range')
        source.seek(start)
        audio = source.read(samples, dtype='int16', always_2d=True)
    return audio[:, channel]


def build(case, sources, cache, output, full):
    output.mkdir(parents=True, exist_ok=True)
    required = {interval['file'] for intervals in case['enrollment_ranges'].values() for interval in intervals}
    required.add(case['room_file'])
    verified = {name:verified_source(name,sources[name],cache) for name in sorted(required)}
    name = case['room_file']
    channel = case['source_channel_zero_based']
    mix = read_clip(verified[name], channel, case['clip_start_sample'], case['clip_samples'], sources[name]['channels'])
    def write(filename, audio):
        path = output/filename
        sf.write(path, audio, 16000, subtype='PCM_16')
        if sha256(path) != case['expected_wave_sha256'][filename]:
            raise ValueError(f'Rebuilt WAV checksum mismatch for {filename}')
    # Some upstream mono containers have additional RIFF chunks. Preserve their headers when using the complete recording.
    if sources[name]['channels'] == 1 and case['clip_start_sample'] == 0 and len(mix) == sf.info(verified[name]).frames:
        shutil.copyfile(verified[name], output/'mixture.wav')
        if sha256(output/'mixture.wav') != case['expected_wave_sha256']['mixture.wav']:
            raise ValueError('Unexpected mixture checksum')
    else:
        write('mixture.wav', mix)
    for label, intervals in case['enrollment_ranges'].items():
        clips = [read_clip(verified[v['file']], v['channel'], v['start'], v['samples'], sources[v['file']]['channels'])
                 for v in intervals]
        write(f'solo_{label}.wav', np.concatenate(clips))
    if full and sources[name]['channels'] > 1:
        with sf.SoundFile(verified[name]) as source, sf.SoundFile(
            output/'meeting_channel1.wav', mode='w', samplerate=16000, channels=1, subtype='PCM_16'
        ) as target:
            for block in source.blocks(blocksize=1<<18,dtype='int16',always_2d=True):
                target.write(block[:,channel])
    count = case['clip_samples']//160
    activity = np.zeros((3,count), dtype=bool)
    truth = case['evaluation_only']
    for i,label in enumerate('ABC'):
        for left,right in truth['ten_ms_active_ranges'][label]:
            activity[i,left:right] = True
    np.save(output/'score_activity_ticks.npy',activity,allow_pickle=False)
    (output/'reference_text.json').write_text(json.dumps({'fixed_text':truth['transcript']},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', help='Case name; repeat to select several')
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--full', action='store_true',help='Also export the complete, original AliMeeting room channel')
    args=parser.parse_args()
    data=json.loads(Path(__file__).with_name('natural_mono_cases.json').read_text(encoding='utf-8'))
    args.cache.mkdir(parents=True,exist_ok=True)
    for key in args.case or data['cases']:
        if key not in data['cases']:
            parser.error(f'Unknown case {key}. Choose from {", ".join(data["cases"])}')
        build(data['cases'][key], data['sources'], args.cache, args.out/key, args.full)


if __name__=='__main__':
    main()
