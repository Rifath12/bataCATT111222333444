"""Score a physical mono meeting against independent speaker activity labels."""
import argparse
from contextlib import ExitStack
from itertools import permutations
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def verify_mixture(source, exported, playback_gain):
    # Probe distributed intervals at common sample-grid boundaries. Ignore filter
    # edge transients where a probe was resampled without its neighbouring audio.
    with sf.SoundFile(source) as native:
        divisor = math.gcd(native.samplerate, exported.samplerate)
        up, down = exported.samplerate//divisor, native.samplerate//divisor
        aligned_count = min(divisor//2, native.frames//down)
        if aligned_count == 0:
            raise ValueError('No complete common sample interval')
        latest = native.frames//down-aligned_count
        verified = 0
        for k in np.unique(np.linspace(0,latest,24,dtype=int)):
            native.seek(int(k)*down)
            expected = native.read(aligned_count*down, dtype='float32')
            if up != down:
                expected = resample_poly(expected,up,down)
            exported.seek(int(k)*up)
            actual = exported.read(len(expected),dtype='float64')
            interior = slice(24, -24) if len(expected)>96 else slice(None)
            difference = actual[interior]-expected[interior]*playback_gain
            if np.max(np.abs(difference),initial=0)>4/32768:
                raise ValueError('Exported mix does not match the supplied recording, gain or sample clock')
            verified += 1
    return verified


def read_levels(output_dir, source, first_frame):
    """Keep 100 ms RMS measurements in memory instead of all audio samples."""
    paths = sorted(output_dir.glob('Speaker_*.wav'))
    if [f.stem for f in paths[:3]] != ['Speaker_A', 'Speaker_B', 'Speaker_C']:
        raise ValueError('Expected Speaker_A.wav, Speaker_B.wav and Speaker_C.wav')
    with ExitStack() as opened:
        waves = [opened.enter_context(sf.SoundFile(f)) for f in [output_dir/'mixed.wav', *paths]]
        mix = waves[0]
        rate = mix.samplerate
        source_info = sf.info(source)
        if source_info.channels != 1 or abs(source_info.frames * rate / source_info.samplerate - mix.frames) > 1:
            raise ValueError('The scoring mixture and exported mixture have different sample clocks')
        if rate % 10 or any((f.samplerate, f.frames, f.channels) != (rate, mix.frames, 1) for f in waves):
            raise ValueError('All output WAVs must be mono, share a clock and span the whole recording')
        span = rate//10
        end_frame = mix.frames//span
        if not 0 <= first_frame < end_frame:
            raise ValueError('Requested scoring interval is empty')
        summary = json.loads((output_dir/'speakers.json').read_text(encoding='utf-8'))
        gain = float(summary['wav_playback_gain'])
        if not 0 < gain <= 1:
            raise ValueError('Invalid exported mixture gain')
        probes = verify_mixture(source,mix,gain)
        levels = np.empty((len(waves), end_frame-first_frame))
        for index, wav in enumerate(waves):
            wav.seek(first_frame * span)
            for start in range(0, levels.shape[1], 256):
                count = min(256, levels.shape[1] - start)
                rows = wav.read(count*span, dtype='float64').reshape(count, span)
                levels[index,start:start+count] = rows.std(axis=1)
    return levels, [f.stem for f in paths], source_info.duration, gain, probes


def full_activity(case_name, input_dir, cache):
    from prepare_natural_mono import verified_source

    manifest = json.loads(Path(__file__).with_name('natural_mono_cases.json').read_text(encoding='utf-8'))
    case = manifest['cases'][case_name]
    refs = case['label_sources']['labels']
    labels = case['clip_speakers_ABC']
    if labels != case['all_recording_speakers'] or case['source_channel_zero_based'] != 0:
        raise ValueError('Choose a case with exactly three annotated speakers in the full meeting')
    cache.mkdir(parents=True, exist_ok=True)
    source = input_dir/'meeting_channel1.wav'
    ticks = int(sf.info(source).duration*100)
    activity = np.zeros((3, ticks), dtype=bool)
    rttm = verified_source(case['room']+'.rttm', refs, cache)
    for raw in rttm.read_text(encoding='utf-8').splitlines():
        fields = raw.split()
        if len(fields) < 8 or fields[0] != 'SPEAKER' or fields[1] != case['room']:
            raise ValueError(f'Unexpected annotation row in {rttm.name}')
        start, duration = map(float, fields[3:5])
        index = labels.index(fields[7])
        activity[index,round(start*100):round((start+duration)*100)] = True
    return source, activity


def analyze(input_dir, output_dir, full_case=None, cache=None, exclude_start_fraction=0):
    if not 0 <= exclude_start_fraction < 1:
        raise ValueError('Start exclusion fraction must be in [0, 1)')
    if full_case is None:
        source = input_dir/'mixture.wav'
        activity = np.load(input_dir/'score_activity_ticks.npy', allow_pickle=False)
    else:
        if cache is None:
            raise ValueError('Full-meeting scoring needs a cache for the pinned RTTM')
        source, activity = full_activity(full_case, input_dir, cache)
    if activity.dtype != np.bool_ or activity.ndim != 2 or activity.shape[0] != 3:
        raise ValueError('Scoring requires three boolean 10 ms speaker activity tracks')
    first_frame = math.ceil(sf.info(source).duration*10*exclude_start_fraction)
    levels, labels, duration, gain, probes = read_levels(output_dir, source, first_frame)
    frames = levels.shape[1]
    end_tick = (first_frame+frames)*10
    if end_tick > activity.shape[1]:
        raise ValueError('Ground truth ends before the final complete scoring block')
    ticks = activity[:,first_frame*10:end_tick]
    reference = ticks.reshape(3,frames,10)
    every = reference.all(2)
    stable = (every | (~reference).all(2)).all(0)
    mixed = levels[0]
    outputs = levels[1:]
    eligible = stable & (mixed/gain>=.00178)
    solo = eligible & (every.sum(0)==1)
    double = eligible & (every.sum(0)==2)
    triple = eligible & (every.sum(0)==3)
    winner = np.argmax(outputs,axis=0)
    unique = (outputs==outputs.max(0)).sum(0)==1
    correct = np.array([[np.count_nonzero(solo & every[s] & unique & (winner==slot))
                         for slot in range(len(outputs))] for s in range(3)])
    relative_level = outputs/np.maximum(mixed,1e-15)

    def stats(order):
        who = {}
        wanted_energy = other_energy = 0.0
        for s,slot in enumerate(order):
            valid = solo & every[s]
            own = outputs[slot,valid]
            other = np.delete(outputs,slot,axis=0)[:,valid]
            wanted_energy += np.sum(own**2)
            other_energy += np.sum(other**2)
            def median_db(denominator):
                return float(np.median(20*np.log10((own+1e-12)/(denominator+1e-12)))) if own.size else None
            who['ABC'[s]] = dict(eligible_100ms_blocks=int(valid.sum()),
                largest_output_on_correct_track=int(correct[s,slot]),
                median_own_to_loudest_other_db=median_db(other.max(0)),
                median_target_rms_vs_mixture_db=median_db(mixed[valid]))
        floors = {}
        for limit in (12,18,24):
            hi_all = relative_level >= 10**(-limit/20)
            hi = hi_all[list(order)]
            phantom = [idx for idx in range(len(outputs)) if idx not in order]
            false_assigned = int((hi[:,solo] & ~every[:,solo]).sum())
            floors[str(limit)] = dict(
                triple_blocks_with_three_outputs_above_floor=int(hi[:,triple].all(0).sum()),
                double_blocks_with_both_active_above_floor=int((hi | ~every)[:,double].all(0).sum()),
                false_active_outputs_on_solo_blocks=false_assigned,
                false_active_outputs_including_extra_tracks_on_solo_blocks=false_assigned+int(hi_all[phantom][:,solo].sum()))
        return dict(assigned_outputs=[labels[i] for i in order], per_speaker=who,
            solo_blocks_correct_loudest=sum(int(correct[s,slot]) for s,slot in enumerate(order)),
            eligible_solo_blocks=int(solo.sum()),eligible_double_blocks=int(double.sum()),eligible_triple_blocks=int(triple.sum()),
            solo_intended_to_all_other_output_energy_db=float(10*np.log10((wanted_energy+1e-15)/(other_energy+1e-15))),
            levels_relative_to_mixed_input_db=floors)

    best = max(permutations(range(len(outputs)),3), key=lambda order:sum(correct[s,slot] for s,slot in enumerate(order)))
    return dict(clip=full_case or input_dir.name, method=(
        'Nonoverlapping 100 ms blocks with stable activity at every 10 ms tick and mixture AC RMS, before common WAV playback gain, at least '
        '0.00178 full scale. All exported tracks compete for solo loudest. Triple track audibility tests energy '
        'and can count bleed. False active outputs are counted among assigned tracks unless including_extra is specified.'),
        labels=labels, physical_output_order=stats((0,1,2)), diagnostic_best_solo_permutation=stats(best),
        duration_s=duration, verified_mixture_alignment_probes=probes, playback_gain=gain, scored_start_s=first_frame/10, scored_end_s=end_tick/100,
        scored_annotated_seconds_by_number_speaking=(np.bincount(ticks.sum(0),minlength=4)/100).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir',type=Path)
    parser.add_argument('output_dir',type=Path)
    parser.add_argument('report',type=Path)
    parser.add_argument('--full-case', choices=['M8014'], help='Score the original full room WAV using its pinned RTTM')
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--exclude-start-fraction', default=0.0,type=float)
    args = parser.parse_args()
    result = analyze(args.input_dir,args.output_dir,args.full_case,args.cache,args.exclude_start_fraction)
    args.report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    row = result['physical_output_order']
    print(f"{result['clip']} estimated correct-track loudest in {row['solo_blocks_correct_loudest']}/{row['eligible_solo_blocks']} annotated solo blocks")


if __name__=='__main__':
    main()
