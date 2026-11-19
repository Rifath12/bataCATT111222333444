"""Build the native AliMeeting Mandarin transcript for the held-out part of M8014."""
import argparse
import json
from pathlib import Path
import re
import unicodedata

from prepare_natural_mono import verified_source


def normalize(text):
    marked = re.sub(r'<[^>]*>|\[[^\]]*\]', '', text)
    canonical = unicodedata.normalize('NFKC',marked)
    return ''.join(c.lower() for c in canonical if unicodedata.category(c)[0] in 'LN')


def parse_textgrid(path):
    content = path.read_text(encoding='utf-8-sig')
    output = {}
    for tier in re.split(r'(?m)^\s*item \[\d+\]:',content)[1:]:
        name = re.search(r'(?m)^\s*name = "([^"]+)"',tier)[1]
        turns = []
        for interval in re.split(r'(?m)^\s*intervals \[\d+\]:',tier)[1:]:
            text = re.search(r'(?ms)^\s*text = "(.*)"\s*$',interval)[1].replace('""','"')
            if normalize(text):
                start = float(re.search(r'(?m)^\s*xmin = ([.\d]+)',interval)[1])
                end = float(re.search(r'(?m)^\s*xmax = ([.\d]+)',interval)[1])
                turns.append((start,end,normalize(text)))
        output[name] = turns
    return output


def build(cache, start):
    manifest = json.loads(Path(__file__).with_name('natural_mono_cases.json').read_text(encoding='utf-8'))
    case = manifest['cases']['M8014']
    refs = case['label_sources']
    room = case['room']
    cache.mkdir(parents=True,exist_ok=True)
    manual = parse_textgrid(verified_source(room+'.TextGrid',refs['transcript'],cache))
    records, fixed, whole = {}, {}, {}
    for label,speaker in zip('ABC',case['clip_speakers_ABC'],strict=True):
        rows = []
        for a,b,text in manual[speaker]:
            if b <= start:
                continue
            partial = a < start
            if partial:
                alignment = verified_source(f'{room}_{speaker}.json',refs[f'timed_characters_{label}'],cache)
                words = json.loads(alignment.read_text(encoding='utf-8'))['word_alignments']
                selected = [(w['start']+w['end'])/2 for w in words if w['end']>w['start']]
                chars = [normalize(w['text']) for w in words if w['end']>w['start']]
                if ''.join(c for mid,c in zip(selected,chars) if a<=mid<b)!=text:
                    raise ValueError('Timed characters disagree with the manual boundary utterance')
                text = ''.join(c for mid,c in zip(selected,chars) if start<=mid<b)
            rows.append(dict(start_s=max(start,a),end_s=b,source_start_s=a,partial_at_start=partial,text=text))
        records[label] = rows
        fixed[label] = ''.join(r['text'] for r in rows)
        whole[label] = ''.join(r['text'] for r in rows if not r['partial_at_start'])
    return dict(room=room, scoring_start_s=start, speaker_ids=dict(zip('ABC',case['clip_speakers_ABC'])),
        method='Use normalized original manual TextGrid text for all whole utterances. For the boundary utterance, retain characters whose aligned midpoints are at or after the cut. The full timed turn must exactly match the manual text. all_whole_turns_text excludes the boundary utterance entirely.',
        fixed_text=fixed,all_whole_turns_text=whole,reference_segments=records,
        transcript_sha256=refs['transcript']['sha256'],
        normalized_character_counts={s:len(t) for s,t in fixed.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--start',type=float,default=484.3)
    args = parser.parse_args()
    result = build(args.cache,args.start)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(result['normalized_character_counts'])


if __name__=='__main__':
    main()
