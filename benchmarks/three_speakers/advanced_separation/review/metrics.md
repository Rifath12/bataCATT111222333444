# Matched Mandarin evaluation audit

Two full AliMeeting Test recordings, M8016 and M8017, were rescored on their original sample clocks. The scripts evaluate prior mono, the mono character pass, prior array, the array context pass and direct microphones. The room-mono and array paths share the same diarization. Each direct microphone uses its own local timing detector.

## Results pooled across both meetings

| Output | cpCER edits / 15,859 characters | Time-constrained edits / 15,859 | Time-constrained edits / 37 approximate exact-three characters | Diarization error | Exact-three speaker recall | Exact three-speaker set |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous room mono | 2,743 | 3,506 | 31 | 10.14% | 88.89% | 66.67% |
| Revised room mono | 2,715 | 3,469 | 30 | 10.14% | 88.89% | 66.67% |
| Previous array | 2,126 | 2,840 | 19 | 10.14% | 88.89% | 66.67% |
| Revised array | 1,894 | 2,306 | 15 | 10.14% | 88.89% | 66.67% |
| Previous three close microphones | 1,505 | 2,142 | 6 | 14.60% | 83.51% | 65.76% |
| Adaptive three close microphones | 1,386 | 2,014 | 6 | 14.13% | 86.83% | 69.71% |
| Refined room-mono timing | | | | 8.78% | 88.89% | 66.67% |

The earlier cpCER numbers reproduced exactly. Matching anonymous labels once by diarization alone and reusing that mapping for the transcripts gave the same character edit counts in all ten cases. The three-close-microphone labels also matched the known A/B/C microphone order in both meetings. All speech counts, insertions and deletions contribute to the pooled scores.

The two source annotations give very different exact-three exposure. Forced-aligned RTTMs yield 1.30 and 1.13 seconds, while broad nonempty TextGrid turns yield 6.52 and 3.05 seconds. The prior close-microphone recall of 50.31% and 61.97% used the broader TextGrid turns. On the shared aligned reference, the same predictions get 81.67% and 85.63% speaker recall. Their exact three-label sets are correct for 60.00% and 72.39% of those two brief intervals. Under the broader definition, the shared mono/array paths and the close path have 31.14% and 23.59% pooled exact speaker set accuracy, respectively. The reference definition must accompany any reported activity score.

Adaptive close-mic processing increased whole-meeting speaker-attributed character accuracy from 90.51% to 91.26%. The two recordings separately went from 86.92% to 88.02% and from 93.90% to 94.32%. The adaptive detector raised strict exact-three recall and also raised false triple alarms. Its strict precision/F1 fell from 53.12%/58.77% to 34.83%/46.45% on the aligned reference, and from 75.07%/35.90% to 49.34%/33.25% on the broader TextGrid. The aligned exact-three character errors stayed at six. The adaptive thresholds were developed on the separate M8013 recording.

The refined mono boundaries were assessed without replacing the transcript. Aligned DER fell from 11.43% to 10.04% and 8.73% to 7.40%. The exact-three prediction was unchanged. Broad pooled TextGrid DER moved from 15.92% to 15.96%. With aligned RTTMs, the 2.43 seconds of triple reference speech span 12 windows. The broader TextGrid reports 9.57 seconds in 26 windows. The timed output includes the boundaries and exact-set fractions for every window.

The 37 exact-three characters are assigned by estimated midpoints in TextGrid turns. The underlying annotations contain no precise Mandarin character timestamps for this calculation. This is too little speech to validate a 90% target across sustained three-way conversations. These two meetings and outputs have been examined in several development rounds. Use a separately frozen final set before claiming generalization.

## Scoring details

- Activity uses half-open continuous intervals on a one-microsecond clock. It counts distinct active speakers, includes simultaneous speech and has no boundary collar. A single Hungarian assignment per recording maximizes total correctly attributed speaking time. The paired replays reuse the previous timing-derived map across every room-audio variant and use the known microphone-to-person assignment for the close-mic pair. Unmapped speaker labels still contribute errors. Intervals are clipped to the recorded timeline.
- Conventional cpCER independently minimizes total pooled edit count over one global text permutation. The fixed-map score reuses the activity or physical microphone mapping so that inconsistent text and audio labels remain visible.
- The time-constrained score permits a diagonal character match or substitution only when the two character midpoints differ by at most one second. The unchanged outputs are also scored with 0.5- and 2-second thresholds to expose sensitivity. Insertions are assigned to a group at the hypothesis timestamp. Deletions and substitutions are assigned at the reference timestamp. Where complete valid per-character timing is unavailable, a uniformly interpolated within-turn midpoint is used. This is a character-level diagnostic following the idea of temporal text alignment.
- The sparse matched-pair solver is equivalent to minimum Levenshtein edit count under the time constraint. Pairing identical characters saves two edits relative to deletion plus insertion. Substituting saves one. A prefix-max tree searches increasing pairs. Ties favor exact matches and then smaller elapsed offsets.

[AliMeeting reference corpus](https://www.openslr.org/119/) · [Contrasting TextGrid boundary precision with forced-aligned RTTMs](https://arxiv.org/abs/2507.09226) · [Pinned RTTM reference](https://github.com/nttcslab-sp/diar-forced-alignment/tree/9527b7c64846fb38316a610f32e9d3466bd6d8b7/AliMeeting/Test_Ali_far) · [MeetEval temporal matching method](https://www.isca-archive.org/chime_2023/neumann23_chime.html)

## Reproduce

Install NumPy, SciPy and RapidFuzz. From the audio workspace, point `manifest.json` at the existing reference TextGrid, aligned RTTM, decoded transcript JSONL and predicted speaker timing JSON. `seconds` is the WAV duration in samples divided by its sample rate. Predicted timing follows the existing JSON shape with a `segments` array. Entries use audio-relative `speaker`, `start` and `end`. Paths in the manifest are relative to the working directory. Supply a `speaker_map` for known assignments. Omit `text` for activity-only outputs.

```bash
python audit_metrics.py manifest.json matched_results.json
python compact.py matched_results.json matched_summary.json
python -m unittest discover -s path/to/audit
```

The tests compare the sparse solver to a dense dynamic program on 300 seeded random cases. Seven additional cases exercise overlapping intervals for one speaker, incorrect and missing labels, fixed physical assignments, exact overlap windows, loss of time alignment, empty references, a supplied single-character timestamp and escaped TextGrid text.
