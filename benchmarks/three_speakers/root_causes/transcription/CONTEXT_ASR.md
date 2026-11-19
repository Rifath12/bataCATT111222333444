# Evaluating speaker-attributed Mandarin recognition

Three speaker files must share a 16 kHz mono sample timeline with the original mono recording. The experiments use the existing SenseVoice Small int8 recognizer in Mandarin mode and the original speaker timing predictions. The recognizer reads each separated waveform and writes transcript files.

The optional continuous pass decodes the mono meeting in roughly eight-second chunks, choosing a low-energy cut between seven and nine seconds after the preceding cut. Each side receives 700 ms of real audio context. Only token centers inside the central interval are retained. The recognizer supplies timestamped tokens. Candidate ownership uses the predicted activity, local energy on each separated track and proximity to a predicted speech interval (up to 400 ms). A development-only search tested overlap limits from 0% to 100% and short-span suppression cutoffs of 0, 0.4, 0.8, and 1.6 seconds. It selected the continuous hypothesis when words were allocated to the speaker, at any fraction of overlap. The stem hypothesis is retained where no continuous words were allocated, except for turns shorter than 0.4 seconds. This fallback can retain a second simultaneously speaking person. The meeting-level error improvement on development audio came with more deletions, so the evaluation reports edit types separately.

The restoration pass recovers a longer stem utterance when the continuous transcription matches a consecutive substring of the stem, with at most ten additional normalized characters and at least one quarter of the stem text covered. The long-turn pass keeps the stem on predicted turns longer than 4.27 seconds when the continuous alternative adds no more than about 1% characters. These decisions are calibrated using only M8013. The long-turn rule was obtained by training a shallow decision tree on per-utterance edit differences and selecting complexity by five contiguous time blocks. All its negative leaves reduce to that single condition.

The separate copy-suppression experiment removes one of two concurrently decoded identical or nested strings only if its own track is quieter where the outputs overlap. The three frozen thresholds are a minimum string length of two normalized characters, at least 3 dB difference, and overlap for at least a quarter of the weaker utterance. Genuine concurrent repetitions can satisfy those conditions, so per-clip and per-speaker results should be inspected.

## Inference

Within the repository, place the helper modules and frozen settings alongside the existing track recognizer. From that directory, run the following command with the three aligned tracks, the timing JSON, original mono WAV and model files. Existing recognition invocation stays available when no optional flag is supplied.

```sh
python recognize.py --audio path/to/stems --timing path/to/timing.json \
  --asr path/to/sensevoice --backend sensevoice --target-rms 0.035 \
  --context-mix path/to/mono.wav --restore-supported-text \
  --protect-long-turns --suppress-duplicates --threads 2 --out results
```

For the independent duplicate experiment, add only `--suppress-duplicates` to the original track-decoding command. The combined benchmark uses every flag shown above. A reference-free end-to-end run on the 42.85-second R8002_M8003 excerpt produced the same chronological transcript byte for byte as the benchmark replay.

## Evaluation protocol

The recordings come from the AliMeeting Mandarin corpus. R8008_M8013 is the development recording. Parameter selection minimizes character edits after a fixed baseline speaker permutation on that recording. R8008_M8016 is a historical evaluation meeting. R8008_M8017 is a new full meeting used for evaluation after the rules were fixed. The two evaluation meetings are from the same room series. The four historical excerpts and three fresh companion excerpts are reported separately from complete meetings.

The reference TextGrid is optional and used only for evaluation. cpCER is the total number of substitutions, insertions and deletions after one optimal whole-meeting permutation of speaker labels, divided by the reference character count. Case, punctuation and corpus event markup are normalized. The outputs contain per-speaker text, a chronological transcript, and timestamped JSONL. Segment start and end times describe predicted turns. Word boundary accuracy was not measured.

## Results

| Audio and evaluation group | Reference characters | Previous edits | Updated edits | Previous cpCER | Updated cpCER |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development M8013, same mono stems | 8,943 | 2,985 | 2,509 | 33.38% | 28.06% |
| Historical full M8016, same mono stems | 7,705 | 2,188 | 1,970 | 28.40% | 25.57% |
| Fresh full M8017, same mono stems | 8,154 | 1,198 | 1,012 | 14.69% | 12.41% |
| Pooled M8016 and M8017, same mono stems | 15,859 | 3,386 | 2,982 | 21.35% | 18.80% |
| Four historical excerpts, same mono stems | 1,492 | 368 | 368 | 24.66% | 24.66% |
| Three fresh excerpts, same mono stems | 1,128 | 398 | 374 | 35.28% | 33.16% |
| Same three fresh excerpts, repaired audio plus updated recognition | 1,128 | 398 | 355 | 35.28% | 31.47% |

The two complete meetings show 404 fewer edits with the same separator. M8016 reductions were concentrated in speakers A and B. The final system still made 1,101 errors on speaker C's 1,580 reference characters. Among four historical excerpts one regressed by six errors, two improved by three each and one tied. The additional mixture recognition pass took 340 seconds on M8016 with two CPU threads and 481 seconds on M8017 with one CPU thread, while other jobs shared the machine. These timings exclude diarization, source separation, and the initial per-track speech recognition.

On the array input, the independently decoded beam tracks were stronger than mono-context replacement. They scored 1,504 and 626 edits for M8016 and M8017 respectively. Passing those transcripts through just the quiet-copy filter removed one error on M8016 and three on M8017. Applying the full mono-context intervention to beam outputs raised the pooled error count from 2,130 to 2,561. The validated continuous-context configuration is for the single-channel speaker stems.


### Complete-meeting input and recognition comparisons

| Audio input and chosen transcription | M8016 edits / 7,705 | M8017 edits / 8,154 | Pooled cpCER |
| --- | ---: | ---: | ---: |
| Original mono stems, cropped recognition | 2,188 | 1,198 | 21.35% |
| Original mono stems, contextual recognition | 1,970 | 1,012 | 18.80% |
| Noise-processed mono stems, contextual recognition | 1,962 | 997 | 18.66% |
| Contribution-masked mono stems, cropped recognition | 1,970 | 1,100 | 19.36% |
| Contribution-masked mono stems, contextual recognition | 1,927 | 988 | 18.38% |
| Array beam tracks, cropped recognition and quiet-copy filter | 1,503 | 623 | 13.41% |
| Array beam tracks with first-channel contextual recognition and quiet-copy filter | 1,721 | 840 | 16.15% |

The contribution-mask row and the noise-processing row are distinct audio treatments. The contextual rows use the full recognition command shown above. The array quiet-copy row is reproduced by omitting all contextual options and keeping `--suppress-duplicates`. The two complete meetings share a room series, so the pooled result does not establish expected accuracy across other rooms, microphones or Mandarin speakers.
