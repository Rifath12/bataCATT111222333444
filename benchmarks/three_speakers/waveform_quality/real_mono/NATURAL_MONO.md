# Natural Mandarin meetings on one microphone

The four excerpts contain 113.67 seconds of naturally recorded conversation from three sessions. Three people speak within every excerpt. Their speech overlaps for 2 or 3 people at a time. The 96–98% target remains unverified.

| Excerpt | Original recording | Duration | Two active | Three active |
| --- | --- | ---: | ---: | ---: |
| M8014 | AliMeeting R8008, physical array microphone 1, original session has three participants | 60.00 s | 4.66 s | 1.15 s |
| M8009 | AliMeeting R8005, physical array microphone 1, fourth participant silent throughout excerpt | 20.00 s | 10.63 s | 4.25 s |
| REAL_T_01 | REAL-T EVAL2 Mandarin, H2 recording, session 260413-2 | 15.66 s | 8.30 s | 4.97 s |
| REAL_T_03 | Same REAL-T H2 recording, different time interval | 18.01 s | 4.89 s | 4.87 s |

AliMeeting selection covered the four three-person test meetings. These contained 5.37, 1.76, 1.30 and 1.13 seconds of annotated triple speech in total. We chose the 60 seconds with the most triple speech in M8014, then added a fourth-speaker-silent 20-second window in another meeting to cover more overlap. Among four initial REAL-T candidates, two supplied nonoverlapping, separately annotated, same-device example audio for all three people. This sample favors overlap and feasible enrollment. It does not estimate prevalence or accuracy in random meetings.

## Ordinary recording without enrollment

Use [the unaided test](UNAIDED_MONO.md) for a recording supplied without voice samples. It uses the shipping voiceprint embeddings. The comparisons below are a separate enrollment experiment that disables those embeddings.

## Enrolled-roster experiment

Each entry gives the number of correct or audible 100 ms blocks divided by the eligible blocks. All model inputs were the original mono recording and, where specified, separate A/B/C voice examples. The activity annotations and transcripts were supplied only to scoring. Legacy names were anonymous. Their fixed A/B/C scores therefore include arbitrary name swaps. The results JSON also includes an oracle diagnostic that renames tracks once per clip for the best solo agreement.

| Mode | Correct track loudest, one speaker | Both active tracks audible, two speakers | All three tracks audible, three speakers | Unintended audible track occurrences in solo blocks, including extra outputs |
| --- | ---: | ---: | ---: | ---: |
| Legacy, three sources requested | 234/557 (42.0%) | 99/237 (41.8%) | 26/134 (19.4%) | 778 |
| Enrolled, automatic local count | 348/557 (62.5%) | 15/237 (6.3%) | 9/134 (6.7%) | 313 |
| Enrolled, fixed three sources | 273/557 (49.0%) | 132/237 (55.7%) | 48/134 (35.8%) | 729 |

More energy on the expected tracks can also come from leakage. Read overlap audibility together with unintended solo-track activity and speaker-attributed transcription. A wrong person audible in an output can still pass this energy test. Natural room recordings have no isolated ground-truth source waveforms here. On M8014, 60 calibration probes per participant found median absolute correlation peaks of 0.33–0.40 between headsets and the room channel. Broadly varying lag and room transfer preclude treating headset PCM as sample-matched ground truth.

The separation application exported four tracks for every tested legacy excerpt and three for each enrolled run. The compact results JSON includes every physical track hash, per-participant assignment, separately identified anonymous-oracle mapping, 12/18/24 dB threshold sweeps, and CPU export times under concurrent workload. The tested commands use the tuned PyTorch model, one CPU thread and disabled ONNX embeddings. Timing includes setup and writing audio. It excludes ASR.

## Reproduce

Use Python 3.11+ with NumPy, SciPy and SoundFile plus the project’s inference dependencies and existing PyTorch weights. Run these examples from the folder containing this document. Set APPLICATION to the checkout’s process_meeting.py and MODELS to its weight directory. Omitting --case prepares all four excerpts. The AliMeeting files are approximately 0.95 GB combined. The prep verifies pinned source bytes and all four emitted WAVs per case.

```sh
python prepare_natural_mono.py --out data --cache cache --case M8014 --full
case_dir=data/M8014
python "$APPLICATION" "$case_dir/mixture.wav" --models "$MODELS" --threads 1 --no-embeddings \
  --participants 3 --speakers 3 --enroll "$case_dir/solo_A.wav" "$case_dir/solo_B.wav" "$case_dir/solo_C.wav" \
  --out runs/M8014_forced
python score_natural.py "$case_dir" runs/M8014_forced M8014_forced.json
```

Omit --speakers 3 to test automatic local count with the enrolled roster. Supply a fresh output folder for every run. To reproduce legacy, use process_meeting.py at commit fe21b33c9c065ee391ad4a7f5a6392784c3581c0 with --speakers 3 and without --participants or --enroll. The legacy names reflect their assignment order. Only the revised enrolled conditions use known A/B/C order.

For the original 32 minute 17 second three-person meeting, separate data/M8014/meeting_channel1.wav with the same enrollment. Run the command below on its full output. The held-out interval starts at 484.3 s. All enrollment examples end by 384.618 s. Scoring the original timeline avoids restarting speaker identities at the evaluation boundary.

```sh
python score_natural.py data/M8014 runs/M8014_full M8014_full.json \
  --full-case M8014 --cache cache --exclude-start-fraction 0.25
```

## Scoring and provenance

The scorer verifies 24 distributed sample-aligned probes against the source. It discards blocks that cross a speaker-activity transition or have mixture AC RMS below 0.00178 full scale before playback gain. On solo blocks the correct track must be strictly louder than every other output. During overlap, an expected track counts as audible if its RMS is within 18 dB of the mixture RMS or exceeds it, i.e. the ratio is at least 0.126. Unintended solo-track counts accumulate each additional track over that same floor. The scored block counts are smaller than annotated speech due to transitions and the input audibility floor. Named counts pool time and can favor the most talkative person. Inspect individual speakers and clips in natural_mono_results.json.

The manifest identifies original recordings, microphone, sample offsets, every enrollment range, human reference text, annotation sources, hashes and labeled time intervals. AliMeeting references use normalized manual TextGrid contents. Partial utterances at crop boundaries use timed character midpoints. All seven affected entire utterances exactly matched the corresponding normalized manual text. REAL-T supplies its own per-person transcripts and timings. AliMeeting enrollment uses prior same-device solo speech. REAL-T enrollment can fall later in the original session and represents an independently available voice example for this offline test.

Source recordings and derived transcript/activity labels in the manifest are provided under CC BY-SA 4.0. Attribution is to Fan Yu et al. for AliMeeting/M2MeT, and Shaole Li, Shuai Wang, Jiangyu Han, Ke Zhang, Wupeng Wang and Haizhou Li for their newly recorded REAL-T EVAL2 corpus. The manifest records the pinned audio, annotation, and time-alignment mirrors. Source audio is downloaded on demand.

- [AliMeeting, OpenSLR 119](https://openslr.org/119/)
- [REAL-T dataset card and EVAL2 provenance](https://huggingface.co/datasets/REAL-TSE/REAL-T)
- [AliMeeting timing annotations](https://github.com/nttcslab-sp/diar-forced-alignment)
- [MultiTalk character alignments](https://huggingface.co/datasets/MultiTalk/MultiTalkBench)
- [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
