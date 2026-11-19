# Ordinary mono recordings without voice examples

We ran both applications from their normal command line on two Mandarin recordings using the tuned PyTorch separator and the supplied Sherpa voiceprint model. The operator supplied one WAV and the number of participants. No microphone, per-person examples or scoring labels were supplied to the app. The two independent excerpts contain 35.66 s of conversation from two meetings, including 9.22 s of annotated three-person overlap.

| Excerpt / mode | Exported tracks | Best whole-clip naming on solo | Exactly one audible on solo | Exactly two on double | Exactly three on triple | CPU export time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AliMeeting M8009, legacy forced three | 4 | 28/37 | 8/37 | 26/87 | 13/33 | 39.35 s |
| AliMeeting M8009, revised automatic, three participants | 3 | 28/37 | 35/37 | 14/87 | 0/33 | 30.08 s |
| REAL-T H2 clip 01, legacy forced three | 4 | 11/20 | 4/20 | 30/80 | 11/45 | 26.73 s |
| REAL-T H2 clip 01, revised automatic, three participants | 3 | 20/20 | 20/20 | 0/80 | 0/45 | 21.39 s |

The table counts 100 ms blocks with stable annotations and audible input. An exported track counts as audible at or above −18 dB relative to the mixture. The exact-count columns include every emitted track and require no naming assumption. Across the two clips, the revised automatic mode preserved a single audible track on 55/57 solo blocks, up from 12/57. It produced exactly two on 14/167 double blocks, down from 56/167, and exactly three on 0/78 triple blocks, down from 24/78. Several outputs can contain the same voice, so passing a count check provides insufficient evidence of source purity.

Best naming uses a single assignment chosen after the run for the complete clip. The primary criterion is the loudest correct track on annotated solo blocks. Ties use balanced speaker-by-speaker activity agreement over solo and two-person periods. M8009 has solo speech for all three participants and both apps have one best mapping there. All eligible H2 solo blocks belong to one person. Two revised and six legacy mappings tie on those solo blocks, so the other identities remain ambiguous from that measurement. Unenrolled track letters identify anonymous session slots. Only a known voice reference or user-provided mapping can supply a person's actual name.

The two tested commands change both the count policy and the participant limit. These paired results describe the complete user-facing configurations. The available measurements do not establish whether an isolated routing change helps the unaided forced-three case. Each run used one CPU thread under shared load. The times exclude ASR. Result JSON contains output WAV hashes and the full distribution of detected audible tracks for every true talker count.

## Reproduce

Run in the folder containing the benchmark. Set APPLICATION to the revised checkout's process_meeting.py, LEGACY to process_meeting.py at commit fe21b33c9c065ee391ad4a7f5a6392784c3581c0 and MODELS to the project weights folder. The pinned SHA-256 of the voiceprint model is in unaided_results.json.

```sh
python prepare_natural_mono.py --cache cache --out data --case M8009 --case REAL_T_01
python "$APPLICATION" data/M8009/mixture.wav --participants 3 --threads 1 --models "$MODELS" --out runs/M8009_revised
python "$LEGACY" data/M8009/mixture.wav --speakers 3 --threads 1 --models "$MODELS" --out runs/M8009_legacy
python score_unaided.py data/M8009 runs/M8009_revised M8009_revised.json
python score_unaided.py data/M8009 runs/M8009_legacy M8009_legacy.json
```

Use fresh output directories on each run. Substitute REAL_T_01 for M8009 to reproduce the H2 comparison. The scorer verifies the original source audio against 24 distributed, sample-aligned pieces of each exported mixture.
