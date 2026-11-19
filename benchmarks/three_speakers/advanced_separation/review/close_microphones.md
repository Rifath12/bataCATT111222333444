# Dedicated microphone activity and transcription

The earlier activity gate normalized each channel to a loud-speech quantile,
required a Silero probability of 0.40 and compared the current channel with the
loudest other channel. Binary opening removed very short detections. On the
forced-aligned RTTM exact-three label in M8016, the old system missed 0.736
speaker-seconds at the detector's 32 ms frame resolution. Every one of those
misses passed the old amplitude ratio and fell below the Silero threshold.
The audit distinguishes energy failures, posterior failures, and temporal
filter failures in `diagnostic.json`.

The released optional profile estimates normalized power, a noise floor and a
three-by-three power leakage matrix from confidently dominant frames in the
input recording. A projected nonnegative solve estimates local energy on each
microphone. A 0.20 speech-probability threshold requires both absolute local
energy of at least 0.0009 and at least 0.25 of the microphone's observed power.
It bridges gaps of up to four 32 ms frames and expands the edge by three frames
in each direction. The profile does not need reference labels, enrollment or
transcripts during inference.

A grid of fixed-ratio and leakage-aware gates was compared on calibration
meeting R8008_M8013 only. Average full DER on the two annotation types, with a
fixed exact-three count F1 and recall objective, selected the adaptive gate.
The profile was frozen before M8016/M8017 scoring. The test runs kept the same
model, shared decoder configuration, sample clock, recording inputs and physical
speaker assignments. A separate gain-normalization probe on M8013 was excluded
from this version.

| Result | M8016 prior | M8016 adaptive | M8017 prior | M8017 adaptive |
| --- | ---: | ---: | ---: | ---: |
| Speaker-attributed character edits | 1,008/7,705 | 923/7,705 | 497/8,154 | 463/8,154 |
| One minus character edit rate | 86.92% | 88.02% | 93.90% | 94.32% |
| RTTM DER | 16.63% | 16.24% | 12.39% | 11.82% |
| RTTM exact-three speaker-time recall | 81.79% | 83.33% | 85.55% | 90.86% |
| RTTM all-three frame F1 | 47.85% | 41.65% | 75.23% | 51.76% |
| TextGrid DER | 18.58% | 17.48% | 14.69% | 13.76% |
| TextGrid exact-three speaker-time recall | 50.31% | 54.35% | 61.97% | 69.62% |
| TextGrid all-three frame F1 | 30.90% | 30.30% | 46.83% | 38.35% |

The transcript metric applies the physical A/B/C assignment before scoring each
speaker. The prior results also reproduce under the optimal global assignment.
Pooling the two recordings, the new system made 1,386 edits over 15,859 Chinese
reference characters, down from 1,505 edits. Deletions fell from 661 to 483.
Substitutions increased from 581 to 598 and insertions increased from 263 to 305.
This is 91.26% versus 90.51% by one minus the pooled character edit rate.

The TextGrid activity uses caption intervals. The RTTM activity supplies the
separate official overlap label and has 1.30 plus 1.13 seconds of exact-three
activity in the two files. The exact-three speaker-time recall is the correctly
reported fraction of reference person-time during those windows. It differs
from all-three detection recall, which counts a window only when all three
people are detected. The more sensitive gate produced more false triple alarms
and reduced F1 under both annotations. The legacy profile remains available
for conservative overlap alerts. These recordings give limited evidence about
sustained three-person overlap.

The M8016 recordings are 1,838.980125 seconds each, the M8017 recordings
1,820.052 seconds each, mono at 16 kHz on their source timeline. Gain normalization
and the frontend parameters match the released SenseVoice int8 recipe. ASR on
prepared timings took 162.09 and 141.66 seconds for the full recordings, with
two model CPU threads per process. The two jobs were concurrent on a shared
Linux x86-64 host. Cached activity postprocessing took approximately 0.1 second
per recording. The original one-thread Silero extraction took about 30 seconds
per file group. No live latency measurement was made in this run.

## Reproduce the optional profile

Use the existing close-microphone CLI with the new frozen profile and inputs
ordered by the person corresponding to A/B/C. Install the existing ONNX Runtime,
Sherpa ONNX, SoundFile, NumPy and SciPy dependencies. See `provenance.json` for
model and corpus checksums and package versions.

```bash
PYTHONPATH=. python advanced_separation/direct_microphones.py \
  close_A.wav close_B.wav close_C.wav \
  --vad /models/silero_vad.onnx --asr /models/sensevoice \
  --profile advanced_separation/direct_microphones_adaptive.json \
  --copy-audio --threads 2 --out improved
```

This profile changes when a voice is recognized. The waveform output names
track the three physical microphones and retain the PCM. The release CLI
rejects a repeated input inode, including aliases and hard links. Distinct
16 kHz mono WAVs must already share the same sample clock.

[AliMeeting corpus and attribution](https://www.openslr.org/119/)
