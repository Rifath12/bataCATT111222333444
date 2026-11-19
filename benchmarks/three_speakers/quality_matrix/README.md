# Three-speaker mono quality matrix

The acceptance check evaluates each metric in its own direction. It records an improvement only when the paired score moves in the required direction. A safer default may improve one measure and preserve the others. Any regression or unverified measure prevents an all-metrics pass. The paired JSON and executable check report the individual decisions.

## Default speed and speaker-slot reliability

The sparse contribution mask produced byte-identical WAVs in both full recordings. Its median time fell from 13.14 to 3.31 seconds on M8016 and from 8.13 to 3.37 seconds on M8017. These times cover the mask stage and its file I/O. All 1,388 ASR input spans matched sample for sample with the previous whole-file reading path. The benchmark retained the one-item recognition batch after testing two and four. [Runtime tests](runtime.md)

The mono entry point also accepts a supplied timing file with explicit `speaker_ids`. An absent known speaker keeps an aligned silent audio track and an empty transcript. The frozen timing exporter includes the ordered list of A/B/C IDs.

## Optional extra recognition

The existing contextual and isolated SenseVoice streams can be reconciled with a Chinese Zipformer recognizer. The frozen rule permits only a small nearby edit backed by a strict vote. Before loading the extra model, the implementation identifies turns for which the Zipformer vote could change that decision. An alternate view that cannot alter the vote consumes no extra decoding. On M8017, decoding 196 of 533 turns added 84.53 seconds on one CPU thread, including model loading and text reconciliation. The pruned run reproduced the frozen transcript exactly. [CPU measurement](recognition/m8017_tight_latency.json) Use the same 16 kHz waveforms, the existing SenseVoice models, and the Chinese Zipformer CTC model directory containing `model.int8.onnx` and `tokens.txt`.

```sh
python benchmarks/three_speakers/root_causes/process_audio.py meeting.wav \
  --timing speaker_timing.json --sensevoice /models/sensevoice \
  --models /models/separation --sepformer-three /models/sepformer-three \
  --consensus-zipformer /models/zipformer-ctc-zh-int8 --out separated
```

| Metric on the same two complete meetings | Before | Optional consensus |
|---|---:|---:|
| Substitutions | 1,528 | 1,500 |
| Deletions | 1,042 | 1,030 |
| Insertions | 221 | 213 |
| Total edits, 15,859 reference characters | 2,791 | 2,743 |
| Speaker-attributed character score, 1 − character error rate | 82.40% | 82.70% |

The rule was calibrated on development material and evaluated on two complete recordings already used in earlier releases. Total edits fell for each A/B/C speaker in both meetings. On M8016, B gained one insertion and C gained one substitution while each of their total error counts fell. These individual regressions matter under the full matrix requirement. On two untouched AISHELL-4 transcription excerpts, consensus changed none of the contextual transcript text, so the result remained 68 edits in 147 characters. The consensus mode is opt-in. [Independent recognition check](recognition/aishell_consensus.json) [Full scores, individual speakers and candidate counts](recognition/scores.json)

## Timing on independent rooms

We froze a new AISHELL-4 timing validation set before inference. It consists of 12 excerpts from three previously unscored recordings, totaling 252.53 seconds with 18.39 seconds of true three-speaker activity. Each extracted excerpt contains three active reference identities. Scoring uses the same alignment, zero collar and one optimal speaker mapping per clip for each method.

| Timing arm | Three-speaker precision | Three-speaker recall | Exact-three F1 | Diarization error |
|---|---:|---:|---:|---:|
| Previous dictionary and narrow boundaries | 35.87% | 38.23% | 37.01% | 24.20% |
| Released structured native and acoustic evidence | 55.03% | 35.67% | 43.29% | 18.58% |
| Released structured model with DiariZen | 55.25% | 30.34% | 39.17% | 19.06% |
| New binary pitch fusion | 49.06% | 35.40% | 41.12% | 19.29% |
| New Sortformer pilot | 44.86% | 34.20% | 38.82% | 19.40% |

Neither candidate meets the timing acceptance check. On the separate 18-clip AliMeeting replay, both released detectors missed 4.40 seconds of true three-way overlap. Perfect arbitration restricted to their combined detections could recover at most 83.24% on that panel. An independent acoustic pass must recover speech missed by both for a further dependable gain. [Source manifest and hashes](data/manifest.json) · [Full timing comparisons](timing/metric_matrix.json) · [Source corpus](https://www.openslr.org/111/)

## Waveform separation

Tests of residual redistribution, leakage subtraction, harmonic suppression, and multi-view combination found gains in some controlled room cases and losses for the weakest or worst isolated voice. The production waveform model remains the evaluated baseline. The matched controlled metrics, including independent Mandarin speakers and exact-three intervals, are retained in the benchmark. [Controlled waveform scores](waveform/frozen_wave.json) · [Additional new trio](waveform/scores.json)

Run the paired scorecard with `python compare_metrics.py paired_metrics.json` from this directory. It keeps each comparison scoped to its test panel. The 90% target remains unmet for three people speaking simultaneously through one microphone.
