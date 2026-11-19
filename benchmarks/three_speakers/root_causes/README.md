# Three-speaker Mandarin audio

This release repairs a failure that duplicated voices across mono tracks, adds an optional spatial separator for recordings made with a microphone array, and measures how much speech is lost at each stage. All examples are audio only. A, B and C are stable anonymous labels within a recording.

## Run on an audio file

Use a 16 kHz audio file and the existing benchmark checkpoints. The file runner detects whether the input contains one channel or a synchronized microphone array, estimates speaker timing, and produces `speaker_A.wav`, `speaker_B.wav`, `speaker_C.wav`, a timing JSON and one Mandarin transcript per person, plus timestamped JSONL. No reference transcript, reference timing or live microphone is needed. The WAVs retain silence outside the detected speaker turns. Real names require a separate mapping provided by the operator.

```sh
bench=benchmarks/three_speakers
python "$bench/root_causes/process_audio.py" meeting.wav \
  --out meeting_result \
  --native-library "$NEMO_RUNTIME/lib/libnemo_speech_asr_c.so" \
  --diar-model "$NEMOTRON_GGUF" --models "$DPRNN_MODELS" \
  --sepformer-three "$SEPFORMER3" --sensevoice "$SENSEVOICE" --threads 2
```

The directory identified by `DPRNN_MODELS` contains the previously used DPRNN and voice-embedding checkpoints. `SEPFORMER3` identifies the optional three-source checkpoint. `SENSEVOICE` contains `model.int8.onnx` and `tokens.txt`. The native Nemotron runtime and model infer the timing. Use `--timing timing.json` in place of those two options to reuse previously computed labels. For array input, the spatial renderer uses the original microphones, channel 1 for timing and predicted solo regions to estimate the direction of each voice. The array file needs at least 0.3 s of usable solo evidence for each voice after boundary erosion. File processing is offline and uses the same recording for speaker identity throughout. [Model setup and previous benchmark](../refinement/README.md)

On mono input the normal mode applies a soft overlap mask to temporary recognition tracks, then uses the longer context pass and preserves close agreement with the per-speaker reading on longer turns. Use `--fast-transcript` to skip this second pass. Use `--no-mask-for-asr` to recognize the directly separated mono tracks. The listening WAVs use the neural separator. Array input uses spatially filtered tracks and direct per-speaker recognition with duplicate removal. Its transcript degraded when the continuous mono reading was added. [Recognition procedure and measured variants](transcription/CONTEXT_ASR.md)

## Which change fixed which failure?

| Diagnosis | Intervention | Measured effect |
|---|---|---|
| Any participant with no 750 ms solo turn could cause the mono neural renderer to abort. The previous fallback copied the mixed waveform into every marked speaker track. | Match available speaker embeddings and use the time pattern of each separated voice to assign the remaining tracks. Record missing cues and ambiguous assignments. | On six previously untranscribed clips from four previously timed meetings, the number of exact duplicated 10 ms overlap pairs fell from 6,232 to zero. Character edits fell from 428 to 425 out of 1,135. Insertions fell from 80 to 20, but deletions rose from 116 to 239. Four clips improved, two regressed. |
| The calibrated mono timing decoder expanded speech at both ends of a turn, often creating false overlap. | Fit a narrower expansion on a separate meeting and retain the prior acoustic verifier for time frames proposed as three voices. Export this result separately for activity analysis. | Diarization error on two full meetings fell from 10.14% before the previous verifier, or 10.08% with it, to 7.65%. On three new speaker-disjoint excerpts it fell from 13.93% to 11.02%. Some real speech was cut earlier. |
| Short isolated clips discard linguistic context in Mandarin. Some text is copied to the quieter track during overlap. | Recognize longer stretches of the mixture, infer token ownership using the stems and detected activity, and check stem text where the two readings agree. | The same held-out mono tracks recognized with more context improved from 3,386 to 2,982 edits across 15,859 characters. Using the soft mask with that context reduced the total to 2,915. Adding a soft mask for the recognition input and the same context rules reduced the previous pipeline's 398 edits to 355 across 1,128 characters in three new speaker-disjoint excerpts. |
| Room recording quality, cross-talk and gated silence account for a substantial share of errors. | Where the original recording has an array, estimate speaker-specific spatial covariances and apply a Souden MVDR beamformer on the original channels before gating. | Keeping ASR, its timing and global assignment protocol fixed reduced edits from 2,188 to 1,504 on M8016 and from 1,198 to 626 on M8017. Independent solo-reference measurements verified extra rejection of competing sources across all six tracks. |

The mono timing change targets recordings similar to AliMeeting. It worsened timing on the original REAL-T short-clip panel. Tested narrower timestamps also sometimes removed speech needed for recognition. The full-file runner uses the wider ASR timing. The [timing exporter](timing/README.md) provides the separately validated activity output. The [sources and data preparation](data/PROTOCOL.md), [paired missing-cue scores](results/partial_enrollment_scores.json), [timing ablations](timing/summary.json) and [error attribution](diagnostics/DIAGNOSIS.md) give denominators and individual cases.

## What reached 90%?

| Full-file transcript method | M8016, 30:39, 7,705 characters | M8017, 30:20, 8,154 characters | Two recordings pooled |
|---|---:|---:|---:|
| Previous mono | 71.60% (2,188 edits) | 85.31% (1,198 edits) | 78.65% (3,386 edits) |
| Revised mono with continuous recognition and agreement checks | 74.43% (1,970 edits) | 87.59% (1,012 edits) | 81.20% (2,982 edits) |
| Revised mono with soft overlap mask for recognition | 74.99% (1,927 edits) | 87.88% (988 edits) | 81.62% (2,915 edits) |
| Eight synchronized room channels, direct recognition and duplicate removal | 80.49% (1,503 edits) | 92.36% (623 edits) | 86.59% (2,126 edits) |

The score above is 100 × (1 − speaker-attributed character error rate). For each complete meeting the scorer picks one mapping between system and reference speakers and totals substitutions, deletions and insertions. A score over 90% on a whole meeting can coexist with missed voices when all three people talk. Before the small transcript duplicate-removal step, on the eight-channel run for M8017 the individual character error rates were 6.85%, 4.84% and 13.95% for A, B and C. In the same direct test on M8016 they were 6.15%, 10.60% and 63.61%. The two meetings came from the same room series.

The two complete recordings total 60 min 59 s and contain only 2.43 s of labeled three-way speech. The best long-meeting activity profile detected all three for 1.62 s, giving 66.7% recall and 23.1% precision. In three fresh, overlap-enriched excerpts totaling 183.89 s and 4.10 s of annotated three-way overlap, precision was 45.9% and recall was 89.8%. A 620 ms overlap interval missed by the long-meeting system accounted for most of the missed three-way time. The listening package includes this interval at 29:09.57 in M8017.

On nine new, fully overlapping read sentences from three trios, experimental dereverberation raised median improvement in signal-to-distortion ratio during three-way activity from +2.76 to +3.72 dB over the mixture. SenseVoice edits dropped from 82 to 73 out of 118 characters and Zipformer edits from 76 to 66. Both recognizers recovered zero of the nine full phrases exactly. On a natural meeting excerpt the same filter increased edits from 91 to 94 out of 634, so it remains experimental. [Isolated, dry, reverberant and noisy controls](waveform/FINDINGS.md)

## Where the errors remain

On M8016, the same recognizer made 2,188 errors using the delivered mono stems and predicted timing. Giving it reference speaker turns reduced that to 1,996. Using individual headset channels with those same reference turns reduced the count further to 548. This last control requires audio from each participant and uses human annotations. Speaker C alone still had 364 errors out of 1,580 characters on their headset. A matched error audit found C's mono track had been muted entirely during 10.47% of their strong, overlapping speech frames.

With automatic timing, the same recognizer on the original microphone in each assigned window made 1,941 errors versus 2,188 on the separated audio. On the same human-annotated turns, transcribing the original mono microphone made 1,747 errors compared with 1,996 on the separated tracks. Hard separation lost speech useful for word recognition. Compared with the raw mono audio, the individually recorded headsets eliminated a further 1,199 edits. A stationary-noise filter on its own removed 23 edits on M8016 but added nine on M8017. A separate spatial activity model failed to recover additional correctly timed triple overlap, so the array renderer retains the original timing. [Full controls](diagnostics/summary.json)

All timing and processing times here were measured on shared CPU resources. File rendering and decoding can finish faster than the recording plays. The reported runs use whole-recording information and do not establish sustained live performance or commit latency.

[AliMeeting corpus and CC BY-SA 4.0 license](https://www.openslr.org/119/) · [Souden MVDR formulation](https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.SoudenMVDR.html)
