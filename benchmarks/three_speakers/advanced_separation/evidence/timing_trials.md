# Three-speaker timing experiments

Both candidates process single-channel audio and produce occupancy probabilities for zero through three active speakers. The existing speaker-identity model assigns anonymous labels A, B and C. Training and score calculation use time-aligned reference activity. Inference uses the waveform and cached model evidence derived from that waveform.

## Small mono counting model

The model learned from room-augmented mixtures of Mandarin utterances from 34 AISHELL-3 speakers and isolated near-microphone utterances from three people in AliMeeting training session M8013. The room simulator varied voice gain, onset, reverberation and background noise. The student learned to predict activity counts with a 100 Hz mel and linear-frequency feature stream. An eight-speaker AISHELL-3 development split selected the checkpoint. Its synthetic exact-three precision, recall and F1 were 55.98%, 75.41% and 64.26%.

The natural counting model incorporates the student's pooled output alongside the existing structured feature set. Six older AliMeeting recordings provided source-held-out calibration groups. The frozen setting came from those groups, then ran unchanged on the two natural evaluation panels. These panels were used in earlier evaluations of the reference system.

| Natural evaluation panel | Timing model | Exact-three precision | Exact-three recall | Exact-three F1 | Diarization error |
| --- | --- | ---: | ---: | ---: | ---: |
| AliMeeting, 18 excerpts | Structured baseline | 72.77% | 72.27% | 72.52% | 11.87% |
| AliMeeting, 18 excerpts | Mono count student | 73.32% | 71.20% | 72.25% | 11.96% |
| AISHELL-4, 12 excerpts | Structured baseline | 55.25% | 30.34% | 39.17% | 19.06% |
| AISHELL-4, 12 excerpts | Mono count student | 59.28% | 24.31% | 34.48% | 18.80% |

The count student increased precision on both panels and reduced recall on both. The structured timing model remains the default.

A separate six-mixture synthetic panel used nine speakers absent from student training and checkpoint selection. The standalone student's exact-three precision, recall and F1 were 55.93%, 82.30% and 66.60% on 16.50 seconds of labeled true triple overlap. The three dry and three room-simulated fixtures used concatenated readings. The approach retained substantial false triples on this additional controlled test.

## WavLM count head

The second model uses the DiariZen WavLM and Conformer representation before the existing classifier. The classifier's activity classes encode at most two simultaneous speakers. The frozen encoder's hidden output provides additional acoustic evidence for a separately trained zero-to-three counting head. This experiment used 60 excerpts from six earlier AliMeeting sessions and calibrated its weight with entire sessions held out. The held-out natural panels were evaluated with the same operating point.

| Natural evaluation panel | Exact-three precision | Exact-three recall | Exact-three F1 | Diarization error |
| --- | ---: | ---: | ---: | ---: |
| AliMeeting, 18 excerpts | 76.60% | 68.08% | 72.09% | 11.86% |
| AISHELL-4, 12 excerpts | 55.16% | 18.60% | 27.82% | 18.86% |

The hidden count increased AliMeeting precision by 3.83 percentage points and reduced its recall by 4.19 points. AISHELL-4 recall fell by 11.75 points. The baseline retained the higher exact-three F1 across both panels.

## Timing error diagnosis

The baseline usually mistakes three active speakers for two. In AliMeeting, it predicted two speakers for 93.7% of missed exact-three time. Within the 200 ms margins around reference transitions, it missed 6.49 seconds, accounting for 89.1% of all missed exact-three time in that panel. Away from those margins, exact-three recall was 91.7% in AliMeeting and 32.5% in AISHELL-4. Further timing work needs to address AISHELL-4's interior misses as well as AliMeeting's boundary errors. Nearly all false triple time in both panels occurred while two people were speaking.

## Reproduction

Run the commands from the workspace that contains the benchmark data and the existing structured timing code. The environment needs Python, Torch, NumPy, SciPy, SoundFile, scikit-learn and threadpoolctl. Training audio filenames, speaker IDs and SHA-256 digests are in `model/training_manifest.json`.

```bash
sh vnext_timing/run_fusion.sh \
  --holdout further_data/corpus/manifest.json further_data/corpus/probabilities further_recovery/dz_fresh \
  --holdout pareto_data/corpus/manifest.json pareto_data/corpus/probabilities pareto_data/corpus/dz_features
```

The frozen model reads `model/counter.pt` and `frozen.json`. Set `TIMING_PYTHON` to the existing project interpreter if required. The optional `--predict` argument takes the mono WAV, its cached diarization `.npz`, its DiariZen evidence `.npy`, and the output JSON path. `diagnose.py` produced the timing error measurements in `errors.json`.

`latent_features.py` produces the hidden feature arrays with the existing DiariZen checkpoint in `three_speaker_work/dz-model`. `latent_fusion.py` evaluates the frozen hidden count head in `latent_frozen.json`. `latent_results.json` contains the pooled and per-session metrics.

The student replay matched the original metrics byte for byte using the frozen tree file. Both frozen candidates emitted valid A/B/C timing JSON in prediction-only mode. `evaluate_synthetic.py` produced `untuned_synthetic.json` from `vnext_separator/fresh/cases.json`.
