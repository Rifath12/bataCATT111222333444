# Structured three-speaker timing

The decoder estimates an active count and three speaker-specific likelihoods every 20 ms. The count has four classes. The speaker-specific model uses the same shared parameters for all three anonymous slots. A dynamic program selects a continuous sequence among the eight possible subsets of A, B and C. It adds a penalty when voice activity switches and a larger penalty when two or more speakers change together. The evaluation also includes a paired independently selected per-frame comparator.

Both variants use the two existing Nemotron mono views, preserving the anonymous slot order after posterior correlation alignment. The independent variant uses a pretrained DiariZen pass as additional acoustic evidence. The tested DiariZen checkpoint directly represents a maximum of two simultaneous speakers, so its triple-overlap contribution comes from its pair likelihoods and ambiguous evidence across local slots. It never supplies reference speakers. We combine the model features with short-time level, spectral energy, periodicity and spectral flux. The existing acoustic calculation runs in bounded 5.12-second blocks and is identical on a sample-by-sample comparison of two control excerpts.

The REAL-T development set is the same 45 annotated excerpts, with no shared speakers or source sessions with the 26 REAL-T held-out excerpts. Three folds group whole source sessions. Per-frame and structured operating points are selected from their out-of-fold predictions. The structured operating point is primary because it achieved the higher out-of-fold objective. When transferring this model to AliMeeting, precision and overall timing error both worsen. REAL-T development trims all reference silence. The two Ali development excerpt panels used for matched training total about 50 minutes, and 306.65 seconds contain no active reference speaker. The Ali profile uses three folds grouped by six source sessions. It trains on one in every three 20 ms feature rows and retains all rows at inference.

## Measured timing

The fresh AliMeeting comparison contains 18 previously unevaluated excerpts from three other source meetings. They total 424.97 seconds, including 26.25 seconds when all three speakers are active. Figures pool speaker-time over the entire panel.

| AliMeeting mode | Exact-three precision | Exact-three recall | Exact-three F1 | Diarization error |
|---|---:|---:|---:|---:|
| Earlier native timing calibration | 50.33% | 81.26% | 62.16% | 17.68% |
| Released dictionary and narrowed boundaries | 52.07% | 81.14% | 63.43% | 16.59% |
| Matched acoustic model and sequence decoding | 73.25% | 69.56% | 71.36% | 11.66% |
| Matched acoustic model and per-frame decoding | 74.04% | 71.16% | 72.57% | 11.76% |
| Matched DiariZen and sequence decoding, primary | 72.77% | 72.27% | 72.52% | 11.87% |
| Matched DiariZen and per-frame decoding | 72.08% | 73.18% | 72.63% | 11.84% |
| REAL-T-trained DiariZen and sequence decoding | 36.10% | 85.60% | 50.78% | 30.22% |

The primary Ali model reduced false three-speaker time from 19.61 seconds to 7.10 seconds and still recognized 18.97 seconds when all three were speaking. It found 90.10% of individual active speaker-time during true three-person overlap. Exact-three recall credits a frame only when all three speakers are found at once. Lower recall accompanied the reduction in false alarms. The stand-alone scoring replay used cached upstream posteriors and took 3.30 seconds in aggregate for feature extraction, prediction and sequence decoding on one CPU thread.

On the 26 independent REAL-T excerpts with 66.21 seconds of actual three-way speech, the REAL-T-trained primary model reached 50.79% exact-three precision, 54.55% recall, 52.60% F1 and 17.74% DER. The older two-view count and acoustic method reached 37.85% precision, 66.50% recall, 48.24% F1 and 22.77% DER on the same group. The released wide timing profile reached 40.59% F1 and 22.84% DER. The independent per-frame comparator reached 53.22% F1 and 17.74% DER. The structured mode was the primary choice on both corpora because it achieved the higher development objective.

The REAL-T model trained on audio cropped to continuous speech. Its transfer to AliMeeting increased false activity, especially during silence. Calibration with separate AliMeeting development recordings restored the precision and timing trade-off on three new meetings. See [all paired metrics](timing_summary.json).

## Inference with frozen parameters

Use the benchmark environment and set `PYTHONPATH` to the existing three-speaker helpers and `target_90` directory. The two-view posterior cache is produced by the benchmark's `batch_probabilities.py`. The optional DiariZen cache is produced by the existing `extract.py` from the independent diarization benchmark. Use the pinned NeMo-Speech.cpp v0.3.0 runtime. The timing program reads numeric JSON trees and performs no refit. The Nemotron Q8 checkpoint SHA-256 is `08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1`. The DiariZen WavLM-base checkpoint SHA-256 is `2f65bc7772c92f90b6ff2ef1d7bdd068584828ed4578e6e48754380220aae979`. [Nemotron Q8 checkpoint](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/main/Nemotron-3-Diarization.q8_0.gguf) · [DiariZen source revision 844f555](https://github.com/BUTSpeechFIT/DiariZen/tree/844f5555b0a98acd0931511fc641a8c5b8ba92c7) · [DiariZen checkpoint](https://huggingface.co/BUT-FIT/diarizen-wavlm-base-s80-md)

```sh
bench=benchmarks/three_speakers
export PYTHONPATH="$PWD/$bench:$PWD/$bench/target_90"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python structured.py --from-saved frozen_ali.json --dz-dir DZ_FEATURES \
  --predict meeting.wav meeting.npz speaker_timing.json
```

The result has A/B/C speech intervals and can feed the existing audio renderer. Omitting `--dz-dir` selects the Nemotron plus short-time acoustic variant. `--per-frame` selects the fixed independent comparison without a switch penalty. The structured version uses the whole recorded stream when finding a state path and is intended for offline timing. The output and reported metrics cover speaker activity timing on recorded mono audio.

To refit the same REAL-T settings for the paired tests, use the published 45-clip training manifest and the pinned alternative-feature files.

```sh
python structured.py --dev REALT_DEV_MANIFEST REALT_POSTERIORS --profile PRIOR_PROFILE \
  --dz-dir OLD_DZ_FEATURES --save frozen_timing.json --out timing_results.json \
  --test realt REALT_TEST_MANIFEST REALT_POSTERIORS
```

To fit the separate Ali model, repeat `--dev` for the 83 and 81 historical AliMeeting panels and add `--train-skip 3 --match-development-precision`. The fresh comparison uses 18 previously unevaluated excerpts from three held-out source meetings with no shared annotated speakers with these development recordings. `prior_controls.py` replays the published wide REAL-T and dictionary-plus-narrow Ali configurations. The `counts` scorer uses one speaker assignment per whole excerpt and includes silence, overlap and a zero-second collar.
