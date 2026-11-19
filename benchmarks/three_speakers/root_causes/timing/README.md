# Speaker-turn timing

On two complete AliMeeting recordings with three participants (60 min 59 s), the earlier decoder put the median onset 60 ms before the annotation and the median end 40 ms after it. Of the 7.18 s of falsely detected three-person speech, 6.84 s fell within 300 ms of an annotated activity change. In 6.68 s of the false detections, two people were speaking. These categories overlap. The onset and offset estimates use each annotated edge’s closest corresponding predicted edge within 750 ms, after one global speaker match per file.

The decoder’s maximum filter extended speech marks by seven 10 ms frames on each side. Changing only the filter width from 15 to 5 reduced the extension to two frames per side. The width was fitted on one separate, previously used 37 min 19 s development meeting (R8008_M8013). The fit considered odd widths 1–15 and maximized exact-three F1 minus diarization error. Candidates had to retain development three-way speaker-time recall within 2.5 percentage points and all-three recall within 5 points of the earlier decoder. The release combines this timing change with the existing frozen acoustic verifier. The verifier keeps control of the active count during every proposed three-speaker interval.

| Test panel | Method | Diarization error | Exact-three precision | Exact-three recall | Exact-three F1 | Three-way speaker-time recall |
|---|---|---:|---:|---:|---:|---:|
| Two complete recordings | Prior decoder | 10.14% | 18.41% | 66.67% | 28.85% | 88.89% |
| Same recordings | Acoustic verifier | 10.08% | 23.08% | 66.67% | 34.29% | 88.89% |
| Same recordings | Acoustic verifier and 5-frame width | 7.65% | 23.08% | 66.67% | 34.29% | 88.75% |
| Three new speaker-disjoint excerpts | Prior decoder | 13.93% | 43.81% | 89.76% | 58.88% | 96.59% |
| Same excerpts | Acoustic verifier | 13.73% | 45.89% | 89.76% | 60.73% | 96.59% |
| Same excerpts | Acoustic verifier and 5-frame width | 11.02% | 45.89% | 89.76% | 60.73% | 96.59% |

The complete recordings contained 2.43 s of true three-way speech across 12 events. The combined profile detected at least one frame in nine events. One missed event in R8008_M8017 at 29:09.57–29:10.19 accounted for 0.62 of the 0.81 s missed across both files. The weakest voice had median posterior scores of 0.052 and 0.013 in the two mono views. Its peak score was 0.085 in the original view. See `missed_events.py` and `rare_event_diagnosis.json` for the per-event audit. The three new excerpts contained 4.10 s in 183.89 s of audio. On the three excerpts the combined profile cut false alarms from 21.27 to 12.80 speaker-seconds and raised misses from 4.50 to 7.45 speaker-seconds. The diarization scorer includes silence and overlap and uses a zero collar and a single optimal speaker mapping per recording.

## Domain and transcription effects

The same change to native width improved timing error from 11.00% to 8.74% on two earlier complete AliMeeting recordings and from 28.74% to 28.49% on 81 AliMeeting excerpts. It increased error from 32.30% to 36.21% on 26 REAL-T excerpts. The REAL-T annotations had a different direction of disagreement with the baseline, with a median 190 ms late onset and 225 ms early end. A separate native width fitted on 45 REAL-T development excerpts was 81 frames. It reduced error to 22.85% on the 26 REAL-T evaluation excerpts and lifted three-way speaker-time recall from 66.81% to 76.45%, while precision for exact-three fell from 49.85% to 47.69%. Applying the REAL-T width to the three new AliMeeting excerpts raised their error to 37.12%. All input profiles are selected explicitly. The REAL-T result concerns the native decoder alone.

A post-processing ablation applied 50 ms boundary contraction and a three-frame turn-handoff rule to existing A/B/C timings. With the same separated tracks and SenseVoice settings on R8008_M8016, character error rose from 28.40% (2,188 errors in 7,705 characters) to 28.93% (2,229 errors). A separate test on R8008_M8017 applied the actual combined verifier and 5-frame width to the original tracks. Character error rose from 14.69% to 15.42% (1,198 to 1,257 errors in 8,154 characters, with 50 additional deletions). Retain the original recognition context when exporting the narrower speaker-activity timestamps.

## Export A/B/C timings

From the repository root, use the Q8 Nemotron weights and shared library listed in the target_90 benchmark. The complete-meeting benchmark used a level-normalized second pass with a 9.28 s shift. The three-excerpt panel used the default level-normalized offline second pass. Supply a 16 kHz mono WAV and a folder for the two-pass cache. The timing exporter reads the audio, its cache and numeric model files. Human-annotated references are optional and only used when scoring.

```bash
bench=benchmarks/three_speakers
trial="$bench/target_90"
acoustic="$bench/refinement/acoustic"
timing="$bench/root_causes/timing"
export PYTHONPATH="$PWD/$bench:$PWD/$trial:$PWD/$acoustic"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
printf '%s\n' "$PWD/meeting.wav" > audio-list.txt
python "$trial/batch_probabilities.py" --audio-list audio-list.txt \
  --library "$RUNTIME/lib/libnemo_speech_asr_c.so" --model "$GGUF" --output cache
python "$timing/complete_timing.py" --model "$acoustic/dictionary_model.json" \
  --profile "$timing/native_profile.json" --cache meeting.wav cache/meeting.npz --out timing-output
```

The output is `timing-output/meeting_timing.json` with start and end times in seconds and anonymous A/B/C labels. Repeat `--cache WAV NPZ` to process several recordings. The `--input-dir` form accepts directories containing `mono.wav`, `probabilities.npz`, `view.npz` and `native.rttm`. For scoring, add `--score` and supply `meeting.rttm` beside the WAV or `reference.rttm` inside the input directory.

## Reproduce the ablations

`ablate_padding.py --fit` fits a width on R8008_M8013, and accepts repeated `--test PANEL MANIFEST CACHE` arguments to score a frozen native profile. `fit_realt.py` accepts the 45-excerpt, three-participant REAL-T development manifest, derives an independent profile and records its grid. The exact-three precision floor is two percentage points below development baseline. `benchmark_boundaries.py` computes signed edge delays and the attribution of false three-voice activity near state changes. `turn_boundaries.py` implements the separate post-processing ablation used for the character-error check. See `summary.json` for pooled metrics and the underlying speaker-time error components.
