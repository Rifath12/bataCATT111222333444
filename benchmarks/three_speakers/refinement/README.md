# Three Mandarin voices from one microphone

This release adds a spectral filter for three-voice separation and a Mandarin SenseVoice recognition option. Both work with the existing three-speaker file pipeline. The delivered per-speaker transcripts use A/B/C timing and utterance timestamps. The strongest gains came from speech recognition and from recovering a quieter voice in controlled mixtures.

## Using the improvements

Start with the model downloads and mono 16 kHz timing procedure in the [existing three-speaker guide](../IMPROVEMENT.md). The spectral filter uses the same SepFormer checkpoint and requires one separator inference per three-way window. Its frozen parameters ship with the code. For recognition, download `model.int8.onnx` and `tokens.txt` from the pinned [SenseVoice Small conversion](https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tree/2365baeacb507f821a0c8120fcee3d484dba7a07). The [transcription benchmark](words/WORDS.md) lists both hashes.

Run from the repository root, setting the file and directory variables to your local inputs.

```bash
bench=benchmarks/three_speakers
python "$bench/render.py" "$mono_wav" --diarization "$timing_json" \
  --out "$tracks" --method dprnn --project . --models "$models" \
  --threads 1 --workers 2 --sepformer-three "$sepformer" --refine-three
python "$bench/recognize.py" --audio "$tracks" --timing "$timing_json" \
  --asr "$sensevoice" --backend sensevoice --target-rms .035 \
  --threads 2 --out "$transcripts"
```

The track names are taken from the timing JSON's speaker IDs. With A/B/C IDs, the output contains `speaker_A.wav`, `speaker_B.wav`, `speaker_C.wav`, a combined TXT and JSONL transcript, and three speaker-specific TXT files. In the listening package the full-length tracks are losslessly compressed as FLAC. This file-mode evaluation uses full-recording context. It does not establish streaming latency.

For complete recordings, an optional spectral-dictionary verifier can export A/B/C timing using the included 42 KiB model. Supply a directory with the mono WAV, native RTTM, base probabilities and the level-normalized shifted probabilities produced by the [two-pass preparation](../target_90/README.md).

```bash
trial="$bench/refinement"
export PYTHONPATH="$PWD/$bench:$PWD/$bench/target_90"
python "$trial/acoustic/new_recording.py" --frozen "$trial/acoustic/dictionary_model.json" \
  --input-dir "$meeting_directory" --out "$timing_directory"
```

The frozen model reproduced all 5,260 candidate probabilities and the complete timing output exactly on one new half-hour recording.

## Who was speaking during overlap?

A new held-out set consists of 81 non-overlapping excerpts totaling 18:36 from two previously unscored AliMeeting sessions. There are 91.92 s when all three voices are active. Each excerpt contains three distinct speakers. Each source meeting has four participants and the extraction rule excludes the fourth from that excerpt. The eight source participants and the six people in the two new continuous meetings have distinct labeled identities from the seven previously evaluated AliMeeting sessions. Compare methods within each panel because the frequency of overlap differs.

| Decoder on the 81 new excerpts | Speaker-time recall in triple overlap | All-three precision | All-three recall | All-three F1 | Overall diarization error |
|---|---:|---:|---:|---:|---:|
| Previous calibrated default | 89.8% | 36.8% | 73.0% | 49.0% | 28.7% |
| Previous high-recall two-pass | 95.5% | 22.1% | 87.7% | 35.3% | 46.8% |
| Two-pass with independent acoustic confirmation | 94.4% | 27.4% | 84.3% | 41.3% | 42.3% |
| Per-speaker classifier trained on short and long recordings | 94.1% | 28.7% | 85.2% | 42.9% | 32.1% |
| Classifier with multiple pitch estimates | 97.1% | 19.5% | 92.9% | 32.2% | 43.1% |

The independent confirmation model reduced false triple declarations from 283.96 to 205.45 seconds relative to the high-recall two-pass system. Its speaker-time recall still gives partial credit whenever it finds just one or two voices during real three-way speech. The calibrated default retained the best balance among these rows. Identity matching for these short tests is fixed once per excerpt. Whole meetings get one identity matching over the entire recording.

## How clean was the separated audio?

On nine controlled groups of three Mandarin speakers, the spectral filter estimates how much to reduce competing voices in each frequency band. On the three new speaker groups, median isolation improvement over the mixed signal for the weakest speaker in simulated rooms increased from **+5.27 dB to +6.44 dB**. Across the six previously tested groups, it increased from +4.94 to +5.22 dB. Across all three voices in simulated rooms, median isolation on the new groups slipped from +3.11 to +3.05 dB. The combined clean and simulated-room whole-clip median improved from +5.02 to +5.53 dB on the new groups.

The nine groups contain only 17.58 s of genuine three-voice overlap. Bootstrap intervals across the three new groups include zero. The portable filter added 1.27 s to 62.72 s of separator inference on four short fixtures on a shared CPU. This excludes model initialization. The [waveform benchmark](waveform/README.md) includes the source manifests, pinned model, per-condition measurements and reproduction steps.

## Mandarin transcription

Both recognizers received identical WAVs and predicted timings in each comparison. Speaker-attributed character error counts deletions, substitutions and insertions after one optimal match of A/B/C to the reference speakers per recording. This measures the user's downstream transcription task. Only mixtures with known clean sources support the waveform-isolation measurements above.

| Evaluation | Reference characters | Previous Zipformer error | SenseVoice error |
|---|---:|---:|---:|
| Previously used 37:19 meeting | 8,943 | 40.83% | 33.38% |
| New full 30:39 meeting R8008_M8016 | 7,705 | 31.67% | 28.40% |
| Four shorter real excerpts, 4:33 combined | 1,492 | 24.80% | 24.66% |

On the new full meeting, there were 2,440 errors with Zipformer and 2,188 with SenseVoice. SenseVoice reduced deletions from 1,749 to 871, while substitutions increased from 670 to 1,224 and insertions from 21 to 93. The reference had 622 far-field TextGrid utterances. The scorer chose one reference-to-output speaker assignment for the whole file. The renderer took 272 seconds on the shared CPU and filtered 15 windows. SenseVoice took another 218 seconds for the full file, excluding model initialization and speaker-timing inference. The delivered listening file used the previous calibrated timing with the new spectral filter and SenseVoice.

We also tested complete read phrases mixed together from nine different Mandarin speakers. These three clips preserve their full reference texts. They contain 24 characters in total and 1.14 seconds of energy-defined three-way overlap. With SenseVoice, the original and filtered SepFormer tracks both correctly matched 3 of 24 reference characters, with 29 total edits in each case. Feeding isolated original voices to the same recognizer recovered 18 of 24. Zipformer made 19 errors on the original separated tracks and 21 with the filter. Both recognizers got five of nine isolated phrases entirely correct and zero of nine separated phrases entirely correct. This small stress test illustrates how much intelligibility can be lost when all three people speak together. [Test protocol and raw counts](words/controlled_word_scores.json)

On the four shorter test excerpts two clips had sufficient independent solo speech for the DPRNN renderer. The other two used the prior timing-based routing fallback for both recognizers. In an additional 134.5 s natural excerpt, enabling the new spectral filter changed SenseVoice’s score from 91 to 92 errors out of 634 characters. It improved the quietest voice in the controlled tests without establishing an improvement in transcript quality by itself.

## Continuous meetings and other alternatives

The two new complete meetings lasted 30:39 and 30:20 and together had **2.43 seconds** of annotated triple overlap. The new dictionary verifier reduced false triple detections from 7.18 to 5.40 s while preserving the 1.62 s of correct all-three detections. Exact-three precision increased from **18.4% to 23.1%**, all-three recall stayed at **66.7%**, and three-way speaker-time recall stayed at **88.9%**. Overall diarization error fell from 10.14% to 10.08%. The independent Sortformer check reduced false calls to 6.05 s, with the same recall. [Dictionary comparison](acoustic/new_complete_meetings_dictionary.json) · [Sortformer comparison](acoustic/new_complete_meetings_neural.json)

The alternative tests included pseudo-labeled remixes generated from each recording's own inferred solo speech, a pitch-based speaker verifier, a three-dictionary acoustic reconstruction model, and an independent Sortformer check that processes uncertain intervals. On the two previously evaluated complete meetings, the Sortformer check reduced false three-voice time from 22.56 to 19.55 seconds and all-three F1 rose from 25.9% to 28.3%. Speaker-time recall was 86.8% for both systems after rounding. These 63:53 of audio contained just 7.13 s when all three were talking together.

The new methods have development-only operating points. The comparisons are exploratory across multiple methods. A/B/C can only track speakers within the evaluated recording. The tests support optional use of the waveform and recognition changes. Reliable 90% simultaneous isolation across real meetings remains unverified.

[Metrics](scores.json) · [Natural audio sources and selection](data/PROTOCOL.md) · [Independent confirmation](diarizen/PROTOCOL.md) · [Pitch experiment](harmonic/PROTOCOL.md) · [Dictionary and neural experiments](acoustic/README.md) · [ASR comparisons](words/word_scores.json)

## Reproduce the tests

The main file-mode commands above produce tracks and transcripts from the supplied model checkpoints and speaker timings. To reproduce the experiments, add the repository root, `benchmarks/three_speakers` and `benchmarks/three_speakers/target_90` to `PYTHONPATH`. The previous benchmark documents the two Nemotron posterior caches, the labeled REAL-T clips and the original long-meeting directory layout. For the new data, use the pinned checksums and instructions in `data`. The natural-audio evaluation uses 10 ms frames with zero collar. All timings were measured on shared Linux CPU resources.

To refit the speaker-activity verifier, run `confirm.py` with `--bundle LOCAL_MODEL --profile PROFILE --dev-manifest REALT_DEV --dev-probs REALT_POSTERIORS --work LONG_DATA --views ALTERNATE_CACHE`. The program fits its thresholds on development recordings and saves a local scikit-learn model. Then use `--bundle LOCAL_MODEL --test NAME TEST_MANIFEST POSTERIORS --report REPORT_JSON` or `--bundle LOCAL_MODEL --predict UNLABELLED_POSTERIOR_NPZ TIMING_JSON`. Only load a model bundle you generated or trust. The optional speaker-activity, pitch, remix and independent-model experiments operate on previously exported mono probabilities. The file-mode waveform renderer and recognizer do not depend on those experimental classifiers.
