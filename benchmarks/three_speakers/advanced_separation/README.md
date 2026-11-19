# Mandarin three-speaker quality pass

This audio-only release provides an optional transcript refinement for the original room microphone, a context-aware recognizer for synchronized eight-microphone beams, and a pipeline for three synchronized close microphones. It also includes a guarded single-microphone waveform model for short controlled overlap clips.

## Audited release

| Audio setup | Previous edits | Reviewed edits | Reviewed 1 − cpCER |
| --- | ---: | ---: | ---: |
| One room microphone, additional recognizer pass | 2,715 | 2,713 | 82.89% |
| Eight physical array microphones, contextual transcript | 1,894 | 1,892 | 88.07% |
| Three physical close microphones, adaptive profile | 1,505 | 1,386 | **91.26%** |

All rows use the same 15,859 reference characters from M8016 and M8017. Each row compares its software stage on the same audio setup. The mono row replays cached recognition hypotheses. The array row re-decodes 34 changed windows against verified unchanged results. The close-mic row reruns recognition over both full recordings. A separate matched activity and time-constrained audit uses one speaker map across text and timing within each recording. The optional timing and waveform changes have their own metrics.

The close-mic score improved in both meetings, from 86.92% to 88.02% in M8016 and 93.90% to 94.32% in M8017. Deletions fell from 661 to 483, while substitutions rose from 581 to 598 and insertions rose from 263 to 305. At exact three-way overlap the approximate time-constrained text score stayed at 6 errors on 37 reference characters. On the forced-aligned activity labels, pooled all-three detection F1 fell from 58.77% to 46.45% as the more sensitive detector predicted more false triples. Keep the previous close-mic profile for applications that prioritize precise overlap alerts.

[Matched scoring and reference sensitivity](review/metrics.md) · [Close-microphone benchmark and reproduction](review/close_microphones.md) · [Array and memory audit](review/array.md)

## Single room microphone

Run the existing [offline pipeline](../root_causes/README.md) to obtain three aligned 16 kHz WAVs, `speaker_timing.json` and the first transcript. The quality pass keeps each speaker's identity and timestamps, and edits interior characters when two independent recognizers agree on the exact change. It can make one more single-character edit if both existing views give the same whole turn, with no additional decoder spans. It preserves punctuation and case outside edited glyphs and clears stale token-level metadata on edited rows. The existing audio files are still the speaker outputs for the two full mono meetings.

From `benchmarks/three_speakers`, with the existing audio dependencies and `rapidfuzz` available:

```bash
PYTHONPATH=. python advanced_separation/mono_text.py \
  --baseline output/transcript.jsonl --tracks output \
  --timing speaker_timing.json --zipformer /models/zipformer-ctc-zh \
  --paraformer /models/paraformer-zh --out output/quality_pass
```

Each model directory has `model.int8.onnx` and `tokens.txt`. Use `--zipformer-jsonl` and/or `--paraformer-jsonl` to reuse previous decoder opinions. The existing offline pipeline can invoke this stage by adding `--consensus-zipformer /models/zipformer-ctc-zh --quality-paraformer /models/paraformer-zh` to its usual mono arguments. The benchmark ran int8 [Paraformer Mandarin 2023-09-14](https://huggingface.co/csukuangfj/sherpa-onnx-paraformer-zh-2023-09-14/tree/def027084691107096b5ebba69785756d63de6c5) and the existing Mandarin Zipformer CTC used for the previous release (`sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03`). See [recognizer evidence](evidence/recognizer_trials.md) for the model hash and timings. The extra Paraformer stage took 73–88 seconds per prepared half-hour meeting on one CPU thread. Zipformer and the original waveform separation carry their own cost.

The [experimental waveform module](mono_wave/README.md) selects RecorrNet or the prior separated audio for a short overlap. It improved median exact-three SI-SDR improvement from −1.19 dB to +13.00 dB in a three-clip simulated room test and used the previous stems on one clip. When inserted into five natural excerpts with fixed speaker attribution, gated RecorrNet tied the existing total of 185 character edits over 489 reference characters. The gate changed 1.5 seconds in one recording. An ungated natural probe increased edits from 39 to 46. Long mono recordings use the existing audio separator and optional transcript pass. The reviewed gate also rejects duplicate or negligible stems. The regular mono renderer preserves the direct audio in solo gaps between joined overlap windows. An optional [timing-only refinement](review/mono_timing.md) uses cached dual model probabilities. Its pooled diarization error improved on a forced-aligned reference (10.14% to 8.78%) and moved slightly upward on broader transcript-timing labels (15.92% to 15.96%). Exact-three decisions stay fixed.

## Eight synchronized array microphones

Add `--array-context` to the [normal offline pipeline](../root_causes/process_audio.py) to render and recognize the continuous beams and then run the fixed duplicate filter. To use this stage alone, first obtain `beam_A.flac`, `beam_B.flac`, `beam_C.flac` with [the array renderer](../root_causes/spatial_render.py) and provide the same timing JSON. This pass provides the recognizer more context around short boundaries and suppresses likely leaked speech while decoding. The matched full waveforms in this transcription comparison were unchanged.

```bash
PYTHONPATH=. python advanced_separation/array_context.py \
  --tracks array_beams --timing speaker_timing.json \
  --model /models/sensevoice --out array_recognition
```

The decoder saves `array_recognition/extended/transcript.jsonl` and separate A/B/C text files. Apply the prior duplicate-text filter with its fixed `duplicate_settings.json` to reproduce the filtered scores. Its `--tracks` directory is the existing array renderer's `speaker_A/B/C.wav` output. Reference TextGrids are optional evaluation inputs, and inference works without them.

Full two-meeting comparison with the matching duplicate filter reduced character edits from 2,126 to 1,894 (86.59% to 88.06% on the same equipment). The unfiltered decoder processed prepared beams in 164–187 seconds on the shared CPU host. This measures a file-mode recognition stage. In the reviewed decoder, streaming beam-power computation cut peak RSS from 642 to 86 MiB for this stage with identical features. It also retains valid speech at recording ends and removes duplicated decoding at overlapping chunk edges. The released arrays have eight physical, synchronized channels. The spatial renderer needs at least 0.3 seconds of usable interior solo speech per speaker for enrollment. Duplicating one mono input cannot reproduce the geometry.

For an optional waveform profile, pass `--profile overlap-mask --ungated` to the array renderer. It uses the same static beam during solo speech and a channel-shape likelihood during predicted overlap. On six controlled triplets across two rooms, its median SI-SDR improvement rose from −4.26 to +0.76 dB and −1.47 to +3.99 dB, respectively. The baseline had positive SI-SDR improvement on 4 of 18 source tracks. The waveform profile reached 15 of 18. Median projected SIR improvement also rose from +3.10 to +3.68 dB and +5.97 to +7.83 dB. Source-tracked mixture controls supply the controlled evaluation references. The reported full-meeting text gain uses the original array waveform profile.

## Three synchronized close microphones

Feed one aligned 16 kHz mono recording per speaker in A/B/C order. The activity detector uses the relative microphone levels to suppress bleed and permits simultaneous activity. The CLI reports each voice under the supplied recording position and can save three named WAVs. Every output timestamp uses the original shared sample clock.

```bash
PYTHONPATH=. python advanced_separation/direct_microphones.py \
  close_A.wav close_B.wav close_C.wav \
  --vad /models/silero_vad.onnx --asr /models/sensevoice \
  --profile advanced_separation/direct_microphones_adaptive.json \
  --copy-audio --out close_recognition
```

Use the standard recurrent 16 kHz Silero ONNX model with a 64-sample context, a 512-sample hop and separate recurrent state for each channel. The `--copy-audio` option saves each corresponding captured microphone signal with its A/B/C name. The detector gates recognition and reports speaker timing. The waveform output retains the captured signals, including any physical bleed. The adaptive profile was calibrated on a different AliMeeting recording (M8013). Choose `direct_microphones_profile.json` for the previously published conservative profile. Independent physical inputs require no inference-time references or voice enrollment.

The previous conservative close-mic profile produced 1,008/7,705 edits in M8016 (86.92%) and 497/8,154 in M8017 (93.90%). Pooled character accuracy was 90.51%, compared with 82.70% for the previous mono release on the same conversations. This is an equipment and software comparison. Activity DER was 18.58% and 14.69%, respectively, under the TextGrid-derived time-overlap metric recorded in the scores. The same annotations mark 6.52 and 3.05 seconds with all three speakers active. Recognition activity recall during those windows was 50.31% and 61.97%, respectively. The close-mic file pipeline took 282 and 248 seconds on one shared CPU thread, including activity detection and recognition.

## How to read the scores

Speaker-attributed character edits are substitutions, deletions and insertions after one global label match per meeting. The reported percentage is one minus total edits divided by total labeled reference characters. This is a text score. SI-SDR improvement compares isolated waveforms to source references and is reported in dB. The real recordings are complete Mandarin AliMeeting Test sessions M8016 (30:39) and M8017 (30:20). Three active voices are rare even in these three-participant meetings, so the [short natural-overlap panel](evidence/natural_mono_wave.json) is reported separately. These tests used files on a Linux CPU host. Live latency and RTX performance remain unverified.

[AliMeeting dataset and license](https://www.openslr.org/119/) · [Guarded waveform model card](https://huggingface.co/playwithmino/recorrnet-v2-1ch-aishell1mix-ver2-alimeeting)
