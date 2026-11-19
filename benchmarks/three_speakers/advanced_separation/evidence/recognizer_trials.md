# Independent Mandarin recognition comparisons

I tested two published recognizers on real three-speaker recordings using the existing system's speaker boundaries and audio. Paraformer-Large decoded each existing speaker track. The multi-speaker MFCCA model decoded room mono by itself and room mono alongside the three speaker tracks derived from that same mono recording. MFCCA outputs serialized utterances. A deterministic match against the previous transcript supplied the anonymous speaker labels. Reference characters were used only for scoring.

Speaker-attributed character edits include insertions, deletions and substitutions after a single global speaker permutation per clip. Smaller numbers are better.

| Matched clips | Existing system | Paraformer-Large int8 | MFCCA from mono | MFCCA from mono with predicted speaker views |
| --- | ---: | ---: | ---: | ---: |
| Two independent AISHELL-4 examples | 68 / 147 | 87 / 147 | 110 / 147 | 79 / 147 |
| Six dense AliMeeting replays | 336 / 1,146 | 338 / 1,146 | | |
| Three dense AliMeeting replays under 20 seconds | 128 / 342 | 134 / 342 | 205 / 342 | 178 / 342 |

The matched AISHELL-4 clips total 26.33 seconds, with 3.46 seconds of annotated three-way overlap. Processing time on one shared CPU thread was 5.10 seconds for Paraformer and 19.77 and 30.25 seconds for each MFCCA arm. These times cover recognition on prepared input. Neither experiment replaced the existing mono speech recognizer.

With a one-second timing collar, the six dense AliMeeting clips had 73 / 153 time-constrained edits at estimated character midpoints during exact-three overlap with the existing recognizer and 74 / 153 with Paraformer. Restricting to entire annotated turns contained inside exact-three overlap gave 49 / 104 and 44 / 104. The official annotation supplies turn timestamps, and the overlap scoring interpolates reference character positions within those turns. MFCCA was excluded from this score because its emitted utterances have no individual speaker timestamps.

The CPU build of MOSS-Transcribe-Diarize ran a two-second operational smoke test in 69.5 seconds. It emitted four labels for three annotated speakers and returned zero duration for three of its four segments. The 10.27-second AliMeeting smoke stopped at decoder memory admission. The portable output provides sequential speaker turns. The records in `alternatives/trials.json` contain the native metadata for both runs.

## Corroborated character edits

A separate probe retained the baseline transcript and required both Zipformer and Paraformer to offer the same interior edit. The edit had to be surrounded by unchanged characters and each recognizer's complete turn hypothesis had to be at least 65% similar to the baseline. This made 55 amendments across the two meetings. The source tracks, diarization and utterance boundaries stayed fixed.

| Panel | Baseline edits | Edits after the probe |
| --- | ---: | ---: |
| Full mono M8016 | 1,818 / 7,705 | 1,806 / 7,705 |
| Full mono M8017 | 925 / 8,154 | 909 / 8,154 |
| Pooled full mono | 2,743 / 15,859 | 2,715 / 15,859 |
| Two independent AISHELL-4 clips | 68 / 147 | 65 / 147 |
| Six dense AliMeeting clips | 336 / 1,146 | 329 / 1,146 |
| AISHELL-4 exact-three midpoints | 23 / 32 | 22 / 32 |
| Dense AliMeeting exact-three midpoints | 73 / 153 | 72 / 153 |

The pooled full-meeting result goes from 82.70% to 82.88% when expressed as one minus speaker-attributed character error. Paraformer took 87.9 and 72.9 seconds of additional prepared-audio decoding on one shared CPU thread, respectively. Those times exclude the prior Zipformer run. The probe decoded all C spans and 60 preselected A/B spans per meeting. The selection only used existing recognized text, and focused on the turns for which Zipformer could propose an edit. `consensus_probe.py` and `consensus_probe.json` record the decision rule and measured counts.

## Why speaker C was difficult in M8016

The table fixes the existing predicted turn boundaries and changes the recognizer or its waveform for one speaker. Each row covers 1,580 reference characters and the recording lasts 30 min 39 sec. The headset is a diagnostic recording with its original clock. A prior audit measured a 60 ms lag for this headset.

| Input and recognizer | Edits | Substitutions | Deletions | Insertions |
| --- | ---: | ---: | ---: | ---: |
| Deployed mono pipeline | 1,031 | 494 | 527 | 10 |
| Same separated mono speaker track, Paraformer | 1,090 | 807 | 242 | 41 |
| Existing physical eight-microphone beam, Paraformer | 983 | 663 | 252 | 68 |
| Individual headset, Paraformer | 541 | 286 | 148 | 107 |

Predicting more characters in the same mono speaker track increased substitutions for M8016. On M8017, changing only this label's recognizer gave 403 / 1,713 vs the baseline 416 / 1,713. The two complete mono meetings together gained 46 additional errors when label C used Paraformer.

A separate time-window diagnostic examined reference turns with at least three characters and allowed 0.16 seconds of boundary slack. For M8016 turns with less than 10% reference overlap, the deployed speaker C text had 358 edits on 595 characters. Continuous ASR of the unprocessed room microphone had 377. The matching speakers A and B had 111 / 1,772 and 139 / 1,676. These window-level counts differ from speaker-attributed full-meeting scores. They show the difficult speaker is affected during predominantly solo speech as well. The microphone comparison provides stronger causal evidence about the available signal, holding the recognizer and speech boundaries fixed.

## Reproduction

The root-relative `overlap_cases.json`, `joint_cases.json` and `full_cases.json` freeze the inputs. `decode_backend.py` and `mfcca_decode.py` produce recognition output. `score_models.py` and `diagnose_speaker.py` produce aggregate and local scores using the previous release's normalizer and scorer. `scores.json` and `score_focus.json` record the detailed counts.

Paraformer uses the Sherpa-ONNX port `csukuangfj/sherpa-onnx-paraformer-zh-2023-09-14` at commit `def027084691107096b5ebba69785756d63de6c5`. The `model.int8.onnx` SHA-256 is `f36a0433bcf096bd6d6f11b80a3ac8bed110bdca632fe0d731df8d1a84475945`. [Sherpa-ONNX documentation](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/paraformer-models.html)

MFCCA uses ModelScope `NPU-ASLP/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950` version 3.0.0 and FunASR v0.3.0 commit `64bd637c301c3e4f771466808c30ec96d5531e45`. The checkpoint SHA-256 is `629ac2b7ca7a4fc38da000b8117d80349405e082958d431726c329f8f43342c0`. Evaluation used beam width four, CTC weight 0.6 and no standalone language model. [MFCCA paper](https://arxiv.org/abs/2210.05265) · [ModelScope checkpoint](https://www.modelscope.cn/models/NPU-ASLP/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950)
