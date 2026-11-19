# Optional Mandarin character consensus

The mono wrapper accepts `--consensus-zipformer PATH` to compare the final SenseVoice contextual text with its existing localized tracks and a selective independent CTC decode. It preserves speaker assignments and the source WAV timeline. The Zipformer directory must contain the Chinese `model.int8.onnx` and `tokens.txt` used in the original recognition benchmarks. The fast transcript path remains independent.

For manual use with already aligned speaker tracks:

```sh
python benchmarks/three_speakers/recognize.py \
  --audio SPEAKER_TRACKS --timing SPEAKER_TIMING.json \
  --backend sensevoice --asr SENSEVOICE_MODEL --context-mix ROOM_MONO.wav \
  --consensus-zipformer CHINESE_ZIPFORMER_MODEL --target-rms 0.035 \
  --threads 2 --out TRANSCRIPTS
```

The original full-turn transcript is the alignment anchor. A slot changes only when more views support the alternative than the original, with separate minimum agreement thresholds for replacement, insertion and deletion. Deleting a character requires independent Zipformer agreement and an immediate repetition. Relative alignment agreement bounds permit skipping speech turns that cannot change for any Zipformer output. The optional pass uses no gold transcript at inference.

The rule family was calibrated once using five chronological blocks from M8013 and frozen before applying it to M8016 and M8017. Model ablation omitted FireRed, so these benchmarks require one optional model. The current contextual and cached refinement sequences are replayed from the same time-aligned inputs.

| Evaluation | Previous S/D/I | Selective consensus S/D/I | Total errors |
|---|---:|---:|---:|
| AliMeeting M8016, 7,705 characters | 993 / 694 / 151 | 983 / 685 / 150 | 1,838 → 1,818 |
| AliMeeting M8017, 8,154 characters | 535 / 348 / 70 | 517 / 345 / 63 | 953 → 925 |
| Pooled | 1,528 / 1,042 / 221 | 1,500 / 1,030 / 213 | 2,791 → 2,743 |

The speaker error totals also went down. M8016 A 389→378, B 417→409, C 1,032→1,031. M8017 A 378→360, B 154→149, C 421→416. M8016 B had one additional insertion and C had one additional substitution. An earlier short excerpt also had one additional total edit. Across six previously evaluated overlap excerpts there were 331 edits in 1,146 characters, down from 336, with one additional deletion. The exact-three subset stayed at 73 edits in 153 aligned characters.

Exact cached abstention replay retained every transcript while reducing Zipformer tracks assessed to 279 of 855 turns on M8016 and 196 of 533 on M8017. Speaker-turn time reduced from 1,661.2 to 763.4 seconds and from 1,498.4 to 710.1 seconds. The selected pass adds model latency to previously completed SenseVoice and mixture decoding. On M8017, the selected pass added 84.53 seconds wall time on one shared CPU thread, including model load and reconciliation. It reproduced the full cached decode transcript exactly. A fresh two-clip AISHELL-4 check stayed at 68 edits in 147 characters, with every pooled edit category unchanged.
