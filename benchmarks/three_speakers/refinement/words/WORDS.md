# Mandarin transcription of three separated tracks

The optional SenseVoice Small recognizer consumes the same three tracks and predicted timing as the Zipformer baseline. It forces Mandarin, disables inverse text normalization, and applies the same peak-limited 0.035 RMS normalization to both recognizers. New audio needs no reference transcript. The program exports a timestamped transcript.jsonl, transcript.txt and a text file for each input speaker ID. Timestamps mark utterances. A supplied TextGrid or JSON reference enables character error scoring after inference.

Download model.int8.onnx and tokens.txt from the [pinned SenseVoice conversion](https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tree/2365baeacb507f821a0c8120fcee3d484dba7a07). The two SHA-256 digests in file order are `c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51` and `f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc`. Tests used sherpa-onnx 2.0.1. Use the existing project’s recognition environment.

From the repository root, set `trial` to the directory containing compare_recognizers.py.

```bash
export PYTHONPATH="$PWD/benchmarks/three_speakers"
python "$trial/compare_recognizers.py" --backend sensevoice \
    --model /path/to/sensevoice --tracks /path/to/separated \
    --timing /path/to/speaker_timing.json --out /path/to/transcripts --threads 2
```

The complete-file comparison used R8008_M8013 and the previous full hybrid render. This meeting supplied timing calibration in earlier work and is a repeat test. Its two recognizers received identical separated WAVs and 16 kHz audio spans. The sensitivity to the recognizer measures a downstream transcription change.

For a second comparison we fixed one excerpt per AliMeeting Eval session selected in the earlier three-speaker benchmark. We chose the longest eligible three-person excerpt in each session, then clipped the outer boundaries inward where an official TextGrid utterance crossed the cut. The same cuts applied to both recognizers. No reference timing was provided inside the evaluated intervals. Run prepare_word_cases.py with the earlier excerpt manifest, posterior cache and balanced profile, plus four official Eval_Ali_far TextGrids named by session. The preparation script uses `multiview_score.py` from the earlier target_90 benchmark. Include that directory on PYTHONPATH during case preparation.

The standard DPRNN renderer supplied both recognizers with the same separated audio wherever the three voices had solo enrollment. When the renderer could not enroll all three voices, both recognizers received the existing timing-routed tracks for that case. In routed overlap, each active track can contain the same mixed speech. All four selections are retained and the render mode is recorded per excerpt. These clips were previously evaluated for timing. The recognizer was fixed before transcription scores on these four clips were observed.

Speaker-attributed character error is computed with one optimal global speaker permutation per recording. The denominator is the number of reference characters. Insertions, deletions and substitutions count as errors. The excerpt score pools edit counts across the four recordings.

## Results on the complete calibration meeting

The shared hybrid render contains 37 minutes 19 seconds and 8,943 official reference characters. Zipformer made 3,651 errors (40.83%). SenseVoice made 2,985 errors (33.38%). Deletions fell from 3,048 to 1,279, substitutions rose from 579 to 1,601 and insertions rose from 24 to 105. SenseVoice decoded the three tracks in 386 seconds on a shared CPU with two inference threads. The two model timings were measured on separate runs under different CPU loads. No live latency was measured.

The example transcript labels the input streams 0, 1 and 2 as Speaker A, B and C, respectively.

## Independent excerpt results

Across 4:32.9 of audio from four meetings, Zipformer made 370 errors out of 1,492 reference characters (24.80%) and SenseVoice made 368 (24.66%). It improved one excerpt and regressed on three. Two cases used DPRNN and two used timing routing. The sizeable gain in the complete calibration meeting therefore has limited independent support.

For an exploratory waveform check, we selected the excerpt with the most predicted three-way activity among the two cases with adequate inferred solo anchors. The 134.5-second audio used 3 three-voice SepFormer windows. With SenseVoice fixed, the standard SepFormer path made 91 errors in 634 characters (14.35%) and the frozen spectral filter made 92 (14.51%). One recording provides limited support for a downstream effect.

## Full phrases with three simultaneous sources

The direct overlap test takes the first complete utterance for each of the nine speakers in the three AISHELL-3 trios already reserved for waveform testing. The speakers are disjoint from the data used to fit the spectral filter. We used the pinned AISHELL-3 test/content.txt at f20d5db4a31fe779ef07bb1af4ea92da5c786622 and verified each source audio checksum. Each mono simulated-room mixture uses a 0/−4/−8 dB distribution, source onset delays of 0/0.12/0.24 seconds rotated by case, and 15 dB signal-to-noise ratio. Every source sample is retained. All estimates are recognized as full utterances, with one best speaker permutation per mixture.

The three mixtures contain 24 reference characters, 5.60 s of recorded mixture, and 1.14 s of simultaneous three-source activity according to the common source-energy detector. Both SepFormer paths produced 29 errors in the full-utterance transcriptions (18 substitutions, 3 deletions, 8 insertions). Three of the 24 reference characters matched. The identical recognizer on the isolated dry inputs yielded 6 substitutions and 18 matches. Insertions cause the separated character error rate to exceed 100%. These are short, difficult two-to-five-character prompts. The character scores cover entire prompts and do not locate each error within the overlap.

To reproduce, run controlled_words.py with the waveform test's fresh.json, its checksum-verified original sources, the pinned AISHELL-3 test/content.txt, and the frozen separator, filter, and recognizer. Add the repository root, three_speakers, target_90 and the directory containing refine_three.py to PYTHONPATH. controlled_word_scores.json contains exact per-case hypotheses, source hashes and edit counts.

Zipformer provided a second recognition check on the identical stored-in-memory inputs. It made 7 errors in 24 characters on dry isolated speech (29.2%), 19 after standard separation (79.2%), and 21 after refined separation (87.5%). SenseVoice's three results reproduced exactly. The same nine dry phrases therefore give considerably lower error with both recognizers than their recovered tracks in the mixture.
