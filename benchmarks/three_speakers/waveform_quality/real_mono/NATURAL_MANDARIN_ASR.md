# Speaker-attributed Mandarin transcription on room microphones

These four excerpts contain 113.67 seconds of real meeting audio from three recordings and 742 reference characters, including 15.24 seconds of annotated simultaneous three-person speech. The two REAL-T excerpts come from the same meeting. Each row compares identical input, enrollment and reference text across runs where enrollment applies.

Each cell gives the number of character edits and character error rate (CER). Lower is better. “Legacy fixed” keeps its arbitrary output filenames in A/B/C order. “Legacy best global” lets that anonymous system choose its most favorable single mapping using the reference after decoding. Named systems must keep their enrolled A/B/C mapping for the fixed score.

| Excerpt | Audio | Three active | Reference characters | Legacy fixed | Legacy best global | Enrolled auto fixed | Enrolled forced fixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| M8009 | 20.00 s | 4.25 s | 193 | 358 (185.5%) | 350 (181.3%) | 175 (90.7%) | 177 (91.7%) |
| M8014 | 60.00 s | 1.15 s | 327 | 706 (215.9%) | 692 (211.6%) | 510 (156.0%) | 624 (190.8%) |
| REAL-T 01 | 15.66 s | 4.97 s | 125 | 296 (236.8%) | 287 (229.6%) | 169 (135.2%) | 340 (272.0%) |
| REAL-T 03 | 18.01 s | 4.87 s | 97 | 241 (248.5%) | 196 (202.1%) | 164 (169.1%) | 169 (174.2%) |
| Pooled | 113.67 s | 15.24 s | 742 | 1601 (215.8%) | 1525 (205.5%) | 1018 (137.2%) | 1310 (176.5%) |

The pooled row adds edits and reference characters before dividing. The enrolled automatic route produced 341 substitutions, 271 deletions and 406 insertions. Its total differs from the anonymous best mapping by 507 fewer edits. Track-specific errors vary substantially. The two independent REAL-T runs allow different anonymous assignments in the main table. If the same assignment must span both clips, the legacy route incurs 485 edits on 222 reference characters (218.5% CER).

## Examples from the measured exports

- In M8009, enrolled auto track A captured “都比都比较自觉”, which appears in reference A. Later, auto track B produced “没有那人了”, close to “没有那人啦” in reference A. A recognizable phrase can reach a different person's file.
- In M8014, enrolled auto track C produced “计较” 52 times. All three references contain zero instances. Such decoder repetitions add insertions. CER can exceed 100%, and it measures the combined separator, assignment and recognizer. It cannot establish a percentage of acoustic isolation.

## Measurement

The AliMeeting reference uses the original per-speaker TextGrid. Characters for utterances cut by an excerpt boundary use the midpoint of independent MultiTalk word alignments. REAL-T uses its released full per-speaker Chinese transcription for the corresponding mixture. References preserve disfluencies. OpenCC converts to simplified Chinese and the scorer lowercases Latin text, removes punctuation and whitespace, and keeps Chinese characters (U+3400–U+9FFF), ASCII letters and digits. Arabic numerals and Chinese number words remain distinct.

Each output WAV is decoded in full with the same pinned multilingual Whisper small model, Chinese language, CPU int8, two threads, beam size 3 and deterministic temperature 0. Prior-text conditioning and voice activity filtering are disabled. The experiment preserves all returned segments, including exceptionally repetitive output. The scorer counts the edit distance between each speaker's complete reference and hypothesis. It charges all words from an unmatched extra output as insertions and treats an omitted speaker as an empty output. Tied minimum edit paths may distribute counts differently across operation types while preserving total CER.

The legacy route uses the three-speaker hint and anonymous IDs. Both enrolled routes get disjoint labeled voice samples of three participants. Auto estimates the active count, and forced supplies the three-speaker hint. The fixed-to-best gap measures how many edits a whole-file relabel can remove. Assigning each edit to recognition, leakage or identity requires additional listening or an independent alignment of each spoken character to a speaker. The assessment covers the complete selected excerpts. Substantial portions of each contain one or two active voices.

## Reproduce the scorer

From a directory containing the script and requirements file, install the pinned packages and fetch the four decoder files. The model revision is pinned to the weights used in the table.

```sh
python -m pip install -r requirements_asr_evaluation.txt
hf download Systran/faster-whisper-small config.json model.bin tokenizer.json vocabulary.txt --revision 536b0662742c02347bc0e980a01041f333bce120 --local-dir whisper-small
python score_mandarin_tracks.py decode --reference reference_text.json --tracks exported_meeting --model whisper-small --output decoded.json
python score_mandarin_tracks.py rescore decoded.json --output rescored.json
```

The reference file contains a `fixed_text` object with full transcript strings under keys `A`, `B` and `C`. The export directory must contain its `speakers.json` and all WAV files it lists. The scorer writes every decoded segment for inspection. The companion JSON records exact speaker-level counts, pairwise distances for cross-checking alternative global assignments, file digests, recording IDs, offsets, and pooled comparisons for all three routes.

Corpus attribution and versions are available from [AliMeeting](https://www.openslr.org/119/), the [AliMeeting TextGrids](https://huggingface.co/datasets/AudioMarathon/AudioMarathon/tree/68dd695283723a78d850895466b9a380f630b561/AliMeeting/Test_Ali/Test_Ali_far/textgrid_dir), [MultiTalk Chinese alignments](https://huggingface.co/datasets/MultiTalk/MultiTalkBench/tree/d7188c179603db8566c50b6e43af5c8a1e7678ba/alignments/zh), [REAL-T EVAL2](https://huggingface.co/datasets/REAL-TSE/REAL-T/tree/39ca3b88ffd4ce6078c1d045a3e8711fa2bd2f17/REAL-T-eval2/EVAL2), and [the Whisper model](https://huggingface.co/Systran/faster-whisper-small/tree/536b0662742c02347bc0e980a01041f333bce120).
