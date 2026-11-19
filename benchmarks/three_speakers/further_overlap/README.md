# Three-speaker Mandarin overlap follow-up

[Latest no-regression checks, runtime improvements and optional recognition](../quality_matrix/README.md).

The offline mono pipeline now compares already-computed contextual recognition against the isolated-speaker reading at the same time. It can retain a stronger reading or restore short leading or trailing phrases where both readings share a character anchor. A frozen selector estimates likely benefit from the turn length, overlap, speech rate, character agreement and the type of change. The same duplicate check rejects newly imported copies from quieter tracks.

| Verified comparison | Reference characters | Previous edits | Refined edits |
|---|---:|---:|---:|
| M8016, 30 min 39 s | 7,705 | 1,927 | 1,838 |
| M8017, 30 min 20 s | 8,154 | 988 | 953 |
| Both complete recordings pooled | 15,859 | 2,915 | 2,791 |
| Three additional previously scored speaker-disjoint excerpts | 1,128 | 355 | 343 |

Speaker-attributed character error on the two complete meetings fell from 18.38% to 17.60%. The 124-edit improvement combines 382 fewer deletions with 204 additional substitutions and 54 additional insertions. Speaker C gained 64 edits on M8016 and 23 on M8017. The six output waveforms in the full-meeting listening comparison come from the previously verified mono separator. Median selector overhead after the existing recognizers was approximately 0.1 s per complete file on the evaluation CPU. The new rule is calibrated on the separate M8013 meeting only. It was selected using five chronological development blocks. [Method, tests and per-speaker counts](CACHED_REFINEMENT.md)

The normal offline mono entry point calls the selector. The standalone recognizer exposes it with `--refine-context` and requires `--context-mix` and `--backend sensevoice`. The CLI uses the sampled room signal and three already separated tracks at 16 kHz, with the same recording time origin. [Offline command and model files](../root_causes/README.md)

## Simultaneous three-speaker evaluation

The new AliMeeting source panel has 18 disjoint mono excerpts from three previously unscored sessions. Each source meeting had four participants. Each extracted excerpt contains exactly three annotated speaker identities and excludes the fourth. The panel totals 424.97 seconds with 26.25 seconds marked as exactly three speakers active. All 18 clips were selected by the reference activity before inference. The cropped character reference comes from MultiTalkBench's token times. Its normalized text for the 12 source speakers matches the official TextGrids with zero edits over 38,733 characters. [AliMeeting](https://www.openslr.org/119/) · [MultiTalkBench alignment revision](https://huggingface.co/datasets/MultiTalk/MultiTalkBench/tree/d7188c179603db8566c50b6e43af5c8a1e7678ba)

On the fresh 18 excerpts, the prior dictionary and narrow-boundary profile detected all three in 21.30 of 26.25 seconds and made 19.61 seconds of false triple detections. A frozen AliMeeting-specific classifier estimated the active speaker count and the likely identity of each speaker using two Nemotron inputs, acoustic features and DiariZen. It then selected among the eight possible activity states over time. The evaluation operating point was selected on earlier sessions.

| Timing on the same 18 fresh excerpts | Exact-three precision | Exact-three recall | Exact-three F1 | Diarization error |
|---|---:|---:|---:|---:|
| Previous dictionary and narrow boundaries | 52.07% | 81.14% | 63.43% | 16.59% |
| New structured AliMeeting profile | 72.77% | 72.27% | 72.52% | 11.87% |
| AliMeeting profile using Nemotron and acoustic features | 73.25% | 69.56% | 71.36% | 11.66% |

The primary new profile produced 7.10 seconds of false three-speaker activity and found all three during 18.97 seconds. Partial speaker-time recall during true triple overlap was 90.10%, which credits each correctly identified active speaker. On REAL-T's separate 26-clip test, REAL-T-specific calibration gave 52.60% exact-three F1 and 17.74% diarization error. Transferring that REAL-T model to the new AliMeeting clips gave 50.78% F1 and 30.22% diarization error. The domain-specific timing is opt-in. On one preselected listening example from each of the three source sessions, passing the new A/B/C intervals to the same audio renderer and direct recognizer reduced 130 to 120 edits out of 342 characters. Its time-constrained subset of 88 characters during true three-way speech improved from 45 to 40 edits. Combining those timing changes with the previous contextual recognizer increased edits from 155 to 158 on the same three examples. For overlap-heavy files, pass the frozen timing JSON to `process_audio.py --timing speaker_timing.json --fast-transcript`. This flag also runs the previous duplicate check. With that check enabled the same three examples improved from 128 to 117 edits, and from 44 to 40 in the time-constrained overlap subset. [Matched audio, timing and recognition scores](data/top1_matched_timing.json) · [Direct recognition comparison](data/top1_structured_stem.json) [Frozen timing models, exact counts and commands](timing/README.md)

The fixed text panel contains the two selected excerpts from each fresh session, for six files and 164.76 seconds in all. It covers 10.87 seconds of three-way speech. The recognition algorithms below use the same A/B/C tracks. The frozen contextual update improved accuracy over the previous contextual implementation. Direct recognition of the masked speaker tracks performed best on this particular overlap-rich panel.

| Recognition mode | Entire six-clip panel, 1,146 chars | True three-way speech, 153 aligned chars |
|---|---:|---:|
| Previous contextual recognition | 439 errors (38.31%) | 103 errors (67.32%) |
| Refined contextual recognition | 369 errors (32.20%) | 88 errors (57.52%) |
| Direct recognition of the speaker tracks | 336 errors (29.32%) | 73 errors (47.71%) |

The offline mono wrapper enables the contextual selector in normal mode. Use `--fast-transcript` to reproduce direct recognition when processing overlap-heavy files. An extra context-selection rule using competing speakers' text, local overlap fraction, and time ownership did not improve its dev data. [Full per-clip text results](data/transcript_comparison.json)

For the overlap-conditioned transcript metric, one global mapping is chosen per excerpt using the time-constrained edit count. A character match requires its reference and hypothesis midpoints to be within one second. Reference token times come from the corpus alignment. Hypothesis positions are uniformly interpolated within ASR turns. Reference characters and substitutions and deletions are classified by the reference activity at their midpoint. Insertions are classified by the hypothesis midpoint. The reported quiet or simultaneous counts can therefore differ from alignment over concatenated text without timestamps. The audio asset supplies per-speaker tracks and timestamped transcript exports from selected real excerpts. [Overlap metric implementation](overlap_characters.py) · [Time-constrained evaluation rationale](https://github.com/fgnt/meeteval/blob/main/doc/tcpwer.md)

## Separation experiments

We tested speaker-enrolled extraction on nine new AISHELL-3 voices recorded independently and mixed three at a time. On 132 reference characters, the current separator produced 71 edits and the enrolled extractor produced 105. Its weakest-voice median improvement over the mixture fell from +8.04 dB to +1.51 dB during three-way speech. A second pass that subtracted louder estimated voices was also worse in development. Phase-based reconstruction added 23–24 edits on the independent 37-minute development recording and regressed on the earlier controlled trios. The new offline release retains the verified waveform engine and can use the optional timing profile to change where that engine assigns audio. [Controlled signal and text results](waveform/README.md)

The 90% threshold for three simultaneous Mandarin voices remains unverified. The full-file results include all other speech and silence conditions. File inference was evaluated offline.
