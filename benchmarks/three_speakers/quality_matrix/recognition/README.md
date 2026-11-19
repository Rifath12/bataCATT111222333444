# Mandarin character consensus

The second recognizer checks individual text differences without changing speaker labels, timestamps or audio. In each speaker turn, the current transcript defines an alignment grid. The program aligns SenseVoice's isolated and mixed views and Zipformer CTC text to that grid. It uses fixed agreement and opposition rules to make character substitutions, recover short additions, and remove repeated characters. Empty auxiliary transcripts provide no votes. The optional research calibration also uses a sparse FireRed sample. The no-FireRed calibration selected the same parameters and is the practical configuration.

The rules were fixed on M8013 with five chronological score blocks. The selection constraint required the sum of substitutions, deletions and insertions each to stay at or below the baseline across those blocks and total errors to improve in at least three blocks. The frozen no-FireRed setting reduced substitutions by 30 and deletions by six, with insertions unchanged. All five blocks improved. These are development-selection results.

| Complete mono recording | Errors | Substitutions | Deletions | Insertions |
| --- | ---: | ---: | ---: | ---: |
| M8016, 7,705 reference characters | 1,838 → 1,818 | 993 → 983 | 694 → 685 | 151 → 150 |
| M8017, 8,154 reference characters | 953 → 925 | 535 → 517 | 348 → 345 | 70 → 63 |
| Both recordings | 2,791 → 2,743 | 1,528 → 1,500 | 1,042 → 1,030 | 221 → 213 |

Every speaker's combined edit count improved on the full-meeting replay. Speaker B in M8016 had one additional insertion and speaker C had one additional substitution. Two of the three additional earlier-session short excerpts improved by a combined five edits. One short excerpt worsened by one. Neither full recording was used to select the setting, but both have previously appeared in the project. The score improvement is 82.40% to 82.70% when computed as one minus speaker-attributed character error. Both recordings had three participants and the evaluation includes solo and overlapping speech.

A 10,000-draw paired bootstrap of fixed speaker mappings and 60-second text blocks found an approximate central 95% range of 18 to 78 fewer pooled errors. The blockwise score recomputes the character alignment within each minute and gives a central change of 47 edits. This diagnostic describes these two recordings. For insertion errors alone, the range includes two extra errors.

The sparse FireRed version reached 2,742 total errors on the same two recordings. Its six-clip overlap replay reduced 336 to 331 pooled errors while adding one deletion. Overlap-conditioned scoring was unchanged at 73 errors for 153 reference characters. FireRed's scope was a selected disagreement pilot, not a complete additional decode. A time-local cross-speaker substring filter failed the development requirement and selected no changes.
