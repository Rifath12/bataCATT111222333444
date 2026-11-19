# Testing whether a speaker cue recovers a quiet third voice

The experiment compares the existing three-speaker SepFormer separator and an independently enrolled WeSep BSRNN. We compared original SepFormer, SepFormer with the earlier weighted prediction error (WPE) dereverberation, and SepFormer with the released spectral refinement. The speaker-conditioned candidates used a different utterance from each voice. The residual candidate attenuated louder estimated voices in the original mixture, then reran extraction for quieter targets. A fixed identity margin also tested choosing the BSRNN estimate only when the enrolled voice matched it more distinctly than the SepFormer estimate.

The development panel used the same three simulated-room AISHELL-3 mixtures as the earlier acoustic check. A second panel was locked by a random seed before inference. It has three trios of speakers absent from the development panel and earlier controlled tests. Each mixture combines utterances with +0, −4 and −8 dB relative active levels, correlated room reflections and independent noise at 15 dB SNR. The application does not know the source audio, source levels, or target reference texts at inference. The experimental extractor does use independent enrollment utterances.

| Development arm | SenseVoice character edits / 103 | Median 3-way improvement over mixture | Median weakest-voice 3-way improvement |
|---|---:|---:|---:|
| Original SepFormer | 52 | +4.44 dB | +2.22 dB |
| SepFormer with WPE | 45 | +5.33 dB | +5.33 dB |
| Released spectral refinement of SepFormer | 58 | +4.29 dB | +2.39 dB |
| BSRNN direct extraction | 67 | +4.52 dB | +0.79 dB |
| BSRNN second pass after louder estimates are subtracted | 75 | +2.31 dB | +0.86 dB |
| Equal waveform average of direct and residual BSRNN | 73 | +4.52 dB | +1.37 dB |
| Enrollment margin selection with released refinement | 58 | +4.29 dB | +2.39 dB |

The development panel has 13.401 s of mixture audio and 3.06 s of all-three-speaker activity measured on the isolated source signals. The waveform metric is scale-invariant signal-to-distortion improvement in the exact three-speaker regions over the original mixture, measured against the reverberant source images and using one waveform assignment per clip. The character metric counts pooled substitutions, deletions and insertions after one transcript assignment per clip. Fixed enrollment-label errors are also reported for cue-based methods. The fixed identity-margin rule made no replacements on development.

The speaker-conditioned separator did not improve waveform isolation and recognition for the weakest voice consistently. On the first development trio it improved weakest-voice isolation from +0.15 to +5.11 dB while overall transcription went from 14 to 19 errors. The remaining trios reversed the waveform benefit. Inference took 52.6–78.6 seconds on one shared CPU thread for clips of 3.7–5.8 seconds when both passes were run. The comparisons concern this specific checkpoint and fixed extraction strategies.

## Fresh nine-voice stress test

The 25.60 seconds of audio contained 2.82 seconds marked as all-three activity by the isolated dry source energy rule. The independently enrolled BSRNN only improved the third trio's character count by one edit and made 34 more errors overall than the released spectral refinement. Its median weakest-voice improvement was also lower. The same SepFormer controls are evaluated with the same mixture and matched source references in each case.

| Held-out arm | SenseVoice character edits / 132 | Exact phrases / 9 | Median 3-way improvement over mixture | Median weakest-voice 3-way improvement |
|---|---:|---:|---:|---:|
| Original SepFormer | 73 | 1 | +4.49 dB | +7.24 dB |
| SepFormer with WPE | 70 | 1 | +4.44 dB | +6.79 dB |
| Released spectral refinement of SepFormer | 71 | 1 | +4.45 dB | +8.04 dB |
| BSRNN direct extraction | 105 | 1 | +0.97 dB | +1.51 dB |

Direct anchored extraction took about 230 seconds to render 25.60 seconds of mix on one CPU thread in a shared run. The results support retaining the released separator. The experiment uses simulated rooms and independent enrollment recordings. Natural meeting performance requires its own evaluation.

## Reproduction

Run `prepare_holdout.py` using the pinned AISHELL-3 corpus metadata and transcript list, then use the paths and dependency setup of the previous acoustic waveform benchmark. `run_anchored.py` has three stages. Use `--part tse` for anchored extraction, `--part sep` for the two SepFormer controls and `--part score --backend sense` to decode with SenseVoice and calculate both metrics. If the previous acoustic benchmark cache is available, provide its directory as `--prior-cache` for the development separation stage. Supply `--voiceprint` to score the fixed-margin routing arm. For the fresh stress test, use `--variant direct` in both the extraction and scoring stages. For development and hold-out data use their separate manifest and speaker source directories.

For development, supply `--whole-extraction` when making the target extraction cache. Use also `--limit 1 --whole-extraction` to reproduce the first target extraction on the fresh data before running the remaining cases. During the fresh stress test the direct extractor evaluated the first case whole. The remaining cases used 3.5-second windows with 400 ms crossfades after a whole-context run of the largest test mixture exceeded available memory. The same enrollment identities anchor all frames, so chunk stitching needs no identity matching.

[AISHELL-3 dataset and license](https://www.openslr.org/93/) · [WeSep challenge baseline](https://github.com/REAL-TSE/REAL-TSE-Challenge)
