# Adaptive multichannel waveform separation

The optional adaptive covariance profile uses real synchronized eight-channel
16 kHz microphone audio and an existing A/B/C timing file. Each speaker needs
at least 0.3 seconds of usable interior solo speech for spatial enrollment,
as in the previous array profiles. The decoder estimates each person's
spatial covariance and a background covariance from that recording. Eight
power updates fit the sum of those components during predicted overlap.
Multichannel Wiener reconstruction estimates each speaker's image at
microphone zero. The A/B/C labels remain tied to the enrollment. Solo beams
stay unchanged before common output-level normalization. The existing MVDR
and likelihood-mask modes remain available. [Full-rank spatial covariance model](https://arxiv.org/abs/0912.0171)

From the repository root:

```bash
PYTHONPATH=benchmarks/three_speakers python \
  benchmarks/three_speakers/root_causes/spatial_render.py \
  --audio room8.wav --timing timing.json --profile adaptive-covariance --out separated
```

The FFT processing carries the original sample clock across chunks and uses
enough neighboring frames to make the mask invariant to chunk boundaries.
Three new regressions verified that partitions agree, solo spectra stay
untouched, an inactive source emits zero, and silent overlapping input emits
zero.

## Fixed A/B/C waveform tests

The candidate was selected on eight controlled development mixtures built
from M8013. The unchanged profile was then scored on six separate exact-three
mixtures created from M8016 and M8017, using the original arrays' recorded
solo source images with held A/B/C labels. The six mixtures contain 11.06
seconds of exact three-speaker overlap and 18 scored output voices. The
current likelihood mask was replayed on the same source images. No speaker
permutation was selected from the evaluation references.

| Median over exact-three outputs | Previous mask | Adaptive covariance |
| --- | ---: | ---: |
| Target-aligned energy, all voices | 38.98% | 42.12% |
| Competing voice energy, all voices | 30.91% | 22.95% |
| Other error energy, all voices | 27.61% | 34.15% |
| Absolute SI-SDR, all voices | −1.95 dB | −1.38 dB |
| Target-aligned energy, quietest voice | 15.65% | 18.41% |
| Retained reference activity, quietest voice | 74.24% | 87.12% |

The adaptive profile reduced competing-voice leakage and improved reference
activity retention pooled across both rooms. Competing-voice energy fell
in M8016 and rose slightly in M8017. The other error component increased
in both. Target-aligned energy improved for 13 of 18 voices. These column
medians are calculated over different voice rankings and need not sum to
100%. Zero of the six trials met 96% or 98% target-aligned energy with the
required speech retention. These are controlled mixtures, so the numbers
should not be used as live or authentic-overlap accuracy estimates.

The physical array provided weak spatial evidence in exactly the frequency
bands that carry much of the speech. Across the held rooms, 43–44% of
reference channel power was below 500 Hz. The median steering matrix
condition number in that band was 12–15. The smaller third singular value
makes separation more sensitive to errors in the enrolled source directions.

On CPU, each 11.475-second controlled recording took 3.08–3.60 seconds to
enroll, render and write three PCM WAVs, with timing supplied. These times
exclude any speaker-timing estimator and transcription. A complete
30:20 M8017 file rendered in 76.28 seconds on one CPU thread. All three
WAVs contained exactly 29,120,832 mono samples at 16 kHz. This full file
verifies timing and export and lacks independent source references for a
full-meeting waveform accuracy score.

[Profile and measured evidence](array_waveform_evidence.json)
