# Locked three-room headset evaluation

The published 2026-11-20 panel contains 156 Mandarin clips with three known target waveforms. The recordings came from three AliMeeting rooms. Each original meeting had four participants. Every donor interval contains exactly one annotated speaker across all four meeting participants. The three selected headset channels preserve the measured crosstalk from that person into the other two microphones. The fourth participant supplies no donor and has no active RTTM interval within any donor span. These controlled remixes have known source contributions. The headset diagonal includes any ambient sound captured during the original donor interval.

| Physical room / meeting | A, B, C in headset order | Clips | Samples with all three voices active, in seconds |
|---|---|---:|---:|
| R8004 / M8005 | 8017, 8018, 8020 | 32 | 22.560 |
| R8005 / M8008 | 8039, 8040, 8041 | 53 | 39.648 |
| R8006 / M8012 | 8070, 8071, 8072 | 71 | 52.256 |
| All rooms | Nine distinct headset speakers | 156 | 114.464 |

Rooms and target IDs were selected before model scoring based on availability of isolated, annotated speech. Earlier published experiments used different portions or modalities from the wider AliMeeting corpus. The three rooms and nine speakers are all distinct from the M8014 supervised training room and the earlier M8017/M8016 close-mic controls. Results describe short constructed overlaps. Each repeated donor pairing would share measurement information, so every speaker interval is used only once. Clips within a meeting still share the same room, microphones and speakers.

## Generation and checks

1. `sources.json` pins nine PCM files to a Hugging Face mirror revision and three RTTMs to a separate annotation repository revision. SHA-256 checks run on every input before sample access. The upstream corpus is AliMeeting, distributed through [OpenSLR 119](https://www.openslr.org/119/).
2. Use only annotated single-speaker stretches after the first 45% of each recording, trimmed by 200 ms at each boundary. Build non-overlapping 1.024-second donors. On each selected direct channel, divide by the RMS of its active frames so that active RMS is 0.06. Activity uses 32 ms frames, the larger of 0.0015 RMS and 16% of the within-donor 70th-percentile RMS. Require at least 70% active frames, active direct RMS ≥ 0.002 before scaling, and competing headset RMS ≤ 70% of direct active RMS. This removes ambiguous donors and limits coverage of extreme contamination already present in the isolated reference. The mask defines overlap at 32 ms resolution.
3. Shuffle eligible donors independently with a documented fixed seed. Set one target at 0, –6, –12 or –20 dB in a fixed cycle, rotating which microphone is quieter. The three spatial images sum to the mixture. Check disjoint source time spans, sufficient active overlap and float headroom. Every trial has known fixed A/B/C headers, full three-channel isolated source images and three direct-channel reference waveforms. Every array file includes both activity and its intersection across the three voices.
4. Generate the physical power calibration using only early speaker-labelled solo intervals from the same room, measured across all three synchronized microphones; the held remixes use only later intervals. This comparison assumes access to calibration with isolated speakers. The released spectral power estimator uses the same fixed percentile and clipping bounds. The scratch files `published_transfer.npy` and `calibration.json` record the matrix and its provenance.
5. Score the original A/B/C channel mapping. A clip passes a threshold when all three voices reach the specified target-aligned energy fraction and each retains at least 96% of eligible active speech time at the existing –12 dB gain floor during triple overlap. An identity permutation is never applied. Filter development and training must finish before examining candidate scores from these rooms.

With Python ≥3.11 and NumPy, SciPy and SoundFile installed, from the panel directory run the following. Set `ADVANCED` to the repository's `benchmarks/three_speakers/advanced_separation` and `QUALITY` to `benchmarks/three_speakers/waveform_quality`.

```sh
python get_sources.py
PYTHONPATH="$ADVANCED:$QUALITY" python build_locked_headsets.py --prepare-power-transfer
OPENBLAS_NUM_THREADS=1 PYTHONPATH="$ADVANCED:$QUALITY" python score_locked_baselines.py
PYTHONPATH="$ADVANCED:$QUALITY" python score_locked_baselines.py --summarize
```

`lock.json` records source-list, generator and per-room manifest hashes. Each generated manifest lists file hashes and deterministic uncompressed waveform fingerprints. Reproduction can compare the payload fingerprints when archive serialization differs. Room groups share no evaluation speakers. Use room-wise and level-wise results alongside the pooled clip counts.
