# Waveform separation target

A/B/C waveform quality is scored on aligned, known-source audio. Keep the intended speaker label fixed wherever its score is computed. The evaluation is independent of language recognition and transcript editing.

## Measurements

| Measure | Interpretation |
| --- | --- |
| Target-aligned output energy | Center reference `s` and output `y` by removing their means. Fit the scalar `g = (s·y)/(s·s)`. The target fraction is `||gs||²/||y||²`. This quantifies the part of the output that agrees with its reference waveform. |
| SI-SDR | Express the same energy decomposition as `10 log10(||gs||² / ||y−gs||²)`. The 96% and 98% fractions correspond to **+13.80 dB** and **+16.90 dB absolute SI-SDR**. SI-SDR improvement is reported separately against the unseparated input. |
| Competing-voice energy | Project the residual onto the other two known source waveforms after removing their component along the target. The rest quantifies other error, including recording artifacts, noise and waveform mismatch. |
| Preserved reference-active time | On fixed nonoverlapping 100 ms blocks with at least 30 ms of reference-active audio, estimate the target projection again. Report the fraction of eligible active time with projected level at least −12 dB relative to the original target. The project uses 96% retained time as an additional quality guard so a very quiet copy cannot pass based on its scale-invariant score. |
| Output on another person's speech | When the source timeline gives such intervals, report output levels during another person's speech while the intended voice is inactive, relative to the intended reference level. This catches leakage outside simultaneous regions. |

A control passes the 96% or 98% grade only when each of the three tracks achieves that target-aligned fraction and passes the 96% preservation guard. Report all cases, the weakest track and quietest voice per case, the duration scored, and per-window scores. The 250 ms analysis shows how much scored overlap time has three simultaneous target passes. Local permutations are a diagnostic for potential label switches and never applied for reported grades.

An inaudible output can have a near-perfect target fraction. Exact silence yields a zero fraction and fails. Reference blocks with negligible centered target energy or insufficient activity are reported separately from eligible coverage. SI-SDR dB is limited to ±120 for finite JSON. The thresholds describe this project's engineering test and do not provide a perceptual or population claim.

[Le Roux et al., SI-SDR](https://research.google/pubs/sdr-half-baked-or-well-done/)

## Replay the released controls

From the project workspace with NumPy, SciPy and SoundFile installed, add the established array and benchmark modules to `PYTHONPATH` and run `python vtarget/eval/replay_prior.py`. The script reconstructs the six existing array-microphone controls from archive provenance and verifies the sample clock and mixture against the saved WAV files. It reads the same audio and exact-three sample masks for the three bundled simulated room mono comparisons. The packaged mono tracks inherit a prior clip-wide offline reference assignment. This replay checks in-clip fixed labels without claiming blind speaker reassignment across calls.

The two test families use different sources, environments and activity definitions, so compare method variants within each family. Array activity is the original controlled interval in which three individually recorded solo excerpts overlap. The mono exact-three mask comes from the existing source activity labels. Two modes share the same source rooms for the array controls. Repeat recordings, per-speaker scores and adjacent blocks are correlated, so the results are descriptive. The scorer only calculates exact binomial intervals when explicitly asked to treat the supplied groups as independently sampled. Real overlapping recordings with a single microphone lack isolated waveform references. The acceptance test cannot confirm a 96–98% result from two long recording durations with only a few seconds of exact overlap.

## Score a candidate

Create an NPZ containing `sample_rate` and sample-aligned arrays `references`, `tracks`, and `activity`, each indexed `[A/B/C, sample]`. Use source-derived Boolean activity. Optional arrays are `mixture` for the common original room signal or three original channels, `exact_three` to select a subset of known overlap samples, and `group` for a shared room or speaker group. Never use isolated candidate reference audio to tune the model, align each scoring window, or guide separation at inference.

```text
python waveform_acceptance.py --out results.json case_1.npz case_2.npz
python -m unittest discover -s . -p 'test_waveform_acceptance.py'
```
