# Array rendering and recognition audit

## Corrections

- The array context decoder now clips speaker turns to the available sample clock. Earlier versions dropped an entire turn when its ending crossed a pilot or recording boundary. A wholly negative ending could also create a negative array slice and mark nearly every frame as active. Invalid labels, nonfinite timestamps and reversed intervals raise a clear error. Optional text evaluation with zero reference speech returns an undefined cpCER.
- When the decoder duration cap prevents merging two spans, nearby padded regions share a single cut point. This removes same-speaker windows that formerly decoded the same speech twice. Fully nested windows already represented in an earlier span are skipped. The change affects context spans, and the original diarization remains unchanged.
- The context decoder processes beam power in short file blocks and loads PCM only for the span being decoded. The three tracks no longer need to fit in memory as full float arrays.
- The optional overlap-mask renderer now reads one neighboring STFT frame on either side of each processing block before temporal likelihood smoothing. This makes masks at implementation boundaries use their correct neighborhood. The static MVDR path keeps its original spectra and output samples.

## Verification

| Comparison | Before | After |
| --- | ---: | ---: |
| Beam-power preparation, M8016 peak process RSS | 642.40 MiB | 86.16 MiB |
| Beam-power preparation, M8016 wall time | 2.21 s | 2.16 s |
| Speaker-attributed character edits, M8016 | 1,361 / 7,705 | 1,358 / 7,705 |
| Speaker-attributed character edits, M8017 | 533 / 8,154 | 534 / 8,154 |
| Pooled array character accuracy | 88.06% | 88.07% |
| Exact-three controlled median SI-SDRi, room M8016 | +0.76358 dB | +0.76375 dB |
| Exact-three controlled median SI-SDRi, room M8017 | +3.98680 dB | +3.98670 dB |

The beam-power arrays had identical SHA-256 hashes across all three half-hour beam files. The peak RSS is the separate preparation stage on one CPU thread with no ASR model loaded. The text comparison reran the same SenseVoice model for all 34 changed span windows, checked a retained span from every speaker, reused cached identical-window results, and applied the same fixed cross-speaker duplicate filter. The old spans duplicated 1.80 s in M8016 and 0.48 s in M8017. Text edits counted here include substitutions, deletions and insertions after one global speaker match.

The overlap-mask waveform comparison uses six original controlled mixtures across two held rooms, for 18 known speaker tracks. The corrected halo changed at most 93 int16 units in any PCM sample and left median projected SIR improvement at +3.68105 dB and +7.82672 dB, respectively. The static two-channel MVDR path separately passed identity and sample-clock checks, with a direct old/new comparison producing identical float32 samples. These controlled waveform tests measure the renderer. The text comparison uses the established static array beams.

Five additional multichannel Wiener profiles were screened on the separate M8013 development room. None improved both median exact-three SI-SDR improvement and median projected SIR improvement. The spatial overlap mask remains the chosen profile.

The seven regression cases cover invalid input, clipping at recording edges, nested and repeated decode spans, streamed feature arithmetic, block-invariant likelihoods and sample-aligned rendering. With the repo's ordinary Python dependencies installed, add `benchmarks/three_speakers`, `benchmarks/three_speakers/advanced_separation`, and `benchmarks/three_speakers/root_causes` to `PYTHONPATH` and run `python test_array_boundaries.py`.
