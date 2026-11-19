# Single-channel waveform isolation audit

These tests use three previously locked 2.8-second room crops and an independent historical 2.8-second hard-room crop, with disjoint speakers across the four crops. There are 5.82 seconds of exact-three activity under the existing isolated-source activity rules. All models receive the same 8 kHz mono mixtures. Scoring keeps one A/B/C alignment per existing whole-window output. That alignment is known from the cache, so this experiment measures waveform isolation given the label alignment. It gives no meeting-length voice-identity accuracy estimate.

Target-aligned energy fraction is the centered squared projection of each output onto its isolated reverberant target, divided by the centered output energy. This measure includes distortion and interference in its denominator. It is the correlation-squared form of SI-SDR. A target fraction of 96% implies 13.80 dB absolute SI-SDR, and 98% implies 16.90 dB. These are absolute levels, while SI-SDR improvement measures the change over the mixture. Source gain and retained active duration are separate checks.

| Fixed mono output during exact-three activity | A target fraction | B target fraction | C target fraction |
| --- | ---: | ---: | ---: |
| Room crop 1, original waveform fallback | 55.5% | 18.3% | 17.8% |
| Room crop 2, accepted RecorrNet | 86.7% | 96.4% | 96.1% |
| Room crop 3, accepted RecorrNet | 90.7% | 64.0% | 96.3% |
| Independent room crop, accepted RecorrNet | 95.6% | 91.4% | 95.8% |

Three of twelve speaker outputs reach 96%. None reach 98%, and no crop passes for all three speakers even before enforcing activity coverage. The median output target fraction is 91.05%. The smallest is 17.77%. Approximate 99.4% reconstruction of the original mixture coexists with only 64.0% target-aligned energy on one separate output. Reconstruction tests aggregate conservation and cannot determine whether each voice was routed correctly.

The shared scorer decomposed this 64.0% case into 2.73% energy aligned with competing voices and 33.31% residual error under its projection convention. Retained active sample duration for that speaker was 100%. The poorest fallback stem had only 26.80% retained active duration, with 17.83% competing-voice energy and 64.40% residual error. See `results/shared_exact_three.json` for all twelve rows.

## Additional candidates

The second RecorrNet view runs the same checkpoint on a reversed waveform and reverses its outputs. Only a previously rejected window is eligible. The existing positive mixture-fit gate remains frozen. An extra check requires a 0.02 difference between the two best incumbent waveform label alignments, the same threshold used by the prior natural diagnostic. No expected target waveform enters inference or selection. One rejected controlled window explained effectively 0% of the mixture. Two natural AliMeeting windows explained 3.28% and 16.48%, versus the frozen 90% minimum. Their label margins were 0.0024 and 0.0043. All three fell back without modifying exported audio. CPU inference added about 13.5–32.7 seconds per attempted window in the shared runner. No natural isolated reference exists for these two windows, so these figures are mixture-fit diagnostics rather than natural separation scores.

The independent dictionary trial draws up to 96 magnitude spectra per speaker from separately recorded, hash-verified enrollment. A fixed nonnegative factorization allocates mixture magnitudes to three person-specific dictionaries and a broadband nuisance. Wiener masks preserve the original mixture phase. The equal blend averages this estimate and the incumbent after positive mixture gain calibration. The median exact-three target fraction was 21.84% for the dictionary and 68.81% for the blend. Their lowest per-track target fractions were 2.11% and 13.13%. Neither qualifies to replace the current mono output.

## Reproduction

Run from the project workspace containing the prior controlled fixture Web releases. The scripts read the canonical caches and the same speaker activity provenance. Propagating the shared acceptance suite over these results yields the additional output-interference decomposition and retention statistics.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/vnext_check/wave:$PWD/vnext_separator:$PWD/vnext_separator/deps"
python vtarget/mono/two_view.py --out vtarget/mono/results --mode reverse --panel all
python vtarget/mono/spectral_cues.py vtarget/mono/results/spectral_cues.json
```

The reproduction uses the pinned 8 kHz Mandarin RecorrNet model and existing cached SepFormer estimates. The natural test uses an AliMeeting excerpt. The dictionary files come from a separate AISHELL-3 speaker enrollment utterance per person. [AISHELL-3 dataset](https://www.openslr.org/93/) · [AliMeeting dataset](https://www.openslr.org/119/) · [RecorrNet model revision](https://huggingface.co/playwithmino/recorrnet-v2-1ch-aishell1mix-ver2-alimeeting/tree/96357df71553dd777f7f775367028bbbdf184573)
