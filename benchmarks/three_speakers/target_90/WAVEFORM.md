# Three-voice waveform isolation evaluation

The trial starts with the project's current Libri3Mix SepFormer. Development compares the official WSJ0-3Mix SepFormer, inference on both mixture polarities, a waveform consensus between the two checkpoints, and a conservative time-frequency mask. The consensus aligns the estimated tracks by waveform correlation, then accounts for polarity and amplitude before averaging. It receives no reference tracks.

The two development triplets and six test triplets use disjoint AISHELL speaker IDs. Four test triplets appeared in earlier project tests, and two are additional three-person combinations. The mixtures themselves are new. Each triplet has a clean condition and a simulated room condition with reverberation and approximately 15 dB mixture-to-noise ratio. Sources are mixed with 0, −4 and −8 dB level offsets. The six test triplets provide 12.34 seconds of simultaneous three-source activity in 40.80 seconds of mixture audio. The waveform references are the three isolated inputs after source-specific room processing where applicable.

The selection criterion is the sum across clean and room conditions of median exact-three SI-SDR improvement plus 0.25 times the corresponding weakest-source median. The test evaluates only the frozen selection and the current baseline. Estimates are matched to reference speakers once per clip for waveform scoring. This measures signal isolation and uses known three-source count. Real speaker tracking requires its own evaluation.

The benchmark records source WAV checksums and only accepts a selection made from the identical benchmark script. The new WSJ0-3Mix checkpoint was validated against its published model-weight hashes. [Pinned model checkpoint](https://huggingface.co/speechbrain/sepformer-wsj03mix/tree/21e78b44a378092c34fff32ba8e788a73584f474)

## Frozen test results

Development selected inverse-polarity consensus with Libri3Mix SepFormer. This runs the same model twice and combines the estimates. All numbers below are median exact-three SI-SDR improvements over the unprocessed mixture.

| Condition | Single pass | Two-polarity consensus |
|---|---:|---:|
| Clean | +11.48 dB | +11.96 dB |
| Simulated room | +3.52 dB | +5.02 dB |
| Simulated room, weakest source | +4.94 dB | +4.82 dB |
| Pooled | +7.21 dB | +7.41 dB |

The triplet bootstrap interval for the +1.50 dB room difference was −0.81 to +2.19 dB. Whole-clip pooled median SI-SDR improvement was +7.61 dB for the single pass and +6.94 dB for the consensus. The consensus required 1.97 times the single-pass inference time under the same test process. The comparison supports an optional controlled-room refinement pending more evidence on the weakest voice and whole-clip quality.

## Reproduction

The source cache is a flat folder of the AISHELL-3 WAVs listed in the case provenance in `metrics.json`. The two model folders contain the pinned encoder, masknet, and decoder checkpoints. Run from the repository root.

```bash
bench=benchmarks/three_speakers
trial="$bench/target_90"
for phase in development test; do
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
    python "$trial/benchmark_waveform.py" "$phase" --project . \
      --previous "$bench" --cache /path/to/source_audio \
      --libri /path/to/sepformer-libri3mix --wsj /path/to/sepformer-wsj03mix \
      --results /path/to/waveform-results
done
python "$trial/summarize_waveform.py" /path/to/waveform-results
```
