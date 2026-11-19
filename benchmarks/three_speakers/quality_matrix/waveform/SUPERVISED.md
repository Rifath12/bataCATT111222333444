# Supervised bounded spectral gain check

This check trains a small postfilter of the released three-output separator. The target is the real projection of the reference STFT on the estimated source spectrum, with a local energy denominator and bounded gain. Inputs at inference are the observed mono mixture and the three released estimates only. Ten features describe energy, frequency, local dominance, intertrack coherence and mixture-phase agreement. The fit uses 45 histogram-boosted trees. Candidate gains are blended into the released waveforms at eight uniformly spaced strengths.

Fit and calibration used 24 and 18 mutually speaker-disjoint voices. These were the old 8 trios in development plus test, and the three old fresh trios plus the later three development trios. The heldout panel used four different trios from previously published room fixtures. Its 12 voices share no identities with fit or calibration. Two of the available older heldout trios were excluded because they shared identities. The same reference-to-output assignment, found from the released outputs on the full clip, was used for both methods and every overlap stratum. True overlap on the controlled signal uses 20 ms frames where all reference voices exceeded their existing source-energy thresholds.

Release eligibility required non-regression for six isolation metrics across both clean and room development conditions. The candidate with the largest minimum delta was frozen even when it did not pass, to audit whether the dev gains transferred. The best strength was 0.125 and its minimum development change was −0.0323 dB.

| Heldout noisy room fixture metric | Change from release |
| --- | ---: |
| Pooled whole-clip SI-SDRi | +0.0008 dB |
| Pooled exact-three SI-SDRi | −0.0104 dB |
| Quietest voice, whole clip | +0.0516 dB |
| Quietest voice, exact-three | +0.0093 dB |
| Worst source, whole clip | +0.0090 dB |
| Worst source, exact-three | −0.0257 dB |

This is a negative result for promotion. It shows how a gain on the physically quietest source can coexist with a loss on the worst separated source and in pooled true overlap. Neither clean heldout superiority nor improved transcription was established for this candidate.

Run from the repository root using the existing fixture caches.

```sh
export PYTHONPATH="$PWD/pareto_wave/supervised:$PWD/trial_next_wave/integration"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python pareto_wave/supervised/estimate.py
```

The final supplemental check replayed the same frozen strength on one entirely project-disjoint trio in both dry and noisy-room fixtures. It used speakers 0632, 0686 and 0427. Dry pooled whole-clip SI-SDRi fell 0.0030 dB and dry worst-source whole-clip SI-SDRi fell 0.0353 dB. In the room, pooled exact-three SI-SDRi rose 0.0214 dB, but the quietest and worst source on that overlap both fell 0.0032 dB. This supplemental check also fails the all-metric gate. Add `--fresh` to replay it after the supplied independent trio caches have been generated.
