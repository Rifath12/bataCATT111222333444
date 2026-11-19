# Three-speaker mono waveform audit

The baseline is the released three-speaker SepFormer followed by its existing spectral refiner. Every trial starts from the same mono audio and the same three estimated streams. No profile uses isolated recordings, annotations, or speaker identities during inference. One whole-clip speaker permutation is fixed for the baseline and all postfilters in each controlled comparison.

We measured SI-SDR improvement over the mixture on six axes. These are pooled speaker scores, quietest-speaker scores and scores for the weakest-performing assigned track, each on the whole file and during true three-speaker activity. The quietest source is determined by reference RMS. Each panel uses speaker medians for pooled scores and case medians for the quietest and worst scores. A profile had to avoid regression on all six aggregate metrics in both the dry and simulated-room development fixtures. The multiview development gate covered room fixtures only. Recognition checks used the same SenseVoice and Zipformer models and scored normalized Mandarin characters.

## Results

| Waveform proposal | Development behavior | Frozen transfer behavior |
|---|---|---|
| Dominance-weighted crosstalk compensation and residual-floor damping | None of 88 settings cleared both acoustic conditions. The strongest minimum-gain setting selected residual-floor damping. | On the previous 3-trio noisy-room sentence panel, pooled whole-clip SI-SDRi rose 0.036 dB, while the quietest speaker lost 0.032 dB during exact overlap. |
| Harmonic suppression using each stem's pitch and competing harmonic energy | None of 24 settings cleared both conditions. The strongest setting gained 0.018–0.067 dB on all six room measures, while the quietest clean source lost 0.044 dB over the whole clip. Development edits fell 58→57 with SenseVoice and 48→44 with Zipformer. | On the previous 3-trio noisy-room sentence panel, pooled exact-three SI-SDRi gained 0.011 dB, while the quietest speaker lost 0.162 dB. |
| Reliability-limited agreement of original and dereverberated SepFormer passes | None of 12 settings cleared the room gate. The selected consensus filter reduced quietest whole-file SI-SDRi by 0.020 dB and SenseVoice errors rose 58→59. | On a separate 3-trio room panel the worst exact-three metric fell 0.014 dB. This alternative also requires a second separator pass. |
| Supervised spectral residual gain | An independent 45-tree scalar gain failed calibration on 18 held-aside speakers after fitting on 24 speakers. | The speaker-disjoint 4-trio room panel gained 0.052 dB for quiet whole-file speech, while pooled and worst exact-three isolation fell 0.010 and 0.026 dB. |

The supplemental locked panel used three AISHELL-3 Test speakers whose IDs do not occur in any of the prior project waveform panel manifests. The same trio was tested both dry and with simulated room reflections and 15 dB added noise. The two conditions total 12.29 seconds and 0.68 seconds of true three-speaker activity by the reference-energy criterion. Harmonic suppression gained 0.193 dB across all simultaneous voices in the simulated room. The dry case lost 0.095 dB on the same metric and 0.171 dB for the worst assigned track. The clean and room results for the supplemental panel are separately enumerated in `summary.json`. These short tests with isolated sources measure waveform isolation. The previous evaluation panels had been observed with published systems before this experiment. The new candidates' settings were tested on these panels unchanged after development selection.

Retain the published waveform separator and spectral refiner. The tested postfilters improve specific quantities under specific acoustic conditions, with no demonstrated common winner across the six waveform measures.

## Reproduction evidence

The paired metrics and source manifest are in this directory. The archived evidence records the controls and the parameters tested with the published separator.
