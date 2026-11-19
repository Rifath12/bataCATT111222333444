# Spectral refinement for a three-speaker waveform separator

This experimental filter learns how much to attenuate each time-frequency component of three SepFormer outputs. Its inputs are the same single microphone mixture and the three initial estimates. A least-squares amplitude fit aligns the estimates to the mixture. Features measure each track’s energy, share of mixture power, and local dominance at two smoothing scales. The fitted regression trees predict a bounded gain, blended at 0.75 strength. The existing file renderer applies the filter to its three-speaker SepFormer windows before assigning them to A/B/C.

The model learned from four fixed clean and simulated-room mixtures of two speaker-disjoint Mandarin trios. Each trio was held out in turn to select gain strength. Refit on both development trios supplied the frozen 55-tree, numeric JSON model. No inference input uses a reference speaker identity or isolated source. The evaluation aligns outputs to reference speakers once per clip. The test includes the six trios from the earlier benchmark and three extra trios with nine different voices. All recordings come from the pinned AISHELL-3 release. Each trio is tested in a clean mixture and a simulated room at about 15 dB SNR. Together the nine trios provide 17.58 seconds of exact three-way overlap in 61.2 seconds of audio.

| Three-speaker waveform result | Earlier six trios | Additional three trios |
|---|---:|---:|
| Weakest speaker in rooms, exact overlap SI-SDRi | 4.94 → 5.22 dB | 5.27 → 6.44 dB |
| All sources in rooms, exact overlap SI-SDRi | 3.52 → 3.47 dB | 3.11 → 3.05 dB |
| All sources, whole clips, clean and rooms | 7.61 → 8.16 dB | 5.02 → 5.53 dB |
| All sources, exact overlap, clean and rooms | 7.21 → 7.47 dB | 5.74 → 6.37 dB |

All values are median SI-SDR improvements above the original mixture. On the three additional trios, the room whole-clip median fell from 4.06 to 3.69 dB. The fresh-trio bootstrap interval for the weakest voice in simulated rooms was −0.97 to +1.17 dB. The separate unsupervised complex-gain fitting experiment yielded small effects. Detailed paired statistics and processing times are in the results files. These controlled mixtures measure waveform isolation. Natural meeting speaker attribution and continuous streaming require separate evaluations.

In a matched run of four 3.4 s fixtures on the same shared CPU, one-pass SepFormer inference totaled 62.72 s and the portable spectral filter added 1.27 s (2.0%). This timing excludes model initialization. A separate functional smoke test on a 10.0 s AliMeeting excerpt emitted three 16 kHz WAVs and refined two of the four overlap windows. The listening ZIP contains one 3.4 s controlled mixture and matched baseline/refined A, B and C tracks.

## Use the fitted filter

Append `--refine-three` to an existing `render.py --method dprnn --sepformer-three /path/to/sepformer3` file-mode command. The parent benchmark documents model setup and the remaining arguments. The frozen model and its NumPy/SciPy predictor are beside the renderer. The render report counts refined windows.

## Reproduce the controlled tests

Start in the repository root. Point `wave` to this directory and `models` to the pinned Libri3Mix SepFormer checkpoint directory from the previous benchmark. The source downloader verifies all 66 public utterances by SHA-256.

```bash
bench=benchmarks/three_speakers
export PYTHONPATH="$PWD:$PWD/$bench:$PWD/$bench/target_90:$PWD/$wave"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
python "$wave/fetch_sources.py" "$work/aishell"
python "$wave/cache.py" --weights "$models" --sources "$work/aishell" \
  --fresh-sources "$work/aishell" --out "$work/cached"
for phase in development test fresh; do
  python "$wave/score_projection.py" "$phase" --cache "$work/cached" --out "$work/results"
  python "$wave/score_shrinkage.py" "$phase" --cache "$work/cached" --out "$work/results"
done
python "$wave/summarize_results.py" "$work/results"
python "$wave/export_filter.py" --cache "$work/cached" \
  --selection "$work/results/shrinkage_selection.json" --out "$work/refine_three.json"
```

The export verifies regression outputs against the scikit-learn teacher and checks reconstructed waveforms with the portable predictor on the first new clean/room pair. The listening comparison uses the first extra simulated-room case, with one fixed speaker permutation across both methods. [AISHELL-3 release and license](https://www.openslr.org/93/)
