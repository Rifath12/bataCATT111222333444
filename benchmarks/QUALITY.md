# Verification and overlap results

The [Windows and Linux GitHub Actions run](https://github.com/Rifath12/bataCATT111222333444/actions/runs/44408314868) passed six regressions and the complete two-speaker inference demo on both CPU runners. The checks cover synchronized PCM files, speaker-labeled Mandarin output, character timing, and the listening page. The GitHub runners used verified model downloads. The destination RTX 5080 machine still needs its own measurement.

## Controlled validation with fresh speakers

The test uses 12 mixtures built from 16 Mandarin speakers outside the margin-development groups. Each condition has four mixtures. Both configurations use 2.8-second blocks, 0.60-second history, 0.35-second lookahead, the same recognition model, and no speaker-count hint. The optional configuration considers one extra source when the trained count scores are close and the candidate waveforms pass energy and similarity checks.

| True voices | Exact overlap | Exact track count, default | Exact track count, optional | Default cpCER | Optional cpCER |
|---|---:|---:|---:|---:|---:|
| 2 | 6.80 s | 2/4 | 1/4 | 64.0% | 72.1% |
| 3 | 5.36 s | 3/4 | 2/4 | 75.2% | 78.4% |
| 4 | 3.50 s | 3/4 | 4/4 | 80.9% | 80.3% |

Speaker-attributed character error rate (cpCER) uses one optimal assignment over each complete clip. The denominator includes missing voices and the numerator includes extra characters. In the four-voice test, the median SI-SDR improvement over the mixture on matched sources during exact overlap was **−0.60 dB** for default (15/16 matched sources) and **−0.07 dB** for the option (16/16). Exact overlap uses the reference tracks' 20 ms speech-energy masks. These measurements indicate substantial leakage and omissions in simultaneous speech.

The margin of 1.4 was chosen using nine development cases. The waveform guard screened out near-duplicates and very weak estimates. The 12 fresh cases were scored after selecting the margin. An exploratory 6-second block comparison is included in the raw files. It improved default counting in two-voice clips to 3/4 while raising the four-voice character error to 85.5% on this slice. The standard configuration remains the default.

## Natural meetings

Four AliMeeting excerpts from three meetings contain four annotated participants and 2.80 seconds of exact four-way overlap altogether. Two excerpts were used in the earlier prototype. Two are newly evaluated. The overlap option raised the output count from two to three on one excerpt. Both configurations returned four tracks on the other three excerpts. Energy-mask diarization error averaged **58.1%** for default and **59.4%** for the option across the four clips, with speaker overlap included. Full speaker-attributed text cannot be scored from the partial target transcripts in these natural clips.

Keep the default for general recordings. Use the optional mode for comparison when three or four people speak over one another. The [four-voice example](../demo/four_voices/README.md) gives the mixture and all recovered tracks with timestamps. Real-meeting four-way recovery remains a research limitation.

## Replay check

On a Linux CPU, reprocessing the 11.93-second natural clip produced four identical PCM WAVs in file and paced replay modes. Across five blocks, replay took 2.60 seconds median and 2.71 seconds at the 95th percentile from block end to commit. That timing covers audio only. [Replay measurements](replay_regression.json)

## Reproduce the measurements

The [input manifest](cases.json) lists exact recordings, labels, offsets, and SHA-256 checksums. Rebuilding it with cached originals produced 72/72 sample-identical controlled WAVs. Short clips were reconstructed from AISHELL-3 human read speech. Natural audio is sourced from the REAL-T AliMeeting subset. Reference data enters scoring only. [AISHELL-3](https://huggingface.co/datasets/AISHELL/AISHELL-3) · [REAL-T](https://huggingface.co/datasets/REAL-TSE/REAL-T) · [AliMeeting](https://www.openslr.org/119/) · [RTTM annotations](https://github.com/nttcslab-sp/diar-forced-alignment)

```sh
python prepare_validation_cases.py --out results/validation_data
python benchmark_release.py --cases results/validation_data/controlled --models models --out results/validation_audio_text.json
```

For the natural comparison, pass the four files in `results/validation_data/natural` to `--natural` and the same directory to `--rttm`. Use `--no-text` and `--pattern no-controlled-inputs`. The per-clip results are in [controlled scores](controlled_scores.json), [natural scores](natural_scores.json), and [6-second exploratory scores](six_second_scores.json).
