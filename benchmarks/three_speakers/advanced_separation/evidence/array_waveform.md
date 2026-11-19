# Mandarin three-speaker array experiments

The recordings are two authentic AliMeeting test sessions with three anonymous Mandarin speakers and eight synchronized far-field microphones. The pipeline uses audio-only predicted A/B/C activity for fitting and rendering. Individual headsets and transcripts are excluded from inference. Reference TextGrids are used for scoring. RTTM and previously predicted solo labels are both used to select and identify source excerpts when constructing the controlled waveform tests.

## Full-meeting character results

Recovering 180 ms of real audio context around diarized speech improved full-meeting character attribution using the existing continuous MVDR beams, unchanged predicted timing, SenseVoice int8, and the same duplicate suppression.

| Recording | Duration | Reference characters | Prior edits | Context edits | Prior 1 − cpCER | Context 1 − cpCER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M8016 | 30:38.98 | 7,705 | 1,503 | 1,361 | 80.49% | 82.34% |
| M8017 | 30:20.05 | 8,154 | 623 | 533 | 92.36% | 93.46% |
| Pooled | 1:00:59.03 | 15,859 | 2,126 | 1,894 | 86.59% | 88.06% |

The 232 avoided edits comprise fewer substitutions (1,209 to 1,076) and fewer deletions (756 to 623), alongside more insertions (161 to 195). All six speaker tracks incurred fewer total edits. On the first five minutes of the separate M8013 development meeting, the same approach decreased edits from 128 to 105 on 1,114 characters. Unprotected padding gave 120 edits and ordinary continuous beams with original span boundaries gave 125.

Files with the fixed suppression applied are `../context_8016_filtered/scores.json` and `../context_8017_filtered/scores.json`. The corresponding baseline files are `../../next_root_asr/dup_array_8016/scores.json` and `../../next_root_asr/dup_array_8017/scores.json`. Decode took 187.17 s and 164.13 s respectively on CPU with two threads, starting from the predicted timing and continuous beam FLACs. This timing excludes diarization and array rendering.

## Controlled overlap waveform results

The separate optional waveform profile preserves the reference-microphone image while multiple voices are predicted active. The MVDR spectrum is selected unchanged in solo frames. The same overlap-add loop preserves the waveform sample clock. With the full authentic inputs and predicted timing available, CPU rendering took 39.28 s for M8016 and 27.61 s for M8017, including all three WAVs and continuous FLACs. The model uses regularized covariance shapes fit from confident solo activity, a diffuse background shape, a mask temperature of 1.3 and an amplitude floor of 0.025. This fixed calibration was selected using a separate M8013 waveform probe.

For each test room, three held sets of separately captured source excerpts were combined digitally into synthetic same-room three-voice overlaps. Each mixture contributes 1.84375 s of simultaneous analysis time and three waveform scores. Source excerpts exclude the separate enrollment signal. The source image at physical microphone zero is the target, and that microphone's mixture is the common baseline. Activity for the controlled mixtures is supplied by construction. Projected SIR measures interference projected onto the two known competing sources. SI-SDR measures both interference and other distortion.

| Test room | Prior median SI-SDRi | New median SI-SDRi | Prior median projected SIR improvement | New median projected SIR improvement |
| --- | ---: | ---: | ---: | ---: |
| M8016 | −4.26 dB | +0.76 dB | +3.10 dB | +3.68 dB |
| M8017 | −1.47 dB | +3.99 dB | +5.97 dB | +7.83 dB |

On the 18 source-track tests, the optional profile yielded 15 positive SI-SDR improvements over the raw channel, versus four for the old renderer. The source excerpts, gain rotations, scores and factor breakdowns are in `../controlled_8016/with_posterior.json` and `../controlled_8017/with_posterior.json`, under the `production_*` keys. The final production renderer generated exactly the same PCM outputs as the frozen benchmark profile and exactly the prior PCM in the pre-overlap solo enrollment, as recorded in `parity.txt`.

The two authentic recordings have only 2.43 seconds of simultaneous three-way speech by their RTTM annotations. The waveform controls directly exercise three simultaneous voices with known source waveforms. They do not establish the same gain on arbitrary authentic mixtures or on single-channel input. The full-meeting transcript table comes from the established static beam profile with revised ASR context. The final calibrated overlap-mask profile has no full-meeting ASR benchmark.

## Run the optional renderer

The existing renderer entry point keeps its default MVDR profile and now accepts `--profile overlap-mask`. Supply a synchronized 8-channel 16 kHz WAV and the repo's predicted timing JSON. `--ungated` also preserves continuous beams for recognizer context.

```bash
python benchmarks/three_speakers/root_causes/spatial_render.py \
  --audio room-array.wav --timing timing.json --out array-tracks \
  --profile overlap-mask --ungated
```

[AliMeeting data](https://www.openslr.org/119/) · [Ito, Araki and Nakatani, directional likelihoods for array masking](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2016/papers/1570256519.pdf)
