# Three-speaker Mandarin waveform quality

This evaluation measures how well the separated audio tracks preserve their intended speaker during three-way overlap. It uses the reference waveform for the same microphone image and clock. It assigns each track a fixed A/B/C label. It never relabels a poor result locally.

A track's target alignment is the fraction of centered output energy explained by the sample-aligned target waveform. The 96% and 98% targets correspond to absolute SI-SDR thresholds of +13.80 and +16.90 dB. A trial passes only if all three tracks reach the threshold and retain at least 96% of the eligible target speech time above the minimum projected level. The level guard is −12 dB in 100 ms blocks. A muted or swapped output cannot pass. These measurements describe waveform alignment and retained speech, independently of transcription accuracy. [Scoring details](SCORING.md) · [SI-SDR reference](https://research.google/pubs/sdr-half-baked-or-well-done/)

## Optional learned close-mic refinements

The close-mic exporter accepts `--refinement gentle` or `--refinement boosted`. Both require three synchronized 16 kHz mono recordings in physical A/B/C order. The profiles were calibrated before testing. Their supervised model data comes from earlier M8014 audio. The baseline transfer for each test room was fitted using only that recording's first 45%.

| Close-mic waveform processing | All three at 96%, independent rooms | All three at 98%, independent rooms |
| --- | ---: | ---: |
| Standard | 21/32 | 16/32 |
| Gentle | 22/32 | 16/32 |
| Boosted | **23/32** | 16/32 |

The 32 control clips are disjoint three-source remixes from two separate meetings (24 M8017, 8 M8016), with 32.608 seconds of exact three-way overlap. Both profiles preserved all 96% and 98% passes in these two rooms. On that room, the weakest output rose from 76.52% to 80.18% with boosted filtering, while median track quality moved from 99.90% to 99.36%. On later M8014 controls, boosted gained two all-three 98% passes and lost one. Gentle reduced the weakest M8016 track from 65.48% to 63.78%. Sustained natural overlap and consistent 96–98% performance remain unverified.

Use a voice activity model, or a stored power transfer array from the same setup, channel order and levels:

```bash
PYTHONPATH=benchmarks/three_speakers \
python benchmarks/three_speakers/advanced_separation/close_waveform.py \
  close_A.wav close_B.wav close_C.wav --power-transfer /calibration/power_transfer.npy \
  --out clean_tracks --refinement boosted
```

A power transfer is a float32 array of shape `(513, 3, 3)` from `fit_power_transfer`. Save the 0.55 quantile array to a `.npy` file with `numpy.save`. For fresh VAD calibration use `--vad /models/silero_vad.onnx` in place of `--power-transfer`. Omit `--refinement` for the standard profile.

The gentle profile exported three aligned WAVs from the full 30:20 M8017 recording in 124 seconds on one CPU thread, using its stored power transfer. Boosted processing retained sample alignment in a 19.712-second block-boundary test, matching the whole-array calculation. Its extra filtering took about 38.5 seconds over 43.0 seconds of controlled input on one CPU thread. These times cover waveform output under the stated conditions.

Two more mono checkpoints were tested on the same short archived clip. The existing guarded track for the weakest voice scored 66.38%. The AISHELL-only ReCorrNet checkpoint lowered that to 58.63%, despite reconstructing 99.09% of the mixture; WavLM SepRQ scored 4.11% for the same voice. Neither alternative was promoted. The benchmark compares known source waveforms on about 1.56 seconds of simultaneous speech, distinct from the close-mic benchmark.

[Gentle scores](evidence/learned_refinement_cross_room.json) · [Boosted scores](evidence/boosted_refinement_cross_room.json) · [Full gentle export](evidence/gentle_refinement_full_export.json) · [AliMeeting dataset](https://www.openslr.org/119/)

## Independent follow-up and capture fix

A newly tested three-person AliMeeting session (R8008_M8014) supplied 24 disjoint solo-derived mixtures and 32.576 seconds of exact three-way overlap. Fixed labels match the physical A/B/C microphone channels. Each source segment occurs after the first 45% used for unannotated room calibration. The 24 cases vary which person is quieter, up to 20 dB lower than the others, and come from one room and one group of speakers.

| Result on the new room | All three at 96% | All three at 98% |
| --- | ---: | ---: |
| Original physical close-mic captures digitally remixed | 13/24 | 5/24 |
| Treated with the released close-mic filter | 14/24 | 7/24 |
| Experimental per-speaker low-frequency cleaner | 14/24 | 4/24 |
| Matched captures with 24 dB less bleed simulated before mixing | 24/24 | 23/24 |
| Matched captures with 30 dB less bleed simulated before mixing | 24/24 | 24/24 |

The last two rows modify the known acoustic bleed component of each source image before digital mixing and keep each intended microphone waveform intact. They isolate how improved capture would help these controlled trials. They do not describe processing of an already mixed recording or results from new physical hardware. The worst case remained at 10.30% with the released filter. The two competing voices on C's own microphone contributed about 9.16 dB more energy than C on the exact overlap samples. C would need about **23 dB** more isolation for the 96% goal and **26 dB** for the 98% goal on that particular control. Other mic arrangements have different requirements.

[Independent benchmark](evidence/compact_evidence.json) · [Fixed fixture manifest](evidence/new_room_manifest.json) · [Capture sensitivity](evidence/new_room_capture_sensitivity.json)

The waveform profile remains the same after these tests. The new conditional mono extractor scored 30.19% median waveform alignment versus the incumbent's 90.31% on the same four locked reference clips, and the guard rejected all four replacements. The competing array profile made a small median gain while adding cross-talk. Nearby solo calibration with phase-sensitive inversion improved the lower decile on a small M8017 replay without changing the number of complete passes. There were no eligible long solo pilots for C on the chosen M8016 partition. Interval-local phase and sparse time-delay cancellation did not improve the fixed development acceptance counts. For implementation background, see [calibrated acoustic transfer estimation](https://aes.org/publications/elibrary/elibrary-page/?id=21148) and [adaptive headset interference reduction](https://link.springer.com/article/10.1186/s13636-020-00180-6).

### Use a measured microphone setup

Route each person to their own close mic, with all three microphones on the same recorder or synchronized audio interface. Keep level controls fixed while recording the A, B, and C solo takes described below. Move the weakest track's microphone closer to the speaker's mouth and use the headset position and seating to reduce bleed from the other two speakers. Re-test after a physical change. Turning up a recorded track raises both the desired voice and the existing bleed.

```bash
python benchmarks/three_speakers/waveform_quality/plan_capture.py \
  solo_A.wav solo_B.wav solo_C.wav --quiet-db 6 12 20 \
  --out capture_plan.json
```

Each solo WAV has all three channels in physical A/B/C order. The JSON identifies the strongest competing microphone source and estimates how much extra isolation each target mic needs, for the measured samples and selected louder-versus-quieter gaps. It reports a second, phase-conservative estimate for the same measured component magnitudes. The simulation changes the modeled physical bleed. The tool does not rewrite any inputs. The prepared demo poses M8016 and M8017 needed up to about 10 dB more isolation for the 96% target or 13 dB for 98% at the tested gaps. The independent M8014 weak cases show why the measurement must be repeated for the actual room and setup.

[Measured capture plan for M8016](evidence/capture_plan_M8016.json) · [Measured capture plan for M8017](evidence/capture_plan_M8017.json)

## Benchmarks

| Audio setup | Previous → reviewed result | All A/B/C at ≥96% | All A/B/C at ≥98% |
| --- | --- | ---: | ---: |
| One mixed track, guarded RecorrNet | 86.72% median target alignment on three prior controls. No candidate replaced it. | 0/3 controls | 0/3 controls |
| Eight synchronized array channels | Median target alignment 38.98% → 42.12%. Median competing-voice energy 30.91% → 22.95%. | 0/6 controls | 0/6 controls |
| One synchronized close mic per speaker | All three tracks at ≥96% increased from 19 to 22 cases and at ≥98% from 15 to 17. | **22/36 controls** | **17/36 controls** |

The setup rows use separate control panels. They characterize each route and cannot be used as a paired hardware comparison. The close-mic controls remix independently selected real solo captures from two AliMeeting meetings into labeled overlap. The 36 controls contain 23.744 seconds of counted exact-three activity, including variants at different voice levels on the same material. The worst close-mic track improved from 47.99% to 48.31%. The median for the quietest voice per clip went from 97.95% to 98.77%. These medians leave significant failure cases. Sustained natural three-way overlap at 96–98% remains unverified.

The array's median other waveform error increased from 27.61% to 34.15%, and 5 of 18 tracks lost some target alignment. This keeps the array candidate optional. On the full 30:20 M8017 array file, it exported three aligned tracks in 76.28 seconds on one CPU thread, given existing speaker timing. The recording confirms runtime and alignment. The controlled trials supply the waveform quality scores.

The mono reverse-pass and enrollment-based spectral alternatives did not surpass the current guard. Cross-stem subtraction had a negligible or negative median change on independent rooms. As a diagnostic, even an oracle 0–1 real-valued spectral mask supplied with the references gave a median of 89.87% on the three earlier clips and had zero of nine outputs above 96%. This result is specific to the tested mask family. The three guarded mono results include one whole-clip evaluation-only label assignment per case, as documented in the scoring evidence.

[Close-mic scores](evidence/close_scores.json) · [Array benchmark](evidence/array.md) · [Mono trials](evidence/mono.md)

## Run the waveform-only close-mic cleanup

Put the three 16 kHz mono WAVs in physical speaker order A/B/C. They need a shared start and sample clock. The profile estimates frequency-dependent leakage from confident solo speech in the first part of the recording. It only attenuates components supported by the measured bleed profile, with a minimum gain that protects quiet target speech. It writes `speaker_A.wav`, `speaker_B.wav`, and `speaker_C.wav` on the same sample clock. The complete 30:39 M8016 export took 37.70 seconds on CPU, including speech detection, calibration, and WAV rendering. The full recording verifies export and alignment. It has no isolated simultaneous references for scoring. Save the raw three captures for comparison. [Full-length runtime and sample count](evidence/close_export_30m.json)

From the repository root:

```bash
PYTHONPATH=benchmarks/three_speakers \
python benchmarks/three_speakers/advanced_separation/close_waveform.py \
  close_A.wav close_B.wav close_C.wav \
  --vad /models/silero_vad.onnx --out cleaned
```

For the eight-channel recording route, the normal offline CLI accepts `--array-profile adaptive-covariance`. The existing default and `--array-overlap-mask` shortcut still run the prior profiles. To export only the three WAVs with existing timing, call the [array renderer](evidence/array.md) directly.

## Check your microphone setup

Record three separate solo takes, with everyone quiet except A, then B, then C. Each take is a **three-channel file** containing all three synchronized close microphones in A/B/C order. Keep hardware gains and positioning fixed. Use a few seconds of speech in each take. The tool digitally combines the captured contributions at equal levels and with each speaker in turn quieter by 6, 12, and 20 dB. It reports the measured bleed matrix and the target alignment and held speech in each output. It uses the direct mic of each solo take as that take's reference, including any device coloration, and requires distinct speech and a common active interval.

```bash
python benchmarks/three_speakers/waveform_quality/calibrate_wiring.py \
  solo_A.wav solo_B.wav solo_C.wav --out waveform_calibration.json
```

This bench checks the recording setup under the acoustic source positions used for the takes. The included M8016 and M8017 setup checks passed 98% on all three tracks in the balanced trial. Once a voice was made 20 dB quieter, the worst track in each room fell below 71%. All three people speaking naturally can differ from the remix. [Setup check data](evidence/solo_setup_check.json) · [AliMeeting corpus and license](https://www.openslr.org/119/)
