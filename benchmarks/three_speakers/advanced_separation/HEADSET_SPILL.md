# Reduce spill on inactive headset tracks

This optional offline command is designed for a meeting recorded on three synchronized physical headsets. It reduces the extra voice or background sound on the headset of someone who stays silent while another person speaks. The command requires all three headset inputs and three solo calibration takes. For an ordinary single-microphone meeting, use the mono separator.

Record one calibration WAV with only A speaking, one with only B speaking, and one with only C speaking. Every calibration WAV contains **all three microphone channels** in physical A/B/C order, at 16 kHz. Keep microphone positions, channel gains and order consistent. About 12 seconds of clear solo audio per person was tested. The three meeting WAVs are separate, equal-length, 16 kHz mono files. From this directory, run the following commands.

```sh
python calibrate_close.py solo_A.wav solo_B.wav solo_C.wav --out setup.npy
python headset_spill.py mic_A.wav mic_B.wav mic_C.wav --pilots solo_A.wav solo_B.wav solo_C.wav --power-transfer setup.npy --out separated
```

Install NumPy, SciPy and SoundFile. Use a fresh destination. Outputs are sample-aligned `speaker_A.wav`, `speaker_B.wav` and `speaker_C.wav`, with elapsed time in `headset_spill.json`. No live microphone or remote service is required for file processing.

The program fits a room-specific presence model from the solo audio. The starting waveform is the standard close-mic filter. Both the room model and the measured acoustic leakage must support lowering a receiver for approximately one second of neighboring audio. The maximum additional level reduction is 6.02 dB. The evidence window uses future audio, so this is an offline process.

## Measured results

A clip passed a threshold only if all three physical outputs reached the specified waveform alignment and preserved enough of each active voice. Removing known source contributions from the same test clips also produced 161 single-talker and 155 two-talker controls. Activity labels and isolated test audio were reserved for scoring.

| Observation | Result |
|---|---:|
| All three reached 96% on standard / with spill reduction | 218/316 and 218/316 |
| All three reached 98% on standard / with spill reduction | 180/316 and 180/316 |
| Median extra reduction on an inactive headset with one talker, 322 headset measurements | 3.02 dB |
| Median extra reduction on the inactive headset with two talkers, 155 measurements | 5.69 dB |
| Largest individual three-way alignment drop in the 316 clips | 0.19 percentage points |
| Largest active-track level change in the one- and two-talker controls | −0.01 dB |

These short, controlled overlaps came from eight AliMeeting sessions. A previously evaluated 244-clip panel and two additional sessions exposed weaknesses in earlier candidates. After the sustained-evidence rule was frozen, a new meeting, M8009, contributed 17 disjoint confirmation clips from three different participants. Its 12 passes at 96% and 11 at 98% were preserved. The fourth meeting participant was excluded from all donor intervals. The three new sessions add 72 controls. The source pins in [headset_spill_sources.json](headset_spill_sources.json) can be used with the [locked headset fixture generator](../waveform_quality/locked_headsets/PROTOCOL.md).

On the 30:20 chronological M8017 replay, the standard and sustained modes both passed 50/72 three-way passages at 96% and 32/72 at 98%. They preserved the same 20/20 single-talker and 71/87 two-talker 96% passes. On the separate, unmodified natural M8017 recording, additional off-mic energy reduction relative to standard was 0.07 dB while A spoke, 1.06 dB while B spoke, and 0.03 dB while C spoke. The measured own-microphone levels were unchanged over 909.3 eligible seconds. Additional natural solo windows in M8006, M8007 and M8009 covered 175.6 eligible seconds, with reductions from 0.01 to 2.56 dB across the nine speaker/session combinations. Energy on an inactive headset includes background sound. The original meetings lack isolated references for measuring waveform accuracy during natural simultaneous speech.

[Matched counts, control scope, natural results and numerical checks](sustained_headset_results.json)
