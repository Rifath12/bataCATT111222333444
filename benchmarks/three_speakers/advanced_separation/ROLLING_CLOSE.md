# Experimental rolling headset calibration

This file-mode option is for three independent, synchronized, 16 kHz mono headset recordings in physical A/B/C order. Supply a power-transfer matrix measured with the same microphones, channel order, positions and input levels. The ordinary calibration command accepts three labeled solo takes.

```bash
python calibrate_close.py --help
python rolling_close.py A.wav B.wav C.wav --power-transfer headset_transfer.npy --out rolling_output
```

Run these commands in `benchmarks/three_speakers/advanced_separation`. Choose a new output directory. The command writes equal-length `speaker_A.wav`, `speaker_B.wav`, `speaker_C.wav` and `trace.json`. The trace lists the amount of evidence available for each physical speaker and the six median transfer levels used for every output block.

The saved initial matrix applies for at least the first 120 seconds. For a genuinely prospective run, measure it before the recording. The benchmark supplied a matrix fitted from the first 120 seconds of the same meeting. Later transfer selections use only completed blocks of this meeting. A 16.384-second block updates from at most 98.304 seconds of history and requires repeated evidence of one dominant microphone. A speaker with insufficient evidence keeps its last estimate. A receiver with no measured incoming path keeps its input samples. The existing offline spectral mask uses boundary context, so the experiment establishes file processing speed only.

In a partially engineered 30:20 replay, standard processing passed the all-three 96% alignment and level criterion in 50/72 controls, and rolling calibration also passed 50/72. At 98%, one additional control passed, raising the count from 32 to 33. Every previously passing control remained above threshold. The weakest-speaker 10th percentile rose from 72.06% to 72.97%. The hardest voice reached 8.13% versus 8.09% with fixed calibration. These numbers describe 87.065 seconds of known three-way overlap inserted in the recording. The identity and accuracy of every sample in unmodified overlapping conversation remain unverified.

The long test exposed an acoustic limit. The worst window had only two qualifying recent frames for the dominant interferer. Its donor microphone energy was poorly predicted by the initial calibration, and the rolling rule lacked enough comparable solo evidence to correct it. The donor also had low cross-channel coherence, so background sound may account for part of the energy attributed to that donor.
