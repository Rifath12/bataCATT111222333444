# Guarded mono waveform audit

The mixture reconstruction gate in the current optional RecorrNet module treated any mathematically positive gain as support for a separate voice. On a controlled regression test, three near-copies of the room mixture all contributed about one-third of the fit and passed because they explained 100% of its energy. A reconstructed third output with only 0.07% of the mixture's L2 norm also passed. Neither result establishes that three voices were separated. The old gate also computed gains from centered samples but applied them to tracks with their original DC offsets.

The revised gate estimates gain with length-normalized columns, measures each calibrated contribution relative to the mixture, and checks the smallest to largest normalized Gram eigenvalue ratio. A candidate can replace the old tracks only when explained energy is at least 90%, every calibrated track contributes at least 1% of the mixture L2 norm, and the eigenvalue ratio is at least 0.01. Accepted tracks are centered before level restoration. Rejected candidates retain the existing sample-aligned tracks. These checks rule out obvious collapse and near-duplicate tracks. They do not establish who spoke or how many words were correctly transcribed.

All six targeted tests passed. The three published 2.8-second simulated room mixtures replayed with unchanged gate selections and median exact-three SI-SDR improvement of +13.00 dB compared with −1.19 dB for the existing separator. The accepted clips had 27.5% and 29.7% minimum source contributions. See `controlled_replay.json`.

## Natural overlap diagnostic

RecorrNet runs natively at 8 kHz. To test whether a full-band gate was rejecting good low-band separation, the same inferred overlap neighborhoods were evaluated with the old fit and a native-band fit. The 10.27-second AliMeeting Test excerpt `R8002_M8002_153879_154906` contains 1.66 seconds of independently annotated exact-three speech. The inference windows used existing predicted timing only. If the candidate passed, its calibrated low band would have been combined with the preserved original stems above its model bandwidth, with tapered boundaries.

| Inference window | Old 16 kHz fit | Native-band fit | Third output fraction | Decision |
| --- | ---: | ---: | ---: | --- |
| 0.00–1.28 s | 1.01% | 1.07% | 0% | Retain old tracks |
| 1.70–3.30 s | 7.20% | 7.35% | 0% | Retain old tracks |

The bandwidth hypothesis did not account for these real extraction failures. Both candidates were rejected and every sample of the fallback audio was preserved. The two forwards took 18.39 and 20.35 seconds on the CPU test. The software guard can be improved without claiming an accuracy gain for natural three-way speech.

Run the lightweight gate tests with `python -m unittest discover -s vnext_check/wave`. In the benchmark workspace, run `python vnext_check/wave/replay_gate.py --out vnext_check/wave/controlled_replay.json` to replay the controlled fixtures. Use `bash vnext_check/wave/run_probe.sh --out vnext_check/wave/probe_first` for the natural diagnostic.
