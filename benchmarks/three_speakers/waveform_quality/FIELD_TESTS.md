# Mandarin waveform isolation with three physical microphones

This release fixes a failure on already separated signals, adds offline calibration for new microphone setups and speeds up boosted WAV export. The trials used three target speakers throughout. The separate single-microphone evaluation is below.

## Five unseen headset sessions

The first panel contains 156 controlled clips from three physical rooms. A second panel contains 88 clips from two different meetings in a fourth room, R8002. Each source is one real Mandarin utterance captured simultaneously by all selected headsets during an annotated solo interval. Speakers, source spans and the calibration cutoff were fixed before scoring. All four originally annotated speakers were checked to exclude other active voices from the donor intervals. The 244 clips contain 182.272 seconds of three-way activity. The 15 target speakers are distinct across the five sessions.

A pass requires **every** physically labeled output to reach the stated waveform alignment and to preserve at least 96% of the eligible active time above the −12 dB level cutoff. The counts below use each clip once. Clips from the same meeting share speakers and acoustics.

| Setting | Three-room panel at 96% | Three-room panel at 98% | Confirmation at 96% | Confirmation at 98% |
|---|---:|---:|---:|---:|
| Raw headset channels | 114/156 | 104/156 | 45/88 | 21/88 |
| Standard | 115/156 | 106/156 | 49/88 | 28/88 |
| Gentle | 119/156 | 106/156 | 49/88 | 29/88 |
| Boosted | 122/156 | 108/156 | 49/88 | 27/88 |

| Relative level of one voice | Clips | Boosted, all three at 96% | Boosted, all three at 98% |
|---|---:|---:|---:|
| Equal nominal levels | 63 | 58 | 44 |
| 6 dB quieter | 61 | 52 | 44 |
| 12 dB quieter | 61 | 40 | 34 |
| 20 dB quieter | 59 | 21 | 13 |

Boosted reached 96% for all three on 171/244 (70.1%) and 98% on 135/244 (55.3%). The 96% grade was achieved in 92.1% of balanced clips and 35.6% of clips with a 20 dB imbalance. These are frequencies of measured waveform passes. The evaluation does not measure Mandarin character accuracy. In the first panel, six target microphones contained more measured energy from the competing voices than from the person wearing the microphone. In the worst such control the excess reached 10.5 dB. The direct-channel references include the background present in the original solo recordings.

[First panel and reproduction](locked_headsets/PROTOCOL.md) · [First panel scores](locked_headsets/baseline_summary.json) · [Confirmation protocol](locked_headsets/confirmation/CONFIRMATION.md) · [Confirmation scores](locked_headsets/confirmation/baseline_summary.json) · [Known source contributions](locked_headsets/capture_report.json)

## A complete 30:20 recording with scoreable overlap

The original M8017 meeting contains only 1.13 seconds annotated with all three people active. We exported its three full, aligned WAVs at the original sample clock. Boosted processing with stored calibration took **8 minutes 29 seconds** on one CPU thread. Across 909.3 seconds of eligible annotated solo speech, each target stayed above the existing level-preservation threshold on all evaluated 100 ms blocks. Those observations do not establish natural overlap purity.

A second version moves and overlays genuine isolated turns from those same people on the original 30:20 clock. No utterance was copied twice. This hybrid has 1,073.61 seconds of untouched original audio, 414.19 seconds vacated where turns were moved, and 332.25 seconds of scored controls. It includes 87.065 seconds of engineered three-way speech in 72 passages. Donors exclude the calibration prefix and all spans from the older short-clip panel.

| Method on the continuous hybrid | At 96%, all three | At 98%, all three | Weakest track | Eligible overlap time meeting 98% in 250 ms windows |
|---|---:|---:|---:|---:|
| Raw physical channels | 48/72 | 33/72 | 8.42% | 50.89% |
| Standard, fixed calibration | 50/72 | 32/72 | 8.09% | 53.04% |
| Standard, rolling calibration | 50/72 | 33/72 | 8.13% | 53.56% |
| Gentle | 50/72 | 32/72 | 8.26% | 53.28% |
| Boosted | 50/72 | 35/72 | 10.44% | 48.42% |

The rolling version updates estimates only after sufficient evidence in previously completed blocks. It gained one 98% passage, preserved all prior passes and raised the quietest-track lower decile from 72.06% to 72.97% on this replay. Take this as a targeted, single-session result. Its offline rendering still uses a small amount of context around block boundaries. The strongest long-run profile depends on whether the application values complete-passage alignment or short-window continuity.

In passage p3_047, interference on C’s mic was 10.48 dB louder than C, and the fixed calibration underestimated energy from the nominal A-only interval on C’s mic by 22.73 dB. Its peak correlation with A’s own mic was only 0.039 over ±100 ms, so the interference cannot be confidently attributed to A’s speech alone. The rolling estimator lacked enough recent confident observations for A to update that path at the critical point. For the 18 passages with one participant 20 dB down, boosted passed 5/18 at 96% and 0/18 at 98%.

To regenerate the continuous input, obtain the pinned M8017 headsets and RTTM from the existing close-mic [provenance](../advanced_separation/review/close_provenance.json). Use the supplied manifest and calibration, then export with the command in the main waveform readme. The raw inputs under `--audio` must be named 8063.wav, 8064.wav and 8065.wav.

```sh
python build_meeting_replay.py --audio path/to/headsets --rttm path/to/room.rttm --manifest evidence/meeting_replay_manifest.json --calibration evidence/meeting_replay_calibration.json --out replay
```

[Complete hybrid results](evidence/meeting_replay_scores.json) · [Worst-case diagnosis](evidence/worst_case_diagnosis.json) · [Rolling method and command](../advanced_separation/ROLLING_CLOSE.md) · [Rolling results](evidence/rolling_m8017.json) · [Natural solo evaluation](evidence/natural_m8017_solo.json)

## CPU speed and numerical checks

On the same five seconds of audio, portable boosted processing took 5.07 seconds. The optional Numba evaluator took 1.92 seconds including initial compilation and 1.02 seconds with compilation already complete. The WAV samples were identical in this comparison. In a longer matched test, both tree evaluators produced identical output samples on ten clips using the same numerical runtime. Across all 244 clips the accelerated run preserved every 96% and 98% decision recorded by the prior implementation. The maximum sample difference between numerical environments using NumPy 2.3.5 and 2.5.5 was 5.97 × 10⁻⁸. Full exports used one CPU thread with other jobs active on the host. A full offline speed measurement does not establish live streaming delay.

[Same-runtime and threshold comparisons](evidence/fast_tree_thresholds.json) · [Cross-environment numerical comparisons](evidence/fast_tree_cross_environment.json)

## Fixed output and calibration behavior

An identity transfer is a valid outcome for isolated microphones or insufficient fitting evidence. The old optional models could still alter those tracks. On 24 clean controls, the worst alignment under boosted processing was 98.35%. The corrected exporter preserved every input sample on all 72 clean tracks. A missing incoming path for one receiver now preserves that receiver's raw waveform while the other receivers can still be filtered. The exporter also rejects a duplicated, nonconstant input signal (including an inverted copy). A copied mono recording cannot supply three physically independent headset tracks.

The three-take offline calibration shares the released spectral estimator. Each labeled solo WAV contains all three headset channels. Twelve seconds of solo speech per target was sufficient to run every new session in the three-room panel. Against the longer early calibration, standard processing kept 115/156 at 96% and moved from 106 to 105 at 98%. In the historically difficult M8014 session, the short calibration cost one boosted pass at 96% and two at 98%. Its report flagged unusually strong leakage. This option is suitable when no stored calibration exists and the user can supply solo takes from their actual setup.

[Clean control waveform comparisons](evidence/clean_passthrough_control.json) · [Short calibration results and edge cases](evidence/short_solo_calibration.json)

## Failure analysis and candidate screening

A second suppression mask raised the 96% count on the first panel by one. It also reduced the quiet-voice lower decile there and lost two historical 98% passes. Static and adaptive phase cancellation did not pass development screening. A calibrated bass filter improved some badly contaminated tracks but mistook natural bass on another track for interference, cutting its alignment from 99.31% to 87.49%. Limiting its action to physically predicted leakage fixed that particular example. On the separate 88-clip confirmation panel the revised filter still lost a 96% pass. These candidates were excluded from the normal exporter.

Fitting a small gain model to remixes of the **same setup's early solo examples** produced a modest positive confirmation result. In the 88-clip panel it raised joint 98% passes from 27 to 29 with the same 49 passing at 96%. Both extra passes involved balanced voices. The weakest voice fell from 7.22% to 6.70%, and one historical 98% pass was lost. Its results are retained as an experimental alternative. The standard, gentle and boosted algorithms retain their original names and operating points.

The experimental room-specific script accepts a locked room directory and can write the three tested WAVs for a selected clip. [Research implementation](locked_headsets/room_gain_experiment.py) · [Complete confirmation replay](locked_headsets/room_gain_panel_replay.json)

[Paired residual-mask scores](evidence/residual_energy_screen.json) · [Phase and bass tests](evidence/phase_and_bass_screen.json) · [Physical-cap confirmation](evidence/energy_cap_confirmation.json) · [Room-specific fit](evidence/room_specific_evidence.json)

## Ordinary single-microphone audio

The single-channel tests summed independent Mandarin sources to create a known waveform reference. Adding short speaker prompts, restoring 4–8 kHz energy, and averaging shifted/polarity-adjusted passes did not demonstrate reliable 96% separation. In one difficult mono case the outputs reconstructed 99.25% of the mixed recording's energy, while their individual alignments with the intended voices were approximately 0.78%, 49.9%, and 69.3%. Passing a mixture reconstruction check cannot certify the contents of an individual WAV. The guarded mono waveform model remains the release choice.

When all three mono stems have passed the separation gate, an optional enrollment step now orders them by A/B/C. It uses separately labeled solo speech from the same session and copies the WAV bytes into the requested order. On two accepted groups totaling eight short clips, it matched 22/24 track assignments and 7/8 complete clip assignments. The raw model slot order matched 9/24 tracks. A third group of four clips failed the waveform gate and contributed no labels to this result. The speaker letters require the user to establish A/B/C through enrollment. No character transcription was scored.

[Mono experiments](evidence/mono_room_experiments.json) · [Enroll and export A/B/C](../advanced_separation/mono_wave/MONO_ENROLLMENT.md) · [Fixed-label verification](../advanced_separation/mono_wave/enrolled_order_new_rooms.json)

## Source audio

AliMeeting provides Mandarin meetings with individual headsets and an eight-channel microphone array. These experiments used the headset audio. The source and derived listening samples are distributed under CC BY-SA 4.0. [Dataset, paper and license](https://openslr.org/119/)
