# Three Mandarin voices on one room microphone

The updated file command exports three time-aligned WAVs for a declared roster, accepts separate solo voice examples, and has an optional waveform correction. Three separated files can still contain a missed or misassigned voice. The 96–98% waveform target remains unmet.

## Known-source acoustic test

Both rooms used a single far-field channel. Solo utterances that had already passed through that microphone and room were mixed at known levels, making the true signal of each talker available to the scorer. Each room contains 12 distinct three-way trials. The weakest talker ranges from equal level to 20 dB below the loudest. R8004_M8006 was used for development. The phase settings were frozen before checking R8005_M8007, with three different speakers in a different room. The app received the mixture and separate, earlier solo examples in A/B/C order. 

| Fixed A/B/C while all three speak | Before phase correction | After phase correction |
|---|---:|---:|
| Development, mean of each trial’s weakest waveform alignment | 10.65% | 11.64% |
| Confirmation, mean of each trial’s weakest waveform alignment | 7.26% | 9.38% |
| Confirmation, weakest voice’s mean fraction of frames retained within 12 dB | 31.05% | 39.93% |
| Confirmation, mean of each trial’s lowest SI-SDR | −16.52 dB | −14.20 dB |
| All three meet the 96% criterion, both rooms | 0/24 | 0/24 |
| All three meet the 98% criterion, both rooms | 0/24 | 0/24 |

Alignment is the squared correlation of estimated and true waveforms during common speech. The strict pass also requires usable level retention on each voice. On the held-out room the median matched change per speaker was +1.06 dB SI-SDR. Of 36 speaker segments, 28 improved and eight worsened. The largest decline was 2.06 dB. These speaker segments come from 12 events in one confirmation room. The remaining errors include within-meeting identity swaps, disappearing quiet voices and speech from other people remaining on an output. [Controlled recordings and criteria](PROTOCOL.md) · [PCM waveform scores](phase_waveform_evidence.json) · [Independent calculation](independent_phase_audit.json)

## Natural meeting speech

Four unaltered excerpts from three physical sessions contain 113.67 seconds, including 15.24 seconds of annotated three-person overlap. No clean isolated waveform exists for the same naturally overlapped utterances. In this panel, automatic counting with enrollment put the loudest track on the physically correct person on 348/557 eligible solo blocks. During 134 eligible triple-speaking blocks, all three enrolled tracks cleared the audibility threshold in 9 blocks using automatic count and 48 with a forced-three decoder. An audible stem can still contain the wrong voice. In one 60-second sample, phase correction increased unwanted energetic output slots during annotated solo speech from 220 to 240 on the enrolled automatic route. Forcing three components throughout a meeting can create duplicate or spurious voices on solo turns. [Natural recordings, exact methods and results](NATURAL_MONO.md)

The app’s own Mandarin recognizer scored 595 character edits on 742 reference characters (80.2% speaker-attributed CER) with the enrolled automatic route before phase correction. The comparison route used the previous forced-three anonymous app with embeddings disabled, and scored 723 edits (97.4%) after a single best global speaker relabeling per excerpt. The enrolled route helped in the pooled result and worsened individual excerpts. A separate Whisper-small check on exactly those WAVs scored 137.2% and 205.5%, respectively, including hallucinated characters. The two recognizers measure downstream transcription with their own error patterns. [Native ASR counts](native_mandarin_characters.json) · [External ASR validation](NATURAL_MANDARIN_ASR.md)

## Input and output fixes

The optional three-person roster prevents a fourth output when speakers return after silence. Separate solo examples can be ordinary mono or stereo files in common sample rates. The original voiceprint and Mandarin recognition models have verified download mirrors. A real Mandarin sample saved as an opposite-polarity stereo duplicate produced the exact same four WAVs and Chinese transcript as its mono counterpart. Each transcript segment reports single-source mixture passthrough, assigned activity and ambiguous identity duration. [End-to-end stereo parity](input_integration_check.json)

## Other evaluated changes

A guarded blend with a 16 kHz Mandarin separator improved pooled medians and caused large losses for several individual speakers. Alibaba MossFormer2, an additional Mandarin SepFormer and recursive quiet-voice extraction also failed to establish a safe gain for the three-voice target. These experimental paths are recorded for reproducibility. A separate three-headset option reduced audio leakage while people were silent and requires three synchronized physical headset channels. [Alternative architectures](ARCHITECTURES.md) · [Recursive test](recursive_confirmation.json) · [Dedicated headsets](../../advanced_separation/HEADSET_SPILL.md)
