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

Using the same underlying enrolled separation estimates and native recognizer, the optional phase correction reduced character edits in all four natural excerpts. The pooled result improved from 595/742 (80.2% CER) to 555/742 (74.8%). For M8009 the external Whisper check moved in the other direction, from 175 to 287 edits on 193 reference characters. [Paired native results and verified PCM checksums](corrected_native.json)

With only a recording and no enrollment examples, two excerpts provided 78 stable triple-speaking blocks. The revised automatic decoder produced three audible tracks in 0/78. The older forced-three command did so in 24/78. On solo blocks the automatic route had exactly one audible output in 55/57, compared with 12/57 when forcing three. Both commands used the restored Mandarin voiceprint model. A/B/C are anonymous for this comparison. [Unaided single-file evaluation](UNAIDED_MONO.md)

## Input and output fixes

The optional three-person roster prevents a fourth output when speakers return after silence. Separate solo examples can be ordinary mono or stereo files in common sample rates. The original voiceprint and Mandarin recognition models have verified download mirrors. A real Mandarin sample saved as an opposite-polarity stereo duplicate produced the exact same four WAVs and Chinese transcript as its mono counterpart. Each transcript segment reports single-source mixture passthrough, assigned activity and ambiguous identity duration. [End-to-end stereo parity](input_integration_check.json)

## Full 32-minute single-microphone meeting

The complete 32:17.09 Mandarin recording contains exactly three participants. Both outputs cover all 15,496,734 samples at 8 kHz, with 692 ordered commits and three finite, unclipped PCM16 speaker tracks. The single-recording run received one mixed WAV and a roster size of three. The enrolled run also received earlier solo examples of each voice on that same room microphone. All enrollment ended before 384.62 s. The table evaluates 484.3 s onward. Each evaluation uses a single physical microphone channel. Both corrected routes use the same automatically estimated voice counts.

| Held-out 24:12.8 portion | Recording only, corrected | Voice samples, before correction | Voice samples, after correction |
|---|---:|---:|---:|
| Correct track loudest during solo speech, 9,996 eligible 100 ms blocks | 4,100 (41.0%) | 5,190 (51.9%) | 5,230 (52.3%) |
| Both expected tracks audible during two-person speech, 661 eligible blocks | 215 (32.5%) | 207 (31.3%) | 244 (36.9%) |
| Three audible tracks during three-person speech, 28 eligible blocks | 6 (21.4%) | 5 (17.9%) | 6 (21.4%) |
| Native Mandarin speaker-attributed character edits, 6,972 reference characters | 5,435 (78.0% CER) | 5,189 (74.4% CER) | 4,950 (71.0% CER) |

Anonymous letters are compared after assigning them to persons once for the whole scoring interval. Each diagnostic selects a single global mapping for its metric. The primary enrolled score always keeps A/B/C fixed. The audible threshold is −18 dB relative to the mixture in a 100 ms block. An audible track can contain the wrong voice. On enrolled outputs the number of unintended audible tracks during solo blocks increased from 8,361 to 9,293 after phase correction. The full recording has 5.37 s of fine-grained annotated triple activity. Coarser manual turn spans overlap for 14.30 s. These durations explain why the separate overlap-heavy and known-source tests matter.

The measured enrolled acoustic run took 809.79 s, followed by 7.90 s for the waveform pass. The combined stage time was 13:38. From the common run start the corrected enrolled files were available around 13:43 and the corrected anonymous files around 14:21. Native text decoding took another 2:45 and 2:57, respectively. This was an offline shared Linux CPU evaluation with up to four independent one-thread window workers, in chronological output order. Both tracking routes reused the same decoded windows, with no annotation or enrollment passed into that front end. The timing excludes the earlier interrupted attempt, downloads and packaging. The ordinary command also supports a bounded `--file-workers` setting. A two-worker normal command on a separate 20-second excerpt matched each serial WAV byte-for-byte and took 17.64 s to export on this host. Sustained microphone capture and live latency were not evaluated.

[Full timing and waveform digests](full_M8014_run_timing.json) · [Corrected enrolled voice activity](full_M8014_enrolled_activity.json) · [Single-recording voice activity](full_M8014_unaided_activity.json) · [Enrolled CER](full_M8014_text_enrolled_score.json) · [Single-recording CER](full_M8014_text_unaided_score.json) · [Parallel command parity](file_worker_parity.json)

To reproduce the native short-clip transcription scores, run `score_native_export.py MODELS EXPORT REFERENCE_TEXT_JSON OUT_JSON` after installing the application's dependencies, RapidFuzz and opencc-python-reimplemented. The full reference builder and scorer use the same original manual text and the 484.3 s cutoff. [Reference builder](full_mandarin_reference.py) · [Whole-meeting character scorer](score_full_text.py)

## Other evaluated changes

A guarded blend with a 16 kHz Mandarin separator improved pooled medians and caused large losses for several individual speakers. Alibaba MossFormer2, an additional Mandarin SepFormer and recursive quiet-voice extraction also failed to establish a safe gain for the three-voice target. These experimental paths are recorded for reproducibility. A separate three-headset option reduced audio leakage while people were silent and requires three synchronized physical headset channels. [Alternative architectures](ARCHITECTURES.md) · [Recursive test](recursive_confirmation.json) · [Dedicated headsets](../../advanced_separation/HEADSET_SPILL.md)

An enrollment-conditioned TD-SpeakerBeam extractor was also tested in both rooms. It recovered some voices better but lowered the quietest-speaker median in the fresh room. Neither it nor the incumbent achieved a joint 96% pass on those eight confirmation clips. [Target-conditioned extraction](TARGET_SPEAKERBEAM.md)
