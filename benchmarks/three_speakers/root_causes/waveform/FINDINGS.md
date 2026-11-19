# Where the three-voice transcription errors entered

Nine longer held-out Mandarin sentences contain 118 reference characters. On isolated recordings, SenseVoice and Zipformer each made two edits after 8 kHz resampling. With the same recordings mixed together, the same recognizers made 42 and 26 edits after dry separation. Adding room reflections increased the counts to 55 and 48. Adding noise at 15 dB SNR increased them to 82 and 76.

| Input, same nine longer sentences | SenseVoice edits / 118 | Zipformer edits / 118 | Whole phrases exact, SenseVoice / Zipformer |
|---|---:|---:|---:|
| Separate originals at 16 kHz | 1 | 2 | 8 / 7 |
| Separate originals at 8 kHz | 2 | 2 | 7 / 7 |
| Separate reverberant sources at 8 kHz | 2 | 2 | 7 / 7 |
| Mixed, dry, then separated | 42 | 26 | 1 / 2 |
| Mixed, with room reflections, then separated | 55 | 48 | 2 / 1 |
| Mixed, with room reflections and noise, then separated | 82 | 76 | 0 / 0 |
| Same noisy-room mixture, WPE before separation | 73 | 66 | 0 / 0 |

The large loss first appears when recovering competing voices. Resampling the isolated utterances accounts for one additional error with SenseVoice and none with Zipformer on this panel. Ground-truth source levels rotate across trios. In the noisy baseline, SenseVoice made 32 edits on the 35 characters of the quietest voices, reduced to 29 after dereverberation. These counts cover complete utterances. They do not localize individual errors to simultaneous regions.

## A repair with a measurable benefit and a regression

Three different development trios provided 103 characters for model selection. Single-channel delayed weighted prediction error (WPE) before SepFormer produced 87 edits summed across both recognizers. Prepending separate same-speaker cue utterances produced 88 and the baseline had 92. Three-way median SI-SDR improvement was 4.44 dB for the baseline, 5.33 dB for WPE and 6.26 dB for cues. The prespecified character criterion selected WPE.

| Comparison after development selection | Reference chars | SenseVoice edits, baseline → WPE | Zipformer edits, baseline → WPE | Median three-way SI-SDR improvement, baseline → WPE |
|---|---:|---:|---:|---:|
| Repeated nine short prompts | 24 | 29 → 26 | 19 → 20 | 4.42 → 4.71 dB |
| Nine longer sentences from new voices | 118 | 82 → 73 | 76 → 66 | 2.76 → 3.72 dB |
| Natural meeting excerpt, 134.5 seconds | 634 | 91 → 94 | Not run | No isolated source recordings |

On new longer sentences, SenseVoice character error fell from 69.5% to 61.9% and Zipformer from 64.4% to 55.9%. The nine-edit SenseVoice gain came entirely from one of the three trios. Zipformer improved on two, with a tie on the third. Whole-sentence exact matches remained zero for both. On the short prompts, SenseVoice had zero exact phrases both times. Zipformer went from zero to one exact phrase despite one more aggregate edit. Insertions can push character error above 100%.

The longer case panel contains 14.95 seconds of mixed speech, including 4.40 seconds of all-three activity by the source-energy criterion. The repeated panel contains 5.60 seconds, including 1.14 seconds of all-three activity. Both panels use the same simulated-room procedure and test read speech. The natural meeting comparison was a previously evaluated excerpt. Re-running the original model reproduced all three modified windows within a single 16-bit PCM quantization step. Using the frozen WPE filter increased the natural excerpt's error from 14.35% to 14.83%.

A fixed two-pass waveform blend of baseline and WPE used both separator estimates. It produced 50/43 SenseVoice/Zipformer edits on development, 26/18 on the repeated prompts and 75/68 on the new longer sentences. It required two evaluations of the separator and performed worse than single-pass WPE on the longer-sentence panel for both recognizers.

Keep the existing real-meeting separator as the default. The WPE result supports further room-specific tests, with the real excerpt regression retained in its evidence record. The controlled tests have not established 90% accuracy for three people speaking simultaneously.
