# Speaker-attributed Mandarin transcription diagnosis

AliMeeting session R8008_M8016 runs for 30 min 39 s. Three people supplied 622 reference speech turns and 7,705 normalized reference characters. Each test used the same int8 SenseVoice model and Chinese decoding settings. Character error rate counts substitutions, deletions and insertions after matching anonymous voice labels across the full recording.

| ASR input and timing | Errors | Character error rate | A errors / 3,380 | B errors / 2,745 | C errors / 1,580 |
|---|---:|---:|---:|---:|---:|
| Separated room tracks + predicted turns | 2,188 | 28.40% | 559 | 518 | 1,111 |
| Single room microphone + predicted turns | 1,941 | 25.19% | 381 | 463 | 1,097 |
| Same tracks + annotated turns | 1,996 | 25.91% | 521 | 430 | 1,045 |
| Individual headsets + annotated turns | 548 | 7.11% | 82 | 102 | 364 |
| Single room microphone + annotated turns | 1,747 | 22.67% | 380 | 387 | 980 |
| Individual headsets + predicted turns | 808 | 10.49% | 122 | 152 | 534 |

The headset and annotation controls use information unavailable from an ordinary room recording. Each headset was aligned with the room recorder using the 50–60 ms delay measured from solo speech. The same utterance merging, length limit, gain, padding, model and scoring method were used across controls.

## What causes the errors

- Undercovered turns had fewer than 75% of reference speech frames marked active for the right voice. These turns contributed 542 errors on 774 reference characters (70.03%). Better-covered turns contributed 1,646 on 6,931 (23.75%). Edits are assigned to whole annotated turns through the full-speaker edit alignment. This measures association with coverage.
- On speaker C’s solo turns with at least 75% predicted timing coverage, the room pipeline yielded 203 errors on 277 characters (73.29%). Headset audio and reference timing yielded 64 (23.10%) on those same turns. This voice was difficult even during solo speech.
- Suppression removed audible signal. During 143.29 speaker-seconds with overlapping annotated speech and strong levels at both microphones, the target track contained exactly zero PCM samples for 4.92 seconds (3.43%). The corresponding solo rate was 4.49 of 659.55 seconds (0.68%). Speaker C lost 2.40 of 22.92 high-energy overlap seconds (10.47%). The detector counts 10 ms frames exceeding their silence energy thresholds by 6 dB in both microphones. Room noise and voice bleed can affect this diagnostic.
- Correcting the ASR turn boundaries on the same room tracks avoided 192 edit operations. Substitutions fell from 1,224 to 871. Deletions rose from 871 to 1,057. The waveform retains any muting applied before decoding.
- With identical annotated turn boundaries, individual headsets avoided 1,448 additional edit operations compared with the delivered room tracks. This measures the combined effects of recorder position, competing voices, room acoustics and waveform processing.
- Repeating the timing comparison on the individual headsets changed 808 edits to 548. Averaging the effects of changing audio and timing in both possible orders assigns 1,414 of the 1,640 avoided edits to the change in audio and 226 to the change in boundaries. These numbers describe this recording and these two interventions.
- With identical annotated turn boundaries, the unprocessed room channel yielded 1,747 errors compared with 1,996 for the three delivered tracks. The untreated channel reduced deletions from 1,057 to 493, and increased insertions from 68 to 219. The room microphone retained competing voices during overlap.
- Only 89 of 1,224 aligned character substitutions had the same primary toneless Mandarin dictionary reading in Unicode 15.1. This total depends on dictionary coverage, polyphonic character handling and the choice of edit alignment.

## Additional correction trials

A short-reply classifier used 640 candidate spans from a separate 37-minute meeting (R8008_M8013). Speaker-held-out development selection chose a probability threshold of 0.45. It altered two outputs from the 30-minute meeting and removed two character errors overall. It made zero changes on four additional short clips. A second method proposed pronunciation-matched substitutions using words repeated by other speakers in the input transcript. Its development-selected threshold approved zero changes on the 30-minute meeting and four short clips. Both trials remain experimental.

## Data and reproduction

OpenSLR distributes the Mandarin meetings and documents the use of a room microphone array and individual headset microphones. The room channel and exported speaker tracks were taken from the existing M8016 benchmark. The three headset WAV files were fetched from the pinned Hugging Face mirror, verified against its LFS SHA-256 values, and matched to anonymous voice labels by maximum speech-time overlap. The ground-truth TextGrid SHA-256 is `68b7e6a42ad94c7d0d87036ae460b04f769696d687386ef732802bd894b45e95`. [AliMeeting corpus](https://www.openslr.org/119/) · [Pinned waveform mirror](https://huggingface.co/datasets/ggfox00000/dia-alimeeting-test/tree/cef2837c4a1a0762074522c1bbbbb90eb4d1acf5/audio/near)

Run `fetch_near.py` to download the three near channels, `energy_audit.py` to estimate their clock offsets and verify muted output frames, `oracle_matrix.py --help` to score each full-meeting control, and `localize.py` for turn-level errors. Provide the repository's `recognize.py` and its dependencies on `PYTHONPATH`. Use a separate output directory for each recording, model version and configuration. The JSON summary contains aggregate counts without transcript content.
