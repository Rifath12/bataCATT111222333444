# Label three separated mono WAVs with enrolled A/B/C voices

This optional offline step reorders the three output tracks of a mono separator after its signal-quality gate accepts the window. Supply one or more known solo recordings per speaker and keep those labels throughout the session. The three input tracks must be sample-aligned mono 16-kHz WAVs of equal length. Use the same microphone setup for the enrollment speech and recording wherever possible. Each speaker needs an audible 0.512-second solo excerpt. The benchmark used multiple examples per speaker, up to 16.38 seconds each.

Install NumPy, SciPy and SoundFile. Give the command a new output directory under an existing parent.

```sh
python enrolled_order.py --tracks slot1.wav slot2.wav slot3.wav \
  --enroll-a solo_A.wav --enroll-b solo_B.wav --enroll-c solo_C.wav \
  --out labeled
```

The export directory contains `Speaker_A.wav`, `Speaker_B.wav`, `Speaker_C.wav`, and `mapping.json`. Each WAV is a byte-for-byte copy of its source. The JSON records each track's original slot, filename, enrollment file(s) and SHA-256. The assignment margin measures the difference in feature-space cost between the best and next permutation. It has no calibrated probability interpretation. Silent outputs, tied assignments, invalid WAVs and unequal sample counts produce an error. Existing output directories are protected against overwriting.

For use inside the pipeline, `fit_templates([speaker_a_clips, speaker_b_clips, speaker_c_clips], 16000)` accepts collections of isolated, normalized mono PCM arrays. Pass the returned templates and a three-by-samples track array to `assign_tracks(templates, tracks, 16000)`. It returns the A/B/C index order and the cost margin. The order indexes into the original array.

## Controlled results

Early solo recordings established each participant's A/B/C label. Evaluation used different, later donor intervals with known source waveforms. The three direct signals from separate headsets were summed into one channel for separation. Reference waveforms were used afterward to determine each output's best assignment for scoring.

| Session | Clips admitted by waveform gate | Raw checkpoint slots placed correctly | Enrollment labels placed correctly | All three enrollment labels correct |
| --- | ---: | ---: | ---: | ---: |
| M8005 | 0 of 4 | — | — | — |
| M8008 | 4 of 4 | 4 of 12 | 12 of 12 | 4 of 4 |
| M8012 | 4 of 4 | 5 of 12 | 10 of 12 | 3 of 4 |
| Total scored | 8 of 12 | 9 of 24 | 22 of 24 | 7 of 8 |

In two earlier sessions from the same room, the method also matched 24 of 24 diagnostic voice assignments. The three-room test used a fixed procedure selected before scoring and three different rooms. The checkpoint's four rejected M8005 windows would use separate fallback processing. Its fallback labels were unevaluated.

The mapper assigns one global permutation per window. Speech isolation, local swaps within a track, unknown speakers and silent or absent participants require separate handling. This evaluation covers short, controlled overlaps with three participants. Continuous 30-minute identity tracking on natural conversations remains unverified.
