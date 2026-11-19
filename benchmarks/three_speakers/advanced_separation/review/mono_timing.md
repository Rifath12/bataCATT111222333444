# Auditing the room microphone timing and renderer

The two complete AliMeeting recordings had poor activity coverage for one anonymous voice. Across the two recordings, A and B had 98.3–99.0% speaker-time recall from the released mono timing. C had 89.30% and 91.62%. A diagnostic assigned labels independently to each three-second window using its reference labels. DER would improve only from 11.434% to 11.403% on M8016 and from 8.731% to 8.677% on M8017. The small remaining attribution opportunity indicates that stable label mapping is already working. The larger errors are missed and false activity, weak audio, and recognition errors.

## Export revised speaker boundaries

The frozen profile adds timestamps where two level/offset variants of the same mono model independently report at least 0.50 support within 450 ms of an existing mark. It trims at most 20 ms of a low-confidence edge when both posteriors are below 0.30 for that voice. It preserves the exact bitmap of all three active speakers frame for frame. All thresholds were selected on the separate M8013 development meeting. The two full recordings and both short panels supplied evaluation scores only. They are replay panels used in earlier rounds.

| Panel | Original DER | Revised DER |
| --- | ---: | ---: |
| M8013 development | 12.79% | 11.91% |
| M8016, 30 min 39 s | 11.43% | 10.04% |
| M8017, 30 min 20 s | 8.73% | 7.40% |
| AliMeeting, 18 excerpts | 17.68% | 16.58% |
| AISHELL-4, 12 excerpts | 22.50% | 21.13% |

On the two full recordings, C precision changed from 83.18% to 84.99% and from 94.19% to 95.17%. Its recall changed from 89.30% to 89.92% and from 91.62% to 91.78%. On the short AliMeeting panel, correctly timed speaker-time during reference three-way overlap decreased from 93.17% to 93.07%. It changed from 72.76% to 72.85% on AISHELL-4. Exact-three F1 was unchanged on all panels.

All DER scores use 10 ms frames, zero collar, overlap included, and one annotation mapping fixed from the original prediction for each recording. Each short-panel metric pools frame counts. The room waveform, recognition boundaries and transcription output were held fixed in this experiment. Independent evaluation of using the revised timing for ASR is still pending. The incremental boundary calculation took about 0.13 seconds per full meeting with the two posterior arrays available.

Use the A/B/C timing output and two-pass NPZ generated from the same input. The cache uses base, alternate, offset, native, and duration_s fields used by the existing batch posterior exporter. The audio verifies expected mono sample layout and shared duration.

    python timing_refinement.py --audio room.wav --timing original_timing.json --cache room_two_view.npz --out refined_activity.json

## Preserve single-speaker PCM

The previous renderer merged nearby overlap regions for model context, then wrote separator outputs over any intervening solo region. The fix limits writes to frames where at least two speakers are predicted active. In the two tested recordings, the affected solo regions totaled 0.92 and 0.01 seconds. The isolated regression test substitutes different model audio into one affected solo gap and verifies that the direct room PCM remains exactly intact for the active speaker and that the adjacent overlap receives model samples. Seven renderer tests and four timing tests pass.
