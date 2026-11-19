# Further evaluation on natural Mandarin

The complete natural meeting is AliMeeting Test R8008_M8017. Its first microphone has 29,120,832 samples at 16 kHz, spanning 30:20.05. Previous experiments evaluated speaker activity throughout this recording. This round introduces scoring against its far-field speaker-attributed transcript. Three participants speak during the meeting, with 1.13 seconds of annotated simultaneous speech by all three.

Three additional excerpts come from previously unused AliMeeting Test recordings. All use channel one of an eight-channel far-field array. The nine excerpt speaker IDs have no intersection with the preceding 39 labeled evaluation and calibration participants. Training exposure of pretrained models is unknown. The parent meetings have four participants. Exactly three have annotated speech within each selected contiguous interval. The selector filters candidates by annotations before any system prediction. It adds the prior benchmark's fourth-speaker absence guard of 250 ms, moves edges inward to the ends of complete transcript utterances, and requires at least 30 seconds. For each eligible source meeting it chooses the qualifying interval with the most annotated three-way overlap, breaking ties by duration. It takes up to three source meetings with the most three-way support. This yields 183.89 seconds and 4.10 seconds of three-way overlap. Integer frame scoring rounds durations up to 10 ms.

| Test source | Selected time in source, seconds | Duration | Annotated three-way overlap |
|---|---|---:|---:|
| R8006_M8012 | 177.27–243.96 | 66.69 s | 2.56 s |
| R8002_M8003 | 276.46–319.31 | 42.85 s | 1.28 s |
| R8004_M8005 | 1609.52–1683.87 | 74.35 s | 0.26 s |

Sources and reproduction:

1. Put the existing benchmark modules containing `recognize.py`, `prepare_alimeeting.py`, `diarize.py`, `improve_overlap.py`, `calibrate_nemotron.py`, and `multiview_score.py` on `PYTHONPATH`. Run `python prepare_new_excerpt.py --out new_excerpts`. This retrieves pinned far-field TextGrids and the selected unmodified PCM samples only by HTTP range, writes the first microphone at the original rate and precision, and produces a per-excerpt manifest including hashes, speaker IDs, positions, and source paths.
2. Feeding the emitted `inputs.txt` to the existing `batch_probabilities.py` creates the paired raw and offline probability export. The frozen original calibrated baseline uses only the base probabilities. Run `python score_extra.py --root new_excerpts --profile previous_profile.json` after placing the exports in `new_excerpts/probabilities`. No reference times or identities enter inference.
3. Use predicted timing with the existing render and SenseVoice Small int8 benchmark implementations. The old path requires a 750 ms solo anchor for every speaker. It failed on R8002_M8003, for which the comparison uses the original mixture-routing fallback. A transient runtime interruption on R8004_M8005 was retried. The new partial-enrollment renderer operates on the identical audio and predicted time lines. The optional ASR overlap mask uses strength 0.8. Both recognizer comparisons use fixed target RMS 0.035. Each WAV set has A/B/C tracks aligned with its input.
4. For character scoring, concatenate normalized Chinese text by speaker in time order and select the complete-recording mapping that minimizes pooled character edits. Report summed substitutions, deletions and insertions divided by reference characters. Speaker-activity scores use 10 ms midpoint frames with no collar, include overlap, and allow one speaker permutation per recording. Speaker-activity and text scores measure different tasks. The available room microphone and TextGrid cannot measure isolated source waveform fidelity.

Pinned audio and TextGrids: https://huggingface.co/datasets/AudioMarathon/AudioMarathon/tree/68dd695283723a78d850895466b9a380f630b561/AliMeeting/Test_Ali/Test_Ali_far

Pinned speaker-alignment references: https://github.com/nttcslab-sp/diar-forced-alignment/tree/9527b7c64846fb38316a610f32e9d3466bd6d8b7/AliMeeting/Test_Ali_far

Corpus protocol: https://openslr.org/119/

The fixed ASR-only waveform comparison uses the frozen 0.8 strength on all three independently selected excerpts. A/B/C listening WAVs continue to use the neural separator. The recognizer can read separate soft-masked WAVs that retain the mixture phase. To export them with bounded memory, use the renderer below with the existing `asr_mask.py` and `diarize.py` modules on `PYTHONPATH`.

```sh
python render_asr_mask.py --mix mono.wav --tracks separated --timing timing.json --out asr_tracks
```

The exporter operates on ten-second waveform blocks with overlapping margins. It uses temporary float WAVs to apply the same shared peak gain to all tracks before writing PCM16. On the 66.69-second test excerpt its three outputs were byte-identical to the in-memory reference implementation. Each of the three new test meetings had lower cpCER with the ASR mask, from 398/1,128 pooled errors (35.28%) to 367/1,128 (32.54%). `guard_comparison.json` includes the counts and direct-separation control for each source.

On the complete natural M8016 meeting, the same mask reduced errors from 2,188 to 1,970 over 7,705 characters (28.40% to 25.57%). On M8017 it reduced errors from 1,198 to 1,100 over 8,154 characters (14.69% to 13.49%). Pooling the two complete meetings yields 3,386 to 3,070 errors over 15,859 characters (21.35% to 19.36%). Deletions decreased from 1,372 to 998 while insertions rose from 145 to 215. The results file is `full_guard_comparison.json`.

The M8017 pair was fully decoded. In the M8016 comparison, `rescore_changes.py` enumerated all 855 original inference spans, verified unchanged timing, compared every sample for each speaker/span pair, reused the original decoding of 378 PCM-identical spans, and ran the same SenseVoice recognizer on the 477 changed spans. All resulting utterances were scored against the complete transcript. A check on the independently decoded M8017 pair found identical text for every one of the 379 nonblank spans with identical waveforms.
