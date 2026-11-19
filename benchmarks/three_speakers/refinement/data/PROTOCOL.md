# Previously unscored AliMeeting Test recordings

All four source arrays were downloaded from the pinned AliMeeting Test mirror revision in `assets.json` and verified with SHA-256 before extraction. Each waveform passed to a model is 16 kHz PCM derived solely from the first microphone. The prior calibration and two previous excerpt and long evaluations use separate AliMeeting source sessions and labeled identities. This split assesses transfer across speakers and meetings from the same corpus. The two new complete sessions use room R8008, also used by earlier long recordings.

| Input | Selected audio | Time with exactly three active voices | Distinct labeled speakers |
|---|---:|---:|---:|
| R8008_M8016, full recording | 30:38.98 | 1.30 s | 3 |
| R8008_M8017, full recording | 30:20.05 | 1.13 s | 3 |
| R8004_M8006, 37 excerpts | 6:45.26 | 38.13 s | 3 per excerpt |
| R8005_M8009, 44 excerpts | 11:51.01 | 53.79 s | 3 per excerpt |

The complete recordings were the remaining two three-participant test sessions in the published annotations. Each of the two excerpt source meetings has four people, with substantial three-person overlap. The previous extraction method selects disjoint intervals at least eight seconds long and requires 400 ms of true three-way overlap. It excludes a fourth identity and retains the other activity in every chosen interval. Every qualifying interval returned by the frozen disjoint-window selector was used. The two complete sessions have little simultaneous three-way speech. The excerpt cohort deliberately enriches that condition. No accuracy threshold or classifier was tuned on these recordings.

The 14 labeled speakers in the new source sessions share zero IDs with the 25 labeled speakers in the seven preceding three-speaker evaluation or calibration sessions. Third-party checkpoint training exposure is unknown. `provenance.json` records the two populations, source sessions and hashes.

To reproduce, run `fetch_arrays.py` beside `assets.json`, then `prepare_fresh.py --root DIRECTORY --references RTTM_DIRECTORY --group short` and the identical command with `--group full`. The scripts require the preceding benchmark's `prepare_alimeeting.py` on `PYTHONPATH`. Use the pinned `nttcslab-sp/diar-forced-alignment` annotations from `AliMeeting/Test_Ali_far` at revision `9527b7c64846fb38316a610f32e9d3466bd6d8b7`. The manifests serve as an independent check of clips, times and IDs. The excerpt two-pass inputs use the preceding batch extraction method. Full recordings use the native probability exporter for both the original and the separately normalized mono file with a 9.28 s offset, as in the earlier full-file trial.
