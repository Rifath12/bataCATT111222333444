# R8002 confirmation panel

This separate panel uses the unchanged locked-headset generator with SHA-256 `3646e00c9b9d97a0844f0f73ae0d04a330f614831873ddc56c77faac6e8f2328`. Two Mandarin meetings in one previously untested physical room supply six speakers outside the supervised M8014 set and the earlier M8005/M8008/M8012/M8016/M8017 close-mic panels. Earlier work using other modalities had access to the wider AliMeeting corpus. These particular headset remixes were first constructed for this confirmation.

| Session | A, B, C speaker IDs | Clips | Seconds with three active voices |
|---|---|---:|---:|
| R8002_M8002 | 8005, 8006, 8007 | 51 | 38.784 |
| R8002_M8003 | 8009, 8011, 8012 | 37 | 29.024 |
| Both | Six distinct speakers | 88 | 67.808 |

Each session has four annotated participants. `selection.json` identifies the three speakers with the most eligible late solo windows before signal quality filtering. Every donor uses exactly one speaker and each source interval is used once. The first 45% of each recording is reserved for recording-specific calibration. Held donors begin after that point. The original fixed seed rule, target levels, 200 ms annotation margin, 32 ms activity mask and 1.024 s clip duration apply. `verify_locked_sources.py` independently re-creates every source image directly from PCM, checks label exclusion and all interval and channel identities. `lock.json` records the generator and manifest hashes.

To reproduce, place this panel's `sources.json` in the working directory with the unchanged `get_sources.py`, `build_locked_headsets.py` and `score_locked_baselines.py` from the original panel. Run the same download, calibration, build and score commands. Use `python verify_locked_sources.py --root <panel>` for independent PCM checks.
