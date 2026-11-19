# Independent AISHELL-4 overlap benchmark

This panel contains 12 disjoint excerpts from three AISHELL-4 test recordings. Each excerpt has exactly three annotated speaker identities. Source recordings have five or six participants. The fixed crop protects against intrusion from every other annotated speaker with a 200 ms inactivity margin. The selection analyzes only the published activity labels. The source rooms were previously indexed for public annotations but were excluded from all existing acoustic inference scores and fitting in this project.

| Official recording | Clips | Duration | Exactly three voices active |
|---|---:|---:|---:|
| L_R003S03C02 | 4 | 88.00 s | 7.16 s |
| M_R003S02C01 | 4 | 88.00 s | 6.38 s |
| S_R003S04C01 | 4 | 76.53 s | 4.85 s |
| Total | 12 | 252.53 s | 18.39 s |

For each preregistered room, the selection maximizes annotated triple-speech time subject to at most four disjoint 6–22 s excerpts, a minimum of 0.40 s of true three-way speech and the outside-speaker margin. Another source meeting was considered by the same rule and yielded zero eligible clips. No inference scores or individual listener assessments enter these choices.

The room audio comes from microphone channel 1 of an original eight-channel FLAC, decoded at the exact sample offsets of the released WAV clock. Polyphase antialias resampling converts 48 kHz to 16 kHz. The resulting files use mono PCM16 WAV. Each file and reference has a SHA-256 in the manifest. `prepare_aishell.py` verifies every compressed byte-range response and exact decoded sample count before writing. Release sources and annotation pointers are explicit URL and content hash fields per row.

`text_shortlist.py` builds a separate text-safe shortlist containing 26.33 s of audio from two independent meetings. Its 147 normalized characters come from fully contained official TextGrid utterances. Trim points were selected within the already-frozen timing excerpts before acoustic evaluation. The two rooms contribute 3.46 seconds of genuine three-way speech. Word intervals are utterance annotations, so uniform positions inside those utterances would only estimate per-character timing. Full text edit measurements for these clips and timing measurements for the 12-clip panel have different denominators.

The primary comparators apply the preceding release without retraining or adapting a threshold to this corpus. The scoring uses the existing framewise three-state evaluator at 10 ms with no collar and a single optimal speaker mapping per clip. The benchmark checks cross-corpus transfer of tuning done on distinct sources. The earliest training sets of third-party pretrained checkpoints were unavailable for an overlap audit.

| Frozen profile | Exact-three precision | Exact-three recall | Exact-three F1 | Diarization error |
|---|---:|---:|---:|---:|
| Width-15 fixed profile | 34.03% | 38.28% | 36.03% | 22.50% |
| Previous dictionary and narrow boundaries | 35.87% | 38.23% | 37.01% | 24.20% |
| Released acoustic model and sequence decoding | 55.03% | 35.67% | 43.29% | 18.58% |
| Released DiariZen and sequence decoding | 55.25% | 30.34% | 39.17% | 19.06% |

Inputs derive from AISHELL-4, published by Beijing Shell Shell Technology. The original distribution states CC BY-SA 4.0 and the Hugging Face metadata labels its mirror Apache-2.0. Treat the original distribution's CC BY-SA 4.0 terms as the governing license for redistributed cropped audio. [OpenSLR SLR111](https://www.openslr.org/111/) · [Pinned upstream files](https://huggingface.co/datasets/AISHELL/AISHELL-4/tree/aada72727856313b19d4a030383c426364931dbf/test)

## Reproduce

Set `PYTHONPATH` to the project's three-speaker modules and diarization benchmarking modules. Run the commands below in a Python environment with NumPy, SciPy, SoundFile and the existing recognition dependencies.

```sh
python prepare_aishell.py --out corpus --references references --rttm-dir references
python text_shortlist.py --root corpus --refs references
python verify_holdout.py --root corpus --manifest corpus/manifest.json --out verification.json
```

Each benchmark script uses the frozen `manifest.json` entries and omits reference data at inference. `run_timing.sh`, `run_dz.sh`, and `run_asr.sh` show the local commands used with the preceding release checkpoints. Their environment path must be adjusted outside the evaluation workspace.
