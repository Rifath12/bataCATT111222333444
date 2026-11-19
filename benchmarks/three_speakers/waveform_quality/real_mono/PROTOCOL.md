# Single-channel Mandarin conversation controls

The development meeting is R8004_M8006 (headsets 8031, 8032 and 8033). The confirmation meeting is R8005_M8007 (8035, 8036 and 8037). These are distinct speakers and physical rooms. Their source recordings had a fourth participant. Every donor is at least 200 ms inside an annotated interval where just its target speaks, accounting for all four original participants. Original source hashes and URLs are in [sources.json](sources.json).

Each meeting yields two matched 33.28-second WAVs, each supplied separately to a single-microphone system. The physical-microphone condition uses channel zero of the actual eight-channel far-field array. All three isolated source images in this condition were captured by that same microphone. The close-speech condition uses each target's own headset for the corresponding utterance. Both conditions mix the same original turns using a shared order, labels, activity mask, and schedule of source levels. Distinct source information available to the scorer is excluded from the single-channel input. A separator may separately use labeled earlier speech for enrollment if the method specifies enrollment.

The controlled timeline contains three solo passages, six two-person passages spanning each pair, and twelve triple passages. Its three-way conditions rotate the four levels 0, −6, −12 and −20 dB among A/B/C. Each event lasts 1.536 seconds, including short margins, and uses disjoint 1.024-second source excerpts. Activity can differ within an event. Three-way activity totals 7.684 seconds in development and 7.659 seconds in confirmation. The paired microphone conditions share these source intervals, so their durations must not be added as additional independent source audio.

Array and headset recorders can have different start times and sampling clocks. We estimate the array-to-headset time map from correlations between short-term energy envelopes in independent 40-second windows. The affine fit had 4.44 ms RMS window delay error in development and 4.43 ms in confirmation. The original array PCM remains untouched by clock compensation. The fit determines which headset samples describe the same utterance. Each capture condition is scored against its own precisely mixed reference waveform.

The first 25% of the source provides eight enrolled utterance excerpts per target (8.192 seconds of recorded excerpts per speaker). All evaluated donors begin after 28% of the source. Donors must contain at least 60% shared activity across array and headset levels, exceed fixed RMS thresholds, match in energy-envelope timing, and have an own-headset advantage. These selection conditions reduce ambiguous donor segments. Isolated turns contain the original room noise, and their ambient sound scales along with the speaker during mixing. Levels are equalized per donor **at the receiving microphone** before applying the stated relative gains. Thus equal nominal levels describe the constructed listening stressor and not the original participants' source loudness. These replayed overlaps have precise mathematical references, while natural overlaps in an untouched meeting have no equivalent clean ground truth.

Each condition has enrollment WAVs, `meeting.wav`, `meeting.npz` with aligned `mixture`, `references` and `activity` arrays, and per-event NPZs under `held`. Reference arrays are fixed in A/B/C order. `mono_manifest.json` records the selected source spans, gain, event position and source hashes. The verifier independently reconstructs every target directly from pinned source PCM, checks disjointness and annotation margins, compares samples and verifies that the three contributions sum to the exact delivered mono WAV. It verified all 102 evaluated utterance donors and 48 enrollment intervals across both rooms. [Verification and payload hashes](verification.json)

## Reproduce

From this directory run the following with Python 3.11+, NumPy and SciPy.

```sh
python get_sources.py
OPENBLAS_NUM_THREADS=1 python build_mono.py
OPENBLAS_NUM_THREADS=1 python verify_mono.py
```

The AliMeeting Mandarin data originates from Alibaba Group and is distributed under CC BY-SA 4.0. [OpenSLR 119](https://openslr.org/119/)

## Score exported WAVs

`score_mono_meeting.py` accepts the condition directory and the physical A/B/C outputs in that order. Supply the existing waveform_quality directory through `PYTHONPATH`. It verifies the complete duration and equal output rates. Native 8 kHz outputs are upsampled to the 16 kHz reference clock with polyphase filtering. It measures event-wise waveform alignment and coverage. Solo and pair scores include a second stricter column requiring every absent output to be at least 20 dB below mean active reference power during their common activity. A duplicated mono signal therefore fails the strict solo measure as well as the joint triple measure. The diagnostic best assignment is a single reference-only permutation for the entire meeting. The released WAVs and primary scores use their original fixed A/B/C labels.

```sh
PYTHONPATH=/path/to/benchmarks/three_speakers/waveform_quality python score_mono_meeting.py R8005_M8007/array_channel0 Speaker_A.wav Speaker_B.wav Speaker_C.wav --out scores.json
```
