# Three-speaker Mandarin separation with a 90% recall target

The experimental decoder reached **92.3% speaker-time recall** during three simultaneous voices on AliMeeting excerpts and **87.5%** on speaker-disjoint REAL-T excerpts. Precision on frames marked as three voices fell on both sets. The results support keeping the previous decoder as the default. The experimental decoder can export A/B/C timing for the existing WAV renderer.

## Natural mono recordings

| Evaluation | Duration with three voices | Decoder | Speaker-time recall during triple overlap | All-three frame recall | All-three frame precision | Overall DER |
|---|---:|---|---:|---:|---:|---:|
| REAL-T Mandarin H2, 26 excerpts from 10 sessions | 66.21 s | Previous calibrated | 66.8% | 34.1% | 49.8% | 32.3% |
| REAL-T Mandarin H2 | 66.21 s | Experimental two-pass | 87.5% | 66.5% | 37.8% | 22.8% |
| AliMeeting Eval, 83 excerpts from four sessions | 73.26 s | Previous calibrated | 91.8% | 77.1% | 47.5% | 19.1% |
| AliMeeting Eval | 73.26 s | Experimental two-pass | 92.3% | 78.5% | 25.7% | 34.7% |
| AliMeeting, two complete three-participant meetings | 7.13 s | Previous calibrated | 86.8% | 61.9% | 16.4% | 11.0% |

Speaker-time recall credits each correctly assigned active speaker during a true three-way interval. All-three frame recall requires all three voices in the same 10 ms frame. All-three frame precision is the share of the system’s three-voice frames that truly contain three voices. DER combines missed speech, extra speech and speaker confusion over the full scored timeline. Scoring uses zero collar and one best identity matching per excerpt or full meeting. It does not establish cross-excerpt identity continuity, waveform isolation, or character recognition accuracy.

The first evaluation contains 7:11 of natural mono speech selected for overlap and no reference silence. The [pinned REAL-T release](https://huggingface.co/datasets/REAL-TSE/REAL-T/tree/39ca3b88ffd4ce6078c1d045a3e8711fa2bd2f17) provided the data. There were 45 development excerpts from nine other three-participant sessions, with no shared source sessions or annotated speaker IDs. The decoder used separate development sessions to select thresholds and made no changes for the test data.

The second evaluation contains 31:42 of mono audio from [AliMeeting Eval](https://www.openslr.org/119/). The original meetings had four participants. Each scored excerpt contains exactly three annotated identities and excludes the fourth with a 250 ms guard wherever the adjacent reference allows it. The selection retains silence and one- and two-voice speech within each excerpt. This is aggregate excerpt duration from several meetings. The frozen REAL-T decoder was also tested without adapting it to AliMeeting.

The two complete meetings were evaluated previously. This round tried a separate confidence blend for long files using a level-normalized pass with a shifted start. Development selected a second-pass weight of 0, yielding the previous result. The same-session REAL-T experiment used additional clips from the same meeting as context and achieved 81.0% speaker-time recall and 40.8% exact-three precision. Those scores apply to clips with that additional context.

## Methods and speed

One mono file is passed through Nemotron twice. The second pass uses level normalization and the native offline inference interface. The experimental decoder aligns their anonymous speaker streams by posterior correlation and combines the diarization outputs with spectral and periodicity features from the same file. A small gradient-boosted classifier predicts the active speaker count on a 10 ms grid. Only the benchmark scorer receives the reference labels for held-out files.

On the REAL-T test, native inference summed to 146 s, and the second pass added 111 s. On AliMeeting excerpts, native inference summed to 419 s, and the second pass added 266 s. These are file-processing times measured on a shared Linux CPU, with file input already available. They exclude training, WAV separation and recognition. Streaming latency and RTX 5080 throughput were not measured in this run.

## Waveform isolation

On six separate three-speaker Mandarin triplets, two-polarity SepFormer improved median isolation over the mixture from **+3.52 to +5.02 dB** during exact triple overlap in simulated rooms. The weakest speaker went from **+4.94 to +4.82 dB**. Across the whole clips, the pooled median fell from +7.61 to +6.94 dB, and inference took 1.97 times as long. The difference in the simulated-room median had a triplet-bootstrap interval of −0.81 to +2.19 dB. These short, controlled tests support retaining the single pass for general use.

A separate matched comparison used the enrollment-conditioned [PS4 checkpoint](https://huggingface.co/TaurenMountain/PS4/tree/8feb7a1b3da1264659f155c42bfb40a5be88d730). On 12 simulated-room mixtures, its median SI-SDR improvement during true triple overlap was −1.25 dB in enrollment order, versus +5.91 dB for SepFormer on the same cases. The PS4 estimate remained below 0 dB after best-case track reassignment. Neither test measures waveform isolation in the two long natural recordings. [Waveform protocol](WAVEFORM.md)

## Reproduce the natural benchmark

Start from the repository root. Use Python 3.11 or newer with NumPy, SciPy, SoundFile, scikit-learn and threadpoolctl. Use the previous benchmark’s Linux NeMo-Speech.cpp v0.3.0 CPU package and Q8 GGUF. The tested binary and model checksums are in the [previous benchmark](../README.md).

```bash
bench=benchmarks/three_speakers
trial="$bench/target_90"
work=/path/to/local/benchmark
runtime=/path/to/nemo-speech-0.3.0-linux-x86_64-cpu
gguf=/path/to/Nemotron-3-Diarization.q8_0.gguf
export PYTHONPATH="$PWD/$bench"
export LD_LIBRARY_PATH="$runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
python "$trial/prepare_realt.py" "$work/realt"
python "$trial/batch_probabilities.py" \
  --audio-list "$work/realt/audio.txt" --library "$runtime/lib/libnemo_speech_asr_c.so" \
  --model "$gguf" --output "$work/realt_probs"
python "$trial/multiview_score.py" \
  --train-manifest "$work/realt/development.json" --train-probs "$work/realt_probs" \
  --profile "$trial/previous_profile.json" \
  --test realt "$work/realt/heldout.json" "$work/realt_probs" --out "$work/realt_scores.json"
```

For AliMeeting, use the Eval archive and the four RTTMs at the revision and paths in the [previous benchmark](../README.md). The session IDs and checksums are in [metrics.json](metrics.json). Run `prepare_alimeeting.py --archive ARCHIVE --references RTTM_DIRECTORY --out DIRECTORY` to produce the WAVs and manifest. List the 83 resulting WAV paths one per line for `batch_probabilities.py`. Add `--test ali DIRECTORY/manifest.json PROBABILITY_DIRECTORY` to the same evaluation command to use the frozen REAL-T settings on AliMeeting.

To export experimental A/B/C timing for another mono 16 kHz file, cache its two views with `batch_probabilities.py` and run the scoring command with `--predict YOUR_WAV YOUR_NPZ TIMING_JSON` in place of `--test` and `--out`. The fit still uses the same 45 labeled development examples. The new audio receives no reference inputs. The output JSON is accepted by the existing `render.py --diarization` option. The standard renderer requires three detected labels and enough predicted solo speech to form speaker anchors for separation.
