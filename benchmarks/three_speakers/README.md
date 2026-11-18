# Long Mandarin meeting with three speakers

Session R8008_M8013_MS807 contains 37 minutes 19 seconds of room audio and exactly three annotated participants. The mono trials used the first microphone channel of the AliMeeting far-field recording. They received the session count of three and generated anonymous speaker identities. The individual headset channels and the annotations were reserved for scoring and a separate diagnostic.

## Results

| Tracker | Overall speaker-time recall | Recall during overlap | Recall during three-way overlap | Diarization error |
|---|---:|---:|---:|---:|
| Pyannote + ERes2Net | 60.6% | 57.2% | 57.9% | 58.2% |
| Reverb + ERes2Net | 49.5% | 38.4% | 33.8% | 72.4% |
| Reverb + CAM++ and input gain | 59.6% | 40.6% | 33.8% | 62.3% |
| Sortformer v2.1 | 61.5% | 52.4% | 49.4% | 42.5% |
| DiariZen WavLM | 70.6% | 70.8% | 63.3% | 49.2% |
| Nemotron 3 | 97.3% | 90.4% | 81.1% | 22.9% |
| Nemotron 3 + Silero speech gate | 87.3% | 84.9% | 74.6% | 23.8% |
| Nemotron 3 calibrated (33:19 test) | 93.9% | 86.5% | 77.8% | 10.1% |
| Nemotron 3 calibrated for overlap (33:19 test) | 96.3% | 91.2% | 81.1% | 13.1% |
| Eight channels with marked examples (33:19 test) | 76.9% | 62.2% | 51.8% | 26.6% |

The full recording contains 211.94 seconds of overlap, including 8.82 seconds of three-way overlap. The exported pipeline used Nemotron 3 Q8, DPRNN overlap separation, Mandarin Zipformer CTC and took 10.70 minutes on the Linux CPU. It produced 42.9% speaker-attributed character error and correctly matched 5,120 of 8,943 reference characters. The individual-headset and reference-timing diagnostic scored 5.1% character error. The exported pipeline's overall speaker-time precision was 82.5%.

Across the controlled three-way mixtures, median SI-SDR improvement over the input was +1.94 dB for DPRNN and +1.87 dB for BSRNN with separate enrollment.

With the reference turn times on the unseparated mono recording, the same recognizer scored 32.6% speaker-attributed character error.

The exported mono run reached the 90% recall target for the full meeting and for overlap.
Recall with all three speakers active remained below 90%.
The audio and transcript are benchmark outputs.

## Scoring

- Speaker-time recall is the fraction of annotated speaker time assigned to the correct track. Each active voice contributes independently during overlap. One optimal identity assignment is fixed across the recording.
- Diarization error rate is the sum of missed speaker time, false speaker time, and confused speaker time divided by annotated speaker time. Speech is scored on 10 ms frames with zero collar and overlap included.
- Speaker-attributed character error is the minimum character edit rate after a single assignment of complete speaker transcripts to reference speakers. Text is normalized with Unicode NFKC, lowercasing, and removal of punctuation and event markup. Character edits are recorded in `metrics.json`.
- SI-SDR improvement is calculated separately on four controlled three-voice mixtures with matched source waveforms. The test includes 5.36 seconds with all three voices active. The target-speaker extractor receives independent enrollment utterances in that diagnostic.

The room-array diagnostic uses all eight channels in the same WAV. The first two minutes supply marked training turns. Minutes two through four select the activity thresholds and smoothing. The following 33 minutes 19 seconds supply evaluation audio. It is a within-meeting adaptation trial that requires additional input.

The two calibrated Nemotron rows also use a within-meeting split. Both fit activity thresholds and smoothing on the first four minutes of annotated mono speech and score the last 33 minutes 19 seconds. The overlap-focused configuration adds a penalty when development recall falls below 94% during overlap. The thresholds are evaluated with no model fine-tuning. The exported three tracks and transcript use the original Nemotron preset.

## Rerun the mono benchmark

Start in the root of this repository with the environment described in the main README. Add `onnxruntime==1.23.2` and `rapidfuzz==3.14.6` to the environment. Run `python fetch_models.py` to retrieve and verify the separator, speaker embedding and Mandarin ASR checkpoints.

Download [Eval_Ali.tar.gz](https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Eval_Ali.tar.gz) and [R8008_M8013.rttm](https://raw.githubusercontent.com/nttcslab-sp/diar-forced-alignment/9527b7c64846fb38316a610f32e9d3466bd6d8b7/AliMeeting/Eval_Ali_far/R8008_M8013.rttm). The RTTM used here has SHA-256 `3d754d9c176eb1115337e4fb93cefc030346d0edf970fb129c187d3575bd1472`. The extracted 16 kHz mono WAV has SHA-256 `303e73615378469b9a35c7ca7fd32d9f6c42f5afec6dc163e79789b4d315b4a1`.

For the exported trial, download the [NeMo-Speech.cpp v0.3.0 CPU release](https://github.com/NVIDIA/NeMo-Speech.cpp/releases/tag/v0.3.0) and [Nemotron-3-Diarization.q8_0.gguf](https://huggingface.co/nvidia/Nemotron-3-Diarization/resolve/main/Nemotron-3-Diarization.q8_0.gguf), SHA-256 `08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1`. The measured runtime used the Linux x86-64 CPU release, which has SHA-256 `255f12b722ab15b2a38a51cf125d56e3fa97058b528f7e895b39d6a473deb06e`. The `v3-offline` preset carries speaker identities across chunks and uses a 30.4 second input buffer. The published measurements cover file-based CPU throughput.

Place the downloads in a directory outside the repository and substitute its path below.

```bash
data=/path/to/data
bench=benchmarks/three_speakers
export OPENBLAS_NUM_THREADS=1
python "$bench/extract.py" "$data/Eval_Ali.tar.gz" --out "$data/session"
python "$bench/nemotron_benchmark.py" "$data/session/mono.wav" \
  --binary "$data/nemo-speech-0.3.0-linux-x86_64-cpu/bin/nemo-speech" \
  --model "$data/Nemotron-3-Diarization.q8_0.gguf" \
  --rttm "$data/R8008_M8013.rttm" --output "$data/nemotron.json"
python "$bench/render.py" "$data/session/mono.wav" \
  --diarization "$data/nemotron.json" --out "$data/separated" \
  --method dprnn --models models --project . --threads 4
python "$bench/recognize.py" --audio "$data/separated" --timing "$data/nemotron.json" \
  --reference "$data/session/far.TextGrid" --project . --out "$data/separated" \
  --asr models/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03 --threads 4
```

`render.py` exports three 16 kHz PCM WAVs with exactly the same sample count as the input. `recognize.py` writes a text transcript and JSONL with speaker and character times. On your own audio, omit the optional RTTM and TextGrid arguments when reference annotations are unavailable.

The Sortformer baseline uses `sortformer_benchmark.py` with the [Sortformer v2.1 ONNX export](https://huggingface.co/tonythethompson/diar-streaming-sortformer-4spk-v2.1-onnx), SHA-256 `82b9c735e1cfc6b36b4ff8a994d9a0573e922d0e80a58a8553b2c58f7aff0c00`, and the [MIT-licensed inference adapter](https://github.com/mefich/sortformer-onnx-python/tree/689282f2528f958d5ce4ae2fddcc049e91ffecef) at commit `689282f2528f958d5ce4ae2fddcc049e91ffecef`.

The optional calibration requires the same NeMo-Speech v0.3.0 dynamic library and accepts the mono recording. `nemotron_probabilities.py` calls the public C interface to export 10 ms probabilities as an NPZ. `calibrate_nemotron.py` takes that NPZ, the JSON from `nemotron_benchmark.py`, and the reference RTTM. Its overlap-focused variant uses `--overlap-recall-target 0.94`. These measurements concern activity labels.

## Attribution

[AliMeeting](https://www.openslr.org/119/) was published by Alibaba Group under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). [Forced-aligned timing](https://github.com/nttcslab-sp/diar-forced-alignment) supplies the diarization reference. The Mandarin character references come from the corpus TextGrid. Model weights and original recordings are downloaded separately under their respective licenses.
