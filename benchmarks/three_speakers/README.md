# Long Mandarin meeting with three speakers

Session R8008_M8013_MS807 contains 37 minutes 19 seconds of room audio and exactly three annotated participants. The mono trials used the first microphone channel of the AliMeeting far-field recording. They received the session count of three and generated anonymous speaker identities. The individual headset channels and the annotations were reserved for scoring and a separate diagnostic.

## Results

| Tracker | Overall speaker-time recall | Recall during overlap | Recall during three-way overlap | Diarization error |
|---|---:|---:|---:|---:|
| Pyannote + ERes2Net | 60.6% | 57.2% | 57.9% | 58.2% |
| Reverb + ERes2Net | 49.5% | 38.4% | 33.8% | 72.4% |
| Reverb + CAM++ and input gain | 59.6% | 40.6% | 33.8% | 62.3% |
| Sortformer v2.1 | 61.5% | 52.4% | 49.4% | 42.5% |
| Eight channels with marked examples (33:19 test) | 76.9% | 62.2% | 51.8% | 26.6% |

The full recording contains 211.94 seconds of overlap, including 8.82 seconds of three-way overlap. The exported pipeline used Sortformer v2.1, overlap DPRNN, Mandarin Zipformer CTC and took 9.69 minutes on the Linux CPU. It produced 56.6% speaker-attributed character error and correctly matched 3,899 of 8,943 reference characters. The individual-headset and reference-timing diagnostic scored 5.1% character error.

Across the controlled three-way mixtures, median SI-SDR improvement over the input was +1.94 dB for DPRNN and +1.87 dB for BSRNN with separate enrollment.

The mono pipeline fell short of the 90% speaker-time recall target. The audio and transcript are benchmark outputs.

## Scoring

- Speaker-time recall is the fraction of annotated speaker time assigned to the correct track. Each active voice contributes independently during overlap. One optimal identity assignment is fixed across the recording.
- Diarization error rate is the sum of missed speaker time, false speaker time, and confused speaker time divided by annotated speaker time. Speech is scored on 10 ms frames with zero collar and overlap included.
- Speaker-attributed character error is the minimum character edit rate after a single assignment of complete speaker transcripts to reference speakers. Text is normalized with Unicode NFKC, lowercasing, and removal of punctuation and event markup. Character edits are recorded in `metrics.json`.
- SI-SDR improvement is calculated separately on four controlled three-voice mixtures with matched source waveforms. The test includes 5.36 seconds with all three voices active. The target-speaker extractor receives independent enrollment utterances in that diagnostic.

The room-array diagnostic uses all eight channels in the same WAV. The first two minutes supply marked training turns. Minutes two through four select the activity thresholds and smoothing. The following 33 minutes 19 seconds supply evaluation audio. It is a within-meeting adaptation trial that requires additional input.

## Rerun the mono benchmark

Start in the root of this repository with the environment described in the main README. Add `onnxruntime==1.23.2` and `rapidfuzz==3.14.6` to the environment. Run `python fetch_models.py` to retrieve and verify the separator, speaker embedding and Mandarin ASR checkpoints.

Download [Eval_Ali.tar.gz](https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Eval_Ali.tar.gz) and [R8008_M8013.rttm](https://raw.githubusercontent.com/nttcslab-sp/diar-forced-alignment/9527b7c64846fb38316a610f32e9d3466bd6d8b7/AliMeeting/Eval_Ali_far/R8008_M8013.rttm). The RTTM used here has SHA-256 `3d754d9c176eb1115337e4fb93cefc030346d0edf970fb129c187d3575bd1472`. The extracted 16 kHz mono WAV has SHA-256 `303e73615378469b9a35c7ca7fd32d9f6c42f5afec6dc163e79789b4d315b4a1`.

For the Sortformer trial, use the [Sortformer v2.1 ONNX export](https://huggingface.co/tonythethompson/diar-streaming-sortformer-4spk-v2.1-onnx) with SHA-256 `82b9c735e1cfc6b36b4ff8a994d9a0573e922d0e80a58a8553b2c58f7aff0c00` and [the MIT-licensed inference adapter](https://github.com/mefich/sortformer-onnx-python/tree/689282f2528f958d5ce4ae2fddcc049e91ffecef) at commit `689282f2528f958d5ce4ae2fddcc049e91ffecef`. Use only three session identities. The streaming diarizer runs on file replay. These measurements cover CPU batch throughput.

Place the downloads in a directory outside the repository and substitute its path below.

```bash
data=/path/to/data
bench=benchmarks/three_speakers
export OPENBLAS_NUM_THREADS=1
git clone https://github.com/mefich/sortformer-onnx-python.git "$data/sortformer-adapter"
git -C "$data/sortformer-adapter" checkout 689282f2528f958d5ce4ae2fddcc049e91ffecef
python "$bench/extract.py" "$data/Eval_Ali.tar.gz" --out "$data/session"
python "$bench/sortformer_benchmark.py" "$data/session/mono.wav" \
  --adapter "$data/sortformer-adapter" --model "$data/sortformer.onnx" \
  --rttm "$data/R8008_M8013.rttm" --output "$data/sortformer.json"
python "$bench/render.py" "$data/session/mono.wav" \
  --diarization "$data/sortformer.json" --out "$data/separated" \
  --method dprnn --models models --project . --threads 3
python "$bench/recognize.py" --audio "$data/separated" --timing "$data/sortformer.json" \
  --reference "$data/session/far.TextGrid" --project . --out "$data/separated" \
  --asr models/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03 --threads 3
```

`render.py` exports three 16 kHz PCM WAVs with exactly the same sample count as the input. `recognize.py` writes a text transcript and JSONL with speaker and character times. On your own audio, omit the optional RTTM and TextGrid arguments when reference annotations are unavailable.

## Attribution

[AliMeeting](https://www.openslr.org/119/) was published by Alibaba Group under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). [Forced-aligned timing](https://github.com/nttcslab-sp/diar-forced-alignment) supplies the diarization reference. The Mandarin character references come from the corpus TextGrid. Model weights and original recordings are downloaded separately under their respective licenses.
