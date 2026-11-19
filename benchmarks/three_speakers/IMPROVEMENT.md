# Three-speaker performance improvement

The baseline uses Nemotron 3 for activity labels, DPRNN for waveform separation, and Zipformer for Mandarin recognition. The candidate uses a calibrated activity decoder, parallel DPRNN windows, optional SepFormer for three-way windows, and optional utterance level normalization for recognition.

## Speaker timing on new meetings

The 37-minute session R8008_M8013 supplied calibration data. The fixed profile was tested on R8008_M8014 and R8008_M8015, which contained six different participants and 63 minutes 53 seconds of room audio. Each run received only the first room microphone channel. This test contained 7.13 seconds with all three participants speaking at once. The two sessions were selected for their annotated three-way overlap duration among eligible test sessions.

| Pooled measurement | Native timing | Calibrated timing |
|---|---:|---:|
| Speaker-time precision overall | 83.3% | 91.8% |
| Speaker-time recall overall | 97.3% | 97.1% |
| Diarization error rate overall | 21.6% | 11.0% |
| Speaker-time recall during overlap | 85.8% | 89.9% |
| Speaker-time recall with all three active | 78.8% | 86.8% |
| Recall of frames with exactly three voices | 38.4% | 61.9% |
| Precision of frames marked as exactly three voices | 25.1% | 16.4% |

The candidate improves overall timing and third-voice recovery. A frame can contribute correct assignments for two active speakers while a false third voice lowers three-way precision. In the second meeting, exact-three F1 dropped from 27.1% to 9.8%.

The evaluation uses 10 ms frames and one fixed identity match for each entire meeting. The calibration widens a weak third-voice detection near stronger evidence for that speaker and filters out three-voice assignments with low confidence. The profile is stored in `overlap_profile.json`.

## Waveform isolation

Each of 24 controlled mono Mandarin mixtures had exactly three known sources. The 12 clean mixtures and their 12 simulated-room counterparts used speakers from four different three-person groups. Together they contained 9.44 seconds of true triple overlap. The compared models received the same mixtures and the count of three. The median scale-invariant signal-to-distortion ratio improvement (SI-SDRi) was measured against the original mixture during true triple overlap.

| Condition | DPRNN FP32 | SepFormer Libri3Mix |
|---|---:|---:|
| Clean | +2.71 dB | +6.16 dB |
| Simulated room with background noise | +2.94 dB | +5.91 dB |
| CPU inference time over all clips | 25.19 s | 78.97 s |

Dynamic int8 DPRNN took 29.44 seconds and scored +2.04 dB in simulated rooms. `render.py` supports `--sepformer-three` for windows where the estimated activity contains at least 50 ms of three-way overlap.

## CPU export and recognition

On the first five minutes of R8008_M8014, the median rendering time over two runs was 24.05 seconds with one worker and four intra-op threads, and 15.17 seconds with four workers and one thread each. Both configurations rendered the same 32 overlap windows. Of 14.4 million exported PCM16 samples, 73 samples differed by one quantization step and the other samples matched.

On the original 37-minute file, applying a peak-limited gain to each utterance reduced speaker-attributed Mandarin character error from 42.9% to 38.9% on the same separated tracks. The combined candidate reduced the audio rendering stage from 5 minutes 16 seconds to 3 minutes 38 seconds and detected 51 windows for SepFormer. The combined run scored 40.8% character error. The original meeting supplied calibration for the timing profile. These CPU file tests measure batch throughput.

## Run the candidate

The candidate takes a mono 16 kHz WAV containing three speakers. The previous benchmark documents how to obtain the native CPU runtime, GGUF, and DPRNN files. [Previous benchmark](README.md)

The optional SepFormer weights are `encoder.ckpt`, `decoder.ckpt`, and `masknet.ckpt`. Install `speechbrain==1.0.3` to use them. [Pinned SepFormer weights](https://huggingface.co/speechbrain/sepformer-libri3mix/tree/1f2d824e2333aec9e60d0030fb144885c1afc426)

```bash
bench=benchmarks/three_speakers
data=/path/to/data
nemo=/path/to/nemo-speech-linux-cpu
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
LD_LIBRARY_PATH="$nemo/lib" python "$bench/nemotron_probabilities.py" "$data/mono.wav" \
  --library "$nemo/lib/libnemo_speech_asr_c.so.1" \
  --model "$data/Nemotron-3-Diarization.q8_0.gguf" \
  --output "$data/probabilities.npz" --default-rttm "$data/native.rttm"
python "$bench/improve_overlap.py" "$data/mono.wav" --probabilities "$data/probabilities.npz" \
  --native-rttm "$data/native.rttm" --profile "$bench/overlap_profile.json" \
  --out "$data/updated_timing.json"
python "$bench/render.py" "$data/mono.wav" --diarization "$data/updated_timing.json" \
  --out "$data/updated" --method dprnn --project . --models models \
  --workers 4 --threads 1 --sepformer-three "$data/sepformer3"
python "$bench/recognize.py" --audio "$data/updated" --timing "$data/updated_timing.json" \
  --project . --out "$data/updated" --asr models/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03 \
  --threads 3 --target-rms .035
```

`improvement_metrics.json` records the source revisions, per-meeting scores and timing checks. `compare_three_separators.py` reproduces the waveform comparison from the fixtures and cache prepared by `prepare_validation_cases.py` in the repository root.

[AliMeeting corpus](https://www.openslr.org/119/) · [Public test audio mirror](https://huggingface.co/datasets/ggfox00000/dia-alimeeting-test/tree/cef2837c4a1a0762074522c1bbbbb90eb4d1acf5) · [Time-aligned scoring labels](https://github.com/nttcslab-sp/diar-forced-alignment/tree/9527b7c64846fb38316a610f32e9d3466bd6d8b7/AliMeeting)
