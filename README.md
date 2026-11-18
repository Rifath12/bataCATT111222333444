# Mandarin speech separation

Process a single mixed audio file locally. Get one WAV per estimated speaker, a Mandarin transcript with estimated character timestamps, and a listening page that opens in a browser. Speaker letters persist within a run. Provide known names separately when you know which letter belongs to whom.

## Run on Windows

1. Install 64-bit Python 3.12 with the Python launcher.
2. Run `setup_cpu.bat` once. It installs a local environment and downloads checksum-verified model files.
3. Open `demo_windows.bat`. It runs the included recording, checks both speaker files and the transcript, and opens the result.

To process a recording, use a fresh output-folder name.

```bat
process_meeting.bat "meeting.wav" --transcribe --out "results\meeting"
```

Open `listen.html` inside the output folder to play the mixture and individual voices. The same folder contains `speakers.json`, `transcript.txt`, and `transcript.jsonl`. Timestamps are relative to the input recording. The waveform separator works at 8 kHz. Audio export finishes before the optional text pass starts.

## Overlap and replay

Default mode uses the model's automatic count. With heavy overlap, try `--overlap-margin 1.4` to consider one additional voice when its decoder score is close. The additional voice must pass minimum signal and waveform similarity checks. The listening page and commit log flag windows where the decoder count changed. The checks measure plausibility and outputs can still contain mixed voices.

```bat
process_meeting.bat "overlap.wav" --overlap-margin 1.4 --transcribe --out "results\overlap"
process_meeting.bat "overlap.wav" --mode replay --write-chunks --out "results\replay"
```

Replay schedules input at the recording rate and logs bounded-window commit delay. A fixed count can be requested with `--speakers 4 --separator original` for an experiment where four voices are expected to overlap in each window. The count override applies per window. To name speakers, supply a local `--names names.json` mapping such as `{"spk_A":"张三","spk_B":"李四"}`.

## Verification

[The demo transcript](demo/transcript.txt) is actual output from the included human-speech mixture. See the [evaluation report](benchmarks/QUALITY.md) for count accuracy, character errors, waveform recovery, and limitations.

```sh
python -m unittest discover -s tests -v
python process_meeting.py demo/mixed.wav --transcribe --out results/demo_check
python verify_demo.py results/demo_check
```

For Linux, use Python 3.12 and install CPU PyTorch before the inference dependencies.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install --no-deps asteroid==0.7.0
python fetch_models.py
```

## Models and demo attribution

- The demo mixes two AISHELL-3 recordings at overlapping offsets, with gain matching and mono resampling. Exact source URLs and reference text appear in [the attribution file](demo/source_attribution.json). [AISHELL-3, Apache 2.0](https://huggingface.co/datasets/AISHELL/AISHELL-3)
- The waveform models use Asteroid's MultiDecoderDPRNN recipe. The inference source retains its [license](ASTEROID_UPSTREAM_LICENSE.txt). [Original model](https://huggingface.co/JunzheJosephZhu/MultiDecoderDPRNN) · [Fine-tuned checkpoint](https://huggingface.co/Namadgi/MultiDecoderDPRNN-finetuned)
- Mandarin decoding uses the sherpa-onnx Zipformer CTC export. Byte-level tokens are decoded to Unicode characters before writing timing fields. [Mandarin model documentation](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/icefall/zipformer.html)
