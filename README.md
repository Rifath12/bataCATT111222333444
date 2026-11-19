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

## One meeting with three Mandarin speakers

For one mixed recording containing three people, use a session roster. It writes aligned `Speaker_A.wav`, `Speaker_B.wav` and `Speaker_C.wav` on the meeting clock, including silence where a slot has no detected activity.

```bat
process_meeting.bat "meeting.wav" --participants 3 --out "results\three_people"
```

For a complete audio-only run with the tested waveform correction and Mandarin transcript, this command accepts a single recording and assumes three participants. It uses anonymous speaker letters.

```bat
process_meeting.bat "meeting.wav" --participants 3 --mixture-phase --transcribe --out "results\three_people_corrected"
```

To keep known identities, supply one independently recorded solo WAV for each speaker. Mono and stereo WAVs at ordinary recording rates, including 16, 44.1 and 48 kHz, are converted automatically. A few representative utterances from the same room and microphone per person are preferable. The enrollment method uses up to 32 audible 0.512 s windows per person. Give the files in A/B/C order. Natural interruptions and substantial overlap can still produce uncertain assignments.

```bat
process_meeting.bat "meeting.wav" --participants 3 --enroll "solo_A.wav" "solo_B.wav" "solo_C.wav" --out "results\enrolled"
```

The default roster lets the separator estimate how many voices are active in each window. On audio with frequent heavy overlap, adding `--speakers 3` makes the decoder estimate three components in each non-silent window. This can recover weak overlapping speech and can introduce duplicate or spurious components on solo turns. Both commands accept `--names names.json` for display names and `--transcribe` for optional Chinese text. On Linux, invoke `python process_meeting.py` with the same arguments.

`speakers.json` gives each track's file and detected activity status. `commits.jsonl` reports the number of decoded sources, the estimated active labels and timestamps as samples at the output sample rate. With enrollment it also contains finer A/B/C intervals and ambiguous assignment flags. These labels are estimates. Without labeled solo WAVs, A/B/C are anonymous recording-local slots; identity across a long pause can be uncertain.

For an optional offline waveform pass, add `--mixture-phase` to the file command. The pass divides each recorded time–frequency bin among the speaker tracks in proportion to their estimated magnitudes. It improved waveform alignment in two known-source room simulations. On natural recordings it also made sound from other speakers more audible on some tracks. Compare the WAVs when bleed matters. In this mode, optional transcription reads the restored audio. The activity log and any `--write-chunks` previews refer to separation before restoration.

To restore a completed export without rerunning the decoder, run the following command with the project's Python environment. It writes a fresh set of speaker WAVs, a listening page, the commit log, and updated metadata. Transcripts from the previous audio are omitted.

```shell
python mixture_phase.py results/three_people --out results/three_people_restored
```

Each transcribed segment in `transcript.jsonl` includes how much of its audio was assigned to that speaker, how much came directly from a window classified as one voice, and how much had an ambiguous identity. The review flag covers both timing and separation warnings. A quiet background talker can still go undetected.

[Single-microphone Mandarin results, including overlap and meeting runtime](benchmarks/three_speakers/waveform_quality/real_mono/RESULTS.md)

For offline processing on a multicore CPU, `--file-workers 2 --threads 1` runs two acoustic windows concurrently with shared model weights. The app assigns speaker identities and commits samples in recording order. One worker is the default. Each additional worker uses working memory. Recorded voice overlap, file length and CPU load affect elapsed time.

```bat
process_meeting.bat "meeting.wav" --participants 3 --file-workers 2 --threads 1 --out "results\meeting_parallel"
```

## Overlap and replay

Default mode uses the model's automatic count. With heavy overlap, try `--overlap-margin 1.4` to consider one additional voice when its decoder score is close. The additional voice must pass minimum signal and waveform similarity checks. The listening page and commit log flag windows where the decoder count changed. The checks measure plausibility and outputs can still contain mixed voices.

```bat
process_meeting.bat "overlap.wav" --overlap-margin 1.4 --transcribe --out "results\overlap"
process_meeting.bat "overlap.wav" --mode replay --write-chunks --out "results\replay"
```

Replay schedules input at the recording rate and logs bounded-window commit delay. A fixed count can be requested with `--speakers 4 --separator original` for an experiment where four voices are expected to overlap in each window. The count override applies per window. To name speakers, supply a local `--names names.json` mapping such as `{"spk_A":"张三","spk_B":"李四"}`.

## Verification

[The demo transcript](demo/transcript.txt) is actual output from the included human-speech mixture. [Hear the four-voice example](demo/four_voices/README.md) and see the [evaluation report](benchmarks/QUALITY.md) for count accuracy, character errors, waveform recovery, and the Windows and Linux checks.

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
