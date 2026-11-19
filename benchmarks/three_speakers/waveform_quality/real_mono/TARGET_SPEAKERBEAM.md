# Audio-enrollment-conditioned target speech extraction

The ESPnet TD-SpeakerBeam model ran three times per single-microphone overlap, each time with earlier isolated speech from a different participant. The target audio affects the separation mask inside the neural network. The resulting tracks retain the order of the A, B and C enrollment cues.

| Enrolled waveform measure | Development incumbent | Development TD-SpeakerBeam | Confirmation incumbent | Confirmation TD-SpeakerBeam |
|---|---:|---:|---:|---:|
| Median alignment, all 24 individual tracks | 61.24% | 17.94% | 15.64% | 22.66% |
| Median alignment, eight quietest speakers | 29.73% | 0.42% | 9.58% | 5.33% |
| All three meet 96% alignment and retention | 0/8 | 0/8 | 0/8 | 0/8 |
| All three meet 98% alignment and retention | 0/8 | 0/8 | 0/8 | 0/8 |

The development rule retained the incumbent. Quiet-speaker extraction was particularly poor. A second approach used the conditioned outputs to decide whether to relabel the incumbent. Its fixed agreement gates rejected every proposed reassignment in both rooms. A fixed blend also lost quiet-speaker fidelity on development. The pinned [ESPnet model](https://huggingface.co/espnet/Wangyou_Zhang_librimix_train_enh_tse_td_speakerbeam_raw) uses two-person LibriMix training data and runs at 8 kHz. Its published two-person results have a different speaker count, language and recording condition.

## Reproduce the conditioned signals

Use the first eight fixed three-way cases, the same enrollment WAVs and known-source scorer in [the far-microphone protocol](ARCHITECTURES.md). Retain their physical A/B/C order when scoring. The ready-to-run command below processes a single known-three-voice WAV and writes the three conditioned outputs. The model runs offline after downloading weights and configuration. Use the benchmark's source audio and enrollment tracks at 16 kHz. The output files also use 16 kHz.

Install PyTorch, NumPy, SciPy, SoundFile, PyYAML, torch-complex and the ESPnet 202402 package. The verified CPU run used PyTorch 2.6.0 and the unmodified ESPnet inference modules. Fetch the two pinned files below. The script verifies both SHA-256 digests before loading them.

```sh
repo=https://huggingface.co/espnet/Wangyou_Zhang_librimix_train_enh_tse_td_speakerbeam_raw
rev=0c7e30518719f3208c81042a869fea6b886a9871
curl -fL "$repo/resolve/$rev/exp/enh_train_raw/99epoch.pth" -o 99epoch.pth
curl -fL "$repo/resolve/$rev/exp/enh_train_raw/config.yaml" -o config.yaml
python target_speakerbeam.py overlap.wav --enroll enroll_A.wav enroll_B.wav enroll_C.wav \
    --config config.yaml --checkpoint 99epoch.pth --out speakerbeam_output
```

Each enrollment is resampled to 8 kHz, divided into 32 ms blocks and filtered by RMS. The cutoff is the larger of −45 dBFS and 2% of the peak block RMS. The model uses up to six seconds of the retained blocks. The measured implementation cached the enrollment embeddings. It compared the cached output to the official two-input ESPnet extractor on the first fixture of each room with absolute tolerance 0.000002 and relative tolerance 0.00001. The study file records the implementation hashes, options, acceptance metrics, all paired single-track acoustic measurements and chronological case IDs. [Case evidence](target_speakerbeam_results.json)
