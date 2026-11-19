# Recovering missing Mandarin text from existing recognition passes

The optional `--refine-context` switch uses the cropped speaker-track and continuous mixed-channel SenseVoice hypotheses already computed by `recognize.py`. In addition to either complete hypothesis, the selector can retain the chosen reading and restore up to five characters at each end from the other reading if they share a character anchor. It uses duration, speaker overlap, text length and rate, and character agreement. It scores only competing readings of the same predicted turn. Speaker IDs and turn boundaries stay aligned with the separated audio.

The switch enables the prior contextual restoration, long-turn protection and quiet-copy checks first. Candidate changes use the same quiet-copy rule. The selector uses the existing model weights and requires no extra model pass.

```sh
python benchmarks/three_speakers/recognize.py \
  --audio /path/to/masked-speaker-tracks --timing /path/to/timing.json \
  --context-mix /path/to/mono.wav --asr /path/to/sensevoice \
  --backend sensevoice --target-rms 0.035 --threads 2 --refine-context \
  --out /path/to/refined-transcripts
```

## Fixed-recording evaluation

| Recording | Reference characters | Published errors | With refinement |
| --- | ---: | ---: | ---: |
| M8013 development | 8,943 | 2,509 | 2,448 |
| M8016 | 7,705 | 1,927 | 1,838 |
| M8017 | 8,154 | 988 | 953 |
| Two complete evaluation meetings | 15,859 | 2,915 | 2,791 |
| Three additional guarded excerpts | 1,128 | 355 | 343 |

Speaker-attributed CER pools Chinese character edits after one optimal speaker mapping per recording. The selector reduces full-meeting CER from 18.38% to 17.60%. On the additional excerpts, two improve by seven edits and one worsens by two. Speaker C improves by 64 edits on M8016 and 23 on M8017. The tests measure recognition only. They reuse the same waveforms and speaker timing.

For each proposed replacement on M8013, calibration estimates its effect by substituting it into that speaker's full character stream and recomputing character edits against the same matched reference speaker. Five contiguous temporal blocks selected the tree depth, minimum leaf size and minimum predicted gain. Approximate out-of-fold edit gains were 6, 25, 6, 7 and 24. The final tree was fitted to all five development blocks and fixed before scoring the two complete evaluation meetings and three excerpts. Calibration used M8013's original mono speaker tracks. Evaluation used the prior published contribution-masked tracks.

Reference-free cached assembly first reproduced the previous complete-meeting transcripts, then evaluated the selector on the same spans and emitted chronological JSONL and speaker A/B/C text. A real 42.85-second waveform decoded with the optional CLI produced the same 34 ordered rows and four text files as cached assembly. The complete waveform recognition took 23.39 seconds with one shared CPU thread.
