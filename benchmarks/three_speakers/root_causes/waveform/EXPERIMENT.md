# Three simultaneous Mandarin voices

Each test input uses three independently recorded, complete Mandarin utterances. Start offsets are 0, 120 and 240 ms. The active source levels differ by 0, 4 and 8 dB and rotate across cases. The simulated room contains source-specific early reflections and a decaying reverberation tail. The noisy condition adds white noise at 15 dB SNR. Predictions are made from the simulated mixture. Isolated target utterances and their transcripts are reserved for scoring. The speaker-cue condition additionally supplies one different, separately recorded utterance from each speaker. Practical use of that condition requires reliable solo samples.

Three development mixtures, three repeated short test mixtures and three new longer test mixtures contain 27 different speakers. Test selections were locked in the manifest before inference. On development speech the score sums character errors from SenseVoice and Zipformer over all three speakers. The lower-error repair advances to both test sets. An exact tie is decided by higher median three-way SI-SDR improvement. Both test sets also run a baseline separator and the prespecified acoustic controls.

The two-pass comparison aligns the baseline and dereverberated outputs by absolute correlation and polarity, normalizes their levels, and averages them equally when their correlation is at least 0.25. This fixed rule was added after the single-pass selection and before held-out word scoring. It requires two separator evaluations.

The first control recognizes dry, isolated recordings at 16 kHz. The second runs the same audio after 8 kHz resampling and the same recognition normalization. The third recognizes each 8 kHz isolated signal after room simulation. Other controls pass the dry mixture, the room mixture without noise, or the room mixture with noise through the identical Libri3Mix SepFormer checkpoint. The earlier short-phrase baseline is the noisy room mixture. These paired conditions measure where errors enter the pipeline. The controlled corpus consists of read speech and a fixed room simulation.

For each mixture, one optimal assignment of entire estimated tracks to source speakers is used for the transcript score. Character errors count substitutions, deletions and insertions. An exact phrase has zero character edits. Waveform scores use a separate whole-clip waveform assignment. On simulated room cases the primary signal target is the reverberant source; dry-source scores are saved separately. Simultaneous regions for the signal metric are determined by each source's energy in 20 ms frames. The transcript score covers the complete utterance. Neither score directly determines speaker identities across files.

## Reproduction

Use the dependency versions for the three-speaker benchmarks and the model directories already used by the published complete-word benchmark. From the repository root, add the directories containing benchmark_examples.py, benchmark_waveform.py, controlled_words.py, recognize.py, sepformer_three.py and refine_three.py to PYTHONPATH. Use an empty output directory when changing any code, membership, or weights. Independent attention batch groups are evaluated four at a time to bound CPU memory. Each neural model runs in a separate process with one thread.

```bash
python fetch_wave_inputs.py --data "$CASE_DIR"
for phase in development test; do
  for stage in sep sensevoice zipformer score; do
    python test_wave.py --data "$CASE_DIR" --weights "$SEPFORMER" \
      --sense "$SENSEVOICE" --zipformer "$ZIPFORMER" --out "$OUTPUT_DIR" \
      --phase "$phase" --stage "$stage"
  done
done
```

For the two-pass comparison, run test_blend.py with the same data, output, SenseVoice and Zipformer arguments. Use --phase development repeat new and each of --stage sensevoice, --stage zipformer and --stage score in that order. For the real-excerpt check, check_real_excerpt.py consumes the original mono audio, inferred timing, cached standard SepFormer tracks, voice embedding and separation weights. It re-renders the affected windows with the original model and checks that they match the published audio within 0.0002 absolute amplitude before scoring the modified tracks with the identical recognizer.

The data directory contains the checked-in manifest and the downloaded source audio. Each source is verified against the SHA-256 value in the manifest. The output includes individual hypotheses, per-source edit counts, signal scores, selection and aggregated results. Timings in these checks reflect a shared CPU under variable load. They do not estimate streaming latency.

The source recordings are from AISHELL-3, published by Beijing Shell Shell Technology Co., Ltd. The manifest pins the dataset revision and contains the source filenames and reference transcripts. [AISHELL-3 dataset](https://www.openslr.org/93/) · [Pinned data release](https://huggingface.co/datasets/AISHELL/AISHELL-3/tree/f20d5db4a31fe779ef07bb1af4ea92da5c786622) · [SpeechBrain model](https://huggingface.co/speechbrain/sepformer-libri3mix) · [WPE research](https://groups.uni-paderborn.de/nt/pubs/2018/ITG_2018_Drude_Paper.pdf)
