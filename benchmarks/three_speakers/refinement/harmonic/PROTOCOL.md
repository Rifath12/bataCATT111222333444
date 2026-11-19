# Three-speaker harmonic verification

The verifier reads a 16 kHz mono mixture and the two Nemotron posterior streams from the preceding benchmark. It searches a short-time spectrum for three different fundamental frequencies, successively subtracting each selected harmonic template’s contribution. Each anonymous speaker receives a pitch distribution estimated from confident single-speaker frames. A gradient-boosted classifier uses the harmonic scores, distribution matches, and temporal posterior statistics to estimate whether three voices are present. An ablation removes the pitch features while keeping training, data, and architecture identical.

The count decision retains the previous calibrated decoder’s labels outside proposed three-way activity. When the verifier rejects a previous third-voice detection, the lowest-confidence voice drops out. A separate guarded operating point only adds a voice to a previous two-speaker frame. Development selected its keep and add thresholds to maximize session-group F1 gain, subject to precision at least as high as the previous decoder on both development domains and no more than 3.5 percentage points of all-three recall loss on either domain.

Training uses 45 REAL-T development clips and the earlier 37-minute AliMeeting development recording R8008_M8013. The three-fold REAL-T split groups by session. Guard calibration also predicts the entire AliMeeting development recording using a verifier trained only on REAL-T. All selection precedes fresh-panel scoring. The additional guarded rule was proposed after examining the initial arm on the repeated benchmark. Only the fresh panel independently evaluates that design change. Anonymous speaker matching is once per whole meeting or excerpt. This measures activity on three preexisting slots and has no isolated-reference waveform score.

The command below assumes the intermediate posterior files and corpus layouts from the previous two benchmark rounds. Set the Python search path to the previous benchmark implementation and three-speaker meeting utilities. For a fresh panel with the same schema add the manifest and paired probability directory.

```bash
python harmonic_verify.py \
  --profile tri90/publish/previous_profile.json --realt tri90_natural/realt_e2_cn_h2 \
  --cache tri90 --ali tri90_ali_clips --previous three_speaker_work \
  --fresh-manifest trial_next_fresh/clips_manifest.json \
  --fresh-probs trial_next_fresh/probabilities --out pitch_results.json
```
