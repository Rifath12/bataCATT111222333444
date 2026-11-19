# Mono three-speaker timing experiment

The tested candidate learns the exact three-speaker count as a binary problem. It combines the existing paired Nemotron views and the separate DiariZen view with pitch salience and short-term support from the dictionary timing. A calibrated filter overrides three-speaker decisions in the structured decoder. It leaves other speaker decisions intact. When removing a triple decision it keeps the two slots with strongest combined Nemotron evidence.

Calibration holds out complete source sessions across the six older AliMeeting recordings that supplied 164 development excerpts. Each fold retrains both the prior structured comparator and the candidate classifier before selecting the candidate thresholds. The criterion maximizes the worst change across exact-three precision, recall and F1 and overall diarization error, measured against both the prior dictionary and the structured approach. A negative worst change means the candidates failed the requested all-metric improvement on calibration. The frozen parameters are ordinary JSON, and no reference transcript or identity appears in its inference input.

For scoring, provide 16 kHz mono WAVs, matching two-pass Nemotron probabilities and aligned DiariZen features. The manifest lists `audio`, `ref`, and `source_session` per clip. Audio and reference paths are relative to the manifest. The reference contains three annotated identities. The evaluator matches anonymous labels to reference speakers once per clip and uses 10 ms frames.

```sh
pareto_timing/benchmark.sh \
  --holdout MANIFEST.json NEMOTRON_CACHE_DIR \
  --dz DIARIZEN_FEATURE_DIR \
  --frozen pareto_timing/frozen.json \
  --out RESULTS.json
```

For inference without scoring, use `--predict AUDIO.wav NEMOTRON_CACHE.npz TIMING.json` in place of `--holdout`. This is an offline experiment. It does not measure reconstruction quality or words attributed to a speaker.

## Frozen evaluation panels

- AliMeeting challenge replay has the same 18 short excerpts used in the prior timing study. The benchmark and overall results were known before this experiment. No clip entered calibration for either candidate.
- AISHELL-4 transfer has 12 newly cropped excerpts from three source sessions and 18.39 seconds of annotated exact three-speaker activity. These recordings were outside calibration. Scope is new pipeline evaluation, since the upstream pretraining composition cannot be verified.
- The Sortformer pilot selected five chronology-spaced excerpts per older source session, before reading any Sortformer score. It adds soft speaker-count cues from the pinned four-slot ONNX model. Calibration uses the same source-grouped fold protocol on this smaller set. Its metric gate includes overall speaker precision and recall, and speaker recall on exact-three frames.

On the AliMeeting replay, neither prior timing system recognized all three speakers for 4.40 of the 26.25 seconds when three spoke. An oracle restricted to the union of their triple predictions could recover at most 83.24% of those frames. The overlap breakdown is in `diagnostic.json`. It uses reference labels solely for analysis. The oracle is never available for inference.

## Outcomes

Values in the table are percentages computed from pooled frame counts. The higher value is better for exact-three precision, recall and F1. Lower diarization error is better.

| Panel | Arm | Exact-three precision | Exact-three recall | Exact-three F1 | Diarization error |
|---|---|---:|---:|---:|---:|
| AliMeeting replay | Dictionary + narrow | 52.07 | 81.14 | 63.43 | 16.59 |
| AliMeeting replay | Frozen structured | 72.77 | 72.27 | 72.52 | 11.87 |
| AliMeeting replay | Binary evidence fusion | 65.78 | 78.36 | 71.52 | 12.26 |
| New AISHELL-4 | Dictionary + narrow | 35.87 | 38.23 | 37.01 | 24.20 |
| New AISHELL-4 | Frozen structured | 55.25 | 30.34 | 39.17 | 19.06 |
| New AISHELL-4 | Binary evidence fusion | 49.06 | 35.40 | 41.12 | 19.29 |
| New AISHELL-4 | Sortformer feature pilot | 44.86 | 34.20 | 38.82 | 19.40 |

The all-metric gate in `metric_matrix.json` requires no regression on any of the seven timing metrics and a strict improvement in at least one. The two experimental candidates failed that gate against both released comparators. Sortformer features are an additional model pass with greater runtime and memory. The extraction script uses the repository's pinned v2.1 ONNX model and adapter. No inference defaults were updated by these experiments. The JSON prediction smoke test contains nine valid anonymous intervals for a 22-second recording.
