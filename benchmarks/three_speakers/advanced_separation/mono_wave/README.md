# Guarded three-speaker Mandarin waveform separator

This optional quality pass separates one mono overlap window with a pretrained RecorrNet checkpoint trained on AISHELL-1 mixtures and part of AliMeeting Train. It retains the existing tracks if the candidate cannot reconstruct at least 90% of the centered mixture's energy using positive gains. When accepted, those gains also restore levels compatible with the original mix. Successful probes needed a gain of about 0.135, since uncalibrated RecorrNet output was about 7.4 times as loud.

## Run

Use Python 3.12 or newer. Within this folder, install the dependencies and download the SHA-256 pinned code and weights.

```bash
python -m pip install -r requirements.txt
python get_models.py --model recorrnet
```

Instantiate the model once and reuse it. This interface expects finite mono PCM. Use the original sample rate as the integer second argument. The returned tracks are exactly the input length and sample rate. RecorrNet processes at 8 kHz internally. Use `speakers=None` to request the checkpoint's own speaker count, or `speakers=3` in a known three-speaker region. The bounded path keeps the original context across a clip while splitting independent internal batches. CPU timings below were measured using one PyTorch compute thread.

```python
from separate import Separator

separator = Separator("recorrnet")
unordered_tracks = separator(mono_pcm, sample_rate=16000, speakers=3)
```

On CPU builds of PyTorch with fused scaled dot-product attention, `Separator("recorrnet", flash_attention=True)` can reduce runtime. One paired 2.8-second probe ran in 40.53 seconds rather than 49.91 seconds, with a relative waveform difference of 1.43 × 10⁻⁶. This option was verified on PyTorch 2.7.1+cpu. Other builds can fall back to standard attention.

Use external diarization or voice embeddings to map the unordered outputs to stable speaker names before applying the gate. Align the fallback stems and candidate on the same PCM timeline. The gate then operates only on the mixture and candidate waveforms.

```python
from gate import select_tracks

tracks, source, fit = select_tracks(mono_pcm, existing_tracks, aligned_candidate)
```

The model and source revisions, exact hashes and native sample rates are in `provenance.json`. The original model and upstream code are MIT licensed. The evidence audio derives from AISHELL-3, released under Apache-2.0. Source identifiers, timestamps and checksums are provided in `../evidence/controlled_mono_holdout.json`.

## Verified controlled comparison

| Measure | Released spectral SepFormer | Guarded RecorrNet |
| --- | ---: | ---: |
| Median SI-SDR improvement over the original mix, whole window | −1.79 dB | +12.80 dB |
| Median SI-SDR improvement, exactly three active speakers | −1.19 dB | +13.00 dB |
| Median quietest speaker improvement, exactly three active speakers | +3.61 dB | +14.45 dB |
| Median worst speaker improvement, exactly three active speakers | −2.93 dB | +11.96 dB |

Three simulated room windows, each 2.8 seconds, use nine distinct Mandarin voices absent from the set of historical benchmark speaker IDs audited. There are 5.48 seconds of genuine three-way activity in the selected windows. Each model sees the exact same 8 kHz mixture. Tracks are matched to the known isolated references once per window for scoring and the listening samples. The actual selection gate uses only the mixture and candidate waveforms and is invariant to track order. CPU timing was 9.17–11.88 seconds for the released model and 44.53–46.29 seconds for the unaccelerated bounded checkpoint, excluding model load. Each speaker trio contributes one independent test comparison. The three-way activity scores improved in two clips and tied in one. The small panel gives no reliable population confidence interval. Real conversation, natural meeting transcription, 30-minute attribution and real-time performance require independent measurement.

Each listening folder contains the mix, the three isolated references, the released A/B/C outputs and the guarded A/B/C outputs. The first case falls back to the released stems. The other two pass the same energy test. Raw ungated CorrNet was severely degraded in the first case (−18.82 dB during exact-three overlap), an example of why the fallback matters. The gate checks shared acoustics. Stable attribution, words and separation from the same person appearing in multiple output stems remain separate tests.
