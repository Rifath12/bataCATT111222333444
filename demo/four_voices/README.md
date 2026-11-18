# Blind four-voice example

This short controlled recording mixes four different Mandarin speakers. The tool inferred three tracks by default. The overlap option recovered four tracks from the same single-channel input without requiring names, voice samples, or a count hint. The source recording IDs and independently provided transcripts appear in [the input manifest](../../benchmarks/cases.json) under `evaluation_G3_4voices`.

- [Mixed audio](mixed.wav)
- [Speaker A](Speaker_A.wav)
- [Speaker B](Speaker_B.wav)
- [Speaker C](Speaker_C.wav)
- [Speaker D](Speaker_D.wav)
- [Mandarin transcript](transcript.txt) · [Character timestamps](transcript.jsonl)

The waveforms share the exact recording clock. This example includes **0.64 s** where all four references are active. Speaker-attributed character error moved from **86.2%** to **82.8%**. The extra track is useful for diagnosis. Crosstalk and incomplete phrases remain. [Full evaluation](../../benchmarks/QUALITY.md)
