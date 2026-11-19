# Faster mono preprocessing with exact audio parity

The contribution mask uses the mixture signal for single-speaker regions. It computes spectral transforms only around overlapping regions, on the original transform grid and with enough guard samples for overlap-add reconstruction. It keeps the previous filter, gain, sample rate, waveform length and file format.

| Recorded mono meeting | Old mask median | Sparse mask median | Output |
|---|---:|---:|---|
| M8016, 30:39 | 13.14 s | 3.31 s | Three byte-identical WAVs |
| M8017, 30:20 | 8.13 s | 3.37 s | Three byte-identical WAVs |

Each arm ran three times, alternating order, with one CPU thread and the same renderer, waveforms and timing. The seconds include temporary PCM I/O for the mask stage. They exclude separation and recognition. The benchmark verifies hashes of all six produced speaker files. A synthetic regression test checks exact float equality for short audio, transform boundaries, dense overlap and sparse activity. [Full recordings and timings](mask_runtime.json)

Recognition now reads each bounded time span directly from the WAV file. A paired check on all 1,388 spans across the two complete meetings verified exact sample equality with the whole-file read. This avoids keeping the whole speaker recording in the recognizer's waveform buffer. Batch sizes 2 and 4 changed decoded output and ran slower on the six earlier overlap excerpts. The offline workflow uses batch size 1. [Batch measurements](asr_batch.json)

A caller can declare known speaker IDs in a supplied mono timing file. The file runner preserves the A/B/C slots even when an expected speaker is silent for the whole excerpt. A smoke test with only B active wrote 40,000 correctly aligned samples per track, exactly silent A and C waveforms, and empty text files for A and C.

The tests use the same command dependencies documented for the offline pipeline. To compare the transforms, put the benchmark directory and its `root_causes` directory on `PYTHONPATH`, then run `test_sparse.py`. The long-file timing fixture in `bench_masks.py` takes the original meeting and already rendered speaker tracks as used in the previous benchmark.
