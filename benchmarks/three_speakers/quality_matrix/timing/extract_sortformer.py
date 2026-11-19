"""Cache independent Sortformer streaming posteriors for mono excerpts."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf

from sortformer_benchmark import audio_features


def stream(wave, model, chunk_frames, subsampling):
    """Use the pinned adapter's standard state update with one fresh recording."""
    model.reset_state()
    feature = audio_features(wave, model.mel_basis)
    stride = chunk_frames * subsampling
    batches = []
    for start in range(0, len(feature), stride):
        chunk = feature[start : start + stride]
        length = len(chunk)
        padded = np.pad(chunk, ((0, stride - length), (0, 0)))
        predicted = model._streaming_update(padded[None], length)[0]
        batches.append(predicted[: (length + subsampling - 1) // subsampling])
    return np.concatenate(batches)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--manifest", action="append", type=Path, default=[])
    ap.add_argument("--audio", action="append", type=Path, default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    if not a.manifest and not a.audio:
        ap.error("Specify --manifest or --audio.")
    sys.path.insert(0, str(a.adapter.resolve() / "src"))
    from sortformer import CHUNK_LEN, SUBSAMPLING, Sortformer

    constructor = ort.SessionOptions

    def options():
        selected = constructor()
        selected.intra_op_num_threads = a.threads
        selected.inter_op_num_threads = 1
        selected.enable_cpu_mem_arena = False
        selected.enable_mem_pattern = False
        selected.add_session_config_entry("session.disable_prepacking", "1")
        return selected

    ort.SessionOptions = options
    try:
        model = Sortformer(
            str(a.model.resolve()), provider="CPUExecutionProvider", log_severity=3
        )
    finally:
        ort.SessionOptions = constructor
    a.out.mkdir(parents=True, exist_ok=True)
    sources = a.audio.copy()
    for manifest in a.manifest:
        rows = json.loads(manifest.read_text())
        rows = rows if isinstance(rows, list) else rows["clips"]
        sources += [manifest.parent / row["audio"] for row in rows]
    started = time.perf_counter()
    for index, path in enumerate(sources, 1):
        output = a.out / f"{path.stem}.npy"
        if output.exists():
            continue
        wave, rate = sf.read(path, dtype="float32")
        if rate != 16000 or wave.ndim != 1 or not len(wave):
            raise ValueError(f"Expected nonempty 16 kHz mono audio: {path}")
        posterior = stream(wave, model, CHUNK_LEN, SUBSAMPLING)
        temp = output.with_suffix(".pending.npy")
        np.save(temp, posterior)
        temp.replace(output)
        if index == 1 or index % 8 == 0 or index == len(sources):
            print(
                index,
                len(sources),
                path.stem,
                len(wave) / rate,
                round(time.perf_counter() - started, 2),
                flush=True,
            )


if __name__ == "__main__":
    main()
