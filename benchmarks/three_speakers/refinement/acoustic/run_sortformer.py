"""Run the independent ONNX probe with bounded memory and resumable streaming state."""

import argparse
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort

from sortformer_benchmark import audio_features


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("wave", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--work", required=True, type=Path)
    p.add_argument("--threads", default=1, type=int)
    a = p.parse_args()
    sys.path.insert(0, str(a.work / "sortformer-adapter/src"))
    from sortformer import Sortformer, CHUNK_LEN, SUBSAMPLING

    constructor = ort.SessionOptions

    def options():
        x = constructor()
        x.intra_op_num_threads = a.threads
        x.inter_op_num_threads = 1
        x.enable_cpu_mem_arena = False
        x.enable_mem_pattern = False
        x.add_session_config_entry("session.disable_prepacking", "1")
        return x

    ort.SessionOptions = options
    try:
        model = Sortformer(
            str(a.work / "sortformer.onnx"),
            provider="CPUExecutionProvider",
            log_severity=3,
        )
    finally:
        ort.SessionOptions = constructor
    feat = audio_features(np.load(a.wave, mmap_mode="r"), model.mel_basis)
    stride = CHUNK_LEN * SUBSAMPLING
    checkpoint = a.output.with_suffix(".state.npz")
    rows = []
    start = 0
    names = (
        "spkcache",
        "spkcache_preds",
        "fifo",
        "fifo_preds",
        "mean_sil_emb",
        "n_sil_frames",
    )
    if checkpoint.exists():
        with np.load(checkpoint) as restore:
            start = int(restore["start"])
            rows = [restore["probs"]]
            for n in names:
                if n in restore:
                    setattr(
                        model, n, int(restore[n]) if n == "n_sil_frames" else restore[n]
                    )
    for i in range(start, len(feat), stride):
        part = feat[i : i + stride]
        length = len(part)
        predicted = model._streaming_update(
            np.pad(part, ((0, stride - length), (0, 0)))[None], length
        )[0]
        rows.append(predicted[: (length + SUBSAMPLING - 1) // SUBSAMPLING])
        values = {n: getattr(model, n) for n in names if getattr(model, n) is not None}
        pending = checkpoint.with_suffix(".pending.npz")
        np.savez(pending, start=i + stride, probs=np.concatenate(rows), **values)
        pending.replace(checkpoint)
        if (i // stride) % 8 == 0:
            print(f"{i}/{len(feat)}", flush=True)
    pending = a.output.with_suffix(".pending.npy")
    np.save(pending, np.concatenate(rows))
    pending.replace(a.output)
    checkpoint.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
