"""Extract permutation-invariant DiariZen features from a single mono microphone."""

import argparse
import itertools
import json
from pathlib import Path
import time
import tomllib

import numpy as np
import soundfile as sf
import torch


def load_model(model_dir):
    from diarizen.models.eend.model_wavlm_conformer import Model

    config = tomllib.loads((model_dir / "config.toml").read_text())
    model = Model(**config["model"]["args"])
    model.load_state_dict(
        torch.load(
            model_dir / "pytorch_model.bin", map_location="cpu", weights_only=True
        )
    )
    return model.eval()


def pair_features(prob, num_slots=4, maximum=2):
    mapping = [
        group
        for size in range(maximum + 1)
        for group in itertools.combinations(range(num_slots), size)
    ]
    if len(mapping) != prob.shape[-1]:
        raise ValueError(f"Unexpected class count {prob.shape[-1]}")
    bits = np.array(
        [[slot in group for slot in range(num_slots)] for group in mapping],
        dtype=np.float32,
    )
    pairs = prob[..., bits.sum(axis=1) == 2]
    marg = np.sort(prob @ bits, axis=-1)
    h = -(prob * np.log(np.maximum(prob, 1e-10))).sum(axis=-1)
    ranked_pairs = np.sort(pairs, axis=-1)
    return np.column_stack(
        [prob[:, 0], pairs.sum(axis=-1), marg, ranked_pairs[:, -2:], h]
    ).astype(np.float32)


@torch.inference_mode()
def extract(model, file):
    wave, sr = sf.read(file, dtype="float32")
    if wave.ndim != 1 or sr != 16000 or not len(wave):
        raise ValueError(file)
    win = 256000
    n = int(np.ceil(len(wave) / 160))
    times = (np.arange(n) + 0.5) / 100
    output = np.zeros((n, 9), np.float32)
    total = np.zeros(n, np.float32)
    starts = list(range(0, max(1, len(wave) - win + 1), win // 2))
    if len(wave) > win and starts[-1] < len(wave) - win:
        starts.append(len(wave) - win)
    for start in starts:
        audio = wave[start : start + win]
        padded = np.pad(audio, (0, max(0, win - len(audio))))
        logits = model(torch.from_numpy(padded)[None, None])[0]
        probabilities = logits.softmax(dim=-1).numpy()
        vals = pair_features(probabilities)
        # WavLM convolutions have a 400-sample receptive field and 320-sample stride.
        local = times - start / sr
        valid = (local >= 0) & (local < len(audio) / sr)
        t = local[valid]
        indices = np.arange(len(vals)) * 0.020 + 0.0125
        weight = np.minimum(1.0, np.minimum(t + 0.1, len(audio) / sr - t + 0.1))
        for j in range(vals.shape[1]):
            output[valid, j] += weight * np.interp(t, indices, vals[:, j])
        total[valid] += weight
    return output / np.maximum(total[:, None], 1e-6)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, action="append", default=[])
    ap.add_argument("--audio", type=Path, action="append", default=[])
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model_dir)
    print("Output classes", model.classifier.out_features, flush=True)
    if not args.manifest and not args.audio:
        ap.error("Supply --audio or --manifest")
    paths = list(args.audio)
    for manifest in args.manifest:
        rows = json.loads(manifest.read_text())
        rows = rows if isinstance(rows, list) else rows["clips"]
        paths.extend(manifest.parent / row["audio"] for row in rows)
    for index, path in enumerate(paths):
        target = args.output / (path.stem + ".npy")
        if target.exists():
            continue
        start = time.perf_counter()
        features = extract(model, path)
        pending = target.with_suffix(".tmp")
        with pending.open("wb") as out:
            np.save(out, features)
        pending.replace(target)
        if index % 10 == 0:
            print(
                index,
                len(paths),
                path.stem,
                len(features),
                round(time.perf_counter() - start, 2),
                flush=True,
            )


if __name__ == "__main__":
    main()
