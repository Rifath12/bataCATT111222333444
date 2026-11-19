"""Compare SenseVoice batch sizes on identical input spans."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import recognize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--reverse", action="store_true")
    a = p.parse_args()
    root = a.manifest.parent
    inventory = json.loads(a.manifest.read_text())
    results = {}
    for row in inventory:
        key = Path(row["audio"]).stem
        case = root / key
        timing = json.loads((case / "timing.json").read_text())["segments"]
        paths = {s: case / "mask" / f"speaker_{s}.wav" for s in "ABC"}
        outputs = {}
        for batch in sorted(a.batches, reverse=a.reverse):
            start = perf_counter()
            items, wall = recognize.transcribe(
                paths,
                timing,
                a.model / "model.int8.onnx",
                a.model / "tokens.txt",
                threads=a.threads,
                batch_size=batch,
                target_rms=0.035,
                backend="sensevoice",
            )
            outputs[batch] = dict(
                items=items,
                seconds=round(perf_counter() - start, 4),
                inference_seconds=wall,
            )
            print(key, batch, len(items), outputs[batch]["seconds"], flush=True)
        for batch, item in outputs.items():
            if 1 in outputs:
                item["identical_to_batch_1"] = item["items"] == outputs[1]["items"]
            if batch != 1:
                del item["items"]
        results[key] = outputs
        a.out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
