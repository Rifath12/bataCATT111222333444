"""Compare unconditional separators and speaker-conditioned extraction on a common input."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.signal import resample_poly
import sherpa_onnx as sx
import torch

from anchored_extract import observe, call_extractor, choose_track
from benchmark_examples import waveform_scores
from controlled_words import recognize
from recognize import text_scores
from refine_three import SpectralRefiner
from target_extract import TargetExtractor
from test_wave import create_sources, low_memory_separator
from acoustic_repair import dereverberate


def transcriptions(asr, truth, outputs):
    hyp = recognize(asr, outputs)
    scores = text_scores(truth, hyp)
    return dict(
        edits=scores["totals"]["errors"],
        characters=scores["totals"]["reference_chars"],
        scores=scores["speaker_scores"],
        phrases_exact=sum(
            s["deletions"] + s["insertions"] + s["substitutions"] == 0
            for s in scores["speaker_scores"]
        ),
        hypotheses=hyp,
    )


def enroll(cues):
    out = []
    n = 2 * 24000
    for cue in cues:
        signal = resample_poly(cue, 2, 1).astype("float32")[:n]
        out.append(np.pad(signal, (0, n - len(signal))))
    return out


def stats(data, backends):
    return {
        arm: dict(
            **{
                f"{backend}_{key}": sum(
                    row["arms"][arm][backend][value] for row in data
                )
                for backend in backends
                for key, value in (("edits", "edits"), ("exact", "phrases_exact"))
            },
            chars=sum(row["arms"][arm][backends[0]]["characters"] for row in data),
            triple_s=sum(row["triple_s"] for row in data),
            trio_s=sum(row["seconds"] for row in data),
            median_all_triple=float(
                np.median(
                    [
                        s["exact_overlap_improvement_dB"]
                        for row in data
                        for s in row["arms"][arm]["waveform"]
                    ]
                )
            ),
            median_weak_triple=float(
                np.median(
                    [
                        row["arms"][arm]["waveform"][row["quiet"]][
                            "exact_overlap_improvement_dB"
                        ]
                        for row in data
                    ]
                )
            ),
        )
        for arm in data[0]["arms"]
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in (
        "manifest",
        "sources",
        "toolkit",
        "speaker",
        "weights",
        "sepformer",
        "sense",
        "zip",
        "out",
    ):
        p.add_argument("--" + field, type=Path, required=True)
    p.add_argument("--group", required=True)
    p.add_argument("--voiceprint", type=Path)
    p.add_argument("--backend", choices=("sense", "zip", "both"), default="both")
    p.add_argument("--limit", type=int)
    p.add_argument("--tag", default="")
    p.add_argument("--prior-cache", type=Path)
    p.add_argument("--whole-extraction", action="store_true")
    p.add_argument("--part", choices=("tse", "sep", "score"), required=True)
    p.add_argument(
        "--variant",
        choices=("direct", "residual", "average", "best_sep", "routed"),
        default=None,
    )
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261119)
    manifest = json.loads(a.manifest.read_text())
    rows = manifest["groups"][a.group][: a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    if a.part == "tse":
        target = TargetExtractor(a.weights, a.toolkit, a.speaker)
    if a.part == "sep" and a.prior_cache is None:
        separator = low_memory_separator(a.sepformer)
    if a.part == "score":
        models = {}
        if a.backend in ("sense", "both"):
            models["sense"] = sx.OfflineRecognizer.from_sense_voice(
                model=str(a.sense / "model.int8.onnx"),
                tokens=str(a.sense / "tokens.txt"),
                language="zh",
                use_itn=False,
                num_threads=1,
            )
        if a.backend in ("zip", "both"):
            models["zip"] = sx.OfflineRecognizer.from_zipformer_ctc(
                model=str(a.zip / "model.int8.onnx"),
                tokens=str(a.zip / "tokens.txt"),
                num_threads=1,
            )
        refine = SpectralRefiner()
        if a.voiceprint:
            from process_meeting import VoiceEmbeddings

            embed = VoiceEmbeddings(a.voiceprint)
    cases = []
    for i, trio in enumerate(rows, 1):
        for entry in trio:
            for kind in ("target", "enroll"):
                path = a.sources / entry[kind]
                if (
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    != manifest["sha256"][path.name]
                ):
                    raise ValueError(f"Input failed manifest checksum {path}")
        _, wet, mix, cues, triple = create_sources(a.sources, trio, i)
        controlpath = a.out / f"{a.group}_{i}_sep.npz"
        targetpath = a.out / f"{a.group}_{i}_tse.npz"
        if a.part == "sep":
            if controlpath.is_file():
                continue
            if a.prior_cache is None:
                normal = separator(mix)
                wpe = separator(dereverberate(mix))
            else:
                normal = np.load(a.prior_cache / f"{a.group}_{i}_room_baseline.npz")[
                    "tracks"
                ]
                wpe = np.load(a.prior_cache / f"{a.group}_{i}_wpe.npz")["tracks"]
            np.savez(controlpath, baseline=normal, wpe=wpe)
            print("sep", a.group, i, flush=True)
        if a.part == "tse":
            if targetpath.is_file():
                continue
            anchors = target.anchors(enroll(cues))
            started = perf_counter()
            if a.variant == "best_sep":
                continue
            if a.variant in ("direct", "routed"):
                direct = call_extractor(target, mix, anchors, whole=a.whole_extraction)
                residual = direct
            else:
                direct, residual = observe(
                    target, mix, anchors, 0.85, whole=a.whole_extraction
                )
            np.savez(
                targetpath,
                direct=direct,
                residual=residual,
                actual_second_pass=np.array(a.variant not in ("direct", "routed")),
                wall_s=perf_counter() - started,
            )
            print("tse", a.group, i, round(perf_counter() - started, 2), flush=True)
        if a.part == "score":
            controls = dict(np.load(controlpath))
            normal = controls["baseline"]
            arms = {**controls, "best_sep": refine(mix, normal)}
            if targetpath.is_file():
                results = dict(np.load(targetpath))
                arms["direct"] = results["direct"]
                if bool(results.get("actual_second_pass", True)):
                    arms["residual"] = results["residual"]
                    arms["average"] = 0.5 * (arms["direct"] + arms["residual"])
                if a.voiceprint:
                    arms["routed"] = choose_track(
                        arms["best_sep"],
                        arms["direct"],
                        [embed(cue) for cue in cues],
                        embed,
                    )
            if a.variant:
                arms = {
                    name: waves
                    for name, waves in arms.items()
                    if name in ("baseline", "wpe", "best_sep", a.variant)
                }
            record = dict(
                trio=trio,
                quiet=int(np.argmin(np.linalg.norm(wet, axis=-1))),
                seconds=len(mix) / 8000,
                triple_s=triple,
                arms={},
            )
            truth = [
                dict(speaker=str(j), text=voice["text"], start=0.0)
                for j, voice in enumerate(trio)
            ]
            for name, waves in arms.items():
                record["arms"][name] = dict(
                    **{
                        backend: transcriptions(asr, truth, waves)
                        for backend, asr in models.items()
                    },
                    waveform=waveform_scores(wet, dict(zip("ABC", waves)), mix),
                )
            cases.append(record)
            print(
                a.group,
                i,
                {
                    k: {backend: v[backend]["edits"] for backend in models}
                    for k, v in record["arms"].items()
                },
                flush=True,
            )
    if cases:
        metrics = stats(cases, list(models))
        print(metrics, flush=True)
        (a.out / f"{a.group}{a.tag}.json").write_text(
            json.dumps(dict(summary=metrics, cases=cases), ensure_ascii=False, indent=2)
            + "\n"
        )


if __name__ == "__main__":
    main()
