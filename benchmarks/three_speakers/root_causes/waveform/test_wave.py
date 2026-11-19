"""Test a development-selected waveform repair and characterize the separate degradation stages."""

import argparse
import json
from pathlib import Path
from fractions import Fraction
from time import perf_counter

import numpy as np
from scipy.signal import resample_poly
import sherpa_onnx as sx
import soundfile as sf
import torch
from benchmark_examples import waveform_scores
from benchmark_waveform import room_response
from controlled_words import make_mixture, recognize
from recognize import text_scores
from sepformer_three import SepFormer
from acoustic_repair import contextualize, dereverberate


def low_memory_separator(weights):
    """Split only independent attention batches to bound peak memory on the CPU."""
    from speechbrain.lobes.models.dual_path import SBTransformerBlock

    print("Loading separator", flush=True)
    model = SepFormer(weights)
    print("Loaded separator", flush=True)
    for block in model.mask.modules():
        if isinstance(block, SBTransformerBlock):
            evaluate = block.forward

            def grouped(x, fn=evaluate):
                return torch.cat([fn(group) for group in x.split(4, dim=0)], dim=0)

            block.forward = grouped
    return model


def create_sources(root, row, idx):
    paths = [root / s["target"] for s in row]
    dry, mix, triple, levels, starts = make_mixture(paths, idx)
    rng = np.random.default_rng(19112026 + idx)
    wet = np.stack([room_response(x, i, rng) for i, x in enumerate(dry)])
    clean = wet.sum(0)
    noise = rng.normal(size=len(clean)).astype(np.float32)
    raw = clean + noise * np.sqrt(np.mean(clean**2) / (np.mean(noise**2) * 10**1.5))
    gain = min(1.0, 0.92 / np.max(np.abs(raw)))
    np.testing.assert_allclose(gain * raw, mix, atol=1e-7)
    dry *= gain
    wet *= gain
    cues = []
    for i, s in enumerate(row):
        pcm, sr = sf.read(root / s["enroll"], dtype="float32")
        ratio = Fraction(8000, sr)
        cue = resample_poly(pcm, ratio.numerator, ratio.denominator)
        cues.append(room_response(cue, i, np.random.default_rng(219 + i)).astype("f"))
    return dry, wet, mix, cues, triple


def native_tracks(root, row, idx, dry):
    """Construct the paired isolated reference at 16 kHz before the separator's 8 kHz conversion."""
    rows = []
    starts = np.roll([0, 1920, 3840], idx)
    for speaker, start, reference in zip(row, starts, dry, strict=True):
        pcm, rate = sf.read(root / speaker["target"], dtype="float32")
        ratio = Fraction(16000, rate)
        signal = resample_poly(pcm, ratio.numerator, ratio.denominator)
        track = np.zeros(2 * dry.shape[1], dtype="float32")
        track[start : start + len(signal)] = signal
        track *= np.sqrt(np.mean(reference**2) / np.mean(track**2))
        rows.append(track)
    return np.stack(rows)


def recognize_native(model, tracks):
    rows = []
    for idx, wave in enumerate(tracks):
        rms = max(float(np.sqrt(np.mean(wave**2))), 1e-7)
        peak = max(float(np.max(np.abs(wave))), 1e-7)
        stream = model.create_stream()
        stream.accept_waveform(
            16000, np.pad(wave * min(0.035 / rms, 0.98 / peak, 20), (800, 1280))
        )
        model.decode_stream(stream)
        rows.append(dict(speaker=str(idx), start=0.0, text=stream.result.text.strip()))
    return rows


def evaluate_cases(rows, which, separator, asrs, root, candidates, cache):
    cases = []
    for idx, row in enumerate(rows, start=1):
        dry, wet, mix, cues, triple = create_sources(root, row, idx)
        truth = [
            dict(speaker=str(i), start=0.0, text=s["text"]) for i, s in enumerate(row)
        ]
        arms = {}
        for arm in candidates:
            started = perf_counter()
            path = cache / f"{which}_{idx}_{arm}.npz"
            if path.is_file():
                with np.load(path) as f:
                    out = f["tracks"]
                    inference = float(f["seconds"])
            elif separator is None:
                raise FileNotFoundError(path)
            elif arm == "isolated_native":
                out = native_tracks(root, row, idx, dry)
            elif arm == "isolated_dry":
                out = dry
            elif arm == "isolated_room":
                out = wet
            else:
                wave, offset = mix, 0
                if arm == "clean_separation":
                    wave = dry.sum(axis=0)
                if arm == "room_without_noise":
                    wave = wet.sum(axis=0)
                if arm == "wpe":
                    wave = dereverberate(mix)
                if arm == "prompt":
                    wave, offset = contextualize(mix, cues)
                print(
                    "Separating",
                    which,
                    idx,
                    arm,
                    round(len(wave) / 8000, 2),
                    flush=True,
                )
                out = separator(wave)[..., offset : offset + len(mix)]
            if not path.is_file():
                inference = perf_counter() - started
                np.savez(path, tracks=out, seconds=inference)
            reference = dry if arm in ["clean_separation", "isolated_dry"] else wet
            mixture_for_scoring = {
                "clean_separation": dry.sum(axis=0),
                "room_without_noise": wet.sum(axis=0),
            }.get(arm, mix)
            metrics = (
                None
                if arm.startswith("isolated")
                else waveform_scores(
                    reference, dict(zip("ABC", out)), mixture_for_scoring
                )
            )
            result = {}
            for backend, model in asrs.items():
                hyp = (
                    recognize_native(model, out)
                    if arm == "isolated_native"
                    else recognize(model, out)
                )
                scores = text_scores(truth, hyp)
                matches = sum(
                    s["deletions"] + s["insertions"] + s["substitutions"] == 0
                    for s in scores["speaker_scores"]
                )
                result[backend] = {
                    "cpCER": scores["cpCER"],
                    "totals": scores["totals"],
                    "per_source": scores["speaker_scores"],
                    "phrases_matched": int(matches),
                    "hypotheses": hyp,
                }
            arms[arm] = {
                "asr": result,
                "scores": metrics,
                "dry_reference_scores": waveform_scores(
                    dry, dict(zip("ABC", out)), mixture_for_scoring
                )
                if metrics is not None
                else None,
                "separator_seconds": inference,
            }
            print(
                which,
                idx,
                arm,
                " ".join(
                    f"{b}:{r['totals']['errors']}/{r['totals']['reference_chars']}"
                    for b, r in result.items()
                ),
                flush=True,
            )
        cases.append(
            {
                "trio": row,
                "source_dB": np.roll([0, -4, -8], idx).tolist(),
                "triple_s": triple,
                "seconds": len(mix) / 8000,
                "arms": arms,
            }
        )
    return cases


def summarize(cases):
    summary = {}
    for arm in cases[0]["arms"]:
        rows = [s["arms"][arm] for s in cases]
        scores = [x for r in rows for x in (r["scores"] or [])]
        summary[arm] = {"time_s": sum(r["separator_seconds"] for r in rows)}
        for name in ["sensevoice", "zipformer"]:
            total = {}
            for key in [
                "reference_chars",
                "errors",
                "substitutions",
                "deletions",
                "insertions",
                "correct_reference_chars",
            ]:
                total[key] = sum(r["asr"][name]["totals"][key] for r in rows)
            total["cpCER"] = total["errors"] / total["reference_chars"]
            total["phrases_matched"] = sum(
                r["asr"][name]["phrases_matched"] for r in rows
            )
            summary[arm][name] = total
        if scores:
            summary[arm]["median_whole_SI_SDRi"] = float(
                np.median([x["whole_improvement_dB"] for x in scores])
            )
            summary[arm]["median_triple_SI_SDRi"] = float(
                np.median(
                    [
                        x["exact_overlap_improvement_dB"]
                        for x in scores
                        if x["exact_overlap_improvement_dB"] is not None
                    ]
                )
            )
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["data", "weights", "sense", "zipformer", "out"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--phase", required=True, choices=["development", "test"])
    p.add_argument(
        "--stage", required=True, choices=["sep", "sensevoice", "zipformer", "score"]
    )
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sep = low_memory_separator(a.weights) if a.stage == "sep" else None
    models = {}
    if a.stage == "sensevoice":
        models[a.stage] = sx.OfflineRecognizer.from_sense_voice(
            model=str(a.sense / "model.int8.onnx"),
            tokens=str(a.sense / "tokens.txt"),
            language="zh",
            use_itn=False,
            num_threads=1,
        )
    if a.stage == "zipformer":
        models[a.stage] = sx.OfflineRecognizer.from_zipformer_ctc(
            model=str(a.zipformer / "model.int8.onnx"),
            tokens=str(a.zipformer / "tokens.txt"),
            num_threads=1,
        )
    groups = json.loads((a.data / "manifest.json").read_text())["groups"]
    a.out.mkdir(exist_ok=True, parents=True)
    cache = a.out / "cache"
    cache.mkdir(exist_ok=True)
    for phase in ["development"] if a.phase == "development" else ["repeat", "new"]:
        arms = (
            ["room_baseline", "wpe", "prompt"]
            if a.phase == "development"
            else list(
                dict.fromkeys(
                    [
                        "isolated_native",
                        "isolated_dry",
                        "isolated_room",
                        "clean_separation",
                        "room_without_noise",
                        "room_baseline",
                        json.loads((a.out / "selection.json").read_text())["candidate"],
                    ]
                )
            )
        )
        start = perf_counter()
        if a.stage != "score":
            cases = evaluate_cases(
                groups[phase], phase, sep, models, a.data / "sources", arms, cache
            )
            (cache / f"{phase}_{a.stage}.json").write_text(
                json.dumps(cases, ensure_ascii=False)
            )
            continue
        cases = json.loads((cache / f"{phase}_sensevoice.json").read_text())
        others = json.loads((cache / f"{phase}_zipformer.json").read_text())
        for row, other in zip(cases, others, strict=True):
            for arm in row["arms"]:
                row["arms"][arm]["asr"]["zipformer"] = other["arms"][arm]["asr"][
                    "zipformer"
                ]
        summary = summarize(cases)
        if a.phase == "development":
            # Compare both remedies on development, freeze better one even if it fails to beat control.
            # Prioritize actual words. Signal quality is the prespecified tie breaker.
            def rank(arm):
                s = summary[arm]
                return (
                    s["sensevoice"]["errors"] + s["zipformer"]["errors"],
                    -s["median_triple_SI_SDRi"],
                )

            selection = min(["wpe", "prompt"], key=rank)
            (a.out / "selection.json").write_text(
                json.dumps({"candidate": selection, "development": summary}, indent=2)
                + "\n"
            )
        (a.out / (phase + ".json")).write_text(
            json.dumps(
                {"summary": summary, "cases": cases, "wall_s": perf_counter() - start},
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        print(phase, "Finished", flush=True)


if __name__ == "__main__":
    main()
