"""Separate one audio file into speaker WAVs and optional Mandarin text."""
from __future__ import annotations

import argparse
import codecs
from html import escape
import json
import math
from pathlib import Path
import re
import time
from urllib.parse import quote

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import resample_poly
import sherpa_onnx
import soundfile as sf
import torch
from torch.nn.functional import pad

from dprnn_architecture import MultiDecoderDPRNN


RATE = 8000
ASR_RATE = 16000
SLOTS = 4
DEFAULT_MODEL = "tuned"
DEFAULT_BLOCK = 2.8
DEFAULT_HISTORY = 0.60
DEFAULT_LOOKAHEAD = 0.35
DEFAULT_OVERLAP_MARGIN = 0.0


class _TrainingMetadata:
    """Stand-in for two legacy callback keys. The callbacks are never run."""


def load_separator(directory, model_name, device):
    prefix = directory / "multi_decoder_dprnn.bin"
    safety = [
        (_TrainingMetadata, "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint"),
        (_TrainingMetadata, "pytorch_lightning.callbacks.early_stopping.EarlyStopping"),
    ]
    with torch.serialization.safe_globals(safety):
        original = torch.load(prefix, weights_only=True, map_location="cpu")
    config = dict(original["model_args"])
    weights = original["state_dict"]
    if model_name == "tuned":
        config["n_srcs"] = [1, 2, 3, 4, 5]
        weights = torch.load(directory / "dprnn_rys_weights.pt", map_location="cpu", weights_only=True)
    model = MultiDecoderDPRNN(**config)
    model.load_state_dict(weights, strict=True)
    return model.eval().to(device)


def resample(y, src, dst):
    if src == dst:
        return y
    divisor = math.gcd(src, dst)
    return resample_poly(y, dst // divisor, src // divisor, axis=-1).astype(np.float32)


def read_audio(path):
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[0] == 0 or not np.isfinite(audio).all():
        raise ValueError("Expected a non-empty audio recording with finite PCM samples.")
    return resample(audio.mean(axis=1), source_rate, RATE), source_rate, len(audio)


def rms(y):
    return float(np.sqrt(np.mean(y.astype(np.float64) ** 2))) if y.size else 0.


def normalized_dot(a, b):
    bottom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / bottom) if bottom > 1e-9 else 0.0


def json_write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def json_lines(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def track_id(idx):
    return f"spk_{chr(65 + idx)}"


class VoiceEmbeddings:
    def __init__(self, path):
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(path), num_threads=1, provider="cpu")
        if not config.validate():
            raise ValueError(f"Speaker embedding model failed validation: {path}")
        self.model = sherpa_onnx.SpeakerEmbeddingExtractor(config)

    def __call__(self, y):
        if len(y) < 0.6 * RATE or rms(y) < 0.001:
            return None
        stream = self.model.create_stream()
        stream.accept_waveform(ASR_RATE, resample(y, RATE, ASR_RATE))
        stream.input_finished()
        if not self.model.is_ready(stream):
            return None
        speaker = np.asarray(self.model.compute(stream), np.float32)
        return speaker / max(float(np.linalg.norm(speaker)), 1e-8)


class Separation:
    def __init__(self, model, device, embeddings=None, overlap_margin=DEFAULT_OVERLAP_MARGIN):
        self.model = model
        self.device = device
        self.embeddings = embeddings
        self.overlap_margin = overlap_margin

    @torch.inference_mode()
    def infer(self, y, hints):
        if rms(y) < 0.00022:
            return np.empty((0, len(y)), np.float32), None, None
        if hints == 1:
            return y[None].copy(), 1, 1
        original_samples = len(y)
        inp = torch.as_tensor(y, device=self.device).reshape(1, 1, -1)
        minimum_samples = 768
        if original_samples < minimum_samples:
            inp = pad(inp, (0, minimum_samples - original_samples))
        feature = self.model.enc_activation(self.model.encoder(inp))
        final = self.model.masker(feature)[-1]
        heads = self.model.decoder_select
        options = [n for n in heads.n_srcs if 1 <= n <= SLOTS]
        def decode(count):
            if count == 1:
                return y[None].copy()
            decoder = heads.decoders[heads.n_src2idx[count]]
            tracks = decoder(final, feature)[0].detach().cpu().numpy()
            if tracks.shape[1] < original_samples:
                tracks = np.pad(tracks, ((0, 0), (0, original_samples - tracks.shape[1])))
            tracks = tracks[:, :original_samples].copy()
            # SI-SDR training leaves arbitrary gains. Fit a regularized
            # rescaling against the actual mixture for this window.
            gram = (tracks @ tracks.T).astype(np.float64)
            penalty = max(1e-9, float(np.trace(gram)) * 0.003 / count)
            gains = np.linalg.solve(gram + penalty * np.eye(count), tracks @ y)
            return (tracks * gains[:, None]).astype(np.float32)

        if hints is not None:
            return decode(hints), hints, hints
        logits = heads.selector(final).flatten().detach().cpu().tolist()
        scores = {n: logits[heads.n_src2idx[n]] for n in options}
        baseline = max(options, key=scores.__getitem__)
        guess = overlap_candidate(scores, baseline, self.overlap_margin)
        if guess > baseline:
            candidate = decode(guess)
            if has_distinct_sources(y, candidate):
                return candidate, guess, baseline
        return decode(baseline), baseline, baseline

    def separate(self, y, block, history, lookahead, paced, hints=None, on_commit=None):
        """Run bounded windows. Each committed block is immutable."""
        start = time.perf_counter()
        samples = len(y)
        estimates = np.zeros((SLOTS, samples), dtype=np.float32)
        features = {}
        seen = set()
        commits = []
        for left in range(0, samples, block):
            right = min(left + block, samples)
            visible_right = min(right + lookahead, samples)
            visible_left = max(0, left - history)
            if paced:
                time.sleep(max(0., visible_right / RATE - (time.perf_counter() - start)))
            local, guessed, baseline_count = self.infer(y[visible_left:visible_right], hints)
            core_start = left - visible_left
            core_stop = right - visible_left
            keys = range(len(local))
            energy = [rms(local[i, core_start:core_stop]) for i in keys]
            floor = max(0.00025, (max(energy) if energy else 0.) * 0.065)
            candidates = [i for i in keys if energy[i] >= floor]
            if self.embeddings and len(y) > block:
                representations = {i: self.embeddings(local[i, core_start:core_stop]) for i in candidates}
            else:
                representations = {i: None for i in candidates}
            # Map the candidates one-to-one before committing any samples for this block.
            old = sorted(seen)
            matched = {}
            if old and candidates:
                score = np.zeros((len(old), len(candidates)), dtype=np.float32)
                for row, slot in enumerate(old):
                    for col, source in enumerate(candidates):
                        prediction = local[source, :core_start]
                        overlap = estimates[slot, visible_left:left]
                        corr = normalized_dot(prediction, overlap) if rms(overlap) > 0.001 else 0.
                        anchor = features.get(slot)
                        candidate = representations[source]
                        sim = normalized_dot(anchor, candidate) if anchor is not None and candidate is not None else 0.
                        score[row, col] = max(abs(corr), sim)
                rows, cols = linear_sum_assignment(-score)
                for row, col in zip(rows, cols, strict=True):
                    if score[row, col] >= 0.48:
                        matched[candidates[col]] = old[row]
            remaining = sorted((i for i in candidates if i not in matched), key=lambda i: -energy[i])
            free = sorted(set(range(SLOTS)) - seen)
            uncertain = []
            for source in remaining:
                if free:
                    matched[source] = free.pop(0)
                    continue
                available = [i for i in old if i not in matched.values()]
                matched[source] = available[0]
                uncertain.append(track_id(available[0]))
            for source, slot in matched.items():
                wave = local[source, core_start:core_stop]
                # Resolve possible sign changes against simultaneous estimates.
                if abs(normalized_dot(local[source, :core_start], estimates[slot, visible_left:left])) > 0.45:
                    if normalized_dot(local[source, :core_start], estimates[slot, visible_left:left]) < 0:
                        wave = -wave
                estimates[slot, left:right] = wave
                candidate = representations[source]
                if candidate is not None and track_id(slot) not in uncertain:
                    previous = features.get(slot)
                    features[slot] = candidate if previous is None else 0.75 * previous + 0.25 * candidate
                seen.add(slot)
            active = sorted(set(matched.values()))
            record = {
                "sample_start": left, "sample_end": right, "input_available_through": visible_right,
                "decoded_source_count": guessed, "speaker_ids": [track_id(i) for i in active],
                "baseline_source_count": baseline_count, "count_review": guessed != baseline_count,
                "identity_needs_review": uncertain, "commit_elapsed_s": time.perf_counter() - start,
            }
            if paced:
                record["delay_from_block_end_s"] = record["commit_elapsed_s"] - right / RATE
            commits.append(record)
            if on_commit is not None:
                on_commit(record, {track_id(i): estimates[i, left:right].copy() for i in active})
        return {track_id(i): estimates[i] for i in sorted(seen)}, commits


def overlap_candidate(scores, baseline, margin):
    """Consider one extra source when the next decoder's score is close."""
    if margin <= 0 or baseline < 2 or baseline + 1 not in scores:
        return baseline
    return baseline + 1 if scores[baseline] - scores[baseline + 1] <= margin else baseline


def has_distinct_sources(mixture, tracks):
    """Reject a weak added source or nearly duplicated waveforms."""
    magnitudes = np.linalg.norm(tracks, axis=1)
    if float(magnitudes.min()) < 0.10 * float(np.linalg.norm(mixture)):
        return False
    vectors = tracks / np.maximum(magnitudes[:, None], 1e-8)
    similarities = np.abs(np.triu(vectors @ vectors.T, k=1))
    return bool(float(similarities.max()) < 0.85)


def active_spans(audio_16k):
    if audio_16k.size == 0:
        return []
    frame = 320
    count = math.ceil(len(audio_16k)/frame)
    frames = np.pad(audio_16k, (0, count*frame - len(audio_16k))).reshape(-1, frame)
    energy = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    indices = np.flatnonzero(energy > max(0.0003, float(energy.max()) * 0.05))
    if not len(indices):
        return []
    slices = np.split(indices, np.where(np.diff(indices) > 25)[0] + 1)
    return [(max(0, int(s[0])-5)*frame, min(len(audio_16k), (int(s[-1])+7)*frame))
            for s in slices if len(s) >= 6]


# Icefall byte-BPE alphabet used by this Zipformer export.
_BYTE_ALPHABET = (
    list(range(256, 288)) + list(range(32, 127)) + list(range(288, 306))
    + list(range(308, 319)) + list(range(321, 329)) + list(range(330, 383))
    + list(range(384, 423))
)
_TOKEN_BYTE = {chr(symbol): value for value, symbol in enumerate(_BYTE_ALPHABET)}
_TOKEN_BYTE.update({"▁": 32, "⁇": 32})


def characters(tokens, timestamps, begin_s, end_s):
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    events = []
    char_start = None
    for encoded, timestamp in zip(tokens, timestamps, strict=True):
        onset = min(end_s, begin_s + max(0., float(timestamp)))
        if encoded in {"<unk>", "<blk>", "<sos/eos>"}:
            continue
        for unit in encoded:
            byte = _TOKEN_BYTE.get(unit)
            if byte is None:
                continue
            if char_start is None:
                char_start = onset
            decoded = decoder.decode(bytes((byte,)))
            if decoded:
                events.extend((item, char_start) for item in decoded if item.strip() and item != "�")
                char_start = None
    output = []
    for idx, (text, start) in enumerate(events):
        next_start = events[idx + 1][1] if idx + 1 < len(events) else end_s
        end = min(end_s, max(start, min(next_start, start + 0.24)))
        output.append({"text": text, "start_ms": round(start * 1000), "end_ms": round(end * 1000)})
    return output


class Mandarin:
    def __init__(self, directory, threads):
        asr = directory / "sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03"
        self.model = sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(
            model=str(asr/"model.int8.onnx"), tokens=str(asr/"tokens.txt"), num_threads=threads,
        )

    def transcribe(self, speaker, display, audio):
        wave = resample(audio, RATE, ASR_RATE)
        results = []
        for lo, hi in active_spans(wave):
            begin_s, end_s = lo / ASR_RATE, hi / ASR_RATE
            stream = self.model.create_stream()
            stream.accept_waveform(ASR_RATE, wave[lo:hi])
            self.model.decode_stream(stream)
            result = stream.result
            if not result.text.strip():
                continue
            timing = characters(result.tokens, result.timestamps, begin_s, end_s)
            aligned = "".join(x["text"] for x in timing) == "".join(result.text.split())
            results.append({"speaker_id": speaker, "speaker": display, "start_ms": round(begin_s*1000),
                            "end_ms": round(end_s*1000), "text": result.text.strip(),
                            "characters": timing if aligned else [],
                            "timing": "estimated_ctc" if aligned else "segment_only",
                            "review_required": not aligned})
        return results


def run(args):
    if min(args.block, args.threads) <= 0 or min(args.history, args.lookahead) < 0:
        raise ValueError("Specify a positive block count and nonnegative context.")
    if not math.isfinite(args.overlap_margin) or args.overlap_margin < 0:
        raise ValueError("The overlap margin must be finite and nonnegative.")
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    y, native_rate, native_samples = read_audio(args.input)
    voice_path = args.models / "voiceprint.onnx"
    embeddings = VoiceEmbeddings(voice_path) if not args.no_embeddings else None
    model = load_separator(args.models, args.separator, args.device)
    engine = Separation(model, args.device, embeddings, args.overlap_margin)
    names = json.loads(args.names.read_text(encoding="utf-8")) if args.names else {}
    if not isinstance(names, dict) or any(not isinstance(value, str) for value in names.values()):
        raise ValueError("The names JSON must map speaker IDs to strings.")
    args.out.mkdir(parents=True, exist_ok=False)
    audio_dir = args.out / "committed_chunks"
    if args.write_chunks:
        audio_dir.mkdir()

    def export_chunk(info, stems):
        if args.write_chunks:
            for sid, wave in stems.items():
                sf.write(audio_dir / f'{info["sample_start"]:010d}_{sid}.wav', wave, RATE, subtype="FLOAT")

    model_setup = time.perf_counter() - started
    tracks, commits = engine.separate(y, max(1, round(args.block*RATE)), round(args.history*RATE),
                                     round(args.lookahead*RATE), args.mode == "replay",
                                     args.speakers, export_chunk)
    speakers = {}
    largest_peak = max(float(np.max(np.abs(wave))) for wave in (y, *tracks.values()))
    playback_gain = min(1.0, 0.97 / largest_peak) if largest_peak else 1.0
    for speaker, wave in tracks.items():
        letter = speaker[-1]
        display = names.get(speaker, f"Speaker {letter}")
        renamed = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", names.get(speaker, "")).strip(" .")[:72]
        suffix = f"_{renamed}" if renamed else ""
        filename = f"Speaker_{letter}{suffix}.wav"
        sf.write(args.out / filename, wave * playback_gain, RATE, subtype="PCM_16")
        speakers[speaker] = {"display_name": display, "file": filename}
    sf.write(args.out/"mixed.wav", y * playback_gain, RATE, subtype="PCM_16")
    audio_export_s = time.perf_counter() - started
    json_lines(args.out/"commits.jsonl", commits)
    info = {
        "input_file": args.input.name, "separator": args.separator, "device": args.device,
        "mode": args.mode, "speaker_count_hint": args.speakers, "input_samples": native_samples,
        "input_sample_rate": native_rate, "output_samples": len(y), "output_sample_rate": RATE,
        "block_s": args.block, "history_s": args.history, "lookahead_s": args.lookahead,
        "overlap_margin": args.overlap_margin, "wav_playback_gain": playback_gain,
        "review_windows": sum(bool(c["identity_needs_review"] or c["count_review"]) for c in commits),
        "model_setup_wall_s": model_setup, "audio_export_wall_s": audio_export_s,
        "speakers": speakers, "asr": "pending" if args.transcribe else "disabled",
    }
    json_write(args.out/"speakers.json", info)
    rows = []
    write_listening_report(args.out, info, rows)
    if args.transcribe:
        recognition = Mandarin(args.models, args.threads)
        rows = [row for sid, wave in tracks.items()
                for row in recognition.transcribe(sid, speakers[sid]["display_name"], wave)]
        rows.sort(key=lambda row: (row["start_ms"], row["speaker_id"]))
        json_lines(args.out/"transcript.jsonl", rows)
        text_lines = [f'[{r["start_ms"]/1000:06.2f}–{r["end_ms"]/1000:06.2f}] {r["speaker"]}  {r["text"]}'
                      for r in rows]
        (args.out/"transcript.txt").write_text("\n".join(text_lines)+"\n", encoding="utf-8")
        info["asr"] = "Zipformer CTC Mandarin, per-speaker file pass"
        info["complete_wall_s"] = time.perf_counter() - started
        json_write(args.out/"speakers.json", info)
        write_listening_report(args.out, info, rows)
    print(json.dumps(info, ensure_ascii=False, indent=2))


def write_listening_report(directory, info, rows):
    def audio(name, filename):
        return (f'<h2>{escape(name)}</h2><audio controls preload="none" '
                f'src="{quote(filename)}"></audio>')

    tracks = audio("Mixed audio", "mixed.wav")
    tracks += "".join(audio(s["display_name"], s["file"]) for s in info["speakers"].values())
    text_rows = "".join(
        f'<tr><td>{escape(row["speaker"])}</td><td>{row["start_ms"]/1000:.2f}–{row["end_ms"]/1000:.2f}</td>'
        f'<td lang="zh">{escape(row["text"])}</td></tr>' for row in rows
    )
    transcript = (
        f"<h2>Mandarin transcript</h2><table><tr><th>Speaker</th><th>Seconds</th><th>Text</th></tr>"
        f"{text_rows}</table>" if rows else ""
    )
    html = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Mandarin speaker tracks</title>
<style>body {{font:16px system-ui,sans-serif; margin:2rem auto; max-width:55rem; padding:0 1rem}}
td,th {{border-bottom:1px solid #ddd;text-align:left; padding:.5rem}} audio {{width:min(100%,24rem)}}</style>
<h1>Mandarin speaker tracks</h1><p>{escape(info["input_file"])}</p>
<p>{info["review_windows"]} audio windows flagged for count or identity review.</p>
{tracks}{transcript}</html>"""
    (directory/"listen.html").write_text(html, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--models", type=Path, default=Path(__file__).parent/"models")
    p.add_argument("--mode", choices=("file", "replay"), default="file")
    p.add_argument("--separator", choices=("original", "tuned"), default=DEFAULT_MODEL)
    p.add_argument("--block", type=float, default=DEFAULT_BLOCK)
    p.add_argument("--history", type=float, default=DEFAULT_HISTORY)
    p.add_argument("--lookahead", type=float, default=DEFAULT_LOOKAHEAD)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--speakers", type=int, choices=range(1, 5), help="Optional count override.")
    p.add_argument("--overlap-margin", type=float, default=DEFAULT_OVERLAP_MARGIN,
                   help="Selector margin for considering one additional voice." )
    p.add_argument("--names", type=Path, help="UTF-8 JSON mapping anonymous speaker IDs to display names.")
    p.add_argument("--write-chunks", action="store_true", help="Save committed blocks as soon as they are ready.")
    p.add_argument("--no-embeddings", action="store_true")
    p.add_argument("--transcribe", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
