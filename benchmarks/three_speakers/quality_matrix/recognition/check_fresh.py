"""Evaluate frozen consensus on new AISHELL-4 rooms, with decoding selected without labels."""

from pathlib import Path
from time import perf_counter
import json

from cached_gate import export
from character_consensus import apply, decoder_keys, prepare
from contextual_asr import candidates, read_jsonl
from diarize import reference
from recognize import text_scores, transcribe
from overlap_characters import panel_score


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    corpus = Path("pareto_data/corpus")
    out = Path("pareto_text/fresh")
    out.mkdir(exist_ok=True)
    model = Path("/tmp/mandarin-audio-run.wfQ7MzyH/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03")
    summaries = []
    for clip in read(corpus/"top_manifest.json"):
        name = clip["id"]
        folder = corpus/name
        stem = read_jsonl(folder/"stem/transcript.jsonl")
        timing = read(folder/"timing.json")["segments"]
        pairs = candidates(timing, folder/"mask", stem, read(folder/"context_tokens.json")["tokens"])
        refined = read_jsonl(folder/"refined/transcript.jsonl")
        options = {"release": prepare(pairs,refined), "direct": prepare(pairs,stem)}
        all_keys = set().union(*(decoder_keys(entries) for entries in options.values()))
        paths = {p.stem.split("_")[-1]:p for p in (folder/"mask").glob("speaker_*.wav")}
        started = perf_counter()
        zip_rows, _ = transcribe(paths,timing,model/"model.int8.onnx",model/"tokens.txt",threads=1,
                               backend="zipformer",target_rms=0.035,selected_keys=all_keys)
        wall = perf_counter()-started
        truth = read(corpus/clip["text_reference"])
        marks = reference(corpus/clip["ref"])
        report = dict(id=name, duration=clip["duration_s"], exact_three_seconds=clip["exact_three_s"],
                     extra_decode_seconds=wall, selected_turns=len(all_keys), all_turns=len(pairs))
        for category, prior in (("release",refined),("direct",stem)):
            entries = options[category]
            updated, changes = apply(entries,zip_rows)
            export(updated,out/name/category,sorted({entry.record["speaker"] for entry in entries}))
            report[category] = {
                "baseline":text_scores(truth,prior),"candidate":text_scores(truth,updated),
                "timed_baseline":panel_score(truth,prior,marks,clip["duration_s"],1.0),
                "timed_candidate":panel_score(truth,updated,marks,clip["duration_s"],1.0),
                "eligible":len(decoder_keys(entries)), "changes":changes,
            }
        summaries.append(report)
        print(name,{k:[report[k][g]["totals"] for g in ("baseline","candidate")] for k in ("direct","release")})
    (out/"scores.json").write_text(json.dumps(summaries,indent=2,ensure_ascii=False)+"\n")


if __name__=="__main__":
    main()
