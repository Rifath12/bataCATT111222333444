from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import unicodedata
import numpy as np
from scipy.optimize import linear_sum_assignment
import soundfile as sf
import torch

from process_meeting import (
    DEFAULT_BLOCK, DEFAULT_HISTORY, DEFAULT_LOOKAHEAD, Mandarin, RATE,
    Separation, VoiceEmbeddings, json_lines, json_write, load_separator,
    read_audio, rms,
)


def norm(s):
    return "".join(c.lower() for c in unicodedata.normalize("NFKC", s)
                   if unicodedata.category(c)[0] in ("L", "N"))


def edit_dist(a, b):
    costs = list(range(len(b)+1))
    for i, ref in enumerate(a, 1):
        row = [i]
        for j, hyp in enumerate(b, 1):
            row.append(min(costs[j]+1, row[-1]+1, costs[j-1]+(ref!=hyp)))
        costs = row
    return costs[-1]


def cperror(refs, hyps):
    n = max(len(refs), len(hyps))
    a = [*refs, *(n-len(refs))*[""]]
    b = [*hyps, *(n-len(hyps))*[""]]
    matrix = np.array([[edit_dist(x, y) for y in b] for x in a])
    i,j=linear_sum_assignment(matrix)
    count=sum(map(len, refs))
    err=sum(int(matrix[x,y]) for x,y in zip(i,j))
    return {"errors":err, "reference_chars":count, "cpCER":err/count,
            "hypothesis_index_by_reference":{str(int(x)):int(y) if y<len(hyps) else None
                for x,y in zip(i,j) if x<len(refs)}}


def sisdr(ref, est):
    if not len(ref):
        return None
    ref = ref.astype(np.float64) - ref.mean()
    est = est.astype(np.float64) - est.mean()
    proj = np.dot(ref,est) / max(float(np.dot(ref,ref)),1e-10) * ref
    error = est - proj
    return float(10*np.log10(max(float(np.dot(proj,proj)),1e-10)
                              /max(float(np.dot(error,error)),1e-10)))


def refmask(wav):
    frames = wav[:len(wav)//160*160].reshape(-1,160)
    level = np.sqrt(np.mean(frames.astype(np.float64)**2, axis=1))
    return level>max(0.0002,0.055*float(level.max()))


def waveform_scores(sources, tracks, mix):
    names=list(tracks)
    hypotheses=list(tracks.values())
    by=np.asarray([[sisdr(x, y) for y in hypotheses] for x in sources])
    pairs=zip(*linear_sum_assignment(-by)) if len(hypotheses) else []
    exact_samples = np.repeat(np.stack([refmask(x) for x in sources]).sum(axis=0)==len(sources),160)
    scored_samples = len(exact_samples)
    def case_metrics(a):
        m=sisdr(sources[a][:scored_samples][exact_samples],mix[:scored_samples][exact_samples])
        return {"source":chr(65+a),"source_rms":rms(sources[a]),"track":None,
                "whole_improvement_dB":None,"exact_overlap_improvement_dB":None,
                "mixture_exact_overlap_SI_SDR_dB":m}
    result=[case_metrics(i) for i in range(len(sources))]
    for i,j in pairs:
        exact=sisdr(sources[i][:scored_samples][exact_samples],hypotheses[j][:scored_samples][exact_samples])
        mix_score=result[i]["mixture_exact_overlap_SI_SDR_dB"]
        result[i].update(track=names[j],whole_improvement_dB=float(by[i,j])-sisdr(sources[i],mix),
            exact_overlap_improvement_dB=exact-mix_score if exact is not None else None)
    return result


def text_by_track(asr, tracks):
    exact = [r for k,v in tracks.items() for r in asr.transcribe(k, f"Speaker {k[-1]}", v)]
    exact.sort(key=lambda r:(r["start_ms"],r["speaker_id"]))
    grouped = {sid:norm("".join(r["text"] for r in exact if r["speaker_id"]==sid)) for sid in tracks}
    return exact, grouped


def save_output(dest, tracks, mixture, commits, asr_rows):
    dest.mkdir(parents=True,exist_ok=True)
    sf.write(dest/"mixed.wav", mixture, RATE, subtype="FLOAT")
    for key,v in tracks.items():
        sf.write(dest/f"Speaker_{key[-1]}.wav", v, RATE, subtype="FLOAT")
    json_lines(dest/"commits.jsonl", commits)
    json_lines(dest/"transcript.jsonl", asr_rows)
    lines = [f'[{r["start_ms"]/1000:05.2f}–{r["end_ms"]/1000:05.2f}] {r["speaker"]}  {r["text"]}' for r in asr_rows]
    (dest/"transcript.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")


def natural_activity(p,mroot):
    prefix,limits=p.stem.split("_mixture_")
    sid=prefix.rsplit("_",1)[0]
    start,end=map(float,limits.split("_"))
    source_file=mroot/f"{sid}.rttm"
    spans=[]
    for row in source_file.read_text().splitlines():
        data=row.split();lo=float(data[3]);hi=lo+float(data[4])
        if max(lo,start)<min(hi,end):
            spans.append((max(0.,lo-start),min(hi,end)-start,data[7]))
    bounds=sorted({0.,end-start,*[t for lo,hi,_ in spans for t in (lo,hi)]})
    overlap=collections.defaultdict(float)
    for lo,hi in zip(bounds,bounds[1:]):
        overlap[len({sid for a,b,sid in spans if a<(lo+hi)/2<b})] += hi-lo
    return sid,start,spans,{str(n):round(overlap[n],3) for n in range(5)}


def diarization_metrics(spans, tracks, samples):
    nframes=samples//160
    times=(np.arange(nframes)+.5)*160/RATE
    speakers={s for _,_,s in spans}
    ref=np.stack([np.logical_or.reduce([(a<=times)&(times<b) for a,b,k in spans if k==s])
                  for s in sorted(speakers)])
    names=list(tracks)
    est=np.stack([refmask(v)[:nframes] for v in tracks.values()]) if len(names) else np.zeros((0,nframes),bool)
    refs=ref.sum(axis=0)
    ests=est.sum(axis=0)
    cost=(ref[:,None,:]&est[None,:,:]).sum(axis=-1)
    i,j=linear_sum_assignment(-cost)
    overlap_count=(ref[i]&est[j]).sum(axis=0)
    extra=int(np.maximum(ests-refs,0).sum())
    missed=int(np.maximum(refs-ests,0).sum())
    confused=int((np.minimum(ests,refs)-overlap_count).sum())
    return {"ref_speaker_seconds":float(refs.sum())*0.02,"missed_speaker_seconds":missed*0.02,
            "false_alarm_seconds":extra*0.02,"confusion_seconds":confused*0.02,
            "energy_mask_DER":(missed+extra+confused)/int(refs.sum()),
            "acoustic_mapping":{s:(names[j[np.where(i==idx)[0][0]]] if idx in i else None)
                                for idx,s in enumerate(sorted(speakers))}}


def run(args):
    torch.set_num_threads(args.threads)
    base=args.inputs
    device="cpu"
    result_dir=args.out
    result_dir.mkdir(parents=True,exist_ok=True)
    settings=(round(DEFAULT_BLOCK*RATE),round(DEFAULT_HISTORY*RATE),round(DEFAULT_LOOKAHEAD*RATE))
    models={name:Separation(load_separator(args.models,name,device),device,
                            VoiceEmbeddings(args.models/"voiceprint.onnx"))
            for name in ["tuned","original"]}
    asr=Mandarin(args.models,args.threads)
    benchmark=[]
    def timing(commits,wall):
        return {"compute_and_tracking_wall_s":wall,"blocks":len(commits),
                "first_committed_audio_block_s":next((x["commit_elapsed_s"] for x in commits if x["speaker_ids"]),None)}
    if args.condition in ("all","controlled"):
        parents = sorted((base/"controlled").glob("*/reference.json"))
        for file in parents:
            case=file.parent.name
            if args.quick and case.startswith("dev"):
                continue
            summary=json.loads(file.read_text(encoding="utf-8"))
            y,_,_=read_audio(file.parent/"mixed.wav")
            sources=[read_audio(file.parent/f"reference_{k}.wav")[0] for k in summary["speakers"]]
            refs=[norm("".join(ln["transcript"] for ln in lines)) for lines in summary["speakers"].values()]
            src_dict={f"spk_{chr(65+i)}": sig for i,sig in enumerate(sources)}
            clean_rows,clean_text=text_by_track(asr,src_dict)
            isolated_cer=sum(edit_dist(ref,clean_text[k]) for k,ref in zip(clean_text,refs))/sum(map(len,refs))
            mix_rows,mix_text=text_by_track(asr,{"spk_A":y})
            block_options=[("blind", "tuned", None),("known_count", "original", len(sources))]
            if case.startswith("turns"):
                block_options=block_options[:1]
            if case.startswith("dev"):
                block_options += [("native_blind","original",None),("tuned_known_count","tuned",len(sources))]
            for mode, model, hint in block_options:
                engine=models[model]
                t=time.perf_counter()
                tracks,commits=engine.separate(y,*settings,False,hint)
                wall=time.perf_counter()-t
                rows,words=text_by_track(asr,tracks)
                result={"clip":case,"condition":mode,"model":model,"count_hint":hint,
                        "source_count":len(sources),"tracks":len(tracks),"duration_s":len(y)/RATE,
                        "concurrency_seconds":summary["exact_concurrency_seconds"],**timing(commits,wall),
                        "warm_audio_real_time_factor":wall/(len(y)/RATE),
                        "isolated_CER":isolated_cer,
                        "mixture_cpCER":cperror(refs,list(mix_text.values())),
                        "text_cpCER":cperror(refs,list(words.values())),
                        "waveform":waveform_scores(sources,tracks,y)}
                benchmark.append(result)
                out=result_dir/"controlled"/case/mode
                save_output(out,tracks,y,commits,rows)
                json_write(out/"score.json",result)
                json_write(out/"human_reference.json",summary)
                json_lines(out/"recognition_on_clean_references.jsonl",clean_rows)
                print(case,mode,len(tracks),round(result["text_cpCER"]["cpCER"],2),
                      [round(a["whole_improvement_dB"],1) if a["whole_improvement_dB"] is not None else None
                       for a in result["waveform"]], flush=True)
            json_write(result_dir/"controlled_scores.json",benchmark)
    if args.condition in ("all","real"):
        dataset=list(csv.DictReader((base/"real-alimeeting.csv").open(encoding="utf-8")))
        natural=[*sorted(base.glob("*_1607.99_1619.92.wav")),*sorted(base.glob("*_1571.24_1587.66.wav"))]
        for p in natural:
            y,_,_=read_audio(p)
            session,start,spans,concurrency = natural_activity(p,base)
            annotation={item["speaker"]:item["ground_truth_transcript"]
                        for item in dataset if item["mixture_utterance"]==p.stem}
            engine=models["tuned"]
            t=time.perf_counter()
            tracks,commits=engine.separate(y,*settings,False)
            wall=time.perf_counter()-t
            rows,words=text_by_track(asr,tracks)
            out=result_dir/"real"/p.stem
            save_output(out,tracks,y,commits,rows)
            diar=diarization_metrics(spans,tracks,len(y))
            _, original=text_by_track(asr,{"spk_A":y})
            transcription=[]
            for name,ref in annotation.items():
                track=diar["acoustic_mapping"].get(name)
                text=words.get(track,"")
                target=norm(ref)
                transcription.append({"reference_speaker":name,"mapped_track":track,
                    "reference_chars":len(target), "mapped_track_CER":edit_dist(target,text)/len(target),
                    "mixture_CER":edit_dist(target,original["spk_A"])/len(target)})
            result={"clip":p.stem,"session":session,"duration_s":len(y)/RATE,"tracks":len(tracks),
                    "recording_participants":len({s for _,_,s in spans}),
                    "concurrency_seconds_by_RTTM":concurrency,**timing(commits,wall),
                    "warm_audio_real_time_factor":wall/(len(y)/RATE),
                    "diarization":diar, "available_target_transcripts_only":transcription}
            if p.stem.endswith("_1607.99_1619.92"):
                paced_tracks,replay=engine.separate(y,*settings,True)
                delays=[r["delay_from_block_end_s"] for r in replay]
                result["paced_replay"]={"commit_delay_median_s":float(np.median(delays)),
                    "commit_delay_p95_s":float(np.percentile(delays,95)), "blocks":len(delays),
                    "audio_sample_exact":list(tracks)==list(paced_tracks) and
                                      all(np.array_equal(v,paced_tracks[k]) for k,v in tracks.items())}
                json_lines(out/"paced_commits.jsonl",replay)
                prefix=round(DEFAULT_BLOCK*RATE)+round(DEFAULT_LOOKAHEAD*RATE)
                modified=y.copy(); modified[prefix:]*=0
                different,_=engine.separate(modified,*settings,False)
                q = round(DEFAULT_BLOCK*RATE)
                result["first_block_suffix_test"] = all(
                    np.array_equal(v[:q],different.get(k,np.zeros_like(v))[:q]) for k,v in tracks.items()
                ) and all(np.array_equal(v[:q],tracks.get(k,np.zeros_like(v))[:q]) for k,v in different.items())
            json_write(out/"score.json",result)
            json_write(out/"limited_reference.json",{"meeting_start_s":start,"activity_spans_relative_s":spans,
                      "target_only_transcript":annotation,
                      "mixture":p.stem, "dataset_revision":"39ca3b88ffd4ce6078c1d045a3e8711fa2bd2f17"})
            print("REAL",p.stem,result,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--inputs",type=Path,required=True)
    p.add_argument("--models",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--condition",choices=["all","controlled","real"],default="all")
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--quick",action="store_true")
    run(p.parse_args())
