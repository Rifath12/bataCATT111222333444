"""Recognize three speakers from their own synchronized microphone recordings."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import time

from mono_text import export_rows

import numpy as np
from scipy.ndimage import binary_closing, binary_opening, maximum_filter1d, uniform_filter1d
import soundfile as sf

RATE=16000
HOP=512
CONTEXT=64
SEATS=tuple("ABC")


def calculate_features(files, model_path):
    """Run one recurrent speech detector per channel in a small batch."""
    import onnxruntime as ort

    file_ids = [(info.st_dev, info.st_ino) for info in (Path(path).stat() for path in files)]
    if len(file_ids) != 3 or len(set(file_ids)) != 3:
        raise ValueError("Use three distinct synchronized microphone recordings")
    options=ort.SessionOptions()
    options.intra_op_num_threads=1
    options.inter_op_num_threads=1
    session=ort.InferenceSession(str(model_path),providers=["CPUExecutionProvider"],sess_options=options)
    with ExitStack() as stack:
        handle_list = [stack.enter_context(sf.SoundFile(path)) for path in files]
        info=[(f.samplerate,f.channels,len(f)) for f in handle_list]
        if len(info)!=3 or any((r,c)!=(RATE,1) or n<1 for r,c,n in info):
            raise ValueError("Supply three nonempty 16 kHz mono inputs")
        lengths=np.array([n for _,_,n in info])
        if len(set(lengths)) != 1:
            raise ValueError("The tracks must already be synchronized and have the same length")
        n=lengths[0]
        batch=3
        state=np.zeros((2,batch,128),np.float32)
        head=np.zeros((batch,CONTEXT),np.float32)
        steps=(n+HOP-1)//HOP
        power=np.empty((batch,steps),np.float32)
        prob=np.empty_like(power)
        # Read in blocks and carry the VAD state across read boundaries.
        sr=np.array(RATE,dtype="int64")
        block_frames=128
        k=0
        for i in range(0,n,HOP*block_frames):
            block=np.stack([f.read(HOP*block_frames,dtype="float32") for f in handle_list])
            if not np.isfinite(block).all():
                raise ValueError("Microphone audio contains a nonfinite sample")
            block=np.pad(block,((0,0),(0,(-block.shape[1])%HOP)))
            for t in range(0,block.shape[1],HOP):
                x=block[:,t:t+HOP]
                power[:,k]=(x*x).mean(axis=1)
                data=np.concatenate((head,x),axis=1)
                pred,state=session.run(None,dict(input=data,state=state,sr=sr))
                prob[:,k]=pred.ravel()
                head=data[:,-CONTEXT:]
                k+=1
        return power,prob,int(n)


def voice_mask(power, probability, bleed_db=15.0, min_prob=.5):
    """Reject bleed using local microphone dominance, retaining simultaneous sources."""
    power=uniform_filter1d(power,3,axis=-1,mode="nearest")
    # Estimate each microphone's loud, reliable speech level independently.
    weights=np.empty(len(power),dtype="float32")
    for i, p in enumerate(power):
        selected=p[probability[i]>=.7]
        weights[i]=max(1e-9,float(np.quantile(selected,.9))) if len(selected) else max(1e-9,float(np.max(p)))
    scaled=power/weights[:,None]
    band=10**(-bleed_db/10.)
    loudest=np.max(scaled,axis=0)
    trigger=(probability>=min_prob)&(scaled>=np.maximum(1e-5,band*loudest))
    for i,m in enumerate(trigger):
        cleaned=binary_opening(m,iterations=1)
        closed=binary_closing(cleaned,iterations=2,border_value=0)
        # A small lead and tail retain consonants on the original synchronized clock.
        trigger[i]=maximum_filter1d(closed,size=7,mode="constant")
    return trigger


def timestamps(activity,length):
    marks=[]
    for name,on in zip(SEATS,activity,strict=True):
        edges=np.flatnonzero(np.diff(np.r_[False,on,False]))*HOP
        for a,b in zip(edges[::2],edges[1::2],strict=True):
            marks.append({"speaker":name,"start":a/RATE,"end":min(b,length)/RATE})
    return sorted(marks,key=lambda r:(r["start"],r["speaker"]))


def copy_speaker_audio(files, output):
    """Write one aligned PCM file per microphone while preserving all captured speech."""
    targets = [output / f"speaker_{label}.wav" for label in SEATS]
    sources = {source.resolve() for source in files}
    if any(target.resolve() in sources and target.resolve() != source.resolve()
           for target, source in zip(targets, files, strict=True)):
        raise ValueError("An output filename already belongs to a different input microphone")
    for label, source, target in zip(SEATS, files, targets, strict=True):
        if target.resolve() == source.resolve():
            continue
        with NamedTemporaryFile(prefix=f".speaker_{label}.", suffix=".wav", dir=output, delete=False) as handle:
            pending = Path(handle.name)
        try:
            with sf.SoundFile(source) as reader:
                subtype = reader.subtype if reader.subtype in {"PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"} else "FLOAT"
                with sf.SoundFile(pending, mode="w", samplerate=RATE, channels=1, subtype=subtype) as writer:
                    for samples in reader.blocks(blocksize=65536, dtype="float64"):
                        writer.write(samples)
            pending.replace(target)
        finally:
            pending.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("microphones",nargs=3,type=Path,metavar="WAV")
    for key in ("vad","asr","out"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--reference",type=Path,help="Optional evaluation-only TextGrid")
    parser.add_argument("--profile",type=Path,help="Choose a separately calibrated legacy or adaptive activity profile")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--copy-audio", action="store_true", help="Also save speaker_A/B/C.wav")
    parser.add_argument("--only-activity",action="store_true",help="Stop after producing blind timing")
    args=parser.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    started=time.perf_counter()
    power, prob, length = calculate_features(args.microphones, args.vad)
    thresholds=json.loads(args.profile.read_text()) if args.profile else {}
    adaptive = thresholds.get("adaptive_activity")
    if adaptive is None:
        activity = voice_mask(
            power, prob, thresholds.get("bleed_db", 15.), thresholds.get("minimum_probability", .5)
        )
    else:
        from close_activity import activity_mask

        activity = activity_mask(power, prob, **adaptive)
    marks=timestamps(activity,length)
    timing=dict(duration_s=length/RATE,speaker_ids=list(SEATS),segments=marks, activity_profile="adaptive" if adaptive is not None else "fixed")
    (args.out/"speaker_timing.json").write_text(json.dumps(timing,indent=2)+"\n")
    timing_wall=time.perf_counter()-started
    print("blind timing",len(marks),round(timing_wall,3),flush=True)
    if args.copy_audio:
        copy_speaker_audio(args.microphones, args.out)
    if args.only_activity:
        return
    from recognize import transcribe,textgrid,text_scores
    rows,asr_wall=transcribe(dict(zip(SEATS,args.microphones,strict=True)),marks,args.asr/"model.int8.onnx",
        args.asr/"tokens.txt",threads=args.threads,target_rms=.035,backend="sensevoice")
    export_rows(rows,args.out)
    result=dict(vad_seconds=timing_wall,asr_seconds=asr_wall,transcribed_turns=len(rows))
    if args.reference:
        from diarize import score
        truth = textgrid(args.reference)
        result["transcription"]=text_scores(truth,rows)
        result["activity"]=score(truth,marks,length/RATE)
    (args.out/"scores.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result,flush=True)


if __name__=="__main__":
    main()
