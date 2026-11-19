"""Score only controlled windows inside a long partly natural headset replay."""
from argparse import ArgumentParser
from contextlib import ExitStack
from pathlib import Path
import json

import numpy as np
import soundfile as sf

from waveform_acceptance import evaluate_case, signal_metrics, gain_coverage, threshold_pass

RATE = 16000
NAMES = "ABC"


def load_wave(wavs, start, length):
    for wave in wavs:
        wave.seek(start)
    return np.stack([wave.read(length,dtype="float32") for wave in wavs])


def score(args):
    manifest = json.loads(args.manifest.read_text())
    samples = manifest["samples"]
    scenes, pooled_ref, pooled_track = [], [], []
    local_time = {k:0.0 for k in ("96","98")}
    local_eligible = 0.
    retained = np.zeros(3, np.int64)
    eligible = retained.copy()
    by_count = {count:[] for count in (1,2)}
    with ExitStack() as stack:
        wavs = [stack.enter_context(sf.SoundFile(p)) for p in args.wavs]
        if any(w.samplerate != RATE or w.channels != 1 or len(w) != samples for w in wavs):
            raise ValueError("Unexpected output length, channel count or sample rate")
        for group in manifest["groups"]:
            with np.load(args.references/(group["id"]+".npz")) as z:
                ref, active=z["references"], z["activity"]
            out = load_wave(wavs, group["destination_start"], group["length"])
            count = sum(entry is not None for entry in group["members"])
            if count == 3:
                triple = active.all(axis=0)
                if triple.sum() < .03 * RATE:
                    continue
                e = evaluate_case(group["id"], RATE, ref, out, active, group="M8017")
                exact = e["voices"]["exact_three"]
                rows=[exact[s] for s in NAMES]
                if any(r is None for r in rows):
                    raise RuntimeError("Missing reference energy")
                for i,r in enumerate(rows):
                    retained[i]+=r["retained_samples"]; eligible[i]+=r["eligible_samples"]
                for local in e["exact_three_windows"]:
                    local_eligible+=local["exact_three_s"]
                    for threshold in local_time:
                        if local["passes_all_three_with_level"][threshold]:
                            local_time[threshold]+=local["exact_three_s"]
                scenes.append({"id":group["id"],"anchor_sample":group["destination_start"],"quiet_speaker":group["quiet_speaker"],
                    "nominal_quiet_db":group["quiet_level_db"],"exact_three_seconds":e["exact_three_s"],
                    "alignment":[round(r["target_aligned_fraction"],8) for r in rows],
                    "retention":[round(r["retained_active_fraction"],8) if r["retained_active_fraction"] is not None else None for r in rows],
                    "pass":e["passes_exact_three"]})
                pooled_ref.append(ref[:,triple]); pooled_track.append(out[:,triple])
            else:
                members=[i for i,s in enumerate(group["members"]) if s]
                mask=active[members].all(axis=0)
                ref64=ref.astype(np.float64); out64=out.astype(np.float64)
                if mask.sum()<.03*RATE: continue
                metrics=[]
                for i in members:
                    record=signal_metrics(ref64[:,mask],out64[i,mask],i)
                    if record:
                        record.update(gain_coverage(ref64[i],out64[i],mask,1600,480))
                        metrics.append(record)
                other=[i for i in range(3) if i not in members]
                target_power=np.mean(ref64[members][:,mask]**2)
                spill=np.mean(out64[other][:,mask]**2)
                by_count[count].append((min(m["target_aligned_fraction"] for m in metrics),
                    threshold_pass(metrics,.96),threshold_pass(metrics,.98),10*np.log10(max(spill,1e-15)/target_power)))
    ref_all=np.concatenate(pooled_ref,axis=1).astype(np.float64)
    track_all=np.concatenate(pooled_track,axis=1).astype(np.float64)
    by_level={}
    for level in [0,-6,-12,-20]:
        subset=[r for r in scenes if r["nominal_quiet_db"]==level]
        by_level[str(level)]={"cases":len(subset),
            "joint_96":sum(r["pass"]["96"] for r in subset),"joint_98":sum(r["pass"]["98"] for r in subset)}
    q=np.array([min(r["alignment"]) for r in scenes])
    result={"cases":len(scenes),"exact_three_s":sum(r["exact_three_seconds"] for r in scenes),
        "joint_96":sum(r["pass"]["96"] for r in scenes),"joint_98":sum(r["pass"]["98"] for r in scenes),
        "weakest_track_min_p10_median":np.quantile(q,[0,.1,.5]).tolist(),
        "worst_case":scenes[int(np.argmin(q))],"by_nominal_quiet_level_db":by_level,
        "pooled_alignment_ABC":[signal_metrics(ref_all,track_all[i],i)["target_aligned_fraction"] for i in range(3)],
        "pooled_activity_retained_ABC":(retained/eligible).tolist(),
        "eligible_exact_three_s_in_250ms_windows":local_eligible,
        "all_three_time_fraction_in_eligible_250ms_windows":{key:value/local_eligible for key,value in local_time.items()},
        "solo_and_dual_cases":{},
        "event_results":scenes}
    for count,data in by_count.items():
        result['solo_and_dual_cases'][str(count)]={
            'cases':len(data),'joint_active_96':sum(r[1] for r in data),'joint_active_98':sum(r[2] for r in data),
            'median_inactive_output_relative_to_active_reference_db':float(np.median([r[3] for r in data]))}
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(args.out, {k:v for k,v in result.items() if k not in ('event_results','worst_case')},flush=True)


if __name__ == "__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--references',type=Path,required=True)
    p.add_argument('--wavs',nargs=3,type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    score(p.parse_args())
