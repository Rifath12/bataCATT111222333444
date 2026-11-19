"""Estimate how much additional microphone isolation an A/B/C setup needs."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from calibrate_wiring import prepare
from waveform_acceptance import LABELS, evaluate_case


RATE = 16000
TARGETS = (0.96, 0.98)


def isolation_needed(target, interferers, threshold):
    """Largest surviving leakage amplitude that passes on the positive-gain branch."""
    voice = target - target.mean()
    competing = interferers - interferers.mean(axis=1, keepdims=True)
    noise = competing.sum(axis=0)
    norm = np.linalg.norm(voice)
    if norm <= 1e-10:
        raise ValueError("Target waveform has no energy during common speech")
    along = float(voice @ noise) / norm**2
    away = float(np.linalg.norm(noise - along * voice)) / norm
    allowed = np.sqrt((1-threshold) / threshold)
    required = max(1.0, away/allowed - along)
    # Maintain the same polarity and the target gain floor from the test protocol.
    required = max(required, -along/(1-10**(-12/20)))
    captured_db = float(20*np.log10(required))
    # Triangle bound allows for arbitrary relative phase between the captured
    # interferers, at the measured per-source magnitude.
    upper = float(np.linalg.norm(competing, axis=1).sum()) / norm
    robust_db = float(20*np.log10(max(1.0, upper/np.sqrt(1-threshold))))
    return captured_db, robust_db


def planning_cases(images, activity, quiet_levels):
    exact = activity.all(axis=0)
    if exact.sum() < RATE*0.25:
        raise ValueError("The solo takes need at least 0.25 seconds of common speech after alignment")
    settings = [(None, 0.0)] + [(i, gap) for gap in quiet_levels for i in range(len(LABELS))]
    for quiet, gap in settings:
        gains = np.ones(len(LABELS))
        if quiet is not None:
            gains[quiet] = 10**(-gap/20)
        contributions = images*gains[:,None,None]
        direct = contributions[np.arange(3),:,np.arange(3)]
        yield quiet,gap,contributions,direct,exact


def estimate(takes, sample_rate, quiet_levels=(6, 12, 20)):
    """Measure each setting and test a hypothetical change to cross-mic leakage."""
    if sample_rate != RATE:
        raise ValueError("Use 16 kHz aligned input WAVs")
    if not quiet_levels or any(not np.isfinite(g) or not 0 < g <= 40 for g in quiet_levels):
        raise ValueError("Quiet-voice gaps must be above 0 and at most 40 dB")
    images,activity,leakage = prepare(takes,sample_rate)
    rows = []
    overall = {str(round(t*100)): {s:0.0 for s in LABELS} for t in TARGETS}
    robust = {k:dict(v) for k,v in overall.items()}
    for quiet,gap,contributions,direct,exact in planning_cases(images,activity,quiet_levels):
        name = "balanced" if quiet is None else f"{LABELS[quiet]}_quiet_{gap:g}dB"
        mixture = contributions.sum(axis=0).T
        score = evaluate_case(name,RATE,direct,mixture,activity)
        per_speaker = {}
        for i,speaker in enumerate(LABELS):
            leak = np.delete(contributions[:,exact,i],i,axis=0)
            voice = direct[i,exact]
            powers = np.mean((leak-leak.mean(axis=1,keepdims=True))**2,axis=1)
            culprit = [x for x in LABELS if x!=speaker][int(np.argmax(powers))]
            record = dict(
                measured_fraction=score["voices"]["exact_three"][speaker]["target_aligned_fraction"],
                strongest_leak_from=culprit if powers.max()>1e-15 else None,
                additional_isolation_db={}, phase_margin_db={},
            )
            for threshold in TARGETS:
                key=str(round(threshold*100))
                observed, conservative = isolation_needed(voice,leak,threshold)
                record["additional_isolation_db"][key] = observed
                record["phase_margin_db"][key] = conservative
                overall[key][speaker] = max(overall[key][speaker],observed)
                robust[key][speaker] = max(robust[key][speaker],conservative)
            per_speaker[speaker] = record
        rows.append(dict(case=name, quiet_speaker=None if quiet is None else LABELS[quiet],
                         gap_db=gap, voices=per_speaker))
    # Validate the algebra on the actual recorded components. This scales only
    # modeled acoustic bleed. It is not a filter for the original mixture.
    model = {}
    for key,goal in overall.items():
        cases=[]
        scales=10**(-np.array([goal[s] for s in LABELS])/20)
        for quiet,gap,contributions,direct,exact in planning_cases(images,activity,quiet_levels):
            leak=contributions.sum(axis=0).T-direct
            estimate_audio=direct + scales[:,None]*leak
            case=evaluate_case("simulated",RATE,direct,estimate_audio,activity)
            cases.append(case)
        model[key] = dict(
            trials_passing=sum(
                all(m["target_aligned_fraction"]>=float(key)/100-1e-10 and
                    m["retained_active_fraction"] is not None and
                    m["retained_active_fraction"]>=.96 for m in case["voices"]["exact_three"].values())
                for case in cases
            ),
            trials=len(cases),
        )
    return dict(
        sample_rate=RATE,
        anti_leakage_db_by_mic=overall,
        phase_conservative_db_by_mic=robust,
        source_to_microphone_db={
            speaker:{mic:float(leakage[i,j]) for j,mic in enumerate(LABELS)}
            for i,speaker in enumerate(LABELS)
        },
        calibration_cases=rows,
        simulation_hypothetical_bleed_reduction=model,
    )


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("solo",nargs=3,type=Path,metavar="SOLO_WAV",
                   help="Separate A/B/C solo takes, each containing all three microphone tracks")
    p.add_argument("--quiet-db",type=float,nargs="+",default=(6,12,20))
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    if len({path.resolve() for path in a.solo})!=3:
        p.error("provide separate solo recordings for A, B and C")
    recordings, rates=zip(*(sf.read(path,dtype="float64",always_2d=True) for path in a.solo))
    if len(set(rates))!=1:
        p.error("sampling rates differ")
    report=estimate(recordings,rates[0],a.quiet_db)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    for goal,row in report["anti_leakage_db_by_mic"].items():
        details=", ".join(f"{s} +{v:.1f} dB" for s,v in row.items())
        print(f"{goal}% target estimated additional isolation: {details}")


if __name__=="__main__":
    main()
