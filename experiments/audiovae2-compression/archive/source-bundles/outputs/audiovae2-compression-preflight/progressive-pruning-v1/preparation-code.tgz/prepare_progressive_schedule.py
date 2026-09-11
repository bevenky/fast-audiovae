"""Extend the original calibration selection into nested, one-boundary cuts."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
import time

import torch
import run_pilot as base
import settings_screen as screen
from progressive_model import validate_schedule


def main():
    parser = argparse.ArgumentParser()
    for name in ('manifest','source-plan','original-selection','assets','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Preserve existing preparation; choose a fresh directory')
    args.out.mkdir(parents=True)
    base.policy()
    manifest,pools,cache_receipt=base.load_data(args.manifest)
    prior=json.loads(args.original_selection.read_text())
    source_plan=json.loads(args.source_plan.read_text())
    teacher=base.FrozenAudioVAE2.from_files(args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',device='cuda')
    if any(p.requires_grad for p in teacher.parameters()):raise RuntimeError('Teacher must be frozen')
    versions={name:(value.data_ptr(),value._version) for name,value in teacher.state_dict().items()}
    grams={'stage2_output':torch.zeros(512,512,device='cuda',dtype=torch.float64),
           'stage3_output':torch.zeros(256,256,device='cuda',dtype=torch.float64)}
    checks=[];started=time.monotonic()
    with torch.no_grad():
        for number,crop in enumerate(pools['calibration']):
            z,target,_,spans=base.batch([crop])
            trace=base.teacher_forward(teacher,z)
            a,b=spans[0]
            actual=trace['waveform'][...,a:b]
            expected=target[...,a:b]
            check={'source_id':crop['source_id'],'bitwise_equal':torch.equal(actual,expected),
                   'within_tolerance':torch.allclose(actual,expected,atol=1e-5,rtol=1e-4),
                   'max_abs':float((actual-expected).abs().max())}
            if not check['within_tolerance']:raise RuntimeError('Original teacher/cache comparison failed')
            checks.append(check)
            for name,rate in (('stage2_output',1200),('stage3_output',6000)):
                value=trace[name][0,:,a//(48000//rate):b//(48000//rate)]
                generator=torch.Generator().manual_seed(20260910+number)
                indices=torch.randperm(value.shape[-1],generator=generator)[:min(256,value.shape[-1])].to(value.device)
                sample=value[:,indices].double()
                grams[name].add_(sample@sample.T)
    final2=base.pivoted_subset(grams['stage2_output'],256)
    final3=base.pivoted_subset(grams['stage3_output'],128)
    if final2!=prior['stage2_indices'] or final3!=prior['stage3_indices']:
        raise RuntimeError('Original calibration selection did not reproduce; do not alter the target coordinates')
    middle2=base.pivoted_subset(grams['stage2_output'],384)
    middle3=base.pivoted_subset(grams['stage3_output'],192)
    steps=[{'stage2_indices':list(range(512)),'stage3_indices':list(range(256))},
           {'stage2_indices':middle2,'stage3_indices':list(range(256))},
           {'stage2_indices':middle2,'stage3_indices':middle3},
           {'stage2_indices':final2,'stage3_indices':middle3},
           {'stage2_indices':final2,'stage3_indices':final3}]
    steps=validate_schedule(steps,final_selection=prior)
    if versions!={name:(value.data_ptr(),value._version) for name,value in teacher.state_dict().items()}:
        raise RuntimeError('Teacher state changed during selection')
    schedule={'version':'audiovae2_progressive_schedule_v1','steps':steps,
        'teacher_source_sha256':base.sha(args.assets/'audio_vae_v2.py'),
        'teacher_checkpoint_sha256':base.sha(args.assets/'audiovae.pth'),
        'original_selection_sha256':base.sha(args.original_selection),
        'manifest_sha256':base.sha(args.manifest),'source_plan_sha256':base.sha(args.source_plan),
        'method':prior['method'],'calibration_source_ids':[c['source_id'] for c in pools['calibration']],
        'original_endpoint_reproduced':True,'teacher_frozen':True,
        'prepare_source_sha256':base.sha(__file__)}
    schedule['identity_sha256']=screen.digest(schedule)
    base.write_json(args.out/'schedule.json',schedule)
    rows=manifest['splits']['fit']['rows']+source_plan['rows'][:9000]
    coverage=Counter();expressive=Counter();datasets=Counter()
    for row in rows:
        source=row['manifest_row'];coverage[source.get('language','unknown')]+=1
        datasets[source.get('dataset','unknown')]+=1
        for label in source.get('verified_source_labels',[]) or []:expressive[label]+=1
    result={'schedule_sha256':base.sha(args.out/'schedule.json'),'identity_sha256':schedule['identity_sha256'],
        'widths':[[len(s['stage2_indices']),len(s['stage3_indices'])] for s in steps],
        'calibration_sources':len(checks),'teacher_checks':checks,'elapsed_seconds':time.monotonic()-started,
        'fitting_sources':len(rows),'languages':dict(coverage),'datasets':dict(datasets),
        'expressive_source_tags':dict(expressive),'old_selection_reproduced':True,'teacher_frozen':True,
        'coverage_interpretation':'Source labels and counts, not durations of individual expressive events.'}
    base.write_json(args.out/'preparation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('teacher_checks','languages','datasets','expressive_source_tags')}))


if __name__=='__main__':main()
