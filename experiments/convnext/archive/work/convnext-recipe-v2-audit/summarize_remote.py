import json, math, statistics, time, hashlib
from pathlib import Path

BASE=Path('/workspace/fast-audiovae-convnext-20260909-r9/training-runs/decoder-recipe-v2')

def percentile(xs,p):
    xs=sorted(xs)
    if not xs:return None
    v=(len(xs)-1)*p; i=int(v); j=min(i+1,len(xs)-1)
    return xs[i]+(xs[j]-xs[i])*(v-i)

def stats(xs):
    xs=list(xs)
    return {'count':len(xs),'mean':statistics.mean(xs),'median':statistics.median(xs),
            'p05':percentile(xs,.05),'p95':percentile(xs,.95),'min':min(xs),'max':max(xs)} if xs else {'count':0}

def read(p):return json.loads(p.read_text())

status=read(BASE/'status.json')
lines=(BASE/'metrics.jsonl').read_text().splitlines()
metrics=[]
for i,line in enumerate(lines):
    try:metrics.append(json.loads(line))
    except json.JSONDecodeError:
        if i!=len(lines)-1:raise
latest=metrics[-1]['step']
names=('teacher_waveform','teacher_mel','feature_matching','adversarial')
scalar_names=('teacher_waveform','teacher_mel','teacher_mel_linear','teacher_mel_log','total',
    'legacy/teacher_waveform','legacy/teacher_mel','legacy/total','discriminator','gradient_norm',
    'discriminator_gradient_norm','combined_norm','learning_rate','scored_samples','perceptual_fraction')
segments={}
for lo,hi in ((1,100),(101,200),(201,400),(401,500),(501,550),(551,600),(601,700),(701,800),
              (801,900),(901,1000),(1001,1100),(1101,1200),(1201,1300),(1301,1400),(max(1001,latest-99),latest)):
    selected=[m for m in metrics if lo<=m['step']<=hi]
    if not selected:continue
    body={'steps':[selected[0]['step'],selected[-1]['step']], 'metrics':{k:stats(m[k] for m in selected if k in m) for k in scalar_names},
        'branches':{},'generator_clip_fraction':sum(m['gradient_norm']>1 for m in selected)/len(selected),
        'generator_clip_factor':stats(min(1,1/m['gradient_norm']) for m in selected),
        'discriminator_clip_fraction':None,'discriminator_clip_factor':{}}
    disc=[m for m in selected if 'discriminator_gradient_norm' in m]
    if disc:
        body['discriminator_clip_fraction']=sum(m['discriminator_gradient_norm']>1 for m in disc)/len(disc)
        body['discriminator_clip_factor']=stats(min(1,1/m['discriminator_gradient_norm']) for m in disc)
    for n in names:
        active=[m for m in selected if n+'/scaled_norm' in m]
        body['branches'][n]={'active_steps':len(active),'saturations':sum(m.get(n+'/saturated',0) for m in active),
            'zero_norms':sum(m.get(n+'/zero_norm',0) for m in active),
            'nominal_share':stats(m[n+'/target_share'] for m in active),
            'realized_component_norm_share':stats(m[n+'/scaled_norm']/sum(m.get(other+'/scaled_norm',0) for other in names) for m in active),
            'raw_norm':stats(m[n+'/raw_norm'] for m in active),
            'raw_to_ema_ratio':stats(m[n+'/raw_norm']/m[n+'/ema_norm'] for m in active if m.get(n+'/ema_norm')),
            'scale':stats(m[n+'/scale'] for m in active)}
    body['loop_seconds_per_step']=(selected[-1]['seconds_in_training_loop']-(metrics[lo-2]['seconds_in_training_loop'] if lo>1 else 0))/len(selected)
    segments[f'{lo}-{hi}']=body

run=read(BASE/'run.json')
metadata=run['heldout_metadata']
heldout_identity=run['data']['heldout']['crops']
identity_summary={'kind':run['kind'],'recipe':run['recipe'],'model_config':run['model_config'],
    'runtime':run['runtime'],'implementation':run['implementation'], 'window_identity':run['window_identity'],
    'teacher_checkpoint_sha256':run['data']['teacher_checkpoint_sha256'],
    'teacher_state_sha256':run['data'].get('teacher_state_sha256'),
    'heldout_crops':heldout_identity,'heldout_metadata':metadata}
del run

def group_summary(rows):
    active=[r for r in rows if r['teacher_rms']>=1e-3]
    quiet=[q for r in rows for q in r['quiet_windows']['windows'] if q['is_quiet']]
    phases=[r['quiet_phase480']['residual_template_rms'] for r in rows if 'residual_template_rms' in r['quiet_phase480']]
    result={'rows':len(rows),'active_rows':len(active),'samples':sum(r['samples'] for r in rows),
        'waveform_cosine':stats(r['waveform_cosine'] for r in active),
        'rms_db_error':stats(r['rms_db_error'] for r in active),
        'absolute_rms_db_error':stats(abs(r['rms_db_error']) for r in active),
        'teacher_waveform':stats(r['teacher_waveform'] for r in rows),
        'teacher_mel':stats(r['teacher_mel'] for r in rows),
        'sample_weighted_raw_mae':sum(r['teacher_waveform_raw']*r['samples'] for r in rows)/sum(r['samples'] for r in rows),
        'sample_weighted_normalized_mae':sum(r['teacher_waveform']*r['samples'] for r in rows)/sum(r['samples'] for r in rows),
        'student_peak_abs':max(r['student_peak_abs'] for r in rows),
        'student_clipped_samples':sum(r['student_clipped_samples'] for r in rows),
        'student_exact_zero_samples':sum(r['student_exact_zero_samples'] for r in rows),
        'active_attenuated_more_than_1db':sum(r['rms_db_error']< -1 for r in active),
        'active_amplified_more_than_1db':sum(r['rms_db_error']>1 for r in active),
        'active_cosine_below_0_1':sum(r['waveform_cosine']<.1 for r in active),
        'active_no_better_than_silence':sum(r['waveform_to_silence_error_ratio']>=1 for r in active),
        'quiet_windows':len(quiet),'quiet_failed':sum(not q['passed'] for q in quiet),
        'quiet_residual_rms':stats(q['residual_rms'] for q in quiet),
        'quiet_output_rms':stats(q['student_rms'] for q in quiet),
        'quiet_teacher_rms':stats(q['teacher_rms'] for q in quiet),
        'quiet_output_over_limit':stats(q['student_rms']/q['output_rms_limit'] for q in quiet),
        'quiet_residual_over_limit':stats(q['residual_rms']/q['residual_limit'] for q in quiet),
        'quiet_output_db_gain':stats(20*math.log10(max(q['student_rms'],1e-15)/max(q['teacher_rms'],1e-15)) for q in quiet),
        'quiet_phase480_residual_rms':stats(phases)}
    return result

evaluations={}
for step in (500,1000):
    source=BASE/f'evaluation-step{step:06d}.json'
    evaluation=read(source); rows=evaluation['rows']
    grouped={'all':rows,'position/beginning':[r for r in rows if r['start_frame']==0],
             'position/interior':[r for r in rows if r['start_frame']>0]}
    for r in rows:
        meta=metadata[r['source_id']]
        for kind in ('condition','language'):
            grouped.setdefault(kind+'/'+str(meta[kind]),[]).append(r)
    quiet=[q for r in rows for q in r['quiet_windows']['windows'] if q['is_quiet']]
    quiet_bins={}
    for name,lo,hi in (('below_minus100',0,1e-5),('minus100_to_minus80',1e-5,1e-4),('minus80_to_minus60',1e-4,1e-3+1e-12)):
        qs=[q for q in quiet if lo<=q['teacher_rms']<hi]
        quiet_bins[name]={key:stats(q[key] for q in qs) for key in ('teacher_rms','student_rms','residual_rms','residual_limit','output_rms_limit')}
        quiet_bins[name]['db_gain']=stats(20*math.log10(max(q['student_rms'],1e-15)/max(q['teacher_rms'],1e-15)) for q in qs)
    evaluations[str(step)]={'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'groups':{k:group_summary(v) for k,v in grouped.items()},'quiet_bins':quiet_bins,
        'rows':[dict({k:v for k,v in r.items() if not isinstance(v,(list,dict))}, **metadata[r['source_id']]) for r in rows],
        'model_states':sorted({r['model_state_sha256'] for r in rows}),
        'row_steps':sorted({r['evaluated_step'] for r in rows}),
        'fixed_statistics_all':all(r['fixed_statistics'] for r in rows),
        'gate_passed':evaluation['gate']['passed']}
calibration=read(BASE/'calibration.json')
c=calibration['calibration']
calibration_summary={k:v for k,v in calibration.items() if k!='calibration'}
calibration_summary['completed_step']=c['completed_step']
calibration_summary['report']={k:v for k,v in c['report'].items() if k!='provenance'}
boundary={str(s):metrics[s-1] for s in (1,100,499,500,501,502,550,750,999,1000,1001,latest) if s<=latest}
result={'captured_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'status_before_metrics_read':status,
    'metrics_cutoff_step':latest,'metrics_count':len(metrics),'finite_all_numeric_metrics':all(math.isfinite(v) for m in metrics for v in m.values() if isinstance(v,(int,float))),
    'step_sequence_exact':all(m['step']==i+1 for i,m in enumerate(metrics)),
    'discriminator_schedule_exact':all(m['discriminator_updates']==max(0,m['step']-500) for m in metrics),
    'examples_32_all':all(m['examples']==32 for m in metrics),
    'stages_perceptual_fraction_exact':all(abs(m['perceptual_fraction']-min(1,max(0,(m['step']-500)/500)))<1e-12 for m in metrics),
    'branches_ever_saturated':{n:sum(m.get(n+'/saturated',0) for m in metrics) for n in names},
    'branches_zero_norm':{n:sum(m.get(n+'/zero_norm',0) for m in metrics) for n in names},
    'interpretation':{'component_norm_share':'Ratio of scaled individual output-gradient norm magnitudes, not final vector projection or parameter-gradient contribution.',
        'parameter_clipping':'Global clip norm threshold1.0; preclip norm is logged. Scale below1 does not directly determine AdamW or Muon parameter displacement.',
        'total':'New train/total is raw teacher waveform+mel diagnostic; excludes adversarial, feature matching and gradient balancing.'},
    'training_segments':segments,'boundary_metrics':boundary,'identity':identity_summary,'calibration':calibration_summary,'evaluations':evaluations}
print(json.dumps(result,sort_keys=True,allow_nan=False))
