"""Read saved pilot metadata only; no checkpoint load or inference."""
import collections, hashlib, json, math, pathlib, statistics
BASE=pathlib.Path('/workspace/fast-audiovae-convnext-20260909-r5/training-runs/representative-reconstruction-v1')
META=json.loads(pathlib.Path('/workspace/fast-audiovae-convnext-20260909-r5/review-group-map.json').read_text())

def mean(a): return statistics.fmean(a) if a else None
def percentile(xs,q):
 a=sorted(xs)
 if not a:return None
 p=(len(a)-1)*q;i=int(p);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(p-i)
def stats(xs):return {'n':len(xs),'mean':mean(xs),'p10':percentile(xs,.1),'median':percentile(xs,.5),'p90':percentile(xs,.9),'min':min(xs) if xs else None,'max':max(xs) if xs else None}
def db(a,b):return 20*math.log10(max(a,1e-15)/max(b,1e-15))
def cosine_summary(rows):
 nq=[r for r in rows if r['teacher_rms']>=.001]
 bysrc=collections.defaultdict(list)
 for r in nq:bysrc[r['source_id']].append(r['waveform_cosine'])
 return {'crops':len(rows),'sources':len(set(r['source_id'] for r in rows)),'nonquiet_count':len(nq),'cosine':stats([r['waveform_cosine'] for r in nq]),'source_macro_cosine':mean([mean(v) for v in bysrc.values()]),'sample_weighted_cosine':sum(r['waveform_cosine']*r['samples'] for r in nq)/sum(r['samples'] for r in nq) if nq else None,'rms_db_error':stats([r['rms_db_error'] for r in nq]),'within_1db':sum(abs(r['rms_db_error'])<=1 for r in nq),'too_quiet_below_minus1db':sum(r['rms_db_error']< -1 for r in nq),'too_loud_above1db':sum(r['rms_db_error']>1 for r in nq),'waveform_mean':mean([r['teacher_waveform'] for r in rows]),'mel_mean':mean([r['teacher_mel'] for r in rows]),'error_to_silence_ratio':stats([r['waveform_to_silence_error_ratio'] for r in nq])}
def window_summary(ws):
 return {'windows':len(ws),'samples':sum(w['valid_samples'] for w in ws),'failed':sum(w.get('passed') is False for w in ws),'teacher_rms':stats([w['teacher_rms'] for w in ws]),'student_rms':stats([w['student_rms'] for w in ws]),'residual_rms':stats([w['residual_rms'] for w in ws]),'student_to_teacher_db':stats([db(w['student_rms'],w['teacher_rms']) for w in ws]),'residual_limit_multiple':stats([w['residual_rms']/w['residual_limit'] for w in ws]),'output_energy_failure_count':sum(w['student_rms']>w['output_rms_limit'] for w in ws),'residual_failure_count':sum(w['residual_rms']>w['residual_limit'] for w in ws)}

def groupname(src):
 g=META['source_groups'][src]
 return 'speech' if g.startswith('speech:') else g.removeprefix('event:')

all_steps=[];compact_rows=[]
for path in sorted(BASE.glob('evaluation-step*.json')):
 d=json.loads(path.read_text()); rows=d['rows'];ws=[]
 for r in rows:
  for w in r['quiet_windows']['windows']:
   ws.append(dict(w,source_id=r['source_id'],start_frame=r['start_frame'],language=META['source_languages'][r['source_id']],condition=groupname(r['source_id'])))
  compact_rows.append(dict({k:v for k,v in r.items() if k!='quiet_windows'},condition=groupname(r['source_id']),language=META['source_languages'][r['source_id']],quiet_summary=window_summary([w for w in r['quiet_windows']['windows'] if w['is_quiet']])))
 quiet=[w for w in ws if w['is_quiet']]
 bins={}
 for name,low,hi in [('below_minus100',0,1e-5),('minus100_to_minus80',1e-5,1e-4),('minus80_to_minus60',1e-4,1e-3),('minus60_to_minus40',1e-3,.01),('at_least_minus40',.01,float('inf'))]:bins[name]=window_summary([w for w in ws if low<=w['teacher_rms']<hi])
 conditions={g:cosine_summary([r for r in rows if groupname(r['source_id'])==g]) for g in sorted(set(groupname(r['source_id']) for r in rows))}
 langs={g:cosine_summary([r for r in rows if META['source_languages'][r['source_id']]==g]) for g in sorted(set(META['source_languages'][r['source_id']] for r in rows))}
 all_steps.append({'step':d['step'],'source_file':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'groups':d['groups'],'all':cosine_summary(rows),'language_macro_cosine_excluding_und':mean([x['cosine']['mean'] for k,x in langs.items() if k!='und']),'conditions':conditions,'languages':langs,'quiet':window_summary(quiet),'quiet_by_condition':{g:window_summary([w for w in quiet if w['condition']==g]) for g in conditions},'window_rms_bins':bins,'beginning':cosine_summary([r for r in rows if r['start_frame']==0]),'interior':cosine_summary([r for r in rows if r['start_frame']!=0]),'quiet_beginning':window_summary([w for w in quiet if w['start_frame']==0]),'quiet_interior':window_summary([w for w in quiet if w['start_frame']!=0]),'worst_quiet_windows_by_output_db':sorted(quiet,key=lambda w:db(w['student_rms'],w['teacher_rms']),reverse=True)[:15]})
metrics=[json.loads(x) for x in (BASE/'metrics.jsonl').read_text().splitlines()]
segments=[]
for lo,hi in [(1,200),(201,400),(401,600),(601,800),(801,1000),(1001,1009)]:
 m=[r for r in metrics if lo<=r['step']<=hi]
 segments.append({'steps':[lo,hi],'entries':len(m),'keys':list(m[0]) if m else [],'teacher_waveform':stats([r['teacher_waveform'] for r in m]),'teacher_mel':stats([r['teacher_mel'] for r in m]),'gradient_norm':stats([r['gradient_norm'] for r in m]),'teacher_waveform_scaled_norm':stats([r['teacher_waveform/scaled_norm'] for r in m]),'teacher_mel_scaled_norm':stats([r['teacher_mel/scaled_norm'] for r in m if 'teacher_mel/scaled_norm' in r]),'last':m[-1] if m else None})
ex=[json.loads(x) for x in (BASE/'exposure.jsonl').read_text().splitlines()]
summary={'source':'saved JSON only; no inference/checkpoint load or training','status':json.loads((BASE/'status.json').read_text()),'teacher_coverage':json.loads((BASE/'teacher-coverage.json').read_text()),'steps':all_steps,'training_segments':segments,'exposure':{'entries':len(ex),'first':ex[0],'last':ex[-1],'metric_entries':len(metrics),'metric_steps_unique':len(set(r['step'] for r in metrics))}}
out=pathlib.Path('/workspace/fast-audiovae-convnext-20260909-r5/completed-review');out.mkdir(exist_ok=True)
(out/'completed-metrics-summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
(out/'completed-crop-metrics.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in compact_rows))
print(json.dumps({'output':str(out),'steps':[{'step':s['step'],'cosine':s['all']['cosine']['mean'],'waveform':s['all']['waveform_mean'],'mel':s['all']['mel_mean'],'quiet_residual_mean':s['quiet']['residual_rms']['mean'],'quiet_gain_median_db':s['quiet']['student_to_teacher_db']['median'],'quiet_output_failure':s['quiet']['output_energy_failure_count']} for s in all_steps]},indent=2))
