"""Read-only completed-run/source audit; CPU tensors only, no neural forwards."""
from __future__ import annotations
import argparse,collections,hashlib,json,math,os,re,statistics,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from fresh_training_data import FreshTrainingData,sha,digest

INDIC=('as','bn','brx','doi','gu','hi','kn','ks','kok','mai','ml','mni','mr','ne','or','pa','sa','sat','sd','ta','te','ur')
def read(p):return json.loads(Path(p).read_text())
def stat(values):
 if not values:return None
 return {'count':len(values),'minimum':min(values),'median':statistics.median(values),'mean':statistics.mean(values),'maximum':max(values)}
def mix(rows,label_map):
 result={'sources':len(rows),'valid_seconds':sum(r['valid_scored_samples'] for r in rows)/48000,
         'by_language':{},'by_dataset':{},'validated_source_labels':{},'labeled_sources':[]}
 union=set();broad=0;unknownlang=0;languages=set()
 for row in rows:
  source=row['manifest_row'];sid=row['source_id'];duration=row['valid_scored_samples']/48000
  language=row.get('normalized_language',source['language'].split('_')[0]);languages.add(language)
  for field,key in [('by_language',language),('by_dataset',source['dataset'])]:
   value=result[field].setdefault(key,{'sources':0,'valid_seconds':0.});value['sources']+=1;value['valid_seconds']+=duration
  labels=list(label_map.get(sid,{}).get('validated_labels',[]))
  if source['dataset']=='thorsten_emotional' and re.search(r'(^|/)whisper/',sid.split(':',1)[-1]):labels.append('explicit_whisper_style')
  if labels:
   union.add(sid);result['labeled_sources'].append({'source_id':sid,'labels':labels,'crop_start_seconds':row['start_frame']*.04,'crop_valid_seconds':duration,'full_source_duration_seconds':source['duration_seconds']})
  for label in labels:
   value=result['validated_source_labels'].setdefault(label,{'sources':0,'scored_seconds_from_labeled_sources':0.});value['sources']+=1;value['scored_seconds_from_labeled_sources']+=duration
  if source['dataset'] not in ('librispeech','fleurs','indicvoices'):broad+=duration
 result.update(indic_languages_present=[l for l in INDIC if l in languages],indic_languages_missing=[l for l in INDIC if l not in languages],
               broad_expressive_dataset_seconds=broad,broad_expressive_dataset_percent=100*broad/result['valid_seconds'],
               distinct_sources_with_validated_event_or_style_labels=len(union),
               startup_sources=sum(r['start_frame']==0 for r in rows),
               source_event_not_crop_proof='Label seconds count selected audio FROM labeled recordings, not verified event duration. No event timestamps or listening used.')
 return result

def quiet_bins(crops):
 counts=[0]*5;samples=[0]*5;total=0;partial=0
 for crop in crops:
  n=crop['valid_scored_samples'];a=crop['context_frames']*1920
  y=crop['teacher_audio'][0,0,a:a+n].double()
  full=n//960;tail=n%960
  rms=y[:full*960].reshape(full,960).square().mean(-1).sqrt()
  if tail:rms=torch.cat([rms,y[full*960:].square().mean().sqrt().reshape(1)]);partial+=1
  bins=torch.bucketize(rms,torch.tensor([0.,1e-5,1e-4,1e-3],dtype=torch.float64),right=False)
  for i in range(5):
   mask=bins==i;counts[i]+=int(mask.sum());samples[i]+=int(mask.sum())*960
   if tail and bool(mask[-1]):samples[i]-=960-tail
  total+=n
 return {'windows':counts,'samples':samples,'total_samples':total,'partial_windows':partial}

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
 started=time.monotonic();torch.set_num_threads(1)
 root=args.root;run=root/'current-lowrate-fresh5000-v1';shards=Path('/dev/shm/fast-audiovae-compression-fresh-shards-v2')
 launch=read(run/'launch.json');completed=read(run/'completed.json');preserved=read(run/'preservation.json');original=read(root/'pilot-selection-v1.json')
 pools={k:v['rows'] for k,v in original['splits'].items()};data=FreshTrainingData(root/'fresh-source-plan-v2/plan.json',root/'pilot-selection-v1.json',pools,shards)
 labels=read(args.labels)
 records=[json.loads(l) for l in (run/'train.jsonl').read_text().splitlines() if l]
 assert [r['step'] for r in records]==list(range(1001,5001))
 actual=[sid for r in records for sid in r['source_ids']];assert actual==list(data.source_ids[:12000])
 for index,r in enumerate(records):assert len(r['source_ids'])==3 and r['fresh_sources']==(index+1)*3 and r['unique_sources']==3000+(index+1)*3
 checkpoint=run/'checkpoint-step5000.pt';assert sha(checkpoint)==completed['checkpoint_sha256']
 state=torch.load(checkpoint,map_location='cpu',weights_only=True)
 oldids=[r['source_id'] for r in pools['fit']];assert state['sources_seen']==oldids+actual
 assert state['step']==5000 and state['fresh_cursor']==12000 and state['fit_cursor']==15000
 assert state['identity']['coefficients']==launch['resume_identity']['coefficients'] and state['identity']['learning_rate']==launch['resume_identity']['learning_rate']
 joined=pools['fit']+data.plan['rows'][:12000]
 identity_counts={}
 for key in ('source_id','audio_sha256','parent_recording_id'):
  values=[row['manifest_row'][key] for row in joined];identity_counts[key]={'count':len(values),'unique':len(set(values)),'duplicates':len(values)-len(set(values))};assert len(values)==len(set(values))
 finite={};allfinite=True
 for key in ('total','waveform','mel','feature','gradient_norm','step_seconds','gpu_memory_gib'):
  values=[r[key] for r in records if key in r];finite[key]=stat(values);allfinite &=all(math.isfinite(v) for v in values)
 assert allfinite
 means=[]
 for start in range(0,4000,500):
  block=records[start:start+500];means.append({'start_step':block[0]['step'],'stop_step':block[-1]['step'],**{k:statistics.mean(r[k] for r in block) for k in ('total','waveform','mel','feature')}})
 logpath=root/'current-lowrate-fresh5000-v1.log';log=logpath.read_text();events=[]
 for line in log.splitlines():
  try:events.append(json.loads(line))
  except json.JSONDecodeError:pass
 waiting=[e for e in events if e.get('stage')=='waiting_for_sealed_training_data']
 waits=collections.Counter(e['step'] for e in waiting)
 warnings=[line for line in log.splitlines() if any(x in line for x in ('Traceback','RuntimeError','ValueError','Error:','NaN','nonfinite'))]
 producer=read(shards/'producer-complete.json');assert producer['complete'] and producer['sealed_source_prefix']==12000
 shard_reports=[];qtotal={'windows':[0]*5,'samples':[0]*5,'total_samples':0,'partial_windows':0};provenance_rows=[]
 for start in range(0,12000,300):
  crops=data.take(start,300);directory=shards/f'{start:06d}-{start+300:06d}';receipt=read(directory/'receipt.json')
  r={'start':start,'stop':start+300,'receipt_sha256':sha(directory/'receipt.json'),'pairs_sha256':receipt['pairs_sha256'],'parameter_updates':receipt['parameter_updates']}
  if start:
   path=directory/'source-provenance.json';assert sha(path)==receipt['source_provenance_sha256'];provenance=read(path);assert len(provenance['sources'])==300
   for item,row,crop in zip(provenance['sources'],data.plan['rows'][start:start+300],crops):
    assert item['source_id']==row['source_id'] and item['source_audio_sha256']==row['manifest_row']['audio_sha256'] and item['input_samples16k']==row['input_samples16k']
    assert digest(item)==crop['cache_key']
   r.update(source_provenance_sha256=receipt['source_provenance_sha256'],teacher_remained_frozen=receipt['teacher_remained_frozen'],singleton_repeat_bitwise=receipt['singleton_decoder_repeat_bitwise'])
  assert sum(c['valid_scored_samples'] for c in crops)==sum(r['valid_scored_samples'] for r in data.plan['rows'][start:start+300])
  qb=quiet_bins(crops)
  for key in ('windows','samples'):qtotal[key]=[a+b for a,b in zip(qtotal[key],qb[key])]
  for key in ('total_samples','partial_windows'):qtotal[key]+=qb[key]
  shard_reports.append(r)
 data.assert_unchanged()
 qtotal['labels']=['rms_exact_zero','0<rms<=1e-5','1e-5<rms<=1e-4','1e-4<rms<=1e-3','rms>1e-3'];qtotal['seconds']=[s/48000 for s in qtotal['samples']]
 qtotal['window_ms']=20;qtotal['source']='Actual cached teacher target tensor scored spans, excluding context and padding, same per-source20ms grids as existing quality checks'
 opt=state['optimizer'];moment_finite=all(bool(torch.isfinite(t).all()) for st in opt['state'].values() for t in st.values() if isinstance(t,torch.Tensor))
 opt_groups=[{k:v for k,v in g.items() if k!='params'}|{'parameter_count':len(g['params'])} for g in opt['param_groups']]
 steps=[int(st['step']) for st in opt['state'].values()];assert set(steps)=={5000};assert moment_finite
 heldout={key:{name:len({r['manifest_row'][key] for r in joined}&{r['manifest_row'][key] for r in pools[name]}) for name in ('calibration','development')} for key in identity_counts};assert all(v==0 for t in heldout.values() for v in t.values())
 code_paths={'resume_sha256':'resume_settings.py','fresh_loader_sha256':'fresh_training_data.py','monitor_sha256':'unified_monitor.py'}
 code_checks={k:sha(root/'code'/v)==launch['resume_identity'][k] for k,v in code_paths.items()};assert all(code_checks.values())
 report={'audit_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'audit_type':'CPU read-only artifacts and cached targets; no model construction or neural forward','audit_script_sha256':sha(__file__),
  'status':{'step':5000,'training_completed':True,'producer_completed':True,'stopped_for_quality_decision':completed['stopped_for_quality_decision'],'requested_target':5000,'training_pid_exists':Path('/proc/1018063').exists(),'producer_pid_exists':Path('/proc/1018105').exists(),'runtime_error_lines':warnings},
  'recipe':{'resume_identity':launch['resume_identity'],'optimizer_groups':opt_groups,'optimizer_state_entries':len(steps),'optimizer_step_counters':stat(steps),'all_optimizer_tensors_finite':moment_finite,'preservation':preserved,'frozen_checkpoint_receipts_all_true':all(read(p)['frozen_state_preserved'] for p in run.glob('checkpoint-step*.json')),'code_identity_checks':code_checks},
  'ledger':{'training_records':len(records),'steps_exact':True,'fresh_order_exact':True,'checkpoint_seen_exact':True,'identity_counts':identity_counts,'heldout_intersections':heldout,'conditional_reserve_sources_consumed':0,'fresh_source_ids_sha256':digest(actual),'checkpoint_sha256':completed['checkpoint_sha256'],'all40_shards_verified':True,'shards':shard_reports,'fresh_plan_identity':data.identity},
  'timing':{'continuation_wall_seconds_through_last_update':records[-1]['continuation_elapsed_seconds'],'producer_wall_seconds':producer['elapsed_seconds'],'producer_new_sources':producer['new_sources_generated'],'waiting_poll_events':len(waiting),'distinct_waited_update_steps':len(waits),'requested_wait_sleep_seconds':len(waiting)*30,'wait_note':'Every logged wait in this successfully completed run is followed by30seconds requested sleep; actual OS sleep may be longer. Diagnostic update time is sampled every25updates, not full GPU-only timings. Final evaluation/save occur after last train record.','diagnostic_update_seconds':finite['step_seconds']},
  'gradient_and_train_values':finite,'training_500step_means':means,'actual_fresh_source_mix':mix(data.plan['rows'][:12000],labels['sources']),'actual_all15000_source_mix':mix(joined,labels['sources']),'fresh_actual_teacher_quiet_bins':qtotal,
  'source_label_policy':labels['policy'],'source_label_inputs_sha256':labels['inputs'],'producer_runtime':producer['runtime'],
  'limitations':['Frozen state is established by recorded in-process version/storage checks and preservation receipts, not an independent post-hoc full teacher tensor dump.','Specific event labels describe recordings. Selected crops can miss a labeled event. Broad expressive datasets include neutral speech.','No human listening or new benchmark was performed. The repeated-source exclusion does not imply speaker-disjointness.']}
 args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'output':str(args.out),'sha256':sha(args.out),'audit_seconds':time.monotonic()-started,'step':5000,'sources':15000,'shards_verified':40,'wait_seconds_requested':len(waiting)*30,'quiet_seconds':qtotal['seconds']}))
if __name__=='__main__':main()
