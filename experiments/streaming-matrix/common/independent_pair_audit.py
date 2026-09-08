"""Offline accounting audit only. Does not load any inference runtime."""
import argparse, collections, hashlib, json, math, random, statistics
from pathlib import Path

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def close(a,b):assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-12),(a,b)
def audit(result_path,monitor_path,out):
 r=json.loads(Path(result_path).read_text());mon=json.loads(Path(monitor_path).read_text())
 assert r['status']=='passed' and r['artifacts_unchanged'] and mon['exit_code']==0
 assert r['protocol']['threads']==1 and r['protocol']['cpu_only'] and r['onnxruntime']=='1.29.0'
 modes=['full','stream_80','stream_160'];uids=['bn_in_00151_1818','en_us_00103_1779','es_419_00060_1994']
 assert r['config'].get('clip_ids',uids)==uids
 rng=random.Random(r['config']['seed']);expected=[]
 for uid in uids:
  for phase,count in [('warmup',2),('measure',5)]:
   for repeat in range(count):
    order=list(modes);rng.shuffle(order)
    for mode in order:
     adapters=['baseline','candidate'];rng.shuffle(adapters)
     for pos,adapter in enumerate(adapters):expected.append((uid,phase,repeat,mode,adapter,f'{uid}/{phase}/{repeat}/{mode}',pos))
 assert len(r['runs'])==len(expected)==126
 keys=['uid','phase','repeat','mode','adapter','pair_id','pair_position']
 refs={x['uid']:x for x in r['references']};assert set(refs)==set(uids)
 pairs=collections.defaultdict(dict);raw_count=0;frames_total=0;samples_total=0
 for index,(row,job) in enumerate(zip(r['runs'],expected)):
  assert tuple(row[k] for k in keys)==job and row['order']==index
  assert row['check']['passed'] and 'error' not in row and row['reference_adapter']=='baseline'
  assert row['check']['exact_required']==r['config']['exact']
  if r['config']['exact']:assert row['check']['bitwise_equal'] and row['check']['max_abs']==0
  frames=row['latent_frames'];hop=1920;samples=frames*hop
  assert samples==row['samples']==row['check']['returned_samples']==refs[row['uid']]['samples']
  close(row['decoded_duration_seconds'],samples/48000)
  stride=frames if row['mode']=='full' else (2 if row['mode']=='stream_80' else 4)
  assert row['chunk_frames']==stride
  n=math.ceil(frames/stride);calls=row['calls'];assert len(calls)==n+(row['mode']!='full')
  prev_end=0.;returned=0
  for j,c in enumerate(calls):
   flush=j==n
   start=min(j*stride,frames);end=min((j+1)*stride,frames)
   assert c['chunk_index']==j and c['kind']==('flush' if flush else ('full' if row['mode']=='full' else 'chunk'))
   assert c['start_frame']==start and c['end_frame']==end
   assert c['start_sample']==start*hop and c['end_sample']==end*hop
   assert c['returned_samples']==(end-start)*hop
   assert c['first']==(j==0) and c['final']==(j==n-1 or flush)
   assert c['start_seconds']>=prev_end and c['end_seconds']>=c['start_seconds']
   close(c['duration_seconds'],c['end_seconds']-c['start_seconds']);assert c['output_copy_seconds']>=0
   prev_end=c['end_seconds'];returned+=c['returned_samples']
  assert returned==samples and row['whole_loop_seconds']>=prev_end
  sec=sum(c['duration_seconds'] for c in calls);close(sec,row['sum_call_seconds']);close(sec/(samples/48000),row['sum_call_rtf'])
  close(row['whole_loop_seconds']/(samples/48000),row['whole_loop_rtf']);close(row['process_cpu_seconds']/row['whole_loop_seconds'],row['process_cpu_to_loop_wall_ratio'])
  close(sum(c['output_copy_seconds'] for c in calls),row['output_copy_seconds']);close(calls[0]['duration_seconds'],row['first_call_seconds'])
  assert row['whole_loop_seconds']>=sec and row['process_cpu_seconds']>=0
  assert row['row_finished_epoch_seconds']>=row['row_started_epoch_seconds']
  raw_count+=len(calls);samples_total+=samples;frames_total+=frames
  if row['mode']=='full':assert row['check']['bitwise_equal'] and row['check']['waveform_sha256']==refs[row['uid']]['waveform_sha256']
  if row['phase']=='measure':
   assert row['adapter'] not in pairs[row['pair_id']];pairs[row['pair_id']][row['adapter']]=row
 assert len(pairs)==45 and all(set(v)=={'baseline','candidate'} for v in pairs.values())
 state_names={'accepted_future_prefix','full_probe','full_alternative','one_frame','two_frames','four_frames','uneven_frames','reset_after_partial','reset_after_flush','interleaved_0','interleaved_1','stream_future_prefix'}
 assert len(r['state_checks'])==72
 state_raw=0
 for uid in uids:
  for adapter in ['baseline','candidate']:
   records=[x for x in r['state_checks'] if x['uid']==uid and x['adapter']==adapter]
   assert len(records)==12 and {x['name'] for x in records}==state_names
   for rec in records:
    assert rec['check']['passed']
    if r['config']['exact']:assert rec['check']['bitwise_equal']
    if 'calls' not in rec:continue
    pos=0
    for call in rec['calls']:
     assert call['start_frame']==pos and call['end_frame']>=pos
     assert call['start_sample']==pos*1920 and call['end_sample']==call['end_frame']*1920
     assert call['returned_samples']==(call['end_frame']-pos)*1920
     pos=call['end_frame']
    assert pos==13 and rec['calls'][-1]['kind']=='flush' and rec['check']['samples']==13*1920
    state_raw+=len(rec['calls'])
 artifacts={};graph_hashes={};library_hashes={}
 for name,a in r['adapters'].items():
  meta=a['metadata'];assert meta['providers']==['CPUExecutionProvider']
  for role in ['full_runtime','stream_runtime']:
   info=meta[role];assert info['threads']==1 and info['providers']==['CPUExecutionProvider'] and info['selected']=='native'
  for p,value in a['artifacts_before'].items():
   if p in artifacts:assert artifacts[p]==value
   artifacts[p]=value
  for attr in a['graph_concurrency_attributes']:
   assert all(type(v) is int and v==1 for v in attr['attributes'].values())
  graph=Path(r['config'][name+'_bundle'])/meta['full_runtime']['model']
  graph_hashes[name]=artifacts[str(graph)]['sha256']
  manifest=json.loads((Path(r['config'][name+'_bundle'])/'bundle.json').read_text())
  native=manifest['native'][meta['full_runtime']['platform']]
  library=Path(r['config'][name+'_bundle'])/native['library'];library_hashes[name]=artifacts[str(library)]['sha256']
 assert artifacts==r['artifacts_after'] and len(set(graph_hashes.values()))==1 and len(set(library_hashes.values()))==1
 for p,value in artifacts.items():assert Path(p).stat().st_size==value['bytes'] and digest(p)==value['sha256'],p
 summaries=[]
 for mode in modes:
  ps=[v for v in pairs.values() if v['baseline']['mode']==mode];assert len(ps)==15
  record={'mode':mode,'complete_pairs':len(ps),'per_clip':[]}
  for uid in uids:
   group=[p for p in ps if p['baseline']['uid']==uid];assert len(group)==5
   b=statistics.median(x['baseline']['sum_call_rtf'] for x in group);c=statistics.median(x['candidate']['sum_call_rtf'] for x in group)
   record['per_clip'].append({'uid':uid,'baseline_rtf':b,'candidate_rtf':c,'reduction_percent':100*(1-c/b)})
  b=statistics.mean(v['baseline_rtf'] for v in record['per_clip']);c=statistics.mean(v['candidate_rtf'] for v in record['per_clip'])
  reductions=[100*(1-p['candidate']['sum_call_seconds']/p['baseline']['sum_call_seconds']) for p in ps]
  record.update(baseline_rtf=b,candidate_rtf=c,reduction_percent=100*(1-c/b),paired_reduction_min=min(reductions),paired_reduction_median=statistics.median(reductions),paired_reduction_max=max(reductions))
  saved=next(x for x in r['paired_summary'] if x['mode']==mode)
  close(b,saved['mean_clip_median_baseline_rtf']);close(c,saved['mean_clip_median_candidate_rtf']);close(record['reduction_percent'],saved['aggregate_reduction_percent']);close(record['paired_reduction_median'],saved['median_paired_reduction_percent'])
  summaries.append(record)
 ratios={}
 for adapter in ['baseline','candidate']:
  for mode in modes:
   rows=[x for x in r['runs'] if x['phase']=='measure' and x['adapter']==adapter and x['mode']==mode]
   v=[x['process_cpu_to_loop_wall_ratio'] for x in rows]
   ratios[adapter+'/'+mode]={'min':min(v),'median':statistics.median(v),'max':max(v),'weighted':sum(x['process_cpu_seconds'] for x in rows)/sum(x['whole_loop_seconds'] for x in rows)}
 rss=[int(s['process_ps'].split()[2]) for s in mon['samples'] if 'process_ps' in s]
 report={'status':'passed','scope':'Offline independent accounting/provenance audit. No inference; waveform verdicts and payload hashes audited from actual timed-output checks, PCM not recomputed.',
  'results_sha256':digest(result_path),'monitor_sha256':digest(monitor_path),'audit_source_sha256':digest(__file__),
  'runs':126,'warmup_runs':36,'measured_runs':90,'complete_measured_pairs':45,'raw_timed_calls':raw_count,'state_checks':72,'state_probe_raw_calls':state_raw,
  'timed_latent_frames':frames_total,'timed_returned_samples':samples_total,'all_recorded_waveform_checks_passed':True,'all_chunk_and_flush_counts_passed':True,
  'time_arithmetic_verified':True,'artifact_count':len(artifacts),'before_after_and_current_artifacts_match':True,'full_graph_sha256':graph_hashes,'full_native_library_sha256':library_hashes,
  'full_call_control_note':'The two full-call models/libraries are byte-identical. Observed full-call timing variation is reported, not attributed to a decoder algorithm change.',
  'max_timed_waveform_error':max(x['check']['max_abs'] for x in r['runs']),'max_state_waveform_error':max(x['check']['max_abs'] for x in r['state_checks']),
  'cpu_process_to_wall_ratios':ratios,'monitor_exit_code':mon['exit_code'],'monitor_max_rss_kib':max(rss) if rss else None,'summary':summaries,'material_blockers':[]}
 assert not Path(out).exists(),out
 Path(out).write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('results');p.add_argument('monitor');p.add_argument('output');a=p.parse_args();audit(a.results,a.monitor,a.output)
