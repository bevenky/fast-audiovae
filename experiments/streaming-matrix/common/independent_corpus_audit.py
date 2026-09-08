"""Offline 60-clip accounting and timing/qualification identity audit; no inference."""
import argparse, ast, collections, hashlib, json, math, struct, zipfile
from pathlib import Path

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def npy_header(archive,name):
 with archive.open(name) as f:
  assert f.read(6)==b'\x93NUMPY';version=f.read(2)
  length=struct.unpack('<H' if version[0]==1 else '<I',f.read(2 if version[0]==1 else 4))[0]
  return ast.literal_eval(f.read(length).decode('latin1'))
def artifact_union(report):
 result={}
 for a in report['adapters'].values():
  for key,v in a['artifacts_before'].items():
   if key in result:assert result[key]==v
   result[key]=v
 assert result==report['artifacts_after'] and report['artifacts_unchanged']
 return result

def audit(corpus,paired,output):
 q=json.loads(Path(corpus).read_text());p=json.loads(Path(paired).read_text())
 assert q['status']==p['status']=='passed' and q['config']['exact']==p['config']['exact']
 assert q['protocol']['threads']==1 and q['protocol']['cpu_only'] and not q['protocol']['timing_claims']
 assert q['protocol']['onnxruntime']=='1.29.0'
 assert q['protocol']['candidate_stream_patterns']==[1,2,4]
 exact=q['config']['exact'];assert type(exact) is bool
 qa=artifact_union(q);pa=artifact_union(p)
 manifest_path=q['config']['latent_manifest'];archive_path=q['config']['latents']
 assert manifest_path==p['config']['latent_manifest'] and archive_path==p['config']['latents']
 for name in ['baseline_bundle','candidate_bundle']:assert q['config'][name]==p['config'][name]
 common=set(qa)&set(pa)
 for name in common:assert qa[name]==pa[name],name
 for name,value in qa.items():assert Path(name).stat().st_size==value['bytes'] and digest(name)==value['sha256'],name
 manifest=json.loads(Path(manifest_path).read_text());cases={x['uid']:x for x in manifest['cases']}
 uids=[x['uid'] for x in manifest['cases']]
 assert len(cases)==len(uids)==60 and q['cohort']['clips']==60 and q['cohort']['uids']==uids
 assert q['cohort']['stored_upstream_references']==60
 assert len(q['references'])==60 and [x['uid'] for x in q['references']]==uids
 assert len(q['runs'])==q['completed_streams']==180
 with zipfile.ZipFile(archive_path) as z:
  assert {n[:-7] for n in z.namelist() if n.endswith('__z.npy')}==set(uids)
  assert {n[:-9] for n in z.namelist() if n.endswith('__ref.npy')}==set(uids)
  for uid in uids:
   latent=npy_header(z,uid+'__z.npy');ref=npy_header(z,uid+'__ref.npy')
   assert latent['descr']=='<f4' and tuple(cases[uid]['latent_shape'])==latent['shape']
   assert ref['descr']=='<f4' and ref['shape']==(1,1,latent['shape'][-1]*1920)
 refs={x['uid']:x for x in q['references']};paired_refs={x['uid']:x for x in p['references']}
 for uid,ref in paired_refs.items():assert ref['waveform_sha256']==refs[uid]['waveform_sha256']
 totalcalls=totalsamples=0;tails=collections.Counter();maxima=collections.defaultdict(float);non_gating_failures=0
 for reference in q['references']:
  uid=reference['uid'];frames=cases[uid]['latent_shape'][-1];samples=frames*1920
  assert reference['adapter']=='baseline' and reference['latent_frames']==frames and reference['samples']==samples
  stored=reference['stored_upstream'];assert stored['gate']==(not exact) and not stored['exact_required']
  assert stored['returned_samples']==samples and stored['waveform_sha256']==reference['waveform_sha256']
  if stored['gate']:assert stored['passed']
  maxima['baseline_vs_stored_upstream']=max(maxima['baseline_vs_stored_upstream'],stored['max_abs'])
 for index,row in enumerate(q['runs']):
  uid=uids[index//3];stride=[1,2,4][index%3];frames=cases[uid]['latent_shape'][-1];samples=frames*1920
  assert row['uid']==uid and row['chunk_frames']==stride and row['latent_frames']==frames
  assert row['passed'] and 'error' not in row and 'close_error' not in row and row['expected_samples']==samples
  n=math.ceil(frames/stride);assert len(row['calls'])==n+1;returned=0
  for j,c in enumerate(row['calls']):
   start=min(j*stride,frames);end=min((j+1)*stride,frames);flush=j==n
   assert c['kind']==('flush' if flush else 'chunk') and c['chunk_index']==j
   assert c['start_frame']==start and c['end_frame']==end
   assert c['start_sample']==start*1920 and c['end_sample']==end*1920
   assert c['returned_samples']==(end-start)*1920
   assert c['first']==(j==0) and c['final']==(j==n-1 or flush)
   returned+=c['returned_samples']
  assert returned==samples
  tails[f'chunk_{stride}_last_{frames-(n-1)*stride}']+=1
  checks=row['checks'];assert set(checks)=={'accepted_full','stored_upstream'}
  assert checks['accepted_full']['gate'] and checks['accepted_full']['exact_required']==exact
  assert checks['stored_upstream']['gate']==(not exact) and not checks['stored_upstream']['exact_required']
  assert checks['accepted_full']['waveform_sha256']==checks['stored_upstream']['waveform_sha256']
  for name,check in checks.items():
   assert check['returned_samples']==samples and math.isfinite(check['max_abs']) and check['max_abs']>=0
   assert math.isfinite(check['rmse']) and check['rmse']>=0
   if check['gate']:assert check['passed']
   if not check['gate'] and not check['passed']:non_gating_failures+=1
   if check['exact_required']:
    assert check['bitwise_equal'] and check['max_abs']==check['rmse']==0 and check['waveform_sha256']==refs[uid]['waveform_sha256']
   maxima['candidate_vs_'+name]=max(maxima['candidate_vs_'+name],check['max_abs'])
  totalcalls+=len(row['calls']);totalsamples+=returned
 # Final proof that the exact streaming graph and registered binary set timed in
 # the paired campaign are the same ones that passed complete-corpus qualification.
 for name in ['baseline','candidate']:
  meta=q['adapters'][name]['metadata'];runtime=meta['runtime'];assert runtime['threads']==1 and runtime['selected']=='native'
  assert runtime['providers']==['CPUExecutionProvider'] and runtime['onnxruntime']=='1.29.0'
  paired_meta=p['adapters'][name]['metadata']
  expected_role='full_runtime' if name=='baseline' else 'stream_runtime'
  assert runtime['model']==paired_meta[expected_role]['model']
  for attr in q['adapters'][name]['graph_concurrency_attributes']:
   assert all(type(v) is int and v==1 for v in attr['attributes'].values())
 graph=Path(q['config']['candidate_bundle'])/p['adapters']['candidate']['metadata']['stream_runtime']['model']
 assert str(graph) in common
 # Match timed candidate 80/160 streams to their corpus counterpart: Apple
 # payload hashes should be reproducible across sessions for this fixed graph.
 matches=0
 for row in p['runs']:
  if row['adapter']!='candidate' or row['mode']=='full':continue
  stride=2 if row['mode']=='stream_80' else 4
  c=next(x for x in q['runs'] if x['uid']==row['uid'] and x['chunk_frames']==stride)
  assert row['check']['waveform_sha256']==c['checks']['accepted_full']['waveform_sha256']
  matches+=1
 assert matches==42
 result={'status':'passed','scope':'Independent offline count/range/gate/provenance audit. No inference. Recorded numerical verdicts and hashes are audited; PCM was not recomputed.',
  'audit_source_sha256':digest(__file__),'corpus_results_sha256':digest(corpus),'paired_results_sha256':digest(paired),
  'clips':60,'stored_upstream_references':60,'streams':180,'patterns_latent_frames':[1,2,4],'raw_calls':totalcalls,'returned_samples':totalsamples,
  'all_required_quality_gates_passed':True,'all_first_final_partial_flush_counts_passed':True,'tail_counts':dict(tails),
  'exact_candidate_vs_accepted_required':exact,'max_abs_errors':dict(maxima),'nongating_stored_upstream_failures':non_gating_failures,
  'artifact_count':len(qa),'common_timing_qualification_artifacts':len(common),'before_after_and_current_artifacts_match':True,
  'dataset_sha256':qa[archive_path]['sha256'],'manifest_sha256':qa[manifest_path]['sha256'],'candidate_stream_graph_sha256':qa[str(graph)]['sha256'],
  'paired_candidate_stream_waveform_hashes_matched':matches,'timing_and_qualification_candidate_identical':True,'material_blockers':[]}
 assert not Path(output).exists()
 Path(output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('corpus');parser.add_argument('paired');parser.add_argument('output');args=parser.parse_args();audit(args.corpus,args.paired,args.output)
