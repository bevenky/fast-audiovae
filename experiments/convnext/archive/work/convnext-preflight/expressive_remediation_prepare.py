"""Verify existing expressive audio and acquire a bounded fresh, split-safe FSD supplement."""
import json,sys,hashlib,shutil,time
from pathlib import Path
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import numpy as np
import soundfile as sf
R1=Path('/workspace/fast-audiovae-convnext-20260908-r1');R9=Path('/workspace/fast-audiovae-convnext-20260909-r9');ROOT=R9/'remediation/expressive'
sys.path.insert(0,str(R9))
from audiovae_student.data import load_manifest,validate_manifest
from audiovae_student.comparison_data import load_comparison_plan,known_identities
from audiovae_student.acquire_expressive import StorageBudget
from audiovae_student.acquire import _json_bytes
from audiovae_student import acquire_fsd_vocal as fsd
from audiovae_student.acquire_emogator import load_tree,verify_blob
BUDGET=400*1024**2; RESERVE=4*1024**3

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def emit(name,value):
 payload=_json_bytes(value);StorageBudget(ROOT,BUDGET,RESERVE).write(ROOT/name,payload)
def dump_rows(name,rows):
 StorageBudget(ROOT,BUDGET,RESERVE).write(ROOT/name,b''.join((json.dumps(r.to_dict(),ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode() for r in rows))
 if load_manifest(ROOT/name)!=rows:raise ValueError('Published manifest round-trip failed '+name)
def verify_audio(r):
 p=Path(r.audio_path); raw=p.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=r.audio_sha256:raise ValueError('Audio SHA changed '+r.source_id)
 audio,sr=sf.read(p,dtype='float32',always_2d=True)
 if sr!=16000 or audio.shape[1]!=1 or audio.shape[0]!=round(r.duration_seconds*16000) or not np.isfinite(audio).all() or not np.any(audio):raise ValueError('Audio contract failed '+r.source_id)
 receipt=json.loads(Path(r.access_record).read_text())
 if receipt['prepared_audio_sha256']!=r.audio_sha256:raise ValueError('Receipt mismatch '+r.source_id)
 return {'source_id':r.source_id,'audio_sha256':r.audio_sha256,'audio_bytes':len(raw),'input_samples':audio.shape[0],'peak_abs':float(np.abs(audio).max()),'rms':float(np.sqrt(np.mean(audio.astype(np.float64)**2))),'access_record_sha256':sha(r.access_record)}

def main():
 if (ROOT/'ready.json').exists():raise ValueError('Immutable stage already prepared')
 plans=[load_comparison_plan(R9/'data/recipe-v2'/x) for x in ['optimization','calibration']]
 active={r.source_id:r for p in plans for r in p['rows']}
 active_people=set().union(*(known_identities(r) for r in active.values()))
 reserved=load_manifest(ROOT/'reserved-all.jsonl');resids=set().union(*(known_identities(r) for r in reserved))
 train=load_manifest(ROOT/'candidate-emogator.jsonl')
 tree=load_tree(R1/'data-expansion-provenance/emogator-tree.json')
 for name in ['README.md','LICENSE']:verify_blob((R1/'data-emogator/provenance'/name).read_bytes(),tree[name])
 if any(known_identities(r)&resids or known_identities(r)&active_people for r in train):raise ValueError('EmoGator overlap')
 validate_manifest(train,training_only=True)
 checks=[]; emotions=Counter();emotionsecs=Counter()
 for i,r in enumerate(train,1):
  check=verify_audio(r);receipt=json.loads(Path(r.access_record).read_text()); check['condition']='emotional_nonverbal';check['intended_emotion']=receipt['intended_emotion']
  if r.license!='Apache-2.0' or receipt['source_revision']!=r.source_revision or receipt['original_git_blob_sha1']!=tree[receipt['source_member']]['sha']:raise ValueError('EmoGator provenance changed')
  checks.append(check);emotions[check['intended_emotion']]+=1;emotionsecs[check['intended_emotion']]+=r.duration_seconds
  if i%5000==0:print(json.dumps({'phase':'verify_existing_emogator','done':i,'total':len(train)}),flush=True)
 dump_rows('emogator-train.jsonl',train)
 emit('emogator-audio-verification.json',{'rows':checks,'scope':'Current prepared WAV hash, full decode finite/nonzero, exact length; pinned original Git blob identity checked against preparation receipt, original MP3 not independently decoded again'})
 emit('emogator-conditions.json',{'by_source':{x['source_id']:{'condition':x['condition'],'intended_emotion':x['intended_emotion']} for x in checks},'label_policy':'Generic emotional nonverbal bursts. Intended emotion never relabeled as a verified laugh, cry, shout, giggle or whistle action.'})
 print(json.dumps({'phase':'emogator_verified','clips':len(train),'hours':sum(r.duration_seconds for r in train)/3600}),flush=True)
 selection=json.loads((ROOT/'fsd-candidate-selection.json').read_text());items=selection['selected']
 devsessions=set()
 for label in ['Whispering','Breathing','Yell','Screaming']:
  options={x['session_id'] for x in items if label in x['target_labels'] and ('session',x['session_id']) not in active_people}
  devsessions.update(sorted(options,key=lambda x:hashlib.sha256(('expressive-remediation-dev-v1:'+x).encode()).hexdigest())[:2])
 if any(('session',s) in active_people for s in devsessions):raise ValueError('New dev overlaps active contributor')
 # This reservation is made before any audio acquisition or training selection.
 emit('new-dev-contributor-reservation.json',{'sessions':sorted(devsessions),'rule':'For each newly available explicit condition, reserve up to two minimum seeded SHA256 contributor groups that never appear anywhere in current training or calibration. Whole contributor group stays dev. Original FSD official heldouts remain excluded.','active_plan_identities':[p['identity']['identity_sha256'] for p in plans]})
 original=fsd.uploader_split
 fsd.uploader_split=lambda username:'dev' if 'freesound:uploader:'+fsd.canonical_uploader(username) in devsessions else original(username)
 for x in items:x['split']=fsd.uploader_split(x['uploader'])
 from huggingface_hub import get_hf_file_metadata,hf_hub_url
 def head(x):
  m=get_hf_file_metadata(hf_hub_url(fsd.REPO,f"clips/dev/{x['id']}.wav",repo_type='dataset',revision=fsd.REVISION),token=False)
  if m.commit_hash!=fsd.REVISION or not isinstance(m.size,int) or not 0<m.size<=fsd.MAX_CLIP_BYTES:raise ValueError('Pinned HEAD contract')
  return x,m.size
 with ThreadPoolExecutor(max_workers=4) as pool: metadata=list(pool.map(head,items))
 # Select dev first then round-robin explicit conditions with a strict original-byte cap.
 buckets=defaultdict(list)
 for x,n in metadata:buckets[(x['split'],x['target_labels'][0])].append((x,n))
 for q in buckets.values():q.sort(key=lambda v:hashlib.sha256(('expressive-remediation-v1:'+v[0]['id']).encode()).hexdigest())
 ordered=[]
 while any(buckets.values()):
  for key in sorted(buckets,key=lambda k:(k[0]!='dev',k[1])):
   if buckets[key]:ordered.append(buckets[key].pop())
 available=BUDGET-StorageBudget(ROOT,BUDGET,RESERVE).used
 origcap=min(110*1024**2,max(0,int((available-20*1024**2)/1.75)))
 chosen=[];estimated=0
 for x,n in ordered:
  if estimated+n<=origcap:chosen.append(x);estimated+=n
 selection.update(selected=chosen,selected_clip_count=len(chosen),counts_by_split=dict(Counter(x['split'] for x in chosen)),counts_by_target=dict(Counter(y for x in chosen for y in x['target_labels'])))
 selection['selection_policy']={**selection['selection_policy'],'uploader_split':'Initial hash split plus immutable new-dev-contributor-reservation.json selected before training use','storage_original_byte_cap':origcap,'original_bytes_from_pinned_HEAD':estimated}
 emit('fsd-reviewed-selection.json',selection)
 root=ROOT/'fsd-fresh-v1'; ex=fsd.Exclusions(prior=[R9/'data/recipe-v2'/s/'train-manifest.jsonl' for s in ['optimization','calibration']],reserved=[ROOT/'reserved-all.jsonl'])
 for p in plans:
  for r in p['excluded']:ex.source_ids.add(r.source_id);ex.parents.add(r.parent_recording_id);ex.hashes.add(r.audio_sha256)
 # Pool requests overlap bounded downloads while the acquisition writes remain sequential.
 with ThreadPoolExecutor(max_workers=4) as pool:
  futures={}
  for x in chosen[:4]:futures[x['id']]=pool.submit(fsd.fetch_pinned_clip,x['id'])
  index={x['id']:i for i,x in enumerate(chosen)}
  def fetch(key):
   result=futures.pop(key).result();n=index[key]+4
   if n<len(chosen):x=chosen[n];futures[x['id']]=pool.submit(fsd.fetch_pinned_clip,x['id'])
   return result
  result=fsd.acquire(selection,root,expected_selection_sha256=hashlib.sha256(_json_bytes(selection)).hexdigest(),exclusions=ex,max_bytes=available-4*1024**2,reserve_bytes=RESERVE,fetch=fetch)
 fresh=load_manifest(root/'prepared/train.jsonl')+load_manifest(root/'prepared/dev.jsonl')
 trainfresh=[r for r in fresh if r.split=='train'];devfresh=[r for r in fresh if r.split=='dev']
 freshdev=set().union(*(known_identities(r) for r in devfresh)) if devfresh else set()
 if any(known_identities(r)&(resids|freshdev) for r in trainfresh):raise ValueError('Prepared training overlaps heldout')
 if any(known_identities(r)&active_people for r in devfresh):raise ValueError('Fresh heldout overlaps active')
 if any(known_identities(r,people=False)&active_people for r in trainfresh):raise ValueError('Fresh training file already planned')
 verification=[verify_audio(r) for r in fresh]
 validate_manifest(train+fresh)
 dump_rows('train-candidates.jsonl',train+trainfresh);dump_rows('fresh-dev.jsonl',devfresh)
 labels={r.source_id:json.loads(Path(r.access_record).read_text())['selected_metadata']['target_labels'] for r in fresh}
 emit('fresh-conditions.json',labels);emit('fresh-audio-verification.json',verification)
 bysplit={}
 for split,rr in [('train',trainfresh),('dev',devfresh)]:
  bysplit[split]={'clips':len(rr),'seconds':sum(r.duration_seconds for r in rr),'contributors':len({r.session_id for r in rr}),'conditions':{label:{'clips':sum(label in labels[r.source_id] for r in rr),'seconds':sum(r.duration_seconds for r in rr if label in labels[r.source_id])} for label in sorted({v for r in rr for v in labels[r.source_id]})}}
 report={'state':'verified_staged_not_trained','created_unix':time.time(),'active_plan_modified':False,'teacher_precomputation_performed':False,'all_audio_on_runpod':True,'existing_emogator':{'clips':len(train),'hours':sum(r.duration_seconds for r in train)/3600,'contributors':len({r.speaker_id for r in train}),'emotion_counts':dict(emotions),'emotion_seconds':dict(emotionsecs),'why_omitted':'Current planner accepts known speech language or explicit action label. EmoGator language und with emotion-only annotations has no action mapping; omission was conservative classification, not missing download. Explicit generic emotional_nonverbal is staged for a deliberate new continuation planner category.','label_limit':'Not counted as verified laugh/cry/giggle/shout/whistle seconds'},'fresh_fsd':bysplit,'fresh_quarantined':result['quarantined'],'no_new_strict_candidates_for':['Crying_and_sobbing','Giggle','Shout','human_whistling'],'planned_current_source_exclusions':len(active),'historical_holdout_sources':len(reserved),'storage_bytes':StorageBudget(ROOT,BUDGET,RESERVE).used,'free_bytes':shutil.disk_usage(ROOT).free,'next_step':'Root must explicitly integrate candidates at a resumable checkpoint using preserved exposure ledger and before final allocation incorporate fresh-dev contributors into all split guards. Existing active schedule unchanged.'}
 emit('ready.json',{**report,'manifest_sha256':{name:sha(ROOT/name) for name in ['train-candidates.jsonl','fresh-dev.jsonl','emogator-train.jsonl','new-dev-contributor-reservation.json','fsd-reviewed-selection.json','fresh-conditions.json','emogator-conditions.json']}})
 print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
