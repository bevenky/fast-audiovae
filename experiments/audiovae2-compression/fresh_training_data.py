"""Read authenticated, immutable continuation shards without device changes."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import torch

PLAN_VERSION='audiovae2_group_continuation_sources_v1'
PAIR_VERSION='audiovae2_group_continuation_pairs_v1'
SOURCE_SHA256='2efdff1708d8ec1471624aae6f232d0f933de26b788b6901c434246847e2d3a8'
CHECKPOINT_SHA256='94b5d51e107e0507d4acc976cfdadb64edd6fd06d1f751dadbf2fd1594274bf1'
GEOMETRY=('source_id','start_frame','context_start_frame','context_frames','scored_frames','valid_scored_samples')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def stat_identity(path):
 s=Path(path).stat()
 return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


class FreshTrainingData:
 def __init__(self,plan_path,original_manifest_path,original_pools,shards_dir):
  self.plan_path=Path(plan_path);self.shards_dir=Path(shards_dir)
  self.plan=json.loads(self.plan_path.read_text());self.identity=self.plan.get('identity_sha256')
  stable={k:v for k,v in self.plan.items() if k!='identity_sha256'}
  if self.plan.get('version')!=PLAN_VERSION or digest(stable)!=self.identity:raise ValueError('Continuation source plan identity differs')
  if self.plan.get('teacher_source_sha256')!=SOURCE_SHA256 or self.plan.get('teacher_checkpoint_sha256')!=CHECKPOINT_SHA256:raise ValueError('Original teacher identity differs')
  if sha(original_manifest_path)!=self.plan.get('original_manifest_sha256'):raise ValueError('Original source manifest changed')
  original=json.loads(Path(original_manifest_path).read_text())
  if digest({k:v for k,v in original.items() if k!='identity_sha256'})!=original['identity_sha256']:raise ValueError('Original manifest identity differs')
  for name in ('fit','calibration','development'):
   if [r['source_id'] for r in original['splits'][name]['rows']]!=[r['source_id'] for r in original_pools[name]]:raise ValueError('Original authenticated pools differ')
  self.source_ids=tuple(self.plan['source_ids']);rows=self.plan['rows']
  if len(rows)!=27000 or self.plan.get('shard_size')!=300 or self.plan.get('approved_range')!=[0,12000] or self.plan.get('conditional_reserve_range')!=[12000,27000]:raise ValueError('Unexpected continuation range')
  if self.source_ids!=tuple(r['source_id'] for r in rows) or digest(list(self.source_ids))!=self.plan['source_ids_sha256']:raise ValueError('Ordered continuation source IDs differ')
  keys=('source_id','audio_sha256','parent_recording_id');seen={k:set() for k in keys};blocked={k:set(self.plan['blocked_identities'][k]) for k in keys}
  for split in original['splits'].values():
   for row in split['rows']:
    for k in keys:
     if row['manifest_row'][k] not in blocked[k]:raise ValueError('Original fitting/validation identity missing from exclusion ledger')
  for index,row in enumerate(rows):
   source=row['manifest_row']
   if row['source_id']!=source['source_id'] or source['split']!='train':raise ValueError('Invalid continuation source identity/split')
   for k in keys:
    value=source[k]
    if not value or value in seen[k] or value in blocked[k]:raise ValueError('Repeated or reserved continuation identity: '+k)
    seen[k].add(value)
   if row['phase']!=('approved' if index<12000 else 'conditional_reserve'):raise ValueError('Continuation source authorization phase differs')
   self._geometry(row)
  self._original_manifest=Path(original_manifest_path)
  self._hashes={str(self.plan_path):sha(self.plan_path),str(self._original_manifest):sha(self._original_manifest)}
  self._stats={path:stat_identity(path) for path in self._hashes}
  self._cached_index=None;self._cached_crops=None

 @staticmethod
 def _geometry(row):
  for key in GEOMETRY[1:]+('input_samples16k',):
   if type(row[key]) is not int or row[key]<0:raise ValueError('Invalid integer source geometry')
  if (row['valid_scored_samples']<4096 or row['valid_scored_samples']>64*1920
      or row['scored_frames']!=math.ceil(row['valid_scored_samples']/1920)
      or row['context_start_frame']+row['context_frames']!=row['start_frame']
      or row['context_frames']!=(0 if row['start_frame']==0 else 30)
      or row['start_frame']*1920+row['valid_scored_samples']>row['input_samples16k']*3):
   raise ValueError('Continuation causal or valid-sample geometry differs')

 def _load(self,start):
  if self._cached_index==start:return self._cached_crops
  stop=min(start+300,len(self.source_ids));directory=self.shards_dir/f'{start:06d}-{stop:06d}'
  receipt_path=directory/'receipt.json'
  if not receipt_path.is_file():raise FileNotFoundError('Continuation shard is not sealed yet: '+str(receipt_path))
  receipt=json.loads(receipt_path.read_text());expected=list(self.source_ids[start:stop])
  if (receipt.get('format')!=PAIR_VERSION or receipt.get('complete') is not True
      or receipt.get('plan_identity_sha256')!=self.identity or receipt.get('start_index')!=start
      or receipt.get('stop_index')!=stop or receipt.get('source_ids')!=expected
      or receipt.get('source_ids_sha256')!=digest(expected)
      or receipt.get('teacher_source_sha256')!=SOURCE_SHA256 or receipt.get('teacher_checkpoint_sha256')!=CHECKPOINT_SHA256
      or receipt.get('parameter_updates')!=0):raise ValueError('Sealed shard identity differs')
  path=directory/'pairs.pt'
  pair_sha=sha(path)
  if Path(receipt['pairs_path']).resolve()!=path.resolve() or pair_sha!=receipt['pairs_sha256']:raise ValueError('Sealed target bytes differ')
  for p,checksum in ((receipt_path,sha(receipt_path)),(path,pair_sha)):
   if str(p) in self._hashes and self._hashes[str(p)]!=checksum:raise ValueError('Previously consumed shard changed')
   self._hashes[str(p)]=checksum
   self._stats[str(p)]=stat_identity(p)
  payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
  if (payload.get('format')!=PAIR_VERSION or payload.get('plan_identity_sha256')!=self.identity
      or payload.get('start_index')!=start or payload.get('stop_index')!=stop
      or len(payload.get('crops',[]))!=stop-start):raise ValueError('Target shard payload identity differs')
  crops=payload['crops']
  for crop,row in zip(crops,self.plan['rows'][start:stop]):
   if any(crop[k]!=row[k] for k in GEOMETRY):raise ValueError('Target source/crop differs from plan')
   frames=row['context_frames']+row['scored_frames']
   for key,shape in [('latents',(1,64,frames)),('teacher_audio',(1,1,frames*1920)),('reference16k',(1,1,frames*640))]:
    tensor=crop.get(key)
    if not isinstance(tensor,torch.Tensor) or tensor.device.type!='cpu' or tensor.dtype!=torch.float32 or tuple(tensor.shape)!=shape or not bool(torch.isfinite(tensor).all()):raise ValueError('Invalid frozen target tensor: '+key)
   if not isinstance(crop.get('cache_key'),str) or len(crop['cache_key'])!=64:raise ValueError('Missing source target identity')
  self._cached_index,self._cached_crops=start,crops
  return crops

 def take(self,start,count):
  if type(start) is not int or type(count) is not int or start<0 or count<1 or start+count>len(self.source_ids):raise ValueError('Invalid fresh-source interval')
  result=[]
  for index in range(start,start+count):
   shard_start=(index//300)*300
   result.append(self._load(shard_start)[index-shard_start])
  if [r['source_id'] for r in result]!=list(self.source_ids[start:start+count]):raise AssertionError('Fresh source ordering changed')
  return result

 def assert_unchanged(self):
  for path,expected in self._hashes.items():
   current=stat_identity(path)
   if current!=self._stats[path]:
    if sha(path)!=expected:raise RuntimeError('Continuation input changed: '+path)
    self._stats[path]=current
