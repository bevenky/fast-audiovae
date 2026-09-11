"""Original frozen-teacher recipe with a bounded checkpoint-released cache."""
from __future__ import annotations
import argparse,fcntl,hashlib,io,json,math,os,shutil,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import soundfile as sf
import torch
from torch.nn import functional as F
import run_pilot as base
from fresh_training_data import PAIR_VERSION,digest,sha
from continuation_data_v2 import ContinuationData, START, STOP

def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def window(t,start,length):return F.pad(t[...,start:start+length],(0,max(0,start+length-t.shape[-1]))).contiguous().clone()
def read_source(row):
 source=row['manifest_row'];payload=Path(source['audio_path']).read_bytes()
 if hashlib.sha256(payload).hexdigest()!=source['audio_sha256']:raise ValueError('Source audio bytes changed: '+row['source_id'])
 native=source['original_sample_rate_hz']==16000 and all(source[k].casefold().split(':',1)[0].strip() in {'none','unchanged'} for k in ('resampler_policy','gain_policy'))
 prepared=source['resampler_policy'] in {'prepared-native-16000-float-v1','prepared-soxr-vhq-to-16000-v1'} and source['gain_policy']=='no-additional-gain-normalization'
 if not native and not prepared:raise ValueError('Unqualified source preparation')
 if prepared and ((source['resampler_policy']=='prepared-native-16000-float-v1')!=(source['original_sample_rate_hz']==16000)):raise ValueError('Source preparation rate differs')
 values,rate=sf.read(io.BytesIO(payload),dtype='float32',always_2d=True)
 if rate!=16000 or values.shape!=(row['input_samples16k'],1):raise ValueError('Prepared source samples/rate/channels differ')
 audio=torch.from_numpy(values.T.copy()).unsqueeze(0)
 if not bool(torch.isfinite(audio).all()):raise ValueError('Nonfinite prepared source')
 return audio

def frozen_versions(model):return {n:(p.data_ptr(),p._version,tuple(p.shape)) for n,p in model.state_dict().items()}


def target_coverage(crops):
 rows=[]
 for crop in crops:
  a,n=crop['context_frames']*1920,crop['valid_scored_samples']
  y=crop['teacher_audio'][0,0,a:a+n].double()
  full,tail=divmod(n,960)
  power=y[:full*960].reshape(full,960).square().sum(-1)
  lengths=torch.full((full,),960,dtype=torch.int64)
  if tail:
   power=torch.cat((power,y[full*960:].square().sum().reshape(1)))
   lengths=torch.cat((lengths,torch.tensor([tail])))
  rms=(power/lengths).sqrt(); quiet=rms<=1e-3;near=rms<=1e-5
  start_samples=crop['start_frame']*1920+torch.arange(len(lengths))*960
  startup=start_samples<38400
  rows.append({'source_id':crop['source_id'],'samples':n,'quiet_samples':int(lengths[quiet].sum()),
   'near_samples':int(lengths[near].sum()),'quiet_windows':int(quiet.sum()),'near_windows':int(near.sum()),
   'near_startup_samples':int(lengths[near&startup].sum()),'near_interior_samples':int(lengths[near&~startup].sum()),
   'start_frame':crop['start_frame'],'context_frames':crop['context_frames']})
 keys=('samples','quiet_samples','near_samples','quiet_windows','near_windows','near_startup_samples','near_interior_samples')
 return {'aggregate':{k:sum(r[k] for r in rows) for k in keys},'rows':rows,
  'definition':'Original teacher RMS on scored20ms windows including valid tails; quiet<=1e-3;near<=1e-5;startup before0.8s absolute source time. Measurement only; no crop or loss change.'}

def main():
 p=argparse.ArgumentParser()
 for name in ('plan','original-manifest','shards','assets'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--stop-index',type=int,default=36000);args=p.parse_args()
 if not START<args.stop_index<=STOP or args.stop_index%300:raise ValueError('Target production must remain within the authorized extension')
 base.policy()
 _,pools,_=base.load_data(args.original_manifest)
 provider=ContinuationData(args.plan,args.original_manifest,pools,args.shards)
 teacher=base.FrozenAudioVAE2.from_files(args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',device='cuda')
 if torch.backends.cudnn.version()!=92501:raise RuntimeError('Producer requires the qualified cuDNN9.25.1 backend')
 initial=frozen_versions(teacher.model)
 implementation_sha=sha(__file__);plan_sha=sha(args.plan)
 runtime={'torch':str(torch.__version__),'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'dtype':'float32','tf32':False,'deterministic':True,'forward_batch_sources':1}
 warmed=set();started=time.monotonic();produced=0
 with ThreadPoolExecutor(max_workers=4) as executor,torch.no_grad():
  for start in range(START,args.stop_index,300):
   stop=start+300
   provider.refresh_committed()
   if stop<=provider.retired_before:continue
   while not provider.can_produce_through(stop):
    base.event('waiting_for_checkpoint_cache_release',next_source=start,retired_before=provider.retired_before,maximum_cached_sources=6000)
    time.sleep(10)
   directory=args.shards/f'{start:06d}-{stop:06d}'
   if (directory/'receipt.json').is_file():provider.take(start,300);continue
   directory.mkdir(exist_ok=True)
   if shutil.disk_usage(args.shards).free < (2<<30):raise OSError('Target shard reserve below2GiB')
   rows=provider.plan['rows'][start:stop]
   # Four concurrent native audio reads, bounded to at most sixteen sources.
   futures={i:executor.submit(read_source,row) for i,row in enumerate(rows[:16])}
   crops,inventory=[],[]
   for i,row in enumerate(rows):
    audio=futures.pop(i).result()
    if i+16<len(rows):futures[i+16]=executor.submit(read_source,rows[i+16])
    frames=math.ceil(audio.shape[-1]/640)
    with torch.backends.cudnn.flags(enabled=False,benchmark=False,deterministic=True,allow_tf32=False):
     z=teacher.encode(audio.to('cuda'))
    if tuple(z.shape)!=(1,64,frames):raise ValueError('Full-source encoder geometry differs')
    if frames not in warmed:
     for _ in range(3):teacher.decode(z)
     warmed.add(frames)
    y=teacher.decode(z)
    # Repeated singleton execution after warmup must be stable before sealing.
    check=teacher.decode(z)
    if not torch.equal(y,check):raise RuntimeError('Full-source decoder repeat differs: '+row['source_id'])
    z,y=z.cpu(),y.cpu()
    provenance={'source_id':row['source_id'],'source_audio_sha256':row['manifest_row']['audio_sha256'],
      'input_samples16k':row['input_samples16k'],'reference_full_sha256':tensor_sha(audio),
      'latents_full_sha256':tensor_sha(z),'teacher_full_sha256':tensor_sha(y),
      'teacher_source_sha256':base.SOURCE_SHA256,'teacher_checkpoint_sha256':base.CHECKPOINT_SHA256}
    key=digest(provenance);length=row['context_frames']+row['scored_frames'];offset=row['context_start_frame']
    crop={k:row[k] for k in ('source_id','start_frame','context_start_frame','context_frames','scored_frames','valid_scored_samples')}
    crop.update(latents=window(z,offset,length),teacher_audio=window(y,offset*1920,length*1920),reference16k=window(audio,offset*640,length*640),cache_key=key)
    crops.append(crop);inventory.append(provenance);produced+=1
    if produced%30==0:base.event('fresh_targets',produced=produced,source_index=start+i+1,elapsed_seconds=time.monotonic()-started)
   if frozen_versions(teacher.model)!=initial or any(p.requires_grad for p in teacher.parameters()) or any(m.training for m in teacher.modules()):raise RuntimeError('Original teacher changed')
   if sha(args.plan)!=plan_sha or sha(__file__)!=implementation_sha:raise RuntimeError('Producer inputs changed')
   payload={'format':PAIR_VERSION,'plan_identity_sha256':provider.identity,'start_index':start,'stop_index':stop,'crops':crops}
   path=directory/'pairs.pt';temporary=directory/'pairs.tmp';torch.save(payload,temporary);temporary.replace(path)
   base.write_json(directory/'source-provenance.json',{'sources':inventory,'teacher':teacher.provenance,'runtime':runtime})
   coverage=target_coverage(crops)
   base.write_json(directory/'target-coverage.json',coverage)
   receipt={'format':PAIR_VERSION,'complete':True,'plan_identity_sha256':provider.identity,'start_index':start,'stop_index':stop,
    'source_ids':list(provider.source_ids[start:stop]),'source_ids_sha256':digest(list(provider.source_ids[start:stop])),
    'pairs_path':str(path),'pairs_sha256':sha(path),'teacher_source_sha256':base.SOURCE_SHA256,'teacher_checkpoint_sha256':base.CHECKPOINT_SHA256,
    'origin':'Authenticated prepared full sources; crop only after full-source teacher encoding and decoding',
    'encoder_policy':'Frozen FP32 raw_mu, full source singleton with cuDNN disabled',
    'decoder_policy':'Frozen FP32 full source singleton original48k decoder before cropping',
    'source_provenance_path':str(directory/'source-provenance.json'),'source_provenance_sha256':sha(directory/'source-provenance.json'),
    'target_coverage_sha256':sha(directory/'target-coverage.json'),'target_coverage':coverage['aggregate'],
    'runtime':runtime,'generator_sha256':implementation_sha,'parameter_updates':0,'teacher_remained_frozen':True,'singleton_decoder_repeat_bitwise':True}
   base.write_json(directory/'receipt.json',receipt)
   provider.take(start,300)
   progress={'complete':False,'sealed_source_prefix':stop,'authorized_source_interval':[START,args.stop_index],'new_sources_generated':produced,'elapsed_seconds':time.monotonic()-started,'plan_identity_sha256':provider.identity}
   base.write_json(args.shards/'producer-progress.json',progress);base.event('fresh_shard_sealed',**progress)
 provider.assert_unchanged()
 base.write_json(args.shards/'producer-complete.json',{'complete':True,'sealed_source_prefix':args.stop_index,'new_sources_generated':produced,'plan_identity_sha256':provider.identity,'elapsed_seconds':time.monotonic()-started,'runtime':runtime,'generator_sha256':implementation_sha,'training_updates':0})

if __name__=='__main__':
 # Separate producer lock: the explicitly concurrent trainer has no writer role.
 import sys
 lock_root=Path(sys.argv[sys.argv.index('--shards')+1]);lock_root.mkdir(parents=True,exist_ok=True)
 with (lock_root/'.producer.lock').open('a+b') as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  main()
