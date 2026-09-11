"""Newly authorized reserve12000:24000, with the original frozen singleton recipe."""
from __future__ import annotations
import argparse,fcntl,hashlib,io,json,math,os,shutil,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import soundfile as sf
import torch
from torch.nn import functional as F
import run_pilot as base
from fresh_training_data import FreshTrainingData,PAIR_VERSION,digest,sha

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

def main():
 p=argparse.ArgumentParser()
 for name in ('plan','original-manifest','shards','assets'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--start-index',type=int,default=12000);p.add_argument('--stop-index',type=int,default=24000);args=p.parse_args()
 if args.start_index!=12000 or args.stop_index!=24000:raise ValueError('Only newly authorized source interval12000:24000 may be produced')
 base.policy()
 _,pools,_=base.load_data(args.original_manifest)
 provider=FreshTrainingData(args.plan,args.original_manifest,pools,args.shards)
 if provider.identity!='3c65151fd2b37900257e179fee62218c3d33055c94d47dbb7c213593777e7c7d':raise ValueError('Original sealed source plan changed')
 provider.take(0,300)
 teacher=base.FrozenAudioVAE2.from_files(args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',device='cuda')
 if torch.backends.cudnn.version()!=92501:raise RuntimeError('Producer requires the qualified cuDNN9.25.1 backend')
 initial=frozen_versions(teacher.model)
 implementation_sha=sha(__file__);plan_sha=sha(args.plan)
 runtime={'torch':str(torch.__version__),'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'dtype':'float32','tf32':False,'deterministic':True,'forward_batch_sources':1}
 warmed=set();started=time.monotonic();produced=0
 with ThreadPoolExecutor(max_workers=4) as executor,torch.no_grad():
  for start in range(args.start_index,args.stop_index,300):
   stop=start+300;directory=args.shards/f'{start:06d}-{stop:06d}'
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
   receipt={'format':PAIR_VERSION,'complete':True,'plan_identity_sha256':provider.identity,'start_index':start,'stop_index':stop,
    'source_ids':list(provider.source_ids[start:stop]),'source_ids_sha256':digest(list(provider.source_ids[start:stop])),
    'pairs_path':str(path),'pairs_sha256':sha(path),'teacher_source_sha256':base.SOURCE_SHA256,'teacher_checkpoint_sha256':base.CHECKPOINT_SHA256,
    'origin':'Authenticated prepared full sources; crop only after full-source teacher encoding and decoding',
    'encoder_policy':'Frozen FP32 raw_mu, full source singleton with cuDNN disabled',
    'decoder_policy':'Frozen FP32 full source singleton original48k decoder before cropping',
    'source_provenance_path':str(directory/'source-provenance.json'),'source_provenance_sha256':sha(directory/'source-provenance.json'),
    'runtime':runtime,'generator_sha256':implementation_sha,'parameter_updates':0,'teacher_remained_frozen':True,'singleton_decoder_repeat_bitwise':True}
   base.write_json(directory/'receipt.json',receipt)
   provider.take(start,300)
   progress={'complete':False,'sealed_source_prefix':stop,'approved_start_index':args.start_index,'approved_stop_index':args.stop_index,'new_sources_generated':produced,'elapsed_seconds':time.monotonic()-started,'plan_identity_sha256':provider.identity}
   base.write_json(args.shards/'producer-reserve-v1-progress.json',progress);base.event('fresh_shard_sealed',**progress)
 provider.assert_unchanged()
 base.write_json(args.shards/'producer-reserve-v1-complete.json',{'complete':True,'sealed_source_prefix':args.stop_index,'new_sources_generated':produced,'authorized_source_interval':[args.start_index,args.stop_index],'plan_identity_sha256':provider.identity,'elapsed_seconds':time.monotonic()-started,'runtime':runtime,'generator_sha256':implementation_sha,'training_updates':0})

if __name__=='__main__':
 # Separate producer lock: the explicitly concurrent trainer has no writer role.
 import sys
 lock_root=Path(sys.argv[sys.argv.index('--shards')+1]);lock_root.mkdir(parents=True,exist_ok=True)
 with (lock_root/'.producer.lock').open('a+b') as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  main()
