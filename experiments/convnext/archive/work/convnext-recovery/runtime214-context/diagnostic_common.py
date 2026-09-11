"""Immutable checkpoint/data loading for a separate diagnostic-only campaign."""
from __future__ import annotations
import copy,json,os,time,math
from pathlib import Path
from dataclasses import asdict
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from audiovae_student.cache import TrainingCrop
from audiovae_student.batching import _validate_crop
from audiovae_student.model import StudentConfig,StudentDecoder
from audiovae_student.recipe_v2 import RecipeV2Config,RecipeV2Engine
from audiovae_student.discriminators import AudioDiscriminators,DiscriminatorConfig
from audiovae_student.fusion_migration import build_fusion_engine
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.restart_data import file_sha,digest
from audiovae_student.preflight_distillation import _crop_identity
from audiovae_student.quiet_audio import QuietAudioConfig,quiet_window_metrics,_window_values

BASE=Path('/workspace/fast-audiovae-convnext-20260909-r9/remediation')
OLD=Path('/workspace/fast-audiovae-convnext-20260908-r1')
NAMES=('parent','targeted','complex')

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');os.replace(tmp,path)

def status(stage,**values):
    print(json.dumps({'stage':stage,**values,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}),flush=True)

def runtime_policy():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)

def crop_key(c):return (c.source_id,c.start_frame)

def crop_mask(c,p):
    mask=torch.zeros_like(p,dtype=torch.bool)
    skip=6 if c.context_start_frame>0 else 0
    mask[...,c.scored_slice.start+skip:c.scored_slice.stop]=True
    return mask

@torch.no_grad()
def pair_metrics(c,p):
    """Exact historical valid grid; diagnostic metric functions only."""
    if p.shape!=c.teacher_audio.shape:raise ValueError('Prediction sample shape mismatch')
    t=c.teacher_audio.to(p);mask=crop_mask(c,p)
    a,b=p[mask].double(),t[mask].double()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):raise ValueError('Nonfinite scored audio')
    error=a-b;n=a.numel();q=quiet_window_metrics(p,t,mask)
    qr=[w for w in q['windows'] if w['is_quiet']]
    qn=sum(w['valid_samples'] for w in qr)
    qe=sum(w['residual_rms']**2*w['valid_samples'] for w in qr)
    den=float(a.norm()*b.norm())
    return {'source_id':c.source_id,'start_frame':c.start_frame,'samples':n,
        'mae':float(error.abs().mean()),'mse':float(error.square().mean()),
        'waveform_cosine':float(a@b)/den if den else None,
        'teacher_rms':float(b.square().mean().sqrt()),'student_rms':float(a.square().mean().sqrt()),
        'quiet_samples':qn,'quiet_error_sum':qe,'quiet_residual_rms':math.sqrt(qe/qn) if qn else None,
        'quiet_windows':len(qr),'quiet_failed_windows':q['quiet_failed_count'],
        'peak':float(a.abs().max()),'teacher_peak':float(b.abs().max()),
        'overshoot_samples':int((a.abs()>1).sum()),
        'peak_excess_mse':float((a.abs()-1).clamp_min(0).square().mean())}

def aggregate_metrics(rows):
    if not rows:return None
    n=sum(r['samples'] for r in rows);qn=sum(r['quiet_samples'] for r in rows)
    active=[r['waveform_cosine'] for r in rows if r['teacher_rms']>=.001 and r['waveform_cosine'] is not None]
    return {'crops':len(rows),'sources':len({r['source_id'] for r in rows}),'samples':n,
        **{k:sum(r[k]*r['samples'] for r in rows)/n for k in ('mae','mse','peak_excess_mse')},
        'waveform_cosine_nonquiet_mean':sum(active)/len(active) if active else None,
        'quiet_samples':qn,'quiet_residual_rms':math.sqrt(sum(r['quiet_error_sum'] for r in rows)/qn) if qn else None,
        'quiet_windows':sum(r['quiet_windows'] for r in rows),'quiet_failed_windows':sum(r['quiet_failed_windows'] for r in rows),
        'peak':max(r['peak'] for r in rows),'overshoot_samples':sum(r['overshoot_samples'] for r in rows)}

@torch.no_grad()
def model_metrics_from_pairs(pairs):
    rows=[pair_metrics(c,p) for c,p in pairs]
    return {'aggregate':aggregate_metrics(rows),'rows':rows}

@torch.no_grad()
def model_metrics(model,crops,device):
    modes=[(m,m.training) for m in model.modules()]
    try:
        model.eval()
        return model_metrics_from_pairs((c,model(c.latents.to(device))) for c in crops)
    finally:
        for m,mode in modes:m.training=mode

class DiagnosticContext:
    def __init__(self):
        runtime_policy();self.out=Path('/tmp/fast-audiovae-recovery-20260909/runtime214-context');self.outPath=self.out
        self.out.mkdir(exist_ok=True)
        screen=BASE/'corrected-screen'
        self.identity=json.loads((screen/'identity.json').read_text())
        self.data=json.loads((screen/'targeted-data.json').read_text())
        if file_sha(screen/'targeted-data.json')!=self.identity['data_plan_sha256']:raise ValueError('Data bytes changed')
        if digest({k:v for k,v in self.data.items() if k!='identity_sha256'})!=self.data['identity_sha256']:raise ValueError('Data identity changed')
        paths={'parent':BASE/'fusion-screen/parent.pt','targeted':screen/'targeted/final.pt','complex':screen/'targeted_complex/final.pt'}
        summary=json.loads((screen/'summary.json').read_text())
        expected={'parent':summary['parent_sha256'],'targeted':summary['arms']['targeted']['checkpoint_sha256'],
                  'complex':summary['arms']['targeted_complex']['checkpoint_sha256']}
        self.paths=paths;self.expected_hashes=expected
        for name,path in paths.items():
            if file_sha(path)!=expected[name]:raise ValueError('Checkpoint changed '+name)
        self.parent=torch.load(paths['parent'],map_location='cpu',weights_only=True,mmap=True)
        self.checkpoints={name:torch.load(path,map_location='cpu',weights_only=True,mmap=True) for name,path in paths.items()}
        receipt=json.loads((screen/'target-cache.json').read_text());cache=Path(receipt['path'])
        status('verifying_frozen_target_cache')
        if file_sha(cache)!=receipt['sha256']:raise ValueError('Target cache changed')
        saved=torch.load(cache,map_location='cpu',weights_only=True,mmap=True)
        if saved['identity']!=receipt['identity']:raise ValueError('Target cache identity differs')
        self.heldout=[TrainingCrop(**c) for c in saved['heldout']]
        self.pools={name:[TrainingCrop(**c) for c in crops] for name,crops in saved['pools'].items()}
        self.metadata=saved['metadata']
        if _crop_identity(self.heldout)!=self.identity['targets']['heldout']['crops']:raise ValueError('Heldout differs')
        for c in self.heldout:_validate_crop(c)
        if len(self.heldout)!=285:raise ValueError('Missing heldout crops')
        self.before_reports={name:json.loads((screen/('regular' if name=='parent' else 'targeted_complex' if name=='complex' else name)/('before.json' if name=='parent' else 'after.json')).read_text()) for name in NAMES}
        self.probe_crops=self._probes()
        self.receipt={'format_version':1,'checkpoints':{name:{'path':str(paths[name]),'sha256':expected[name],
                      'step':self.checkpoints[name]['engine']['step']} for name in NAMES},
            'target_cache_sha256':receipt['sha256'],'heldout_crops':len(self.heldout),
            'heldout_sources':len({c.source_id for c in self.heldout}),
            'probe_crops':[{'source_id':c.source_id,'start_frame':c.start_frame,'metadata':self.metadata[c.source_id]} for c in self.probe_crops],
            'probe_selection':'Union overshoot sources, real encoded fixtures, source-diverse expressive cases and speech controls; fixed across checkpoints',
            'student_parameter_updates_to_retained_models':0,'original_run':'paused',
            'device_use':'H100 training diagnostics, not an RTF benchmark','torch_version':str(torch.__version__)}
        existing=self.out/'context.json'
        if existing.exists() and json.loads(existing.read_text())!=self.receipt:raise ValueError('Diagnostic context differs')
        atomic_json(existing,self.receipt)

    def _probes(self):
        by_key={crop_key(c):c for c in self.heldout};selected={};seen=set()
        def add(c):
            if c.source_id not in seen:selected[crop_key(c)]=c;seen.add(c.source_id)
        overs={}
        for report in self.before_reports.values():
            for row in report['rows']:
                if row['student_overshoot_samples']:
                    old=overs.get(row['source_id'])
                    if old is None or row['student_peak_abs']>old['student_peak_abs']:overs[row['source_id']]=row
        for r in sorted(overs.values(),key=lambda r:r['source_id']):add(by_key[(r['source_id'],r['start_frame'])])
        for c in self.heldout:
            if c.source_id.startswith('encoded_'):add(c)
        for condition in ('Laughter','Screaming','Breathing','Whispering','human_whistling_source_description','Yell'):
            for c in self.heldout:
                if self.metadata[c.source_id].get('condition')==condition and c.source_id not in seen:
                    add(c);break
        for c in self.heldout:
            if len(selected)>=18:break
            if self.metadata[c.source_id].get('condition')=='speech':add(c)
        return list(selected.values())

    def engine(self,name,device='cuda'):
        if name not in NAMES:raise ValueError('Unknown checkpoint')
        state=copy.deepcopy(self.checkpoints[name]['engine'])
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device=='cuda' else []):
            if name=='parent':
                model=StudentDecoder(StudentConfig(**state['model_config'])).to(device)
                bank=AudioDiscriminators(DiscriminatorConfig(**state['discriminator_config'])).to(device)
                engine=RecipeV2Engine(model,recipe=RecipeV2Config(**state['recipe']),discriminators=bank)
                engine.load_state_dict(state)
            else:
                engine,_=build_fusion_engine(self.parent['engine'],'complex' if name=='complex' else 'control',device=device)
                for key,obj in [('model',engine.model),('discriminators',engine.discriminators),('optimizer',engine.optimizer),
                                ('discriminator_optimizer',engine.discriminator_optimizer),('balancer',engine.balancer)]:
                    obj.load_state_dict(state[key])
                engine.crop_generator.set_state(state['crop_rng'].cpu())
                for key in ('step','calibration','perceptual_start','discriminator_updates','fusion_migration'):
                    setattr(engine,key,copy.deepcopy(state[key]))
            if state_fingerprint(engine.state_dict())!=state_fingerprint(state):raise ValueError('Exact engine restore failed '+name)
        return engine

    def teacher(self):
        from audiovae_student.teacher import FrozenAudioVAE2
        teacher=FrozenAudioVAE2.from_files(OLD/'assets/audio_vae_v2.py',OLD/'assets/audiovae.pth',device='cuda')
        if state_fingerprint(teacher.model.state_dict())!=self.parent['identity']['data']['teacher_state_sha256']:
            raise ValueError('Teacher state differs')
        return teacher

    def source_audio(self,crop):
        if crop.reference16k is None:raise ValueError('Missing reference input')
        valid=(crop.context_frames*1920+crop.valid_scored_samples)//3
        return crop.reference16k[...,:valid].clone(),{'true_start':crop.context_start_frame==0,
            'valid_input_samples':valid,'absolute_context_start_frame':crop.context_start_frame,
            'scope':'Supplied real waveform prefix/window, not an assertion of complete original recording'}

    def verify_files(self):
        for name,path in self.paths.items():
            if file_sha(path)!=self.expected_hashes[name]:raise ValueError('Retained checkpoint changed')
        return {'retained_checkpoint_hashes_unchanged':True}

def load_context():return DiagnosticContext()
