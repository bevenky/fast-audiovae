"""Read a consistent saved checkpoint and audit deployment/training invariants on CPU."""
import hashlib,json,os,shutil,time
from pathlib import Path
from dataclasses import asdict
import torch
from audiovae_student.cache import TrainingCrop
from audiovae_student.model import StudentDecoder,StudentConfig
from audiovae_student.recipe_v2 import RecipeV2Engine,RecipeV2Config,scored_batch_v2
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.restart_data import canonical
BASE=Path('/workspace/fast-audiovae-convnext-20260909-r9')
OUT=BASE/'audit-step1000'
OUT.mkdir(exist_ok=True)
torch.set_num_threads(1)
# The open descriptor remains one immutable checkpoint even if latest.pt is replaced.
with (BASE/'training-runs/decoder-recipe-v2/latest.pt').open('rb') as handle:
    payload=torch.load(handle,map_location='cpu',weights_only=True)
    step=payload['engine']['step']
    path=OUT/f'checkpoint-step{step:06d}.pt'
    if path.exists(): raise ValueError('Audit snapshot already exists')
    handle.seek(0)
    h=hashlib.sha256()
    with path.open('xb') as output:
        while block:=handle.read(8*1024*1024):
            output.write(block);h.update(block)
report={'step':step,'checkpoint_path':str(path),'checkpoint_sha256':h.hexdigest(),
        'checkpoint_bytes':path.stat().st_size,'device':'cpu','optimizer_updates_during_audit':0}
engine=RecipeV2Engine(StudentDecoder(StudentConfig(**payload['engine']['model_config'])),
    recipe=RecipeV2Config(**payload['engine']['recipe']))
engine.load_state_dict(payload['engine'])
engine.model.eval()
model=engine.model
report.update(exact_recipe_load_passed=True,discriminator_updates=engine.discriminator_updates,
              sampler_cursor=payload['sampler']['cursor'],calibration=engine.calibration,
              model_state_sha256=state_fingerprint(model.state_dict()),gradient_balancer=engine.balancer.state_dict())
def stats(x):
    x=x.detach().float().flatten()
    return {'min':float(x.min()),'p01':float(torch.quantile(x,.01)),'median':float(x.median()),
            'p99':float(torch.quantile(x,.99)),'max':float(x.max()),'rms':float(x.square().mean().sqrt())}
report['normalization']={}
for name in ('stem_norm','affine'):
    norm=getattr(model,name); scale,bias=norm.fixed_affine()
    report['normalization'][name]={'statistics_frozen':bool(norm.statistics_frozen),
        'running_mean':stats(norm.running_mean),'running_var':stats(norm.running_var),
        'affine_scale':stats(scale),'affine_bias':stats(bias),'gamma':stats(norm.weight)}
report['phase_bias']=stats(model.adapter.phase_bias)
report['layer_scales']=[stats(block.scale) for block in model.blocks]
report['parameter_groups']={name:{'parameters':sum(p.numel() for g in opt.param_groups for p in g['params']),
    'learning_rates':[g['lr'] for g in opt.param_groups], 'steps':sorted(set(int(v['step']) for v in opt.state.values() if 'step' in v))}
    for name,opt in engine.optimizer.optimizers.items()}
# Select deterministic diagnostic examples by existing labels and stored teacher level.
saved=torch.load(BASE/'heldout-targets.pt',map_location='cpu',weights_only=True)
crops=[TrainingCrop(**v) for v in saved['crops']]
meta=payload['identity']['heldout_metadata']
bycond={}
for c in crops:
    key=meta[c.source_id].get('condition')
    bycond.setdefault(key,[]).append(c)
choices=[]
for key, pattern in [('speech','speech'),('laughter','laugh'),('screaming','scream'),('whistling','whistl')]:
    matching=[c for condition,items in bycond.items() if pattern in str(condition).lower() for c in items]
    if matching: choices.append((key,matching[0]))
quiet=min(crops,key=lambda c:float(c.teacher_audio[...,c.scored_slice].square().mean()))
choices.append(('quietest_panel_crop',quiet))
report['diagnostic_examples']=[]
model_before=state_fingerprint(model.state_dict())
folded=model.fold_normalization()
for condition,crop in choices:
    row={'condition':condition,'source_id':crop.source_id,'start_frame':crop.start_frame,
         'latent_rms':float(crop.latents.square().mean().sqrt()),'valid_samples':crop.valid_scored_samples}
    z=crop.latents
    with torch.no_grad():
        full=model(z);state=model.initial_state();parts=[]
        for i in range(z.shape[-1]):
            y,state=model.forward_stream(z[...,i:i+1],state);parts.append(y)
        stream=torch.cat(parts,-1);fold=folded(z)
        row.update(streaming_max_abs_error=float((full-stream).abs().max()),
                   folded_max_abs_error=float((full-fold).abs().max()),
                   predicted_rms=float(full[...,crop.scored_slice].square().mean().sqrt()),
                   teacher_rms=float(crop.teacher_audio[...,crop.scored_slice].square().mean().sqrt()))
        torch.testing.assert_close(full,stream,atol=3e-5,rtol=3e-5)
        torch.testing.assert_close(full,fold,atol=3e-5,rtol=3e-5)
    batch,result=scored_batch_v2(model,[crop],engine.reconstruction,'cpu')
    row['losses']={k:float(v.detach()) for k,v in result.losses.items()}
    row['output_gradient_norms']={}
    for loss in ('teacher_waveform','teacher_mel'):
        gradient,=torch.autograd.grad(result.losses[loss],batch.prediction,retain_graph=True)
        row['output_gradient_norms'][loss]=float(gradient.norm())
    report['diagnostic_examples'].append(row)
from audiovae_student.distillation_training import evaluate_crops, waveform_gate
from audiovae_student.representative_pilot import summarize_evaluation
values=evaluate_crops(engine,crops,include_signal_checks=True)
full_evaluation={'step':step,'checkpoint_sha256':report['checkpoint_sha256'],'device':'cpu',
 'rows':values,'groups':summarize_evaluation(values,meta),'gate':waveform_gate(engine,values)}
(OUT/f'evaluation-step{step:06d}.json').write_bytes(canonical(full_evaluation))
report['evaluation_summary']=full_evaluation['groups']
report['final_reconstruction_goal_passed']=full_evaluation['gate']['passed']
report['model_unchanged_by_audit']=state_fingerprint(model.state_dict())==model_before
report['stream_and_fold_parity_passed']=True
report['state']='passed'
# Do not copy the very large source-corpus provenance into the concise report.
cal=report['calibration']
report['calibration']={'completed_step':cal['completed_step'],'report_sha256':cal['report_sha256'],
 'fixed_buffers_sha256':cal['fixed_buffers_sha256'],'parameters_unchanged':cal['report']['parameters_unchanged'],
 'stem':cal['report']['stem'],'final':cal['report']['final'],'split':cal['report']['provenance']['split']}
(OUT/'checkpoint-audit.json').write_bytes(canonical(report))
print(json.dumps({k:v for k,v in report.items() if k in ['step','state','checkpoint_path','checkpoint_sha256',
 'discriminator_updates','sampler_cursor','model_unchanged_by_audit','stream_and_fold_parity_passed','diagnostic_examples']}))
