"""Two-source FP64 closure check with the ACTUAL FP32 operator coefficients.

Both original teacher and saved sliced-step0 effective WN weights are frozen
before casting. This does not reslice or repair the saved student's coefficients.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import time

import torch
from torch import nn
import diagnose_progressive_silence as audit

group, base, control = audit.group, audit.base, audit.control
VERSION = 'audiovae2_progressive_restoration_precision_v1'
CASES = ('as_in:train:13453488421825608782.wav',
         'thorsten_emotional:whisper/37773d61d5c91c12d49e55e2b041c0cd.wav')
TIGHT_ATOL = TIGHT_RTOL = 1e-10


@torch.no_grad()
def _fold_double(decoder):
    cloned = group.clone_teacher(decoder)
    rows = []
    for name,source in decoder.named_modules():
        if not isinstance(source,(nn.Conv1d,nn.ConvTranspose1d)): continue
        target = cloned.get_submodule(name)
        expected = group.effective_weight(source).detach()
        if expected.dtype != torch.float32: raise ValueError('Precision control starts with actual FP32 coefficients')
        if group._legacy_weight_hook(target) is not None:
            torch.nn.utils.remove_weight_norm(target)
        target.weight.copy_(expected)
        if not torch.equal(target.weight,expected): raise RuntimeError('Materialization changed an effective coefficient')
        rows.append({'name':name,'shape':list(expected.shape),'fp32_coefficients_preserved_exactly':True})
    cloned.double().requires_grad_(False).eval()
    for name,source in decoder.named_modules():
        target = cloned.get_submodule(name)
        if isinstance(source,(nn.Conv1d,nn.ConvTranspose1d)):
            if not torch.equal(target.weight,group.effective_weight(source).detach().double()):
                raise RuntimeError('Casting changed the exactly represented original FP32 coefficient')
            if target.bias is not None and not torch.equal(target.bias,source.bias.double()):
                raise RuntimeError('Casting changed a bias coefficient')
    return cloned,rows


@torch.no_grad()
def fold_existing_pair(teacher_decoder,pristine_student,selection):
    authenticated = audit.validate_pristine(teacher_decoder,pristine_student,selection)
    teacher64,trows = _fold_double(teacher_decoder)
    decoder64,srows = _fold_double(pristine_student.decoder)
    student64 = group.CompressedDecoderGroup(decoder64,selection,48000).requires_grad_(False).eval()
    # A new explicit proof scope, tied to materialized original step0, rather
    # than bypassing the WN-based original restoration authorization token.
    token = (id(teacher64),id(student64),audit._versions(teacher64),audit._versions(student64.decoder))
    student64._precision_original_step0_token = token
    return teacher64,student64,{'original_step0':audit._public_pristine(authenticated),
        'teacher_materialized':trows,'student_materialized':srows,
        'all_fp32_coefficients_preserved_exactly':True,
        'student_resliced_in_fp64':False,'forward_dtype':'torch.float64'}


def _check_pair(teacher64,student64):
    actual = (id(teacher64),id(student64),audit._versions(teacher64),audit._versions(student64.decoder))
    if getattr(student64,'_precision_original_step0_token',None) != actual:
        raise ValueError('FP64 proof requires unchanged materialized authenticated step0 coefficients')


def detailed_compare(actual,expected):
    result = control.compare_tensors(actual,expected)
    a,b = actual.detach(),expected.detach(); difference = (a-b).abs()
    bad = difference > audit.ATOL + audit.RTOL*b.abs()
    flat = difference.reshape(-1); location = int(flat.argmax())
    channel = location//actual.shape[-1] % actual.shape[1]; frame = location%actual.shape[-1]
    coordinates = torch.nonzero(bad,as_tuple=False)
    result.update(tight_fp64_passed=bool(torch.allclose(a,b,atol=TIGHT_ATOL,rtol=TIGHT_RTOL)),
        tight_atol=TIGHT_ATOL,tight_rtol=TIGHT_RTOL,existing_tolerance_failed_elements=int(bad.sum()),
        maximum_error_channel=channel,maximum_error_frame=frame,
        maximum_error_expected=float(b.reshape(-1)[location]),maximum_error_actual=float(a.reshape(-1)[location]),
        first_failed_coordinates=coordinates[:8].cpu().tolist(),
        relative_rms=result['residual_rms']/result['expected_rms'] if result['expected_rms'] else None)
    return result


@torch.no_grad()
def precision_forward(teacher64,student64,z,selection):
    _check_pair(teacher64,student64)
    if z.dtype != torch.float64: raise ValueError('The original cached latent values must be cast exactly to FP64')
    sites = audit.four_sites(teacher64,selection)
    with audit.capture_sites(teacher64,sites) as captured, control.capture_boundaries(teacher64) as native_boundaries:
        native = control.native_decoder_forward(teacher64,z)
    native_boundaries['waveform'] = native
    weight_rows = {}
    for site in sites:
        source = teacher64.get_submodule(site.path); target = student64.decoder.get_submodule(site.path)
        w = group.effective_weight(source)
        selected = w[site.kept_inputs][:,site.kept_outputs] if isinstance(source,nn.ConvTranspose1d) else w[site.kept_outputs][:,site.kept_inputs]
        weight_rows[site.name] = {'weight':control.compare_tensors(group.effective_weight(target),selected),
                                 'bias':control.compare_tensors(target.bias,source.bias[site.kept_outputs])}
    with ExitStack() as stack:
        for site in sites:
            missing = audit.common.dropped_response(teacher64.get_submodule(site.path),captured[site.name]['input'],site)
            stack.enter_context(audit.common.temporary_output_change(student64.decoder.get_submodule(site.path),
                lambda output,value=missing:output+value))
        with audit.capture_sites(student64.decoder,sites) as restored_sites, control.capture_boundaries(student64.decoder) as restored_boundaries:
            restored = student64.forward_from_latents(z)['waveform']
    restored_boundaries['waveform'] = restored
    checks = {}
    for name,value in native_boundaries.items():
        expected = value[:,selection['stage2_indices']] if name in ('stage2_output','stage3_conditioned_input') else value
        checks[name] = detailed_compare(restored_boundaries[name],expected)
    local = {site.name:detailed_compare(restored_sites[site.name]['output'],captured[site.name]['output'][:,site.kept_outputs]) for site in sites}
    _check_pair(teacher64,student64)
    return {'all_boundaries_original_tolerance':all(v['allclose_existing'] for v in checks.values()),
            'all_boundaries_tight_fp64':all(v['tight_fp64_passed'] for v in checks.values()),
            'boundaries':checks,'restored_local_mixers':local,'actual_selected_weight_differences':weight_rows,
            'scope':'Preserves each actual FP32 effective operator coefficient; includes inherited WN slicing roundoff, no teacher/student coefficient repair'}


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('step0','manifest','assets','out'): parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists(): raise FileExistsError('Use a fresh precision-proof directory')
    if base.sha(args.step0)!=audit.STEP0_SHA: raise ValueError('Expected preserved current-cut step0 checkpoint')
    if base.sha(args.assets/'audio_vae_v2.py')!=base.SOURCE_SHA256 or base.sha(args.assets/'audiovae.pth')!=base.CHECKPOINT_SHA256:
        raise ValueError('Original teacher assets differ')
    process=control.gpu_idle_snapshot()
    manifest,pools,_=base.load_data(args.manifest)
    by_id={c['source_id']:c for c in pools['development']}
    crops=[by_id[s] for s in CASES]
    payload=torch.load(args.step0,map_location='cpu',weights_only=True,mmap=True)
    selection=payload['selection']
    if payload['cut_updates']!=0 or payload['cut_index']!=1 or tuple(len(selection[k]) for k in audit.progressive.KEYS)!=(384,256):
        raise ValueError('This proof is restricted to the original384/256 sliced step0')
    paths=[args.step0,args.manifest,args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',
           Path(manifest['cache_path']),Path(__file__),Path(audit.__file__),Path(group.__file__)]
    protected={str(p.resolve()):base.sha(p) for p in paths}
    base.policy();args.out.mkdir(parents=True)
    base.write_json(args.out/'launch.json',{'version':VERSION,'protected':protected,'process':process,'source_ids':list(CASES),
        'torch':str(torch.__version__),'cudnn':torch.backends.cudnn.version(),'parameter_updates':0,
        'original_atol':audit.ATOL,'original_rtol':audit.RTOL,'tight_atol':TIGHT_ATOL,'tight_rtol':TIGHT_RTOL})
    started=time.monotonic();failure=None;states={};models={};reports=[]
    try:
        teacher=base.FrozenAudioVAE2.from_files(args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',device='cuda')
        initial=audit.progressive.initialize_from_teacher(teacher.model.decoder,selection)
        initial.load_group_state_dict(payload['group']);initial.eval()
        teacher64,student64,receipt=fold_existing_pair(teacher.model.decoder,initial,selection)
        models={'teacher32':teacher.model.decoder,'student32':initial.decoder,'teacher64':teacher64,'student64':student64.decoder}
        states={k:control.state_hash(v) for k,v in models.items()}
        base.write_json(args.out/'coefficient-preservation.json',receipt)
        for crop in crops:
            z=crop['latents'].to(device='cuda',dtype=torch.float64)
            if not torch.equal(z.float().cpu(),crop['latents']): raise ValueError('Cached latent cast changed values')
            for _ in range(3):
                control.native_decoder_forward(teacher64,z);student64.forward_from_latents(z)
            report=precision_forward(teacher64,student64,z,selection)
            report.update(source_id=crop['source_id'],latent_shape=list(z.shape),start_frame=crop['start_frame'],
                          context_start_frame=crop['context_start_frame'],context_frames=crop['context_frames'],
                          valid_scored_samples=crop['valid_scored_samples'],extra_warmup_forwards=6)
            reports.append(report);base.write_json(args.out/'precision-results.json',reports)
    except BaseException as exc:
        failure=repr(exc);raise
    finally:
        files=all(base.sha(p)==s for p,s in protected.items())
        unchanged={k:control.state_hash(v)==states[k] for k,v in models.items()}
        base.write_json(args.out/'completed.json',{'version':VERSION,'failure':failure,'sources':len(reports),
            'complete':failure is None and len(reports)==2 and files and all(unchanged.values()),
            'all_boundaries_original_tolerance':bool(reports) and all(r['all_boundaries_original_tolerance'] for r in reports),
            'all_boundaries_tight_fp64':bool(reports) and all(r['all_boundaries_tight_fp64'] for r in reports),
            'files_preserved':files,'states_preserved':unchanged,'initial_state_hashes':states,
            'elapsed_seconds':time.monotonic()-started,'parameter_updates':0})
        if not files or not all(unchanged.values()): raise RuntimeError('Precision control changed protected originals')


if __name__=='__main__': main()
