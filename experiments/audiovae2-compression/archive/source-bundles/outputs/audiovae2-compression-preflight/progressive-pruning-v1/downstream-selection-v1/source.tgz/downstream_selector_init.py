"""Four-consumer teacher-conditioned selection, then unchanged native B repair.

One calibration-only 512->384 support, no neural optimizer or fallback search.
The original step0 checkpoint authenticates provenance/RNG only: its old group
tensors must never initialize a candidate with the newly selected channels.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import sys
import time

import torch

import reconstruction_aware_init as ordinary

audit, common = ordinary.audit, ordinary.common
group, base, control, progressive = ordinary.group, ordinary.base, ordinary.control, ordinary.progressive
VERSION = 'audiovae2_downstream_selector_initialization_v1'
VARIANT = 'downstream_selection_B'
PATHS = ('model.3.block.2.block.3', 'model.3.block.3.block.3',
         'model.3.block.4.block.3', 'model.4.block.1')
NAMES = ('stage2_ru1', 'stage2_ru2', 'stage2_ru3', 'stage3_up')
B_FITTER_SHA = 'de5362361b1dfd387c9506fefa93e13435d772f784fcc344390791265efa1913'
CHUNK_ROWS = 1024
ROUNDING_FACTOR = 4096  # Fixed FP64 arithmetic bound, not a waveform threshold.


def teacher_sites(decoder):
    """Capture complete teacher consumers; output coordinates never shrink."""
    group._validate_decoder(decoder)
    c2, c3 = (decoder.model[i].block[1].out_channels for i in (3, 4))
    sites = [common.Operation(name, path, list(range(c2)), list(range(c2)), 1200)
             for name, path in zip(NAMES[:3], PATHS[:3])]
    sites.append(common.Operation(NAMES[-1], PATHS[-1], list(range(c2)), list(range(c3)), 6000, 5))
    for site in sites:
        m = decoder.get_submodule(site.path)
        native = ordinary._validate_module(m)
        if m.in_channels != c2 or m.out_channels != (c3 if native else c2):
            raise ValueError('Teacher consumer does not retain its complete channel axes')
    return sites


@torch.no_grad()
def accumulate_consumer(stats, module, x, weights, *, chunk_rows=CHUNK_ROWS):
    """Sum a complete bias-free output contribution Gram, without centering.

    Native phase p uses x[t]*W[p] + x[t-1]*W[p+5]. Masking happens after
    fetching previous context; only actual tensor-start history is zero.
    """
    native = ordinary._validate_module(module)
    stride = 5 if native else 1
    if (x.ndim != 3 or not x.is_floating_point() or x.shape[1] != module.in_channels
            or x.shape[-1] < 1 or weights.shape != (x.shape[0], 1, x.shape[-1]*stride)
            or not torch.isfinite(weights).all() or (weights < 0).any()
            or (weights > (8 if native else 40)).any()
            or type(chunk_rows) is not int or chunk_rows < 1):
        raise ValueError('Invalid complete consumer input or scored-cell weights')
    weight = group.effective_weight(module).detach().double()
    if not torch.isfinite(weight).all(): raise ValueError('Nonfinite teacher effective coefficient')
    if stats is None:
        stats = {'gram': weight.new_zeros((module.in_channels, module.in_channels)),
                 'energy': weight.new_zeros(()), 'weight': weight.clone(),
                 'native': native, 'weighted_samples': 0., 'valid_rows': 0,
                 'partial_rows': 0, 'observations': 0, 'output_channels': module.out_channels}
        if native:
            stats['products'] = [(weight[:,:,p]@weight[:,:,p].T,
                                  weight[:,:,p+5]@weight[:,:,p+5].T,
                                  weight[:,:,p]@weight[:,:,p+5].T) for p in range(5)]
        else:
            w = weight[...,0]; stats['products'] = [w.T@w]
    if (stats['native'] != native or stats['weight'].shape != weight.shape
            or not torch.equal(stats['weight'], weight)):
        raise ValueError('Consumer weights changed during teacher-conditioned selection')
    if stats['gram'].device != x.device: raise ValueError('Consumer and activations must share a device')
    for start in range(0, x.shape[-1], chunk_rows):
        stop = min(x.shape[-1], start+chunk_rows)
        current = x[...,start:stop].detach().movedim(1,-1).reshape(-1,module.in_channels).double()
        if native:
            frames = torch.arange(start,stop,device=x.device)
            previous = x.index_select(-1,(frames-1).clamp_min(0)).detach().double()
            previous = torch.where((frames>0)[None,None,:], previous, torch.zeros_like(previous))
            previous = previous.movedim(1,-1).reshape_as(current)
        for phase in range(stride):
            q = weights[...,start*stride+phase:stop*stride:stride].reshape(-1).double()
            valid = q>0
            if not valid.any(): continue
            q, a = q[valid], current[valid]
            if not torch.isfinite(a).all(): raise ValueError('Nonfinite scored teacher consumer input')
            aa = a.T@(a*q[:,None])
            if native:
                b = previous[valid]
                if not torch.isfinite(b).all(): raise ValueError('Nonfinite scored previous-frame context')
                bb, ab = b.T@(b*q[:,None]), a.T@(b*q[:,None])
                wa, wb, wab = stats['products'][phase]
                cross = ab*wab
                stats['gram'].add_(aa*wa + bb*wb + cross + cross.T)
                output = a@weight[:,:,phase] + b@weight[:,:,phase+5]
            else:
                stats['gram'].add_(aa*stats['products'][0])
                output = a@weight[...,0].T
            # Independently computed, complete bias-free output energy.
            stats['energy'].add_((output.square()*q[:,None]).sum())
            stats['weighted_samples'] += float(q.sum())
            stats['valid_rows'] += len(q)
            stats['partial_rows'] += int((q < (8 if native else 40)).sum())
    stats['observations'] += 1
    return stats


def normalize_grams(statistics):
    """Equal-site relative output error; preserve all signed cross terms."""
    if set(statistics) != set(NAMES): raise ValueError('Exactly the four declared consumers are required')
    result = {}; matrices = []; width = None
    eps = torch.finfo(torch.float64).eps
    for name in NAMES:
        s = statistics[name]; gram = s['gram'].detach().cpu().double(); energy = float(s['energy'])
        if (gram.ndim != 2 or gram.shape[0] != gram.shape[1] or not torch.isfinite(gram).all()
                or not math.isfinite(energy) or energy <= 0 or s['valid_rows'] <= 0):
            raise ValueError('Each full consumer needs finite positive output energy and valid rows')
        if width is not None and gram.shape[0] != width: raise ValueError('Consumers must share the same input support')
        width = gram.shape[0]
        absolute_contraction = float(gram.abs().sum())
        bound = ROUNDING_FACTOR*eps*max(absolute_contraction,energy)
        symmetry = float((gram-gram.T).abs().max())
        gap = abs(float(gram.sum())-energy)
        if symmetry > bound or gap > bound or float(gram.diag().min()) < -bound:
            raise ValueError('Contribution Gram fails the fixed FP64 energy/symmetry identity')
        gram = (gram+gram.T)/2
        normalized = gram/energy; matrices.append(normalized)
        result[name] = {'gram':gram, 'normalizer':energy, 'normalized_gram':normalized,
                        'energy_from_gram':float(gram.sum()), 'energy_identity_abs_error':gap,
                        'arithmetic_bound':bound, 'symmetry_max_abs':symmetry,
                        **{k:s[k] for k in ('weighted_samples','valid_rows','partial_rows','observations','output_channels')}}
    k = sum(matrices)/4
    if not torch.isfinite(k).all(): raise ValueError('Nonfinite normalized selector objective')
    return k, result


def greedy_remove(k, keep):
    """Deterministic CPU FP64 removal; exact ties pick smallest original ID."""
    if (k.device.type != 'cpu' or k.dtype != torch.float64 or k.ndim != 2
            or k.shape[0] != k.shape[1] or not torch.isfinite(k).all()
            or not torch.equal(k,k.T) or type(keep) is not int or not 0<keep<k.shape[0]):
        raise ValueError('Use a finite symmetric saved CPU FP64 Gram and strict width cut')
    n=k.shape[0]; removed=[]; trace=[]; available=torch.ones(n,dtype=torch.bool)
    scores=k.diag().clone()
    for _ in range(n-keep):
        choices=torch.where(available,scores,torch.full_like(scores,math.inf))
        channel=int(torch.argmin(choices)); increment=float(scores[channel])
        if not available[channel] or not math.isfinite(increment): raise RuntimeError('No finite selection candidate')
        removed.append(channel); available[channel]=False
        objective=float(k[removed][:,removed].sum())
        bound=ROUNDING_FACTOR*torch.finfo(k.dtype).eps*float(k[removed][:,removed].abs().sum())
        if objective < -bound: raise ValueError('Selected quadratic output energy is materially negative')
        trace.append({'removed_channel':channel,'increment':increment,'objective':objective,
                      'negative_roundoff_bound':bound,'negative_roundoff_reported':objective<0,
                      'candidate_scores_before_removal':[float(v) if math.isfinite(float(v)) else None for v in choices]})
        scores.add_(2*k[:,channel])
    return {'selected_indices':torch.nonzero(available).flatten().tolist(),
            'removed_indices':removed,'trace':trace,'final_objective':trace[-1]['objective'],
            'tie_break':'smallest original channel ID for exactly equal FP64 score',
            'arithmetic':'CPU float64 from saved fixed K; no model calls or search restarts'}


@torch.no_grad()
def collect_statistics(teacher, calibration, *, on_progress=None):
    sites=teacher_sites(teacher.model.decoder); statistics={name:None for name in NAMES}
    for index,crop in enumerate(calibration):
        z,_,valid,_=base.batch([crop])
        with audit.capture_sites(teacher.model.decoder,sites) as captured:
            base.teacher_forward(teacher,z)
        for site in sites:
            row=captured[site.name]; module=teacher.model.decoder.get_submodule(site.path)
            weights=common.cell_weights(valid,row['output'].shape[-1])
            statistics[site.name]=accumulate_consumer(statistics[site.name],module,row['input'],weights)
        if on_progress is not None:on_progress(index+1)
    return normalize_grams(statistics)


def export_artifact(model, teacher_decoder, calibration_ids, *, base_selected_state_sha256,
                    statistics_sha256, selection_receipt_sha256):
    """Only selection plus four native states; everything else is fresh teacher."""
    if len(calibration_ids)!=72 or len(set(calibration_ids))!=72:
        raise ValueError('Export requires the authenticated 72 distinct calibration sources')
    sites=audit.four_sites(teacher_decoder,model.selections)
    if tuple(s.path for s in sites)!=PATHS: raise ValueError('Native operator scope differs')
    states={path:{k:v.detach().cpu().clone() for k,v in model.decoder.get_submodule(path).state_dict().items()}
            for path in PATHS}
    if any(not torch.isfinite(t).all() for state in states.values() for t in state.values()):
        raise ValueError('Nonfinite native artifact')
    return {'format':VERSION,'variant':VARIANT,'original_step0_sha256':audit.STEP0_SHA,
            'teacher_checkpoint_sha256':base.CHECKPOINT_SHA256,'teacher_source_sha256':base.SOURCE_SHA256,
            'teacher_state_sha256':control.state_hash(teacher_decoder),'selection':model.selections,
            'fit_source_ids':list(calibration_ids),'ridge':ordinary.RIDGE,'operators':states,
            'base_selected_state_sha256':base_selected_state_sha256,
            'candidate_state_sha256':control.state_hash(model.decoder),
            'selection_statistics_sha256':statistics_sha256,'selection_receipt_sha256':selection_receipt_sha256,
            'automatic_promotion':False}


def _cache_gate(records):
    if not records or not all(r['allclose_original_tolerance'] for r in records):
        raise RuntimeError('Live original teacher differs from authenticated scored cached targets')


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('step0','checkpoint','manifest','assets','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Use a new isolated selector initializer directory')
    if any(args.out.resolve().is_relative_to(p.resolve()) for p in (args.step0.parent,args.checkpoint.parent)):
        raise ValueError('Output must not overlap protected previous runs')
    if base.sha(Path(ordinary.__file__))!=B_FITTER_SHA:
        raise ValueError('The original B native fitter changed; no substitute fitter is allowed')
    zero,_,authenticated=audit.checkpoint_inputs(args)
    if base.sha(args.manifest) not in authenticated['launch']['protected'].values():
        raise ValueError('Use the original authenticated calibration/development manifest')
    process=control.gpu_idle_snapshot()
    manifest,pools,_=base.load_data(args.manifest)
    calibration,development=pools['calibration'],pools['development']
    if len(calibration)!=72 or len(development)!=96:raise ValueError('Expected original72 calibration and96 development sources')
    split=common.validate_fit_sources(calibration,development)
    paths=[args.step0,args.checkpoint,args.manifest,args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',
           Path(manifest['cache_path']),Path(manifest['overlay_receipt_path']),Path(__file__),Path(ordinary.__file__),
           Path(audit.__file__),Path(common.__file__),Path(control.__file__),Path(group.__file__),
           Path(progressive.__file__),Path(base.__file__),Path(common.replay.__file__)]
    protected={**authenticated['protected'],**{str(p.resolve()):base.sha(p) for p in paths}}
    base.policy()
    if (str(torch.__version__)!=authenticated['launch']['torch']
            or torch.backends.cudnn.version()!=authenticated['launch']['cudnn']):
        raise ValueError('Use the preserved singleton FP32 runtime and backend')
    args.out.mkdir(parents=True)
    launch={'version':VERSION,'variant':VARIANT,'protected':protected,**split,
        'original_step0_sha256':audit.STEP0_SHA,'preserved_step5000_sha256':audit.STEP5000_SHA,
        'teacher_checkpoint_sha256':base.CHECKPOINT_SHA256,'teacher_source_sha256':base.SOURCE_SHA256,
        'torch':str(torch.__version__),'cudnn':torch.backends.cudnn.version(),'python':sys.version,
        'platform':platform.platform(),'argv':sys.argv,'backend':common.replay.backend_state(),
        'process_snapshot':process,'selector':'four equally weighted normalized complete-consumer output Grams',
        'source_pass_budget':{'teacher_selection':72,'unchanged_B_operator_fits':4*72,'development':96},
        'warmups':'Existing teacher/student per-shape warmups separately from scored passes',
        'selection_dtype':'float64; deterministic greedy on saved CPU K',
        'selection_full_output_channels':[512,512,512,256],'widths':[384,256,128],
        'ridge_factor':ordinary.RIDGE,'unchanged_B_fitter_sha256':B_FITTER_SHA,
        'neural_training_updates':0,'optimizer_created':False,'extra_inference_modules':0,
        'automatic_promotion':False,'candidate_start':'Fresh original teacher sliced with NEW support; old step0 group is never loaded'}
    base.write_json(args.out/'launch.json',launch)
    teacher=model=None; teacher_hash=None; before=None; sites=None; results={}; receipt=None
    failure=None; status='initializing'; cache_records=[]; started=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        teacher=base.FrozenAudioVAE2.from_files(args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',device='cuda')
        teacher_hash=control.state_hash(teacher.model)
        if tuple(teacher.model.decoder.model[i].block[1].out_channels for i in (3,4,5))!=(512,256,128):
            raise ValueError('Expected exact original teacher512/256/128 geometry')
        coverage=ordinary.calibration_coverage(calibration)
        base.write_json(args.out/'calibration-coverage.json',coverage)
        selection_started=time.monotonic()
        def progress(count):
            if count%12==0:base.event('downstream_selection_sources',sources=count,expected=72)
        with common.replay.observe_teacher_cache(calibration) as checks:
            k,statistics=collect_statistics(teacher,calibration,on_progress=progress)
        cache_records.extend(checks);_cache_gate(checks)
        chosen=greedy_remove(k,384)
        selection={'stage2_indices':chosen['selected_indices'],'stage3_indices':list(range(256))}
        statistics_path=args.out/'selection-statistics.pt'
        torch.save({'format':VERSION,'K':k,'consumers':statistics,'fit_source_ids':split['fit_source_ids'],
            'teacher_state_sha256':control.state_hash(teacher.model.decoder),
            'teacher_checkpoint_sha256':base.CHECKPOINT_SHA256,'teacher_source_sha256':base.SOURCE_SHA256,
            'manifest_sha256':base.sha(args.manifest),'source_sha256':base.sha(Path(__file__))},statistics_path)
        saved=torch.load(statistics_path,weights_only=True,map_location='cpu')
        if greedy_remove(saved['K'],384)!=chosen:raise RuntimeError('Saved CPU matrix does not exactly reproduce selection')
        selection_receipt={'selection':selection,**chosen,'statistics_sha256':base.sha(statistics_path),
            'normalizers':{name:s['normalizer'] for name,s in statistics.items()},
            'numerical_checks':{name:{key:s[key] for key in ('energy_identity_abs_error','arithmetic_bound','symmetry_max_abs')}
                                for name,s in statistics.items()},
            'elapsed_seconds':time.monotonic()-selection_started,'saved_K_replay_exact':True,
            'same_support_as_old_step0':selection==zero['selection'],'all_valid_rows':True,
            'shared_support':True,'no_development_selection':True,'no_fallback':True}
        selection_receipt_path=args.out/'selection-receipt.json';base.write_json(selection_receipt_path,selection_receipt)
        launch.update(selection=selection,selection_statistics_sha256=base.sha(statistics_path),
                      selection_receipt_sha256=base.sha(selection_receipt_path))
        base.write_json(args.out/'launch.json',launch)
        del statistics,saved,k
        model=progressive.initialize_from_teacher(teacher.model.decoder,selection).eval()
        pristine=audit.validate_pristine(teacher.model.decoder,model,selection)
        base_hash=control.state_hash(model.decoder);before=audit._versions(model.decoder)
        sites=audit.four_sites(teacher.model.decoder,selection)
        if len(model.trainable_group_parameters())!=90:raise ValueError('Native90-parameter group scope differs')
        base.event('downstream_support_selected',retained=384,removed=128,
                   same_as_previous=selection_receipt['same_support_as_old_step0'],objective=chosen['final_objective'])
        def on_fit(reports):
            base.write_json(args.out/'selector-fits.json',reports)
            last=reports[-1]
            base.event('downstream_B_operator_fitted',site=last['site'],
                       solver_predicted_error_before=last['weighted_error_before'],
                       solver_predicted_error_after=last['weighted_error_after'])
        with common.replay.observe_teacher_cache(calibration*4) as checks:
            ordinary.sequential_fit(model,teacher,calibration,sites,ordinary.observe_operation,on_fit)
        cache_records.extend(checks);_cache_gate(checks)
        ordinary._assert_only_sites_changed(before,model,sites)
        payload=export_artifact(model,teacher.model.decoder,split['fit_source_ids'],
            base_selected_state_sha256=base_hash,statistics_sha256=base.sha(statistics_path),
            selection_receipt_sha256=base.sha(selection_receipt_path))
        artifact_path=args.out/'selector-native-operators.pt';torch.save(payload,artifact_path)
        candidate_hash=payload['candidate_state_sha256']
        objective=base.objective();metadata={r['source_id']:r for r in manifest['splits']['development']['rows']}
        warmup=audit.warm_models(teacher,[model],development)
        base.write_json(args.out/'development-warmup.json',warmup)
        with common.replay.observe_teacher_cache(development) as checks:
            results['selector']=ordinary.evaluate_candidate(model,teacher,development,objective,metadata,args.out,'selector')
        cache_records.extend(checks);_cache_gate(checks)
        if control.state_hash(model.decoder)!=candidate_hash:raise RuntimeError('Evaluation changed the candidate')
        ordinary._assert_only_sites_changed(before,model,sites)
        receipt={'operators_sha256':base.sha(artifact_path),'base_selected_state_sha256':base_hash,
            'candidate_state_sha256':candidate_hash,'teacher_state_sha256':control.state_hash(teacher.model.decoder),
            'original_selected_pristine':audit._public_pristine(pristine),'selection':selection,
            'selection_statistics_sha256':base.sha(statistics_path),'selection_receipt_sha256':base.sha(selection_receipt_path),
            'changed_native_paths':list(PATHS),'full_group_widths':[384,256,128],
            'all9_residual_units_preserved':True,'unchanged_frozen_and_unselected_tensors':True,
            'native_writeback_passed':True,'neural_training_updates':0,'extra_inference_modules':0,'automatic_promotion':False}
        base.write_json(args.out/'selector-receipt.json',receipt)
        status='evaluated';base.event('downstream_selector_evaluated',development_sources=96)
    except BaseException as exc:
        failure=repr(exc);status='failed';raise
    finally:
        files=all(base.sha(path)==digest for path,digest in protected.items())
        states={'teacher':teacher is None or teacher_hash is None or control.state_hash(teacher.model)==teacher_hash,
                'frozen_and_unselected':True}
        if model is not None and before is not None:
            try:ordinary._assert_only_sites_changed(before,model,sites)
            except RuntimeError as exc:
                states['frozen_and_unselected']=False
                if failure is None:failure=repr(exc)
        control._write_rows(args.out/'teacher-cache-checks.jsonl.gz',cache_records)
        complete=(status=='evaluated' and failure is None and files and all(states.values())
                  and len(cache_records)==456 and set(results)=={'selector'} and receipt is not None)
        base.write_json(args.out/'completed.json',{'version':VERSION,'variant':VARIANT,'complete':complete,
            'status':status,'failure':failure,'files_preserved':files,'states_preserved':states,
            'calibration_sources':72,'development_sources':96,'neural_training_updates':0,
            'optimizer_created':False,'automatic_promotion':False,'candidate_receipt':receipt,
            'teacher_cache_comparisons':len(cache_records),
            'teacher_cache_all_pass':bool(cache_records) and all(r['allclose_original_tolerance'] for r in cache_records),
            'teacher_cache_nonexact':sum(not r['bitwise_equal'] for r in cache_records),
            'elapsed_seconds':time.monotonic()-started,'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'results':{name:{key:r[key] for key in ('aggregate','quiet_regions','overview_window_metrics','recovery_window_metrics')}
                       for name,r in results.items()}})
        if not files or not all(states.values()):raise RuntimeError('Initializer altered protected original state/files')


if __name__=='__main__':main()
