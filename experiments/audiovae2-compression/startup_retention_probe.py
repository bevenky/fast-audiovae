"""Disposable fresh-C startup retention probe; aggregate output, no checkpoint.

Run unchanged ordinary updates on the original source prefix, stopping at the
first held-out startup failure or 32 updates. P/U/R raw-state hybrids are causal
diagnostics, not deployable candidates or additive waveform attributions.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
from pathlib import Path
import time

import torch
import startup_retention_diagnostic as helper
import quiet_constraint_warmup as warmup

quiet, prior, recovery = helper.quiet, helper.prior, helper.recovery
base, screen, replay = prior.base, prior.screen, prior.replay
VERSION = 'audiovae2_startup_retention_probe_v1'


def partition(name):
    if name.startswith('model.4.block.1.'):
        return 'U'
    if name.startswith(('model.3.', 'sr_cond_model.3.', 'sr_cond_model.4.', 'model.4.block.0.')):
        return 'P'
    if name.startswith(('model.4.block.2.', 'model.4.block.3.', 'model.4.block.4.',
                        'model.5.', 'sr_cond_model.5.')):
        return 'R'
    raise ValueError('Group tensor is outside the declared P/U/R partition')


def parameter_kind(name):
    if name.startswith('sr_cond_model.'):
        return 'conditioning'
    for kind in ('weight_g', 'weight_v', 'bias', 'alpha'):
        if name.endswith('.' + kind):
            return kind
    raise ValueError('Unrecognized group parameter kind')


def startup_panel(teacher, crops, expected):
    """Select by the original teacher-only absolute source grid, never predictions."""
    chosen = []
    for crop in crops:
        _, target, _, _ = base.batch([crop])
        rows = quiet.window_layout(crop, target)['windows']
        startup = [r for r in rows if r['cohort'] == 'near_startup'
                   and r['source_start_sample'] == 0 and r['source_stop_sample'] == 960
                   and r['valid_samples'] == 960]
        if startup:
            if len(startup) != 1:
                raise RuntimeError('Startup window is not unique within its source')
            chosen.append(crop)
    if len(chosen) != expected:
        raise RuntimeError('Authenticated calibration/development startup support changed')
    entries, checks = helper.prepare(teacher, chosen)
    for entry in entries:
        entry['windows'] = [r for r in entry['windows'] if r['cohort'] == 'near_startup'
                            and r['source_start_sample'] == 0 and r['source_stop_sample'] == 960]
        if len(entry['windows']) != 1:
            raise RuntimeError('Prepared startup grid differs from cached selection')
    return entries, checks


@torch.no_grad()
def score_panel(model, entries):
    result = dict(windows=0, passed=0, residual_only_failed=0, amplitude_only_failed=0,
                  both_failed=0, samples=0, residual_energy=0., student_energy=0., teacher_energy=0.,
                  residual_dc_energy=0., residual_ac_energy=0., residual_sum=0.,
                  residual_max_excess=None, amplitude_max_excess=None)
    for entry in entries:
        prediction = quiet._predict(model, entry)
        records = quiet.window_excesses(prediction, entry)
        a, _ = entry['span']
        by_index = {r['window_index']: r for r in entry['windows']}
        for record in records:
            row = by_index[record['window_index']]
            p = prediction[..., a+row['start_sample']:a+row['stop_sample']].double()
            t = entry['target'][..., a+row['start_sample']:a+row['stop_sample']].double()
            d = p-t
            if p.numel() != row['valid_samples'] or not torch.isfinite(d).all():
                raise RuntimeError('Startup score geometry or finite values changed')
            excess = [float(v) for v in record['values']]
            if not all(math.isfinite(v) for v in excess):
                raise RuntimeError('Nonfinite canonical startup excess')
            rf, af = excess[0] > 0, excess[1] > 0
            result['windows'] += 1
            result['passed'] += int(record['passed'])
            result['residual_only_failed'] += int(rf and not af)
            result['amplitude_only_failed'] += int(af and not rf)
            result['both_failed'] += int(rf and af)
            result['samples'] += d.numel()
            result['residual_energy'] += float(d.square().sum())
            result['student_energy'] += float(p.square().sum())
            result['teacher_energy'] += float(t.square().sum())
            result['residual_dc_energy'] += float(d.mean().square())*d.numel()
            result['residual_ac_energy'] += float((d-d.mean()).square().sum())
            result['residual_sum'] += float(d.sum())
            for kind, value in zip(quiet.KINDS, excess):
                key = kind+'_max_excess'
                result[key] = value if result[key] is None else max(result[key], value)
    n = result['samples']
    if not n:
        raise RuntimeError('Startup panel is empty')
    for name in ('residual', 'student', 'teacher', 'residual_dc', 'residual_ac'):
        result[name+'_rms'] = math.sqrt(result.pop(name+'_energy')/n)
    result['residual_mean'] = result.pop('residual_sum')/n
    return result


def compact_warmup(report):
    return {'passed': report['passed'], 'preserved': report['preserved'],
            'entries_checked': report['entries_checked'], 'new_shapes': report['new_shapes'],
            'warmup_grad_forwards': report['warmup_grad_forwards'],
            'verification_forwards': report['verification_forwards'], 'backwards': report['backwards'],
            'cold_first_vs_third_nonexact_shapes': sum(not r['first_vs_third']['passed']
                                                       for r in report['cold_comparisons'])}


def movement_stats(named, before):
    """Actual ordinary .grad and realized FP32 Adam movement, not lr*g."""
    output = {}
    for field, classify in (('partition', partition), ('kind', parameter_kind)):
        groups = {}
        for (name, p), old in zip(named, before):
            if p.grad is None or not torch.isfinite(p.grad).all() or not torch.isfinite(p).all():
                raise RuntimeError('Ordinary update gradient/state is missing or nonfinite')
            row = groups.setdefault(classify(name), dict(tensors=0, elements=0, changed_elements=0,
                gradient_square_sum=0., gradient_max_abs=0., delta_square_sum=0., delta_max_abs=0.,
                gradient_abs_above_adam_epsilon=0))
            g, d = p.grad.detach().double(), p.detach().double()-old.double()
            row['tensors'] += 1; row['elements'] += p.numel()
            row['changed_elements'] += int((p.detach()!=old).sum())
            row['gradient_square_sum'] += float(g.square().sum())
            row['delta_square_sum'] += float(d.square().sum())
            row['gradient_max_abs'] = max(row['gradient_max_abs'], float(g.abs().max()))
            row['delta_max_abs'] = max(row['delta_max_abs'], float(d.abs().max()))
            row['gradient_abs_above_adam_epsilon'] += int((g.abs()>1e-8).sum())
        for row in groups.values():
            row['gradient_l2'] = math.sqrt(row.pop('gradient_square_sum'))
            row['actual_delta_l2'] = math.sqrt(row.pop('delta_square_sum'))
        output[field] = groups
    return output


def directional_report(named, rows, before_score, after_score, before):
    delta = [p.detach().double()-old.double() for (_, p), old in zip(named, before)]
    signature = {ref: values for ref, values, _ in after_score['signature']}
    after_max = dict(after_score['maxima'])
    output = []
    for (key, old), gradients in zip(before_score['maxima'], rows):
        kind = key[1]; index = quiet.KINDS.index(kind)
        current = after_max[key]
        by_partition = {}
        for part in ('P', 'U', 'R'):
            ix = [i for i, (name, _) in enumerate(named) if partition(name)==part]
            by_partition[part] = quiet._dot([gradients[i] for i in ix], [delta[i] for i in ix])
        output.append({'cohort':key[0], 'kind':kind,
            'gradient_l2':math.sqrt(quiet._dot(gradients, gradients)),
            'gradient_dot_actual_delta':quiet._dot(gradients, delta),
            'gradient_dot_actual_delta_by_partition':by_partition,
            'before_max_excess':old['value'], 'after_same_window_excess':signature[old['ref']][index],
            'actual_same_window_change':signature[old['ref']][index]-old['value'],
            'after_max_excess':current['value'], 'actual_max_change':current['value']-old['value'],
            'maximizing_window_changed':current['ref']!=old['ref']})
    return output


@torch.no_grad()
def ordinary_objective(model, teacher, crops, common):
    denominator = base.reconstruction_denominators(crops, common)
    values = {key:0. for key in prior.COEFFICIENTS}
    with replay.observe_teacher_cache(crops) as checks:
        for crop in crops:
            z, target, valid, spans = base.batch([crop])
            trace = base.teacher_forward(teacher, z)
            h = model.group_from_input(trace['group_input'])
            p = model.suffix_from_group(h)
            for key, loss in base.losses(p, target, h, trace['group_output'], valid, spans, common, denominator).items():
                values[key] += float(loss)
    if len(checks)!=12 or not all(c['allclose_original_tolerance'] for c in checks):
        raise RuntimeError('Post-update objective cache identity failed')
    values['total'] = sum(prior.COEFFICIENTS[key]*values[key] for key in prior.COEFFICIENTS)
    if not all(math.isfinite(v) for v in values.values()):
        raise RuntimeError('Nonfinite post-update original objective')
    return values


def hybrid_scores(model, initial, endpoint, calibration, development):
    saved = {k:v.clone() for k,v in model.group_state_dict().items()}
    saved_hash = recovery.control.state_hash(model.decoder)
    result = []
    try:
        for bits in itertools.product((0,1), repeat=3):
            choice = dict(zip(('P','U','R'), bits))
            state = {name:(endpoint if choice[partition(name)] else initial)[name] for name in initial}
            model.load_group_state_dict(state)
            result.append({'endpoint_partitions':choice, 'calibration':score_panel(model, calibration),
                           'development':score_panel(model, development)})
    finally:
        model.load_group_state_dict(saved)
    if recovery.control.state_hash(model.decoder)!=saved_hash:
        raise RuntimeError('Hybrid diagnostic failed to restore endpoint state')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-updates', type=int, default=32)
    args = parser.parse_args()
    if not 1<=args.max_updates<=32 or args.out.exists():
        raise ValueError('Require a new output path and one to32 disposable updates')
    started = time.monotonic()
    _, teacher, model, optimizer, pools, data, protected = helper.build(json.loads(args.config.read_text()))
    named = model.group_named_parameters(); params = [p for _,p in named]
    counts = {part:sum(partition(n)==part for n,_ in named) for part in ('P','U','R')}
    if (counts!={'P':33,'U':3,'R':54} or len(named)!=90 or optimizer.state
            or [id(p) for group in optimizer.param_groups for p in group['params']]!=[id(p) for p in params]
            or any(p.requires_grad for p in teacher.parameters())):
        raise RuntimeError('Fresh optimizer or all90 P/U/R parameter contract failed')
    initial = {k:v.clone() for k,v in model.group_state_dict().items()}
    if set(initial)!={n for n,_ in named}:
        raise RuntimeError('P/U/R state and parameter partition differ')
    original_hash = recovery.control.state_hash(model.decoder)
    teacher_hash = recovery.control.state_hash(teacher.model)
    frozen = screen.frozen_versions(model)
    original_rng = screen.rng_state(); original_optimizer = copy.deepcopy(optimizer.state_dict())
    protected = dict(protected)
    for path in (args.config, Path(__file__), Path(helper.__file__), Path(warmup.__file__), Path(quiet.__file__)):
        protected[str(path.resolve())] = base.sha(path)
    report = {'version':VERSION, 'complete':False, 'maximum_updates':args.max_updates,
        'updates':0, 'failure_category':None, 'partition_tensor_counts':counts,
        'initial_candidate_sha256':original_hash, 'original_teacher_sha256':teacher_hash,
        'config_sha256':base.sha(args.config),
        'source_code_sha256':{Path(p).name:s for p,s in protected.items() if p.endswith('.py')},
        'torch':str(torch.__version__), 'cudnn':torch.backends.cudnn.version(),
        'precision':'Original FP32/TF32-off policy; canonical FP32 RMS then FP64 excess',
        'startup_score_arithmetic':{'pass_fail_and_excess':'Unchanged evaluator FP32 RMS; RMS squared in FP64 for excess',
            'continuous_rms_mean_dc_ac':'Direct FP64 sample energies/means, pooled across selected windows; DC removed separately per window'},
        'coefficients':dict(prior.COEFFICIENTS), 'optimizer':dict(name='AdamW',lr=3e-5,betas=[.9,.99],eps=1e-8,weight_decay=0),
        'physical_batch':1, 'accumulation':12, 'warmups':[], 'reviews':[],
        'scope':'Fresh C disposable causal diagnosis; no retained checkpoint, replay training, changed loss, Q projection or promotion',
        'interpretation':'Hybrids measure parameter-block interactions; directional derivatives are local and do not identify an additive waveform or loss-branch cause.'}
    seen = []; warmed = set(); completed_checks = 0; cache_nonexact = 0
    try:
        calibration, cc = startup_panel(teacher, pools['calibration'], 6)
        development, dc = startup_panel(teacher, pools['development'], 13)
        report['startup_target_cache'] = {'calibration':cc, 'development':dc}
        report['warmups'].append(compact_warmup(warmup.warm_quiet_entries(
            model, teacher, calibration+development, warmed, optimizer=optimizer)))
        initial_cal, initial_dev = score_panel(model, calibration), score_panel(model, development)
        report['reviews'].append({'step':0, 'calibration':initial_cal, 'development':initial_dev})
        if initial_dev['passed']!=13:
            raise RuntimeError('Fresh C no longer reproduces all13 startup passes')
        baseline = quiet.score_entries(model, calibration)
        rows = quiet.constraint_gradients(model, calibration, baseline, params)
        if len(rows)!=2 or any(p.grad is not None for p in params):
            raise RuntimeError('Calibration constraints polluted ordinary gradient slots')
        common = base.objective()
        for step in range(1, args.max_updates+1):
            crops = data.take((step-1)*12, 12)
            ids = [c['source_id'] for c in crops]
            if ids!=list(data.source_ids[(step-1)*12:step*12]) or set(ids)&set(seen):
                raise RuntimeError('Original ordered distinct source prefix changed')
            entries, checks = helper.prepare(teacher, crops)
            # Warm any new execution geometry before the ordinary gradient path.
            new_entries = []; new_shapes = set()
            for entry in entries:
                key = warmup._shape_key(model, entry)
                if key not in warmed and key not in new_shapes:
                    new_entries.append(entry); new_shapes.add(key)
            if new_entries:
                report['warmups'].append(compact_warmup(warmup.warm_quiet_entries(
                    model, teacher, new_entries, warmed, optimizer=optimizer)))
            if step==1 and quiet.score_entries(model, calibration)!=baseline:
                raise RuntimeError('Shape warming changed the initial calibration constraint values')
            exposure = sum(r['cohort']=='near_startup' for e in entries for r in e['windows'])
            del entries
            before = [p.detach().clone() for p in params] if step==1 else None
            tick = time.monotonic()
            values, update_checks = prior.perform_update(model, teacher, crops, common, optimizer, diagnostics=True)
            duration = time.monotonic()-tick
            seen.extend(ids); completed_checks += len(update_checks)+checks['comparisons']
            cache_nonexact += sum(not c['bitwise_equal'] for c in update_checks)+checks['nonexact']
            if set(optimizer.state)!=set(params) or any(float(s['step'])!=step for s in optimizer.state.values()):
                raise RuntimeError('Ordinary Adam counter/scope changed')
            if (any(not torch.isfinite(p).all() for p in params)
                    or any(not torch.isfinite(s[k]).all() for s in optimizer.state.values()
                           for k in ('exp_avg','exp_avg_sq'))):
                raise RuntimeError('Ordinary Adam parameter/moment state is nonfinite')
            report['updates'] = step
            current_cal, current_dev = score_panel(model, calibration), score_panel(model, development)
            review = {'step':step, 'calibration':current_cal, 'development':current_dev,
                      'ordinary_objective_before_update':{k:values[k] for k in (*prior.COEFFICIENTS,'total')},
                      'training_near_startup_windows':exposure, 'training_update_seconds':duration}
            report['reviews'].append(review)
            if step==1:
                report['first_update'] = {'constraint_directions':directional_report(
                    named, rows, baseline, quiet.score_entries(model, calibration), before),
                    'ordinary_gradient_and_actual_movement':movement_stats(named, before),
                    'ordinary_objective_after_update_same_batch':ordinary_objective(model, teacher, crops, common),
                    'loss_component_gradients_separately_measured':False}
                del rows, before
            if current_dev['passed']<13:
                report['first_heldout_startup_failure_update'] = step
                break
        endpoint = {k:v.clone() for k,v in model.group_state_dict().items()}
        report['endpoint_candidate_sha256'] = recovery.control.state_hash(model.decoder)
        report['hybrids'] = hybrid_scores(model, initial, endpoint, calibration, development)
        if (report['hybrids'][0]['calibration']!=initial_cal or report['hybrids'][0]['development']!=initial_dev
                or report['hybrids'][-1]['calibration']!=current_cal or report['hybrids'][-1]['development']!=current_dev):
            raise RuntimeError('Hybrid baseline/endpoint does not exactly reproduce canonical startup scores')
        report['source_count'] = len(seen)
        report['executed_source_prefix_sha256'] = screen.digest(seen)
        report['source_prefix_unique_and_exact'] = len(set(seen))==len(seen) and seen==list(data.source_ids[:len(seen)])
        report['training_and_preparation_cache_checks'] = dict(comparisons=completed_checks, nonexact=cache_nonexact, all_pass=True)
        report['complete'] = True
    except BaseException as exc:
        report['failure_category'] = type(exc).__name__
        raise
    finally:
        model.load_group_state_dict(initial)
        optimizer.load_state_dict(original_optimizer)
        optimizer.zero_grad(set_to_none=True)
        screen.restore_rng(original_rng)
        report['fresh_c_state_restored'] = recovery.control.state_hash(model.decoder)==original_hash
        report['fresh_optimizer_restored'] = replay.compare_tree(optimizer.state_dict(), original_optimizer)['equal']
        report['original_rng_restored'] = replay.compare_tree(screen.rng_state(), original_rng)['equal']
        report['teacher_and_frozen_preserved'] = (recovery.control.state_hash(teacher.model)==teacher_hash
                                                   and screen.frozen_versions(model)==frozen)
        data.assert_unchanged()
        report['protected_files_preserved'] = all(base.sha(path)==sha for path,sha in protected.items())
        report['elapsed_seconds'] = time.monotonic()-started
        report['no_checkpoint_written'] = True
        report['all_preservation_checks_passed'] = all(report[k] for k in ('fresh_c_state_restored',
            'fresh_optimizer_restored','original_rng_restored','teacher_and_frozen_preserved','protected_files_preserved'))
        if not report['all_preservation_checks_passed']:
            report['complete'] = False
            report['failure_category'] = 'PreservationFailure'
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        if not report['all_preservation_checks_passed']:
            raise RuntimeError('Disposable diagnostic preservation guard failed')
    print(json.dumps({'complete':report['complete'], 'updates':report['updates'],
        'heldout_startup_passed':report['reviews'][-1]['development']['passed'],
        'preserved':report['all_preservation_checks_passed']}))


if __name__=='__main__':
    main()
