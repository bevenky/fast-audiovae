"""Independent saved-artifact verification and descriptive paired comparison."""
from __future__ import annotations
import argparse,json,gc,statistics,math
from pathlib import Path
import numpy as np
import torch
from summarize_fusion_screen import (file_sha,load_json,digest,state_fingerprint,
    cohorts,delta,paired,assert_matching_reports,synthetic,row_map,json_metadata,validate_report)

NAMES=('regular','targeted','targeted_magnitude','targeted_complex')
BASE=Path('/workspace/fast-audiovae-convnext-20260909-r9/remediation/corrected-screen')


def gradient_summary(path, *, expected_steps=None, expected_examples=None,
                     expected_sample_counts=None, parent_discriminator_updates=None):
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:raise ValueError('Empty generator metrics')
    if any(not np.isfinite(v) for row in rows for v in row.values() if isinstance(v,(float,int))):
        raise ValueError('Nonfinite generator metrics')
    actual_steps=[row.get('step') for row in rows]
    if any(type(step) is not int for step in actual_steps):raise ValueError('Invalid generator update numbers')
    if expected_steps is None:expected_steps=list(range(actual_steps[0],actual_steps[0]+len(rows)))
    if actual_steps!=list(expected_steps):raise ValueError('Missing, duplicated or reordered generator updates')
    if expected_sample_counts is not None and len(expected_sample_counts)!=len(rows):
        raise ValueError('Sample-count plan length differs from generator metrics')
    for i,row in enumerate(rows):
        if expected_examples is not None and row.get('examples')!=expected_examples:
            raise ValueError('Generator silently changed batch size')
        if expected_sample_counts is not None and row.get('scored_samples')!=expected_sample_counts[i]:
            raise ValueError('Generator exposure differs from planned valid samples')
        if parent_discriminator_updates is not None and row.get('discriminator_updates')!=parent_discriminator_updates+i+1:
            raise ValueError('Generator-step discriminator counter is discontinuous')
    keys=('teacher_waveform','teacher_mel','feature_matching','adversarial')
    def part(items):
        shares=[]
        for row in items:
            values={key:row[key+'/scaled_norm'] for key in keys}
            if any(v<0 for v in values.values()):raise ValueError('Negative gradient norm')
            total=sum(values.values())
            if total<=0:raise ValueError('Zero pre-sum gradient norm')
            shares.append({key:value/total for key,value in values.items()})
        return {key:statistics.mean(r[key] for r in shares) for key in keys}
    return {'updates':len(rows),'steps':[rows[0]['step'],rows[-1]['step']],
        'all':part(rows),'first25':part(rows[:25]),'last50':part(rows[-50:]),
        'semantics':'Mean per-update pre-sum audio-gradient norm proportions, not parameter-update shares',
        'nonfinite_values':sum(not np.isfinite(v) for row in rows for v in row.values() if isinstance(v,(float,int)))}


def validate_streams(stream):
    expected={'0_2':2,'0_4':4,'1_2':2,'1_4':4}
    if not isinstance(stream,dict) or stream.keys()!=expected.keys():
        raise ValueError('Expected four complete CPU stream checks')
    for key,frames in expected.items():
        row=stream[key]
        if (row.get('passed') is not True or row.get('sample_count_passed') is not True
                or row.get('frames_per_chunk')!=frames or row.get('numerical_tolerance')!=2e-6
                or row.get('rtf_measured') is not False
                or type(row.get('input_frames')) is not int or row['input_frames']<1):
            raise ValueError('Invalid CPU streaming qualification')
        expected_samples=row['input_frames']*1920
        if any(row.get(k)!=expected_samples for k in ('expected_samples','batch_samples','stream_samples')):
            raise ValueError('Streaming sample totals differ from input frame count')
        if row.get('flush_samples')!=0 or not row.get('chunks'):
            raise ValueError('Streaming flush/chunk accounting is missing')
        remaining=row['input_frames']
        for chunk in row['chunks']:
            n=min(frames,remaining)
            if (n<=0 or chunk.get('input_frames')!=n or chunk.get('expected_samples')!=n*1920
                    or chunk.get('output_samples')!=n*1920):
                raise ValueError('A streaming chunk was lost, duplicated or resized')
            remaining-=n
        if remaining or not math.isfinite(row.get('max_abs_error',math.nan)) or not 0<=row['max_abs_error']<=2e-6:
            raise ValueError('Streaming parity failed or input frames were omitted')


def validate_gradient_audit(audit, *, step, model_sha, optimizer_sha):
    fixed=audit.get('fixed_state',{})
    if (audit.get('method')!='fixed_weight_output_gradient_audit' or audit.get('parameter_updates')!=0
            or audit.get('optimizer_updates')!=0 or fixed.get('step')!=step
            or fixed.get('model')!=model_sha or fixed.get('optimizer')!=optimizer_sha):
        raise ValueError('Gradient audit is not tied to the reported model and optimizer')


def validate_calibration(report, parent_engine, *, data_sha, batches, examples, warmup):
    if (report.get('method')!='fixed_weight_selected_loss_ema_reset'
            or report.get('calibrated_loss_names')!=['feature_matching','adversarial']
            or report.get('parameter_updates')!=0 or report.get('optimizer_updates')!=0
            or report.get('batches')!=batches or report.get('examples')!=examples
            or report.get('balancer_training_updates_preserved') is not True):
        raise ValueError('Wrong calibration method, budget or selected objectives')
    provenance=report.get('provenance',{})
    if (provenance.get('split')!='train' or provenance.get('pool')!='gradient_calibration'
            or provenance.get('data_plan_sha256')!=data_sha
            or provenance.get('fresh_discriminator_warmup_updates')!=warmup):
        raise ValueError('Gradient calibration provenance changed')
    before,after=report['balancer_before'],report['balancer_after']
    if before!=json_metadata(parent_engine['balancer']):raise ValueError('Calibration did not start from parent balancer')
    if {k:v for k,v in before.items() if k!='ema'}!={k:v for k,v in after.items() if k!='ema'}:
        raise ValueError('Calibration changed balancer clocks, config or weights')
    if before['ema'].keys()!=after['ema'].keys():raise ValueError('Calibration changed declared loss identities')
    for name,values in after['ema'].items():
        if name not in report['calibrated_loss_names']:
            if values!=before['ema'][name]:raise ValueError('Unselected EMA changed')
        elif (set(values)!={'total','weight'} or any(not math.isfinite(v) or v<=0 for v in values.values())
              or report['loss_statistics'][name]['positive_observations']<1):
            raise ValueError('Recalibrated loss has no finite positive observations')
    fixed=report.get('fixed_state',{})
    for field in ('model','optimizer','calibration'):
        if fixed.get(field)!=state_fingerprint(parent_engine[field]):
            raise ValueError('Calibration changed the trained student or its optimizer/normalization')
    if fixed.get('step')!=parent_engine['step'] or fixed.get('discriminator_updates')!=parent_engine['discriminator_updates']:
        raise ValueError('Calibration changed training clocks')
    if len(report['before'])!=batches or len(report['after'])!=batches:
        raise ValueError('Calibration observations are incomplete')


def source_bootstrap(candidate,reference):
    """Resample source groups; retain all paired crop contributions per source."""
    assert_matching_reports(candidate,reference)
    ca,ra=row_map(candidate),row_map(reference)
    grouped={}
    for key,c in ca.items():
        if synthetic(c):continue
        n=c['samples'];r=ra[key]
        values=grouped.setdefault(c['source_id'],np.zeros(3))
        values+=np.array([n*c['waveform_raw_mae'],n*r['waveform_raw_mae'],n])
    matrix=np.asarray(list(grouped.values()))
    if not len(matrix) or matrix[:,1].sum()<=0:
        raise ValueError('Source bootstrap requires natural sources and positive reference error')
    rng=np.random.default_rng(39170)
    ix=rng.integers(0,len(matrix),size=(2000,len(matrix)))
    sums=matrix[ix].sum(1)
    change=100*(sums[:,0]/sums[:,1]-1)
    return {'metric':'natural_sample_weighted_raw_mae_relative_change_percent',
        'paired_resampling_unit':'source_id','sources':len(matrix),'replicates':2000,'seed':39170,
        'percentile95':[float(v) for v in np.quantile(change,[.025,.975])],
        'limitation':'Heldout source sampling only; not speaker clustering, training-seed uncertainty or perceptual equivalence'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=BASE);args=p.parse_args()
    root=args.root;identity=load_json(root/'identity.json');data=load_json(root/'targeted-data.json')
    if file_sha(root/'targeted-data.json')!=identity['data_plan_sha256']:raise ValueError('Data plan changed')
    if digest({k:v for k,v in data.items() if k!='identity_sha256'})!=data['identity_sha256']:raise ValueError('Invalid sealed data digest')
    receipt=load_json(root.parent/'fusion-screen/parent-receipt.json')
    parent=torch.load(root.parent/'fusion-screen/parent.pt',map_location='cpu',weights_only=True,mmap=True)
    if file_sha(root.parent/'fusion-screen/parent.pt')!=receipt['checkpoint_sha256']:raise ValueError('Parent changed')
    psha=state_fingerprint(parent['engine']['model'])
    parent_step=identity['parent_step'];steps=identity['generator_updates_per_arm'];batch_size=identity['batch_size']
    if (parent_step!=8090 or parent['engine']['step']!=parent_step
            or identity['parent_sha256']!=receipt['checkpoint_sha256']
            or data['parent_sha256']!=receipt['checkpoint_sha256'] or data['parent_step']!=parent_step):
        raise ValueError('Inconsistent experiment parent identity')
    arms={name:(variant,pool,calibrate) for name,variant,pool,calibrate in identity['arms']}
    if list(arms)!=list(NAMES):raise ValueError('Experiment arm identities changed')
    expected_crops={(c['source_id'],c['start_frame']):c for c in identity['targets']['heldout']['crops']}
    if len(expected_crops)!=identity['targets']['heldout']['count']:
        raise ValueError('Sealed heldout membership has duplicate or missing crops')
    if identity['targets']['teacher_state_sha256']!=parent['identity']['data']['teacher_state_sha256']:
        raise ValueError('Teacher differs from the frozen parent teacher')
    parent_engine_sha=state_fingerprint(parent['engine'])
    output={'format_version':1,'parent_step':8090,'parent_sha256':receipt['checkpoint_sha256'],
            'experiment_identity_sha256':digest(identity),'data_summary':data['summary'],
            'teacher_repeatability':identity['targets']['teacher_repeatability'],'arms':{},'comparisons':{},'pending':[]}
    reports={}
    for name in NAMES:
        path=root/name
        if not (path/'complete.json').exists():output['pending'].append(name);continue
        if (path/'inflight.json').exists():raise ValueError('Completed arm has unfinished exposure')
        complete=load_json(path/'complete.json');before=load_json(path/'before.json');after=load_json(path/'after.json')
        migration=load_json(path/'migration.json')
        assert_matching_reports(before,after)
        if before['model_identity']['parameter_state_sha256']!=psha:raise ValueError('Different initial student')
        variant,pool_name,calibrated=arms[name]
        warmup=identity['fresh_discriminator_warmup_updates'] if calibrated else 0
        ck=torch.load(path/'final.pt',map_location='cpu',weights_only=True,mmap=True)
        if (ck['format_version']!='corrected_screen_v1' or ck['arm']!=name or ck['generator_updates']!=steps
            or ck['engine']['step']!=parent_step+steps or ck['parent_sha256']!=receipt['checkpoint_sha256']
            or ck['experiment_identity_sha256']!=digest(identity)
            or ck['data_plan_sha256']!=identity['data_plan_sha256']
            or ck['discriminator_only_updates']!=warmup
            or complete['arm']!=name or complete['updates']!=steps or complete['step']!=parent_step+steps
            or file_sha(path/'final.pt')!=complete['checkpoint_sha256']):raise ValueError('Wrong final checkpoint')
        final_model_sha=state_fingerprint(ck['engine']['model'])
        if final_model_sha!=after['model_identity']['parameter_state_sha256']:
            raise ValueError('Reported evaluation differs from saved model')
        if (json_metadata(ck['migration'])!=migration
                or ck['engine'].get('format_version')!='fusion_recipe_v1'
                or ck['engine'].get('fusion_variant')!=variant
                or json_metadata(ck['engine'].get('fusion_migration'))!=migration
                or migration.get('variant')!=variant
                or migration.get('parent_step')!=parent_step
                or migration.get('parent_engine_state_sha256')!=parent_engine_sha
                or migration.get('original_student_state_sha256')!=psha):
            raise ValueError('Changed migration receipt')
        validate_report(before,expected_crops,parent_step,psha,variant=variant,
            base_config=parent['engine']['model_config'],migration=migration)
        validate_report(after,expected_crops,parent_step+steps,final_model_sha,variant=variant,
            base_config=ck['engine']['model_config'],migration=migration)
        if complete['before_summary']!=before['summary'] or complete['after_summary']!=after['summary']:
            raise ValueError('Completion summary differs from full evaluation report')
        for norm in ('stem_norm','affine'):
            for field in ('running_mean','running_var','num_batches_tracked','statistics_frozen'):
                key=norm+'.'+field
                if not torch.equal(ck['engine']['model'][key],parent['engine']['model'][key]):
                    raise ValueError('Model normalization statistics changed')
        if state_fingerprint(ck['engine']['calibration'])!=state_fingerprint(parent['engine']['calibration']):
            raise ValueError('Model calibration receipt changed')
        expected=data['pools'][pool_name]
        if len(expected)!=steps*batch_size:raise ValueError('Wrong planned generator exposure count')
        gs=gradient_summary(path/'metrics.jsonl',expected_steps=range(parent_step+1,parent_step+steps+1),
            expected_examples=batch_size,parent_discriminator_updates=parent['engine']['discriminator_updates'],
            expected_sample_counts=[sum(e['window']['valid_output_samples48k'] for e in expected[i:i+batch_size])
                                    for i in range(0,len(expected),batch_size)])
        views=[json.loads(line) for line in (path/'views.jsonl').read_text().splitlines()]
        for i,row in enumerate(views):
            if row['update']!=i+1 or len(row['views'])!=batch_size:raise ValueError('Missing discriminator views')
            for got,e in zip(row['views'],expected[i*batch_size:(i+1)*batch_size],strict=True):
                w=e['window']
                if got['source_id']!=w['source_id'] or got['start_frame']!=w['start_frame']:
                    raise ValueError('Discriminator exposure differs from planned sequence')
                if not 0<=got['offset48k']<=w['valid_output_samples48k']-9120:raise ValueError('Invalid D interval')
                if name!='regular' and e['discriminator_start_sample48k'] is not None:
                    if got['offset48k']!=e['discriminator_start_sample48k']:raise ValueError('Planned event view missed')
        if len(views)!=steps:raise ValueError('Missing logged D views')
        stream=load_json(path/'streaming.json')
        validate_streams(stream)
        audit_before=load_json(path/'gradients-before.json');audit_after=load_json(path/'gradients-after.json')
        validate_gradient_audit(audit_before,step=parent_step,model_sha=psha,
                                optimizer_sha=state_fingerprint(parent['engine']['optimizer']))
        validate_gradient_audit(audit_after,step=parent_step+steps,model_sha=final_model_sha,
                                optimizer_sha=state_fingerprint(ck['engine']['optimizer']))
        calibration=None
        if calibrated:
            calibration=load_json(path/'calibration.json')
            n=identity['fixed_weight_calibration_batches']
            validate_calibration(calibration,parent['engine'],data_sha=identity['data_plan_sha256'],
                                 batches=n,examples=n*batch_size,warmup=warmup)
            warm_rows=[json.loads(line) for line in (path/'warmup.jsonl').read_text().splitlines()]
            if ([r['update'] for r in warm_rows]!=list(range(1,warmup+1))
                    or any(not math.isfinite(r[k]) for r in warm_rows for k in ('loss','gradient_norm'))):
                raise ValueError('Missing or invalid discriminator-only warmup observations')
        elif (path/'calibration.json').exists() or (path/'warmup.jsonl').exists():
            raise ValueError('Unplanned calibration or discriminator warmup artifacts')
        reports[name]=after
        output['arms'][name]={'before':cohorts(before),'after':cohorts(after),'gradient_shares':gs,
            'within_arm':paired(after,before),'checkpoint_sha256':complete['checkpoint_sha256'],
            'training_seconds':complete['training_seconds'],'cpu_stream_checks':stream,
            'components_before':load_json(path/'components-before.json'),
            'components_after':load_json(path/'components-after.json'),
            'gradient_audit_before':audit_before,
            'gradient_audit_after':audit_after,
            'calibration':calibration}
        del ck;gc.collect()
    pairs=(('targeted','regular'),('targeted_magnitude','targeted'),('targeted_complex','targeted_magnitude'))
    for candidate,reference in pairs:
        if candidate not in reports or reference not in reports:continue
        c,r=output['arms'][candidate]['after'],output['arms'][reference]['after']
        output['comparisons'][candidate+'_vs_'+reference]={
            group:delta(c[group],r[group]) for group in ('natural','speech','expressive','synthetic')}
        output['comparisons'][candidate+'_vs_'+reference].update(
            paired=paired(reports[candidate],reports[reference]),
            raw_mae_source_bootstrap=source_bootstrap(reports[candidate],reports[reference]))
    output['interpretation']={'automatic_winner_selection':False,'new_inference_operations':0,
        'benchmark_or_model_execution_by_summarizer':False,
        'regular_control':'Matched ordinary-speech control, not exact original mixture replay',
        'targeted_package':'Changed data selection and event-view placement together',
        'fresh_discriminator_package':'Initialization, additional D updates and calibration together',
        'complex_comparison':'Same targeted data, preparation budget, calibrated objectives and evaluation',
        'source_labels':'Source-level labels plus activity, not semantic interval ground truth',
        'overshoot_counts':'Samples/crops may overlap; these are not unique physical event counts',
        'gradient_shares':'Pre-sum output-gradient norm proportions, not parameter-update shares',
        'fixed_mel':'Unchanged common diagnostic criterion, equal crop means, not candidate training loss',
        'quiet':'Natural and synthetic separate; pooled RMS weights valid quiet samples',
        'statistical_scope':'One training trajectory per arm; source bootstrap does not establish perceptual equivalence'}
    temporary=root/'summary.json.tmp'
    temporary.write_text(json.dumps(output,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temporary.replace(root/'summary.json')
    print(json.dumps({'pending':output['pending'],'arms':{k:{'quality':v['after'],'gradients':v['gradient_shares']}
        for k,v in output['arms'].items()},'comparisons':{k:{a:b for a,b in v.items() if a!='paired'}
        for k,v in output['comparisons'].items()}},indent=2,allow_nan=False))


if __name__=='__main__':main()
