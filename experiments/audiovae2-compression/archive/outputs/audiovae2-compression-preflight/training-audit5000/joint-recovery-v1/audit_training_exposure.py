"""Read-only CPU reductions of sealed teacher targets and the completed journal."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import torch

START, STOP, CELL = 12000, 24000, 960
PLAN_ID = '3c65151fd2b37900257e179fee62218c3d33055c94d47dbb7c213593777e7c7d'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def stat(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def occupancy(crop):
    a, n = crop['context_frames'] * 1920, crop['valid_scored_samples']
    waveform = crop['teacher_audio']
    if waveform.device.type != 'cpu' or waveform.dtype != torch.float32:
        raise ValueError('Only original CPU FP32 cached targets are allowed')
    y = waveform[0, 0, a:a+n].double()
    if y.numel() != n or not bool(torch.isfinite(y).all()):
        raise ValueError('Invalid scored teacher audio')
    full, tail = divmod(n, CELL)
    power = y[:full*CELL].reshape(full, CELL).square().sum(-1)
    lengths = torch.full((full,), CELL, dtype=torch.int64)
    if tail:
        power = torch.cat((power, y[full*CELL:].square().sum().reshape(1)))
        lengths = torch.cat((lengths, torch.tensor([tail])))
    rms = (power / lengths).sqrt()
    quiet, near, zero = rms <= 1e-3, rms <= 1e-5, power == 0
    result = {'samples': n, 'windows': int(lengths.numel()), 'teacher_square_sum': float(power.sum()),
              'teacher_rms': float(y.square().mean().sqrt()), 'context_frames': crop['context_frames']}
    for name, mask in (('quiet', quiet), ('near', near), ('zero', zero)):
        result[name + '_samples'] = int(lengths[mask].sum())
        result[name + '_windows'] = int(mask.sum())
        result[name + '_teacher_square_sum'] = float(power[mask].sum())
    return result


def aggregate(rows):
    keys = ('samples', 'windows', 'quiet_samples', 'quiet_windows', 'near_samples', 'near_windows',
            'zero_samples', 'zero_windows', 'teacher_square_sum', 'quiet_teacher_square_sum',
            'near_teacher_square_sum', 'zero_teacher_square_sum')
    result = {key: sum(row[key] for row in rows) for key in keys}
    result['sources'] = len(rows)
    result['scored_hours'] = result['samples'] / 48000 / 3600
    for name in ('quiet', 'near', 'zero'):
        result[name + '_sample_percent'] = 100 * result[name + '_samples'] / result['samples']
        result[name + '_sources_any'] = sum(row[name + '_samples'] > 0 for row in rows)
        result[name + '_sources_all'] = sum(row[name + '_samples'] == row['samples'] for row in rows)
        result[name + '_seconds'] = result[name + '_samples'] / 48000
    return result


def main():
    parser = argparse.ArgumentParser()
    for key in ('plan', 'shards', 'train', 'completed', 'master', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    begun = time.monotonic()
    plan = json.loads(args.plan.read_text())
    if plan['identity_sha256'] != PLAN_ID or digest({k:v for k,v in plan.items() if k != 'identity_sha256'}) != PLAN_ID:
        raise ValueError('Sealed plan differs')
    records = [json.loads(line) for line in args.train.read_text().splitlines() if line.strip()]
    completed = json.loads(args.completed.read_text())
    if len(records) != 1000 or [r['step'] for r in records] != list(range(4626, 5626)):
        raise ValueError('Completed update journal differs')
    ids = [source for row in records for source in row['source_ids']]
    if (ids != plan['source_ids'][START:STOP] or len(set(ids)) != 12000
            or completed['scored_samples'] != records[-1]['scored_samples']):
        raise ValueError('Journal and sealed source lineage differ')
    protected = {str(p): {'sha256': sha(p), 'stat': stat(p)} for p in (args.plan, args.train, args.completed)}
    rows, shard_checks = [], []
    for index in range(START, STOP, 300):
        directory = args.shards / f'{index:06d}-{index+300:06d}'
        receipt_path, pairs_path = directory / 'receipt.json', directory / 'pairs.pt'
        receipt = json.loads(receipt_path.read_text())
        pair_sha = sha(pairs_path)
        if (receipt['plan_identity_sha256'] != PLAN_ID or not receipt['complete']
                or receipt['start_index'] != index or receipt['stop_index'] != index + 300
                or receipt['source_ids'] != ids[index-START:index-START+300]
                or receipt['pairs_sha256'] != pair_sha):
            raise ValueError('Sealed shard authentication differs')
        before = stat(pairs_path)
        payload = torch.load(pairs_path, map_location='cpu', weights_only=True, mmap=True)
        if payload['plan_identity_sha256'] != PLAN_ID or len(payload['crops']) != 300:
            raise ValueError('Shard payload differs')
        for crop, expected in zip(payload['crops'], plan['rows'][index:index+300]):
            for key in ('source_id', 'context_frames', 'start_frame', 'valid_scored_samples'):
                if crop[key] != expected[key]:
                    raise ValueError('Crop geometry or identity differs')
            row = occupancy(crop)
            row.update(source_id=crop['source_id'], dataset=expected['manifest_row']['dataset'],
                       language=expected['normalized_language'], condition=expected.get('condition'),
                       broad_expressive_source=expected['broad_expressive_source'],
                       selection_reason=expected['selection_reason'], global_source_index=len(rows)+START)
            rows.append(row)
        if stat(pairs_path) != before:
            raise RuntimeError('Cached target changed during the read-only audit')
        shard_checks.append({'start': index, 'stop': index+300, 'pairs_sha256': pair_sha,
                             'receipt_sha256': sha(receipt_path), 'unchanged': True})
        del payload
        print(json.dumps({'audited_sources': len(rows), 'total': 12000, 'seconds': time.monotonic()-begun}), flush=True)
    totals = aggregate(rows)
    if totals['samples'] != completed['scored_samples']:
        raise ValueError('Counted samples differ from completed training receipt')
    batches = []
    for i, record in enumerate(records):
        batch = aggregate(rows[i*12:(i+1)*12])
        batch.update(step=record['step'], waveform=record['waveform'], mel=record['mel'],
                     feature=record['feature'], total=record['total'])
        if sum(r['samples'] for r in rows[:(i+1)*12]) != record['scored_samples']:
            raise ValueError('Per-update scored sample cursor differs')
        batches.append(batch)
    groups = {}
    for key in ('dataset', 'language', 'broad_expressive_source', 'selection_reason'):
        buckets = defaultdict(list)
        for row in rows: buckets[str(row[key])].append(row)
        groups[key] = {k: aggregate(v) for k,v in buckets.items()}
    coefficients = json.loads((args.completed.parent / 'launch.json').read_text())['coefficients']
    quarters = []
    for q in range(4):
        quarter = aggregate(rows[q*3000:(q+1)*3000])
        quarter['updates'] = [q*250+1, (q+1)*250]
        quarter['mean_loss'] = {k: statistics.mean(b[k] for b in batches[q*250:(q+1)*250])
                                for k in ('waveform', 'mel', 'feature', 'total')}
        quarter['mean_weighted_loss'] = {k:coefficients[k]*quarter['mean_loss'][k]
                                         for k in ('waveform', 'mel', 'feature')}
        quarters.append(quarter)
    # This inventories unselected manifest records only. It does not reserve or cache them.
    blocked = {key: set(plan['blocked_identities'][key]) for key in ('source_id', 'audio_sha256', 'parent_recording_id')}
    for row in plan['rows']:
        for key in blocked: blocked[key].add(row['manifest_row'][key])
    master_sha = sha(args.master)
    inventory = Counter()
    datasets, languages = Counter(), Counter()
    hours = 0.
    present = missing = 0
    for line in args.master.open():
        record = json.loads(line)
        inventory['all_rows'] += 1
        if record['split'] != 'train':
            continue
        inventory['train_rows'] += 1
        if any(record[key] in blocked[key] for key in blocked):
            inventory['excluded_or_duplicate'] += 1
            continue
        if record['duration_seconds'] * 48000 < 4096:
            inventory['too_short_for_current_fft'] += 1
            continue
        for key in blocked: blocked[key].add(record[key])
        inventory['eligible_unselected_rows'] += 1
        datasets[record['dataset']] += 1
        languages[record['language']] += 1
        hours += record['duration_seconds'] / 3600
        p = Path(record['audio_path'])
        if p.is_file() and p.stat().st_size: present += 1
        else: missing += 1
    result = {'version':'joint_recovery_training_exposure_audit_v1', 'cpu_only':True,
        'model_imports':False, 'model_forwards':0, 'parameter_updates':0,
        'plan_identity_sha256':PLAN_ID, 'protected_inputs':protected, 'shards':shard_checks,
        'threshold_definition':'RMS of each contiguous scored 960-sample/20ms teacher window; valid final tail counted by its length; quiet<=0.001, near<=0.00001, zero exactly0',
        'aggregate':totals, 'quarters':quarters, 'groups':groups, 'batches':batches, 'rows':rows,
        'coefficients':coefficients,
        'update_occupancy':{name:{'zero_occupancy_updates':sum(b[name+'_samples']==0 for b in batches),
            'mean_sample_percent':statistics.mean(b[name+'_sample_percent'] for b in batches),
            'median_sample_percent':statistics.median(b[name+'_sample_percent'] for b in batches)}
            for name in ('quiet','near','zero')},
        'unused_master_inventory':{'manifest_path':str(args.master),'manifest_sha256':master_sha,
            'counts':dict(inventory),'manifest_hours':hours,'nonempty_paths':present,'missing_or_empty_paths':missing,
            'dataset_rows':dict(datasets),'language_rows':dict(languages),
            'exclusion_policy':'Known plan blocked source/hash/parent identities plus ALL27000 planned sources; deduplicated source/hash/parent; train split and >=4096 output samples; missing speaker/session identity not represented as verified separation',
            'sealed_plan_remaining':3000,'additional_to_optimizer_step10000':4375,'new_sources_needed':52500,
            'new_sources_beyond_existing_plan_needed':49500},
        'elapsed_seconds':time.monotonic()-begun}
    for p, before in protected.items():
        if stat(p) != before['stat']:
            raise RuntimeError('Input changed during audit: '+p)
    result['inputs_unchanged'] = True
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({'complete':True,'sources':len(rows),'samples':totals['samples'],
                      'quiet_percent':totals['quiet_sample_percent'],'near_percent':totals['near_sample_percent'],
                      'eligible_unselected_master_sources':inventory['eligible_unselected_rows'],
                      'seconds':result['elapsed_seconds']}),flush=True)


if __name__ == '__main__':
    main()
