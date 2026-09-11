"""Seal an additive source-only continuation plan from downloaded audio."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import heapq
import json
import math
from pathlib import Path
import torch
from continuation_data_v2 import VERSION, PARENT_ID, START, PARENT_STOP, STOP, KEYS, atomic_json
from fresh_training_data import digest, sha

CROP_SEED = '20260910-wholegroup-unique-continuation-v1'
ORDER_SEED = '20260910-joint-continuation-10000-v2'


def rank(value, seed=ORDER_SEED):
    return hashlib.sha256((seed+'|'+value).encode()).hexdigest()


def geometry(source):
    n = round(source['duration_seconds']*16000)
    if abs(n/16000-source['duration_seconds']) > 1/16000+1e-10:
        raise ValueError('Source duration is not an exact prepared sample count')
    sid = source['source_id']
    latest = max(0, (n-40960)//640)
    start = 0 if latest < 30 or int(rank(sid+'/startup', CROP_SEED)[:8],16)%3 == 0 else 30+int(rank(sid+'/crop', CROP_SEED)[:8],16)%(latest-29)
    valid = min(40960,n-start*640)*3
    if valid < 4096: raise ValueError('Source cannot satisfy existing spectral length')
    return dict(input_samples16k=n,start_frame=start,context_start_frame=max(0,start-30),
                context_frames=min(30,start),scored_frames=math.ceil(valid/1920),valid_scored_samples=valid)


def select_unique(candidates, blocked, weights, count):
    blocked = {key:set(values) for key,values in blocked.items()}
    groups = defaultdict(list)
    for row in candidates:
        if row['split'] != 'train' or row['duration_seconds']*48000 < 4096: continue
        if any(row[key] in blocked[key] for key in KEYS): continue
        groups[row['language'].split('_')[0]].append(row)
    groups = {lang:deque(sorted(rows,key=lambda r:rank('order/'+r['source_id']))) for lang,rows in groups.items()}
    selected, seen = [], Counter()
    heap = [(1/max(1,weights.get(lang,10)),lang) for lang in groups]
    heapq.heapify(heap)
    while len(selected) < count:
        if not heap: raise ValueError('Insufficient distinct, unreserved training sources')
        _,lang = heapq.heappop(heap)
        row = groups[lang].popleft()
        if not any(row[key] in blocked[key] for key in KEYS):
            selected.append(row);seen[lang] += 1
            for key in KEYS: blocked[key].add(row[key])
        if groups[lang]: heapq.heappush(heap,((seen[lang]+1)/max(1,weights.get(lang,10)),lang))
    return selected


def summary(rows):
    samples = sum(r['valid_scored_samples'] for r in rows)
    return {'sources':len(rows),'scored_hours':samples/48000/3600,
            'full_source_hours':sum(r['input_samples16k'] for r in rows)/16000/3600,
            'languages':dict(Counter(r['normalized_language'] for r in rows)),
            'datasets':dict(Counter(r['manifest_row']['dataset'] for r in rows)),
            'broad_expressive_scored_percent':100*sum(r['valid_scored_samples'] for r in rows if r['broad_expressive_source'])/samples,
            'startup_sources':sum(r['start_frame']==0 for r in rows)}


def stratified_interleave(rows):
    groups=defaultdict(list)
    for row in rows: groups[row['language'].split('_')[0]].append(row)
    groups={key:deque(sorted(value,key=lambda r:rank('phase-interleave/'+r['source_id']))) for key,value in groups.items()}
    totals={key:len(value) for key,value in groups.items()}
    used=Counter();heap=[(.5/totals[key],key) for key in groups];heapq.heapify(heap)
    ordered=[]
    while heap:
        _,key=heapq.heappop(heap);ordered.append(groups[key].popleft());used[key]+=1
        if groups[key]:heapq.heappush(heap,((used[key]+.5)/totals[key],key))
    return ordered


def main():
    parser = argparse.ArgumentParser()
    for name in ('parent-plan','original-manifest','master','checkpoint','checkpoint-receipt','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    parent = json.loads(args.parent_plan.read_text())
    if parent['identity_sha256'] != PARENT_ID or digest({k:v for k,v in parent.items() if k!='identity_sha256'}) != PARENT_ID:
        raise ValueError('Original plan identity differs')
    original = json.loads(args.original_manifest.read_text())
    if sha(args.original_manifest) != parent['original_manifest_sha256']:
        raise ValueError('Original training manifest changed')
    receipt = json.loads(args.checkpoint_receipt.read_text())
    checkpoint_sha = sha(args.checkpoint)
    if receipt['checkpoint_sha256'] != checkpoint_sha or receipt['optimizer_step'] != 5625 or not receipt['frozen_state_preserved']:
        raise ValueError('Expected the preserved5625 continuation checkpoint')
    checkpoint = torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    expected_history = [r['source_id'] for r in original['splits']['fit']['rows']]+parent['source_ids'][:START]
    if (checkpoint.get('optimizer_step')!=5625 or checkpoint.get('accumulation')!=12
            or checkpoint['historical_sources_seen']+checkpoint['additional_sources_seen']!=expected_history):
        raise ValueError('Checkpoint history differs from the exact27000 accepted-source prefix')
    del checkpoint
    blocked = {key:set(parent['blocked_identities'][key]) for key in KEYS}
    for row in parent['rows']:
        for key in KEYS: blocked[key].add(row['manifest_row'][key])
    candidates = [json.loads(line) for line in args.master.open()]
    selected = select_unique(candidates, blocked, original['summaries']['fit']['normalized_language_counts'], STOP-PARENT_STOP)
    selected = stratified_interleave(selected)
    appended = []
    for source in selected:
        path = Path(source['audio_path'])
        if not path.is_file() or not path.stat().st_size: raise FileNotFoundError(path)
        appended.append({'source_id':source['source_id'],'manifest_row':source,**geometry(source),
            'condition':None,'phase':'authorized_10000_extension','selection_reason':'prior_language_weighted_unique_source',
            'normalized_language':source['language'].split('_')[0],
            'broad_expressive_source':source['dataset'] not in ('librispeech','fleurs','indicvoices'), 'cached_pool_index':None})
    # As in the original planner, interleave the selected weighted mixture so
    # finite low-resource groups are not exhausted at the start of training.
    all_rows = parent['rows']+appended
    all_ids = parent['source_ids']+[r['source_id'] for r in appended]
    stages = [(START,min(START+12000,STOP))]
    while stages[-1][1] < STOP:
        start = stages[-1][1];stages.append((start,min(start+12000,STOP)))
    summaries={f'{a}:{b}':summary(all_rows[a:b]) for a,b in stages}
    indic=set('as bn brx doi gu hi kn kok ks mai ml mni mr ne or pa sa sat sd ta te ur'.split())
    if any(indic-set(s['languages']) for s in summaries.values()):
        raise ValueError('Every continuation segment must retain all22 Indic languages')
    plan = {'version':VERSION,'parent_plan_path':str(args.parent_plan.resolve()),'parent_plan_sha256':sha(args.parent_plan),
            'parent_plan_identity_sha256':PARENT_ID,'original_manifest_sha256':sha(args.original_manifest),
            'teacher_source_sha256':parent['teacher_source_sha256'],'teacher_checkpoint_sha256':parent['teacher_checkpoint_sha256'],
            'starting_checkpoint_sha256':checkpoint_sha,'starting_optimizer_step':5625,
            'appended_start_index':PARENT_STOP,'appended_rows':appended,'total_fresh_sources':STOP,
            'source_ids_sha256':digest(all_ids),'authorized_source_interval':[START,STOP],
            'target_optimizer_step':10000,'gradient_accumulation':12,'execution_batch_size':1,
            'input_hashes':{str(p):sha(p) for p in (args.parent_plan,args.original_manifest,args.master,args.checkpoint_receipt)},
            'generator_sha256':sha(__file__),'ordering_seed':ORDER_SEED,'crop_seed':CROP_SEED,
            'summaries':summaries,
            'language_weights':original['summaries']['fit']['normalized_language_counts'],
            'disjointness':{'source_id':True,'audio_sha256':True,'parent_recording_id':True,'speaker_disjointness_claimed':False},
            'quiet_coverage':'Measured on exact cached teacher targets per shard; crop choice and objectives unchanged',
            'expressive_limit':'Dataset-level labels do not guarantee timestamped vocal events within the chosen crop'}
    plan['identity_sha256'] = digest(plan)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(args.out,plan)
    small = {k:v for k,v in plan.items() if k not in ('appended_rows','language_weights')}
    small['plan_sha256'] = sha(args.out)
    atomic_json(args.out.with_name('summary.json'),small)
    print(json.dumps(small),flush=True)


if __name__ == '__main__': main()
