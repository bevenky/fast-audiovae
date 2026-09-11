"""CPU tests for unique source selection and checkpoint-bound cache retirement."""
import copy
import ast
import importlib.util
import json
from pathlib import Path
import sys
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import continuation_data_v2 as data
# Remote packaging puts the sidecar beside the loader; local project keeps work scripts outside the repository.
PLANNER = HERE/'prepare_continuation_extension_v2.py'
if not PLANNER.exists(): PLANNER = HERE.parents[2]/'audiovae2-compression-preflight/prepare_continuation_extension_v2.py'
spec = importlib.util.spec_from_file_location('extension_planner_test',PLANNER)
planner = importlib.util.module_from_spec(spec);spec.loader.exec_module(planner)


def source(number,language='en'):
    return {'source_id':f's{number}','audio_sha256':f'h{number}','parent_recording_id':f'p{number}',
            'split':'train','duration_seconds':10.,'language':language}


def test_selection_preserves_weights_and_removes_all_three_reserved_identities():
    rows = [source(i,'en' if i<40 else 'hi') for i in range(80)]
    rows += [dict(source(100),audio_sha256='h0'),dict(source(101),parent_recording_id='p1')]
    blocked = {'source_id':{'s2'},'audio_sha256':{'h0'},'parent_recording_id':{'p1'}}
    before = copy.deepcopy(rows)
    selected = planner.select_unique(rows,blocked,{'en':3,'hi':1},20)
    assert sum(r['language']=='en' for r in selected) == 15
    assert sum(r['language']=='hi' for r in selected) == 5
    for key in data.KEYS:
        assert not {r[key] for r in selected}&blocked[key]
        assert len({r[key] for r in selected}) == 20
    assert selected == planner.select_unique(list(reversed(rows)),blocked,{'en':3,'hi':1},20)
    assert rows == before
    with pytest.raises(ValueError,match='Insufficient'):
        planner.select_unique(rows,blocked,{'en':3,'hi':1},100)


def test_stratified_interleave_preserves_small_language_until_final_segment():
    rows=[source(i,'en' if i<21 else 'doi') for i in range(28)]
    ordered=planner.stratified_interleave(rows)
    assert {row['source_id'] for row in rows}=={row['source_id'] for row in ordered}
    assert ordered==planner.stratified_interleave(list(reversed(rows)))
    assert all(sum(r['language']=='doi' for r in ordered[a:a+4])==1 for a in range(0,28,4))


@pytest.mark.parametrize('seconds',[.1,1.,2.56,3.8,10.,90.])
def test_crop_geometry_keeps_original_context_and_no_padding_in_scored_length(seconds):
    row = dict(source(17),duration_seconds=seconds)
    geometry = planner.geometry(row)
    data.FreshTrainingData._geometry({'source_id':row['source_id'],**geometry})
    assert geometry['context_frames'] in (0,30)
    assert geometry['context_start_frame']+geometry['context_frames']==geometry['start_frame']
    assert geometry['valid_scored_samples']==min(40960,round(seconds*16000)-geometry['start_frame']*640)*3
    assert geometry == planner.geometry(row)


def checkpoint(cursor=30000):
    ids=[f'fresh{i}' for i in range(data.STOP)]
    original=[f'old{i}' for i in range(3000)]
    payload={'fresh_cursor':cursor,'optimizer_step':5625+(cursor-data.START)//12,'accumulation':12,
             'total_source_count':3000+cursor,
             'historical_sources_seen':original+ids[:data.START],
             'additional_sources_seen':ids[data.START:cursor],
             'identity':{'source_plan_identity_sha256':'plan'}}
    return payload,ids,original


@pytest.mark.parametrize('damage',['repeat','omission','cursor','optimizer','plan','total','accumulation'])
def test_retirement_rejects_any_checkpoint_ledger_or_recipe_difference(damage):
    payload,ids,original=checkpoint()
    if damage=='repeat': payload['additional_sources_seen'][-1]=payload['additional_sources_seen'][0]
    if damage=='omission': payload['additional_sources_seen'].pop()
    if damage=='cursor': payload['fresh_cursor']-=12
    if damage=='optimizer': payload['optimizer_step']+=1
    if damage=='plan': payload['identity']['source_plan_identity_sha256']='other'
    if damage=='total': payload['total_source_count']-=1
    if damage=='accumulation': payload['accumulation']=3
    with pytest.raises(ValueError):
        data.validate_committed_checkpoint(payload,30000,ids,original,'plan')


def cache_fixture(tmp_path):
    payload,ids,original=checkpoint()
    provider=data.ContinuationData.__new__(data.ContinuationData)
    provider.shards_dir=tmp_path/'new-cache';provider.shards_dir.mkdir()
    provider.source_ids=tuple(ids);provider.original_ids=original;provider.identity='plan'
    provider.retired_before=data.START;provider.committed_cursor=data.START
    provider._hashes={};provider._stats={};provider._cached_index=None;provider._cached_crops=None
    for start in range(24000,30300,300):
        directory=provider.shards_dir/f'{start:06d}-{start+300:06d}';directory.mkdir()
        pairs=directory/'pairs.pt';pairs.write_bytes(f'cached-{start}'.encode())
        receipt={'plan_identity_sha256':'plan','start_index':start,'stop_index':start+300,
                 'source_ids':ids[start:start+300],'pairs_sha256':data.sha(pairs)}
        data.atomic_json(directory/'receipt.json',receipt)
        provider._hashes[str(pairs)]=data.sha(pairs);provider._stats[str(pairs)]=data.stat_identity(pairs)
    path=tmp_path/'checkpoint.pt';torch.save(payload,path)
    data.atomic_json(path.with_suffix('.json'),{'checkpoint_sha256':data.sha(path),'fresh_cursor':30000,
                    'optimizer_and_rng_saved':True,'frozen_state_preserved':True})
    return provider,path


def test_cache_retirement_keeps_recent_and_future_sources_and_all_receipts(tmp_path):
    provider,path=cache_fixture(tmp_path)
    old=tmp_path/'old-cache';old.mkdir();(old/'pairs.pt').write_bytes(b'keep historical cache')
    marker=provider.checkpoint_committed(path,data.sha(path),30000)
    assert marker['retired_before']==30000
    assert provider.committed_cursor==30000 and provider.retired_before==30000
    for start in range(24000,30300,300):
        directory=provider.shards_dir/f'{start:06d}-{start+300:06d}'
        assert (directory/'receipt.json').is_file()
        assert (directory/'pairs.pt').exists()==(start>=30000)
        assert (directory/'retired.json').exists()==(start<30000)
    assert (old/'pairs.pt').read_bytes()==b'keep historical cache'
    provider.assert_unchanged()
    # Calling with the same committed state is safe and does not extend deletion.
    assert provider.checkpoint_committed(path,data.sha(path),30000)==marker
    with pytest.raises(ValueError,match='historical'):
        provider.take(29988,12)


def test_producer_can_fill_the_full_next_review_after_verified_checkpoint(tmp_path):
    provider,path=cache_fixture(tmp_path)
    producer=copy.copy(provider)
    producer._hashes=dict(provider._hashes);producer._stats=dict(provider._stats)
    assert producer.can_produce_through(30000)
    assert not producer.can_produce_through(30300)
    provider.checkpoint_committed(path,data.sha(path),30000)
    assert producer.can_produce_through(36000)
    assert not producer.can_produce_through(36300)
    assert producer.retired_before==30000
    producer.assert_unchanged()
    assert all(Path(p).exists() for p in producer._hashes)


@pytest.mark.parametrize('damage',['bad_sha','missing_receipt','changed_shard','wrong_receipt'])
def test_unverified_commit_cannot_retire_cached_targets(tmp_path,damage):
    provider,path=cache_fixture(tmp_path)
    checksum=data.sha(path)
    if damage=='bad_sha': checksum='0'*64
    elif damage=='missing_receipt': path.with_suffix('.json').unlink()
    elif damage=='wrong_receipt':
        r=json.loads(path.with_suffix('.json').read_text());r['fresh_cursor']=30012;data.atomic_json(path.with_suffix('.json'),r)
    elif damage=='changed_shard': (provider.shards_dir/'024000-024300/pairs.pt').write_bytes(b'changed')
    with pytest.raises((ValueError,FileNotFoundError)):
        provider.checkpoint_committed(path,checksum,30000)
    assert all((provider.shards_dir/f'{s:06d}-{s+300:06d}'/'pairs.pt').exists() for s in range(24000,30300,300))
    assert not (provider.shards_dir/'committed-cursor.json').exists()


def test_producer_full_source_teacher_construction_is_unchanged():
    folder=PLANNER.parent
    old_path=folder/'produce_continuation_pairs.py'
    if not old_path.exists(): old_path=Path('/workspace/fast-audiovae-compression-20260910-v1/produce_continuation_pairs.py')
    old=ast.parse(old_path.read_text())
    new=ast.parse((folder/'produce_continuation_extension_v2.py').read_text())
    def source_loop(tree):
        return next(node for node in ast.walk(tree) if isinstance(node,ast.For)
                    and isinstance(node.target,ast.Tuple)
                    and [value.id for value in node.target.elts]==['i','row'])
    assert ast.dump(source_loop(old),include_attributes=False)==ast.dump(source_loop(new),include_attributes=False)
    for name in ('read_source','window','tensor_sha','frozen_versions'):
        a=next(node for node in old.body if isinstance(node,ast.FunctionDef) and node.name==name)
        b=next(node for node in new.body if isinstance(node,ast.FunctionDef) and node.name==name)
        assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)


def test_target_coverage_measures_valid_tail_and_absolute_startup_without_model_imports():
    tree=ast.parse((PLANNER.parent/'produce_continuation_extension_v2.py').read_text())
    fn=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='target_coverage')
    namespace={'torch':torch}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<coverage>','exec'),namespace)
    rows=[{'source_id':'startup','context_frames':0,'start_frame':0,'valid_scored_samples':1197,
           'teacher_audio':torch.full((1,1,1920),5e-6)},
          {'source_id':'interior','context_frames':30,'start_frame':35,'valid_scored_samples':960,
           'teacher_audio':torch.full((1,1,31*1920),5e-6)}]
    before=[row['teacher_audio'].clone() for row in rows]
    report=namespace['target_coverage'](rows)
    assert report['aggregate']['samples']==2157
    assert report['aggregate']['near_samples']==2157
    assert report['aggregate']['near_startup_samples']==1197
    assert report['aggregate']['near_interior_samples']==960
    assert report['aggregate']['near_windows']==3
    for row,saved in zip(rows,before):torch.testing.assert_close(row['teacher_audio'],saved,rtol=0,atol=0)
