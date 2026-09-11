"""CPU-only CLI checks of the separate progressive source-plan preparation."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'fast-audiovae'/'experiments'/'audiovae2-compression'))
spec = importlib.util.spec_from_file_location('prepare_progressive_extension_under_test', HERE/'prepare_progressive_extension_5000.py')
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def seal(path, value):
    value['identity_sha256'] = prep.digest({k:v for k,v in value.items() if k != 'identity_sha256'})
    path.write_text(json.dumps(value))
    return value


@pytest.fixture(scope='module')
def fixture(tmp_path_factory):
    directory = tmp_path_factory.mktemp('synthetic-progressive-selection')
    audio = directory/'existing-audio'; audio.write_bytes(b'existence-only planner fixture')
    languages = 'as bn brx doi gu hi kn kok ks mai ml mni mr ne or pa sa sat sd ta te ur'.split()
    def rows(prefix, count):
        return [{'source_id':f'{prefix}{i}', 'manifest_row':{
                    'source_id':f'{prefix}{i}', 'audio_sha256':f'hash-{prefix}{i}',
                    'parent_recording_id':f'parent-{prefix}{i}', 'audio_path':str(audio),
                    'split':'train', 'language':languages[i%22]+'_in', 'dataset':'indicvoices'},
                 'input_samples16k':2560, 'start_frame':0, 'context_start_frame':0,
                 'context_frames':0, 'scored_frames':4, 'valid_scored_samples':7680,
                 'broad_expressive_source':False} for i in range(count)]
    fitting, old_rows, appended = rows('fit-',3000), rows('old-',27000), rows('new-',30001)
    manifest = directory/'manifest.json'
    manifest.write_text(json.dumps({'splits':{'fit':{'rows':fitting}}}))
    parent = directory/'parent.json'
    parent_value = seal(parent, {'rows':old_rows, 'source_ids':[r['source_id'] for r in old_rows],
        'blocked_identities':{key:[r['manifest_row'][key] for r in fitting] for key in prep.KEYS},
        'teacher_source_sha256':'teacher-source', 'teacher_checkpoint_sha256':'teacher-weights',
        'original_manifest_sha256':prep.sha(manifest)})
    selection = directory/'selection.json'
    selected = seal(selection, {'parent_plan_identity_sha256':parent_value['identity_sha256'],
        'parent_plan_sha256':prep.sha(parent), 'original_manifest_sha256':prep.sha(manifest),
        'appended_rows':appended})
    checkpoint = directory/'checkpoint.pt'; checkpoint.write_bytes(b'synthetic checkpoint bytes, never loaded')
    return {'directory':directory, 'manifest':manifest, 'parent':parent, 'selection':selection,
            'selected':selected, 'checkpoint':checkpoint, 'fit':fitting, 'old':old_rows}


def run(fixture, tmp_path, monkeypatch, value=None):
    chosen = fixture['selection']
    identity = fixture['selected']['identity_sha256']
    if value is not None:
        chosen = tmp_path/'changed-selection.json'
        identity = seal(chosen, value)['identity_sha256']
    # Production pins remain unchanged; tiny fixture bytes stand in for them.
    monkeypatch.setattr(prep, 'SELECTION', identity)
    monkeypatch.setattr(prep, 'CHECKPOINT', prep.sha(fixture['checkpoint']))
    out = tmp_path/'output'/'plan.json'
    monkeypatch.setattr(sys, 'argv', ['prepare', '--parent-plan',str(fixture['parent']),
        '--manifest',str(fixture['manifest']), '--existing-selection',str(chosen),
        '--checkpoint',str(fixture['checkpoint']), '--out',str(out)])
    prep.main()
    return out


def test_complete_cli_keeps_exact_first30000_order_geometry_and_new_identity(fixture, tmp_path, monkeypatch):
    preserved = {k:prep.sha(fixture[k]) for k in ('parent','selection','manifest','checkpoint')}
    path = run(fixture,tmp_path,monkeypatch)
    plan = json.loads(path.read_text())
    assert plan['appended_rows'] == fixture['selected']['appended_rows'][:30000]
    expected = [r['source_id'] for r in fixture['fit']+fixture['old']+plan['appended_rows']]
    assert len(expected) == len(set(expected)) == 60000
    assert plan['source_ids_sha256'] == prep.digest(expected)
    assert plan['identity_sha256'] == prep.digest({k:v for k,v in plan.items() if k!='identity_sha256'})
    assert plan['identity_sha256'] != fixture['selected']['identity_sha256']
    assert plan['authorized_source_interval'] == [24000,60000]
    assert plan['appended_start_index'] == 30000
    assert list(plan['summaries']) == [f'{i}:{i+6000}' for i in range(24000,60000,6000)]
    assert all(s['sources']==6000 and len(s['languages'])==22 for s in plan['summaries'].values())
    assert preserved == {k:prep.sha(fixture[k]) for k in preserved}


@pytest.mark.parametrize('key', prep.KEYS)
def test_collision_with_original_fit_rejected_before_new_plan(fixture,tmp_path,monkeypatch,key):
    value = dict(fixture['selected'])
    rows = list(value['appended_rows'])
    first = dict(rows[0]); first['manifest_row'] = dict(first['manifest_row'])
    first['manifest_row'][key] = fixture['fit'][0]['manifest_row'][key]
    if key == 'source_id': first['source_id'] = first['manifest_row']['source_id']
    rows[0] = first; value['appended_rows'] = rows
    with pytest.raises(ValueError,match='Repeated or reserved'):
        run(fixture,tmp_path,monkeypatch,value)
    assert not (tmp_path/'output'/'plan.json').exists()


def test_invalid_crop_context_rejected_before_new_plan(fixture,tmp_path,monkeypatch):
    value = dict(fixture['selected']); rows = list(value['appended_rows'])
    rows[0] = {**rows[0], 'context_frames':30}
    value['appended_rows'] = rows
    with pytest.raises(ValueError,match='causal or valid-sample'):
        run(fixture,tmp_path,monkeypatch,value)
    assert not (tmp_path/'output'/'plan.json').exists()


def test_changed_parent_selection_hash_rejected_before_plan(fixture,tmp_path,monkeypatch):
    value = {**fixture['selected'], 'parent_plan_sha256':'different'}
    with pytest.raises(ValueError,match='identity differs'):
        run(fixture,tmp_path,monkeypatch,value)
    assert not (tmp_path/'output'/'plan.json').exists()
