"""Pure receipt/process tests; never starts a model, worker or server."""
from copy import deepcopy
from pathlib import Path
import json
import signal
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import independent_trial_queue as q

D = 'audiovae2_fresh_candidate_recovery_v1'
G = 'audiovae2_grail_candidate_recovery_v1'
Q = 'audiovae2_quiet_candidate_recovery_v1'


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def recovery(root, version=G):
    root.mkdir(parents=True, exist_ok=True)
    ckpt = root / 'checkpoint-step2000.pt'
    ckpt.write_bytes(b'test checkpoint, not a model')
    quality = {'nonquiet_cosine_mean': .99, 'teacher_mae_mean': .002}
    data = dict(version=version, status='awaiting_review', failure=None, step=2000,
                source_count=24000, frozen_state_preserved=True,
                original_files_preserved=True, automatic_next_cut=False,
                optimizer_reset_after_start=False, last_checkpoint_sha256=q.sha(ckpt), final=quality)
    dump(root/'checkpoint-step2000.json', dict(checkpoint_sha256=q.sha(ckpt), step=2000,
         sources_seen=24000, frozen_state_preserved=True, quality=quality))
    dump(root/'completed.json', data)
    return data


def initializer(root):
    root.mkdir(parents=True, exist_ok=True)
    artifact = root/'grail-native-operators.pt'
    artifact.write_bytes(b'test artifact, not a model')
    receipt = dict(operators_sha256=q.sha(artifact), candidate_state_sha256='a'*64)
    data = dict(version='audiovae2_grail_hidden_initialization_v1', variant='grail_shared_hidden',
                complete=True, status='evaluated', failure=None, neural_training_updates=0,
                files_preserved=True, states_preserved={'teacher': True, 'student': True},
                teacher_cache_comparisons=384, teacher_cache_all_pass=True,
                candidate_receipt=receipt, results={'grail': {'aggregate': {'teacher_mae_mean': .01}}})
    dump(root/'grail-receipt.json', receipt)
    dump(root/'completed.json', data)
    return data


@pytest.mark.parametrize('version', [D, G, Q])
def test_completed_recovery_is_bound_to_its_specific_version_and_checkpoint(tmp_path, version):
    recovery(tmp_path, version)
    got = q.verify_completion(tmp_path/'completed.json', kind='recovery', expected_version=version)
    assert got['steps'] == 2000 and got['source_count'] == 24000
    assert got['checkpoint_sha256'] == q.sha(tmp_path/'checkpoint-step2000.pt')
    other = D if version != D else G
    with pytest.raises(RuntimeError):
        q.verify_completion(tmp_path/'completed.json', kind='recovery', expected_version=other)
    with pytest.raises(ValueError):
        q.verify_completion(tmp_path/'completed.json', kind='recovery')


@pytest.mark.parametrize('field,value', [
    ('status', 'running'), ('failure', 'failed'), ('step', 1999), ('source_count', 23999),
    ('frozen_state_preserved', False), ('original_files_preserved', False),
    ('automatic_next_cut', True), ('optimizer_reset_after_start', True),
    ('last_checkpoint_sha256', 'bad'), ('final', {'teacher_mae_mean': 123}),
])
def test_recovery_rejects_incomplete_or_mutated_receipts(tmp_path, field, value):
    data = recovery(tmp_path); data[field] = value; dump(tmp_path/'completed.json', data)
    with pytest.raises(RuntimeError):
        q.verify_completion(tmp_path/'completed.json', kind='recovery', expected_version=G)


@pytest.mark.parametrize('field,value', [
    ('step', 1000), ('sources_seen', 12000), ('checkpoint_sha256', 'bad'),
    ('frozen_state_preserved', False), ('quality', {'teacher_mae_mean': 0}),
])
def test_recovery_rejects_checkpoint_receipt_disagreement(tmp_path, field, value):
    recovery(tmp_path)
    receipt = json.loads((tmp_path/'checkpoint-step2000.json').read_text())
    receipt[field] = value; dump(tmp_path/'checkpoint-step2000.json', receipt)
    with pytest.raises(RuntimeError):
        q.verify_completion(tmp_path/'completed.json', kind='recovery', expected_version=G)


def test_recovery_rejects_checkpoint_byte_mutation(tmp_path):
    recovery(tmp_path); (tmp_path/'checkpoint-step2000.pt').write_bytes(b'changed')
    with pytest.raises(RuntimeError):
        q.verify_completion(tmp_path/'completed.json', kind='recovery', expected_version=G)


@pytest.mark.parametrize('damage', ['update', 'empty_states', 'state', 'cache_count', 'cache_fail', 'receipt', 'artifact'])
def test_initializer_rejects_incomplete_or_disagreeing_provenance(tmp_path, damage):
    data = initializer(tmp_path)
    if damage == 'update': data['neural_training_updates'] = 1
    elif damage == 'empty_states': data['states_preserved'] = {}
    elif damage == 'state': data['states_preserved']['teacher'] = False
    elif damage == 'cache_count': data['teacher_cache_comparisons'] = 383
    elif damage == 'cache_fail': data['teacher_cache_all_pass'] = False
    elif damage == 'receipt':
        changed = deepcopy(data['candidate_receipt']); changed['candidate_state_sha256'] = 'b'*64
        dump(tmp_path/'grail-receipt.json', changed)
    elif damage == 'artifact': (tmp_path/'grail-native-operators.pt').write_bytes(b'changed')
    dump(tmp_path/'completed.json', data)
    with pytest.raises(RuntimeError):
        q.verify_completion(tmp_path/'completed.json', kind='initializer')


def test_dynamic_g_identity_comes_only_from_completed_authenticated_initializer(tmp_path):
    data = initializer(tmp_path)
    stage = {'argv': ['python', 'g.py', '@GRAIL_ARTIFACT_SHA@', '@GRAIL_STATE_SHA@']}
    original = deepcopy(stage)
    resolved = q.resolve_argv(stage, tmp_path)
    assert resolved[-2:] == [data['candidate_receipt']['operators_sha256'], 'a'*64]
    assert stage == original
    assert q.verify_completion(tmp_path/'completed.json', kind='initializer')['complete']
    with pytest.raises(ValueError): q.resolve_argv({'argv': ['@GRAIL_UNKNOWN@']}, tmp_path)
    (tmp_path/'grail-native-operators.pt').write_bytes(b'changed')
    with pytest.raises(RuntimeError): q.resolve_argv(stage, tmp_path)


def test_sealed_code_is_checked_without_executing_it(tmp_path):
    code = tmp_path/'child.py'; code.write_text('raise AssertionError("must not execute")')
    stage = {'argv': ['python', str(code)], 'kind': 'recovery', 'protected_code': {str(code): q.sha(code)}}
    q.authenticate_code(stage)
    code.write_text('changed')
    with pytest.raises(RuntimeError): q.authenticate_code(stage)


def stat_text(state='S', ticks='999'):
    # Linux fields after the closing comm: state is 3, starttime is 22.
    fields = [state] + ['0']*18 + [ticks] + ['0']*10
    return '123 (worker ) with spaces) ' + ' '.join(fields)


def test_proc_identity_handles_exit_race_without_exists_probe(monkeypatch):
    def gone(*a, **k): raise FileNotFoundError('exited between observations')
    monkeypatch.setattr(q.Path, 'read_text', gone)
    assert q.proc_identity(123) is None and not q.active(123)


@pytest.mark.parametrize('state,expected', [('S', True), ('R', True), ('T', True), ('Z', False), ('X', False)])
def test_proc_comm_spaces_and_parentheses_do_not_shift_starttime(monkeypatch, state, expected):
    monkeypatch.setattr(q.Path, 'read_text', lambda *a, **k: stat_text(state, '6789'))
    assert q.proc_identity(123) == {'state': state, 'start_ticks': '6789'}
    assert q.active(123) is expected


def queue_config(tmp_path):
    predecessor = tmp_path/'d'/'results'; recovery(predecessor, D)
    code = tmp_path/'sealed.py'; code.write_text('# never executed')
    stages = []
    for name, kind, version in [('grail_initialization', 'initializer', None),
                                ('grail_recovery', 'recovery', G), ('quiet_recovery', 'recovery', Q)]:
        root = tmp_path/name
        argv = ['fakepython', str(code), name]
        if name == 'grail_recovery': argv += ['@GRAIL_ARTIFACT_SHA@', '@GRAIL_STATE_SHA@']
        stages.append(dict(name=name, kind=kind, expected_version=version, root=str(root),
                           completion=str(root/'results'/'completed.json'), argv=argv,
                           protected_code={str(code): q.sha(code)}, environment={}))
    config = dict(version=q.VERSION, tensorboard_pid=17, tensorboard_logdir=str(tmp_path/'old-tb'),
                  tensorboard_executable='fake-tensorboard',
                  predecessor={'pid': 111, 'start_ticks': '42', 'completion': str(predecessor/'completed.json'),
                               'expected_version': D}, stages=stages, grail_init=str(tmp_path/'grail_initialization'/'results'))
    path = tmp_path/'queue'/'config.json'; dump(path, config)
    return path, config


def fake_queue(monkeypatch, config_path, config, fail=None):
    calls = []
    monkeypatch.setattr(sys, 'argv', ['queue', '--config', str(config_path)])
    monkeypatch.setattr(q, 'proc_identity', lambda pid: None)
    monkeypatch.setattr(q.time, 'sleep', lambda seconds: None)
    def popen(argv, **kwargs):
        name = argv[2]; calls.append(name)
        stage = next(s for s in config['stages'] if s['name'] == name)
        root = Path(stage['completion']).parent
        assert not root.exists()
        if name == 'grail_initialization': initializer(root)
        else:
            if name == 'grail_recovery':
                receipt = json.loads((Path(config['grail_init'])/'grail-receipt.json').read_text())
                assert argv[-2:] == [receipt['operators_sha256'], receipt['candidate_state_sha256']]
            recovery(root, stage['expected_version'])
        rc = 1 if name == fail else 0
        return SimpleNamespace(pid=1000+len(calls), returncode=rc, poll=lambda: rc)
    monkeypatch.setattr(q.subprocess, 'Popen', popen)
    return calls


def test_exact_finite_order_dynamic_identity_and_no_restart_or_overwrite(tmp_path, monkeypatch):
    path, config = queue_config(tmp_path); calls = fake_queue(monkeypatch, path, config)
    old = tmp_path/'old-tb'/'events.old'; old.parent.mkdir(); old.write_bytes(b'preserve')
    q.main()
    assert calls == ['grail_initialization', 'grail_recovery', 'quiet_recovery']
    state = json.loads((path.parent/'status.json').read_text())
    assert state['status'] == 'completed' and len(state['completed']) == 3
    assert state['predecessor']['steps'] == 2000
    assert not state['automatic_extension'] and not state['automatic_promotion']
    before = (path.parent/'status.json').read_bytes()
    with pytest.raises(FileExistsError): q.main()
    assert len(calls) == 3 and (path.parent/'status.json').read_bytes() == before
    assert old.read_bytes() == b'preserve'


@pytest.mark.parametrize('failed,expected_count', [('grail_initialization', 1), ('grail_recovery', 2), ('quiet_recovery', 3)])
def test_failed_child_stops_queue_without_retry(tmp_path, monkeypatch, failed, expected_count):
    path, config = queue_config(tmp_path); calls = fake_queue(monkeypatch, path, config, fail=failed)
    with pytest.raises(RuntimeError, match='Child process failed'): q.main()
    state = json.loads((path.parent/'status.json').read_text())
    assert len(calls) == expected_count and state['status'] == 'failed'
    assert len(state['completed']) == expected_count-1


@pytest.mark.parametrize('kind', ['output', 'lock', 'log'])
def test_preexisting_child_files_are_not_overwritten(tmp_path, monkeypatch, kind):
    path, config = queue_config(tmp_path); calls = fake_queue(monkeypatch, path, config)
    stage = config['stages'][0]; root = Path(stage['root']); root.mkdir()
    if kind == 'output': old = Path(stage['completion']); dump(old, {'original': True})
    else: old = root/(stage['name']+'.'+kind); old.write_bytes(b'original')
    before = old.read_bytes()
    with pytest.raises(FileExistsError): q.main()
    assert not calls and old.read_bytes() == before


def test_predecessor_pid_reuse_stops_before_any_child(tmp_path, monkeypatch):
    path, config = queue_config(tmp_path); calls = fake_queue(monkeypatch, path, config)
    monkeypatch.setattr(q, 'proc_identity', lambda pid: {'state': 'S', 'start_ticks': 'different'})
    with pytest.raises(RuntimeError, match='identity changed'): q.main()
    assert not calls


def test_predecessor_exit_between_observations_uses_sealed_completion(tmp_path, monkeypatch):
    path, config = queue_config(tmp_path); calls = fake_queue(monkeypatch, path, config)
    observations = iter([{'state': 'S', 'start_ticks': '42'}, None])
    monkeypatch.setattr(q, 'proc_identity', lambda pid: next(observations))
    q.main(); assert len(calls) == 3


def dashboard_setup(tmp_path):
    root = tmp_path/'new'; out = root/'results'; out.mkdir(parents=True)
    (out/'train.jsonl').write_text('{}\n')
    dump(out/'initial-parity.json', {'quality': {'passed': True}, 'rng': {'equal': True},
                                   'optimizer': {'equal': True}, 'candidate_state_exact': True})
    old = tmp_path/'old-events'; old.mkdir(); (old/'events').write_bytes(b'old events')
    stage = {'root': str(root), 'completion': str(out/'completed.json'), 'tensorboard': str(out/'tensorboard')}
    return stage, {'tensorboard_executable': 'tensorboard'}, {'tensorboard_pid': 7, 'tensorboard_logdir': str(old)}, old


def test_dashboard_waits_for_initial_parity_before_touching_server(tmp_path, monkeypatch):
    stage, config, state, old = dashboard_setup(tmp_path)
    def forbidden(*a, **k): raise AssertionError('must not touch processes')
    monkeypatch.setattr(q.os, 'kill', forbidden); monkeypatch.setattr(q.subprocess, 'Popen', forbidden)
    out = Path(stage['completion']).parent
    (out/'train.jsonl').unlink()
    assert q.switch_dashboard(stage, config, state) is False
    (out/'train.jsonl').write_text('{}\n')
    dump(out/'initial-parity.json', {'quality': {'passed': False}, 'rng': {'equal': True},
                                   'optimizer': {'equal': True}, 'candidate_state_exact': True})
    with pytest.raises(RuntimeError): q.switch_dashboard(stage, config, state)
    assert (old/'events').read_bytes() == b'old events'


def test_dashboard_rejects_unrelated_process(tmp_path, monkeypatch):
    stage, config, state, old = dashboard_setup(tmp_path)
    monkeypatch.setattr(q, 'active', lambda pid: True)
    monkeypatch.setattr(q.Path, 'read_bytes', lambda path: b'other\0--port\08889\0')
    def forbidden(*a, **k): raise AssertionError('must not touch unrelated process')
    monkeypatch.setattr(q.os, 'kill', forbidden); monkeypatch.setattr(q.subprocess, 'Popen', forbidden)
    with pytest.raises(RuntimeError, match='unrelated'): q.switch_dashboard(stage, config, state)


def test_dashboard_switch_preserves_event_files_and_starts_once(tmp_path, monkeypatch):
    stage, config, state, old = dashboard_setup(tmp_path)
    original_read = q.Path.read_bytes
    def read(path):
        if str(path) == '/proc/7/cmdline':
            return b'python\0tensorboard\0--port\08888\0--logdir\0'+str(old).encode()+b'\0'
        return original_read(path)
    monkeypatch.setattr(q.Path, 'read_bytes', read)
    live = {7: True}; killed = []; launched = []
    monkeypatch.setattr(q, 'active', lambda pid: live.get(pid, False))
    def kill(pid, sig): killed.append((pid, sig)); live[pid] = False
    monkeypatch.setattr(q.os, 'kill', kill)
    def popen(argv, **kwargs): launched.append(argv); return SimpleNamespace(pid=8)
    monkeypatch.setattr(q.subprocess, 'Popen', popen)
    assert q.switch_dashboard(stage, config, state)
    assert killed == [(7, signal.SIGTERM)] and len(launched) == 1
    assert state['tensorboard_pid'] == 8 and state['tensorboard_logdir'] == stage['tensorboard']
    assert (old/'events').read_bytes() == b'old events'
    assert json.loads((Path(stage['root'])/'tensorboard-launch.json').read_text())['previous_events_preserved']
    with pytest.raises(FileExistsError): q.switch_dashboard(stage, config, state)
    assert len(launched) == 1


@pytest.mark.parametrize('damage', ['unpaired_port', 'unpaired_logdir', 'wrong_executable'])
def test_dashboard_token_membership_cannot_authorize_unrelated_server(tmp_path, monkeypatch, damage):
    stage, config, state, old = dashboard_setup(tmp_path)
    tokens = [b'python', b'tensorboard', b'--port', b'8888', b'--logdir', str(old).encode()]
    if damage == 'unpaired_port': tokens[3] = b'9999'; tokens += [b'--other', b'8888']
    elif damage == 'unpaired_logdir': tokens[5] = b'elsewhere'; tokens += [b'--other', str(old).encode()]
    else: tokens[1] = b'unrelated'
    monkeypatch.setattr(q.Path, 'read_bytes', lambda path: b'\0'.join(tokens)+b'\0')
    monkeypatch.setattr(q, 'active', lambda pid: True)
    def forbidden(*a, **k): raise AssertionError('must not stop unrelated server')
    monkeypatch.setattr(q.os, 'kill', forbidden)
    with pytest.raises(RuntimeError, match='unrelated'): q.switch_dashboard(stage, config, state)


@pytest.mark.parametrize('name', ['tensorboard.log', 'tensorboard-launch.json'])
def test_existing_dashboard_artifact_rejects_before_stopping_old_server(tmp_path, monkeypatch, name):
    stage, config, state, old = dashboard_setup(tmp_path)
    retained = Path(stage['root'])/name; retained.write_bytes(b'original')
    def forbidden(*a, **k): raise AssertionError('must not inspect or stop existing server')
    monkeypatch.setattr(q, 'active', forbidden); monkeypatch.setattr(q.os, 'kill', forbidden)
    monkeypatch.setattr(q.subprocess, 'Popen', forbidden)
    with pytest.raises(FileExistsError): q.switch_dashboard(stage, config, state)
    assert retained.read_bytes() == b'original' and (old/'events').read_bytes() == b'old events'


def test_predecessor_incomplete_blocks_every_child(tmp_path, monkeypatch):
    path, config = queue_config(tmp_path); calls = fake_queue(monkeypatch, path, config)
    predecessor = Path(config['predecessor']['completion'])
    data = json.loads(predecessor.read_text()); data['step'] = 1999; dump(predecessor, data)
    with pytest.raises(RuntimeError): q.main()
    assert not calls


def test_changed_sealed_source_blocks_first_child(tmp_path, monkeypatch):
    path, config = queue_config(tmp_path); calls = fake_queue(monkeypatch, path, config)
    code = Path(next(iter(config['stages'][0]['protected_code'])))
    code.write_text('changed after config was sealed')
    with pytest.raises(RuntimeError, match='Sealed source changed'): q.main()
    assert not calls


@pytest.mark.parametrize('damage', ['reorder', 'extra', 'wrong_kind'])
def test_queue_rejects_any_other_trial_sequence_before_lock(tmp_path, monkeypatch, damage):
    path, config = queue_config(tmp_path)
    if damage == 'reorder': config['stages'][1:] = reversed(config['stages'][1:])
    elif damage == 'extra': config['stages'].append(deepcopy(config['stages'][-1]))
    else: config['stages'][0]['kind'] = 'recovery'
    dump(path, config); calls = fake_queue(monkeypatch, path, config)
    with pytest.raises(ValueError): q.main()
    assert not calls and not (path.parent/'queue.lock').exists()
