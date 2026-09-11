"""No subprocesses or models are started by these launch-guard tests."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('identity_launch_under_test', HERE / 'launch_control.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def ready(tmp_path):
    root, active, project, proc = [tmp_path / name for name in ('control', 'active', 'project', 'proc')]
    for path in (root / 'code', active / 'segment-2000-5000', project / 'code', proc):
        path.mkdir(parents=True)
    (root / 'launch_control.py').write_bytes((HERE / 'launch_control.py').read_bytes())
    for name in module.FROZEN_FILES:
        (root / name).write_text('# sealed fixture\n')
    (project / 'code/group_model.py').write_bytes((root / 'code/group_model.py').read_bytes())
    write(root / 'source-freeze.json', {name: module.sha(root / name) for name in module.FROZEN_FILES})
    (root / 'cpu-tests.txt').write_text('28 passed in 2.00s\n')
    write(root / 'cpu-tests-receipt.json', {
        'version': 'identical_teacher_control_cpu_tests_v1', 'passed': True, 'exit_code': 0,
        'python': sys.executable, 'tests_passed': 28,
        'test_files': ['code/test_identical_teacher_control.py'],
        'source_freeze_sha256': module.sha(root / 'source-freeze.json'),
        'test_log': 'cpu-tests.txt', 'test_log_sha256': module.sha(root / 'cpu-tests.txt'),
        'launch_helper_sha256': module.sha(root / 'launch_control.py')})
    checkpoint = active / 'segment-2000-5000/checkpoint-step5000.pt'
    checkpoint.write_bytes(b'qualified checkpoint fixture')
    write(checkpoint.with_suffix('.json'), {'cut_updates': 5000, 'global_updates': 5000,
        'sources_seen': 60000, 'frozen_state_preserved': True, 'checkpoint_sha256': module.sha(checkpoint)})
    write(checkpoint.parent / 'completed.json', {'status': 'awaiting_review', 'failure': None,
        'step': 5000, 'cut_updates': 5000, 'sources_seen': 60000, 'frozen_state_preserved': True,
        'original_files_preserved': True, 'last_checkpoint_sha256': module.sha(checkpoint)})
    write(active / 'producer-completed.json', {'complete': True, 'phases': 2, 'training_updates': 0})
    return dict(root=root, active=active, project=project, python=Path(sys.executable), proc_root=proc)


def test_not_ready_or_stale_evidence_never_starts_a_process_and_leaves_no_launch_record(ready):
    root, active = ready['root'], ready['active']
    assert module.readiness(**ready)['training_step'] == 5000
    def never(*args, **kwargs):
        pytest.fail('A failing readiness gate must never call Popen')
    failures = [(active / 'segment-2000-5000/completed.json', b'{"status":"running"}'),
                (active / 'producer-completed.json', b'{"complete":false}'),
                (root / 'code/identical_teacher_control.py', b'# altered\n'),
                (ready['project'] / 'code/group_model.py', b'# different pinned dependency\n'),
                (root / 'cpu-tests.txt', b'28 failed\n')]
    for path, damaged in failures:
        original = path.read_bytes()
        path.write_bytes(damaged)
        with pytest.raises(ValueError): module.launch(**ready, popen=never)
        assert not (root / '.launch.lock').exists()
        assert not (root / 'process-launch.json').exists()
        assert not (root / 'control.log').exists()
        assert not (root / 'results').exists()
        path.write_bytes(original)


def test_exact_process_matching_and_exclusive_lock_prevent_overlapping_launches(ready):
    proc = ready['proc_root'] / '123'
    proc.mkdir()
    script = str(ready['active'] / 'code/progressive_continue_5000.py').encode()
    (proc / 'cmdline').write_bytes(b'python\0' + script + b'\0--out\0somewhere\0')
    with pytest.raises(RuntimeError, match='PID 123'): module.readiness(**ready)
    # A log viewer mentioning the path is not that executable's argv token.
    (proc / 'cmdline').write_bytes(b'viewer\0message containing ' + script + b'\0')
    assert module.readiness(**ready)['sources'] == 60000
    lock = ready['root'] / '.launch.lock'
    lock.write_text('another launcher owns this')
    with pytest.raises(FileExistsError): module.launch(**ready, popen=lambda *a, **k: pytest.fail('locked'))
    assert lock.read_text() == 'another launcher owns this'


def test_success_is_detached_argv_only_and_cannot_duplicate_or_mutate_training(ready):
    root, active = ready['root'], ready['active']
    preserved = {str(path): module.sha(path) for path in active.rglob('*') if path.is_file()}
    calls = []
    def fake_popen(command, **kwargs):
        assert isinstance(command, list) and command[0] == sys.executable
        assert kwargs['stdin'] == module.subprocess.DEVNULL
        assert kwargs['start_new_session'] is True and kwargs.get('shell', False) is False
        assert kwargs['cwd'] == root
        assert kwargs['stdout'].name == str(root / 'control.log')
        assert kwargs['env']['PYTHONPATH'].split(':')[0] == str(root / 'code')
        assert command[command.index('--out') + 1] == str(root / 'results')
        calls.append(command)
        return SimpleNamespace(pid=4321)
    result = module.launch(**ready, popen=fake_popen)
    assert result['pid'] == 4321 and len(calls) == 1
    assert module.read_json(root / 'process-launch.json')['command'] == calls[0]
    assert (root / '.launch.lock').exists()
    assert not (root / 'results').exists()  # Only the actual control creates its output.
    with pytest.raises(FileExistsError): module.launch(**ready, popen=fake_popen)
    assert len(calls) == 1
    assert {str(path): module.sha(path) for path in active.rglob('*') if path.is_file()} == preserved
