"""Small integration checks for fresh-C startup-anchor recovery and its monitor."""
from pathlib import Path
import sys

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'convnext'))
import startup_anchor_recovery as run
import startup_anchor_monitor as monitor_api
import test_quiet_candidate_recovery as inherited
import test_startup_anchor_update as fixtures


@pytest.fixture(autouse=True)
def cpu_policy():
    rng, threads = run.screen.rng_state(), torch.get_num_threads()
    torch.manual_seed(801)
    torch.set_num_threads(1)
    yield
    run.screen.restore_rng(rng)
    torch.set_num_threads(threads)


def test_new_runner_fresh_factory_and_checkpoint_contract_with_existing_tiny_fixtures(tmp_path, monkeypatch):
    # Execute the existing behavioral checks against the NEW module; old tests
    # and production modules stay unchanged. The poison old group must not load.
    monkeypatch.setattr(inherited, 'run', run)
    inherited.test_fresh_factory_b_c_never_reads_or_installs_old_group(monkeypatch)
    inherited.test_checkpoints_count_adam_batches_even_when_weights_did_not_move(tmp_path)
    assert run.ACCUMULATION == 12 and run.UPDATES == 2000
    assert run.REVIEW_STEPS == (0, 250, 500, 1000, 1500, 2000)
    files = list(tmp_path.glob('*.pt'))
    assert len(files) == 3 and sum(p.stat().st_size for p in files) < 16 * 1024**2
    for path in files:
        saved = torch.load(path, weights_only=True)
        assert saved['format'] == run.VERSION == 'audiovae2_startup_anchor_recovery_v1'
        assert len(saved['optimizer']['state']) == (90 if saved['step'] else 0)
        assert len(saved['sources_seen']) == saved['step'] * 12
        assert saved['sources_seen'] == saved['identity']['source_ids'][:saved['step'] * 12]
        assert saved['coefficients'] == run.prior.COEFFICIENTS
        assert saved['optimizer_reset_after_start'] is False


def test_monitor_accepts_real_anchor_update_fields_and_explains_training_only_scope(tmp_path, monkeypatch):
    model, teacher = fixtures.AnchorWave(scale=.1), fixtures.teacher()
    optimizer = fixtures.optimizer(model)
    monkeypatch.setattr(fixtures.q.prior, 'perform_update', fixtures.ordinary_update)
    updater = run.quiet.StartupAnchorUpdate(model, teacher, fixtures.anchors(), optimizer)
    assert updater.receipt == updater.initialization_receipt
    assert updater.receipt['anchor_windows'] == 6 and updater.receipt['anchor_samples'] == 5760
    assert updater.receipt['current_batch_quiet_constraints'] is False
    assert updater.receipt['state_and_rng_preserved'] and not optimizer.state
    values, _ = updater.perform_update(model, teacher, fixtures.crops(), None, optimizer)
    monitor = monitor_api.RecoveryMonitor(tmp_path / 'tb', writer_factory=inherited.Writer)
    report = inherited.quality()
    monitor.log_validation(report, 0, inherited.quality(2.), inherited.quality(.7))
    record = {'step': 1, 'total': .1, 'waveform': .2, 'mel': .3, 'feature': .4,
              'unique_sources': 12, **values}
    monitor.log_training(record, 1)
    scalars = {tag: (value, step) for tag, value, step in monitor.details.scalars}
    for name, value in values.items():
        if name.startswith('startup_anchor_') and value is not None:
            assert scalars['startup_anchor/' + name[len('startup_anchor_'):]] == (value, 1)
        elif name.startswith('q_') and value is not None:
            assert scalars['quiet_update/' + name[2:]] == (value, 1)
    assert scalars['startup_anchor/after_passed'] == (6., 1)
    assert scalars['startup_anchor/current_batch_constraints'] == (0., 1)
    assert scalars['quiet_update/constraints'] == (2., 1)
    guide = next(text for tag, text, _ in monitor.details.text if tag == 'Guide/Reading this recovery')
    assert 'Six training-only calibration starts' in guide
    assert 'repeated use is counted separately' in guide
    assert 'No development input influences updates' in guide
    assert 'Current-batch quiet constraints are not used' in guide
    assert 'does not guarantee unseen startup retention' in guide
    progress = next(writer for name, writer in monitor.writers.items() if ' 13 ' in name).scalars
    assert progress[-1][1:] == (.05, 1)
    with pytest.raises(ValueError):
        monitor.log_training(record, 1)
    monitor.close()
