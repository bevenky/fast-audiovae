from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from audiovae_student.cache import DECODER_HOP, TrainingCrop
from audiovae_student.discriminators import AudioDiscriminators, DiscriminatorConfig
from audiovae_student.distillation_training import DistillationEngine, DistillationTrainingConfig
from audiovae_student.losses_distillation import DistillationLossConfig
from audiovae_student.model import StudentConfig, StudentDecoder
from audiovae_student.preflight_distillation import PreflightConfig, fixed_crops, run_preflight
from audiovae_student.teacher import CHECKPOINT_SHA256

torch.set_num_threads(1)


def engine():
    torch.manual_seed(77)
    model = StudentDecoder(StudentConfig(hidden_channels=4, expansion_channels=8, head_channels=8,
        dilations=(1,), layer_scale_init=.1, normalization_mode="masked_batch_norm"))
    return DistillationEngine(model, config=DistillationTrainingConfig(optimizer="adamw",
        total_steps=4, warmup_steps=1, freeze_normalization_step=1),
        loss_config=DistillationLossConfig(fft_sizes=(256,), mel_bands=(8,)),
        discriminators=AudioDiscriminators(DiscriminatorConfig(periods=(2,),
            mpd_channels=(2, 2, 4, 4, 4), fft_sizes=(64,), mrd_channels=2)))


def crops(source):
    generator = torch.Generator().manual_seed(9 if source == "diagnostic" else 11)
    result = []
    for start in (0, 3):
        frames = start + 3
        target = torch.randn(1, 1, frames * DECODER_HOP, generator=generator) * .03
        result.append(TrainingCrop(torch.randn(1, 64, frames, generator=generator), target, None,
            "a" * 64, source, start, 0, start, 3, 3 * DECODER_HOP))
    return result


def config(**overrides):
    return PreflightConfig(scored_frames=3, evaluation_interval=2, checkpoint_interval=2,
                           save_audio=False, **overrides)


IDENTITY = {"teacher_checkpoint_sha256": CHECKPOINT_SHA256, "fixture": "runner-contract-only"}


def test_runner_stops_at_gate_and_never_updates_sentinels_or_discriminators(tmp_path):
    trainer = engine()
    diagnostic, sentinel = crops("diagnostic"), crops("sentinel")
    for crop in sentinel:
        crop.teacher_audio.requires_grad_(True)
    discriminator_state = deepcopy(trainer.discriminators.state_dict())
    result = run_preflight(trainer, diagnostic, sentinel, tmp_path / "run",
                           data_identity=IDENTITY, config=config())
    assert trainer.step == 4 and trainer.perceptual_start is None
    assert result['state'] == 'gate_failed' and result['main_exposure_hours'] == 0
    assert result['automatically_launched_training'] is False
    assert result['fold_streaming']['passed'] and result['fold_streaming']['rtf_measured'] is False
    assert set(result['gates']['diagnostic']) == {'all', 'beginning', 'interior'}
    assert all(c.teacher_audio.grad is None for c in sentinel)
    for name, value in trainer.discriminators.state_dict().items():
        torch.testing.assert_close(value, discriminator_state[name], rtol=0, atol=0)
    assert (tmp_path / 'run' / 'gradient-initial.json').exists()
    assert (tmp_path / 'run' / 'gradient-final.json').exists()
    assert len(list((tmp_path / 'run').glob('checkpoint-step*.pt'))) == 3


def test_exact_resume_preserves_next_updates_and_completed_run_is_idle(tmp_path):
    diagnostic, sentinel = crops("diagnostic"), crops("sentinel")
    complete = engine()
    run_preflight(complete, diagnostic, sentinel, tmp_path / 'complete', data_identity=IDENTITY, config=config())
    partial = engine()
    paused = run_preflight(partial, diagnostic, sentinel, tmp_path / 'resume',
        data_identity=IDENTITY, config=config(), max_updates_this_call=2)
    assert paused['state'] == 'paused' and paused['step'] == 2
    restored = engine()
    result = run_preflight(restored, diagnostic, sentinel, tmp_path / 'resume', data_identity=IDENTITY,
        config=config(), resume_from=tmp_path / 'resume' / paused['latest_checkpoint']['path'])
    for name, value in complete.model.state_dict().items():
        torch.testing.assert_close(value, restored.model.state_dict()[name], rtol=0, atol=0)
    assert restored.balancer.state_dict() == complete.balancer.state_dict()
    checkpoint = tmp_path / 'resume' / result['latest_checkpoint']['path']
    before = checkpoint.read_bytes()
    replay = run_preflight(engine(), diagnostic, sentinel, tmp_path / 'resume', data_identity=IDENTITY,
        config=config(), resume_from=checkpoint)
    assert replay == result and checkpoint.read_bytes() == before


def test_runner_rejects_overwrite_changed_data_and_train_dev_overlap(tmp_path):
    diagnostic, sentinel = crops("diagnostic"), crops("sentinel")
    trainer = engine()
    paused = run_preflight(trainer, diagnostic, sentinel, tmp_path / 'run', data_identity=IDENTITY,
        config=config(), max_updates_this_call=2)
    with pytest.raises(ValueError, match='exact identified resume'):
        run_preflight(engine(), diagnostic, sentinel, tmp_path / 'run', data_identity=IDENTITY, config=config())
    path = tmp_path / 'run' / paused['latest_checkpoint']['path']
    with pytest.raises(ValueError, match='exact identified resume'):
        run_preflight(engine(), diagnostic, sentinel, tmp_path / 'run',
            data_identity={**IDENTITY, 'changed': True}, config=config(), resume_from=path)
    with pytest.raises(ValueError, match='Sentinel identities'):
        run_preflight(engine(), diagnostic, diagnostic, tmp_path / 'other', data_identity=IDENTITY, config=config())


def test_fixed_crops_keep_original_timeline_and_split_roles():
    latent_frames = 50
    record = SimpleNamespace(metadata={'identity': {'source': {'source_id': 'item', 'split': 'dev'}}},
        latent_frames=latent_frames, valid_output_samples=latent_frames * DECODER_HOP,
        latents=torch.arange(latent_frames).float().reshape(1, 1, -1).expand(1, 64, -1),
        teacher_audio=torch.arange(latent_frames * DECODER_HOP).float().reshape(1, 1, -1),
        cache_key='b' * 64)
    beginning, interior = fixed_crops(record, role='sentinel')
    assert beginning.start_frame == 0 and interior.start_frame == 17
    assert interior.teacher_audio[..., interior.scored_slice][0, 0, 0] == 17 * DECODER_HOP
    assert interior.latents[0, 0, interior.context_frames] == 17
    assert beginning.valid_scored_samples == interior.valid_scored_samples == 16 * DECODER_HOP
    with pytest.raises(ValueError, match='source split'):
        fixed_crops(record, role='diagnostic')


def test_logging_has_real_evaluation_and_no_duplicate_resume_events(tmp_path):
    event_accumulator = pytest.importorskip('tensorboard.backend.event_processing.event_accumulator')
    diagnostic, sentinel = crops('diagnostic'), crops('sentinel')
    paused = run_preflight(engine(), diagnostic, sentinel, tmp_path / 'run', data_identity=IDENTITY,
        config=config(check_fold_streaming=False), max_updates_this_call=2,
        log_dir=tmp_path / 'logs', run_name='real-preflight')
    run_preflight(engine(), diagnostic, sentinel, tmp_path / 'run', data_identity=IDENTITY,
        config=config(check_fold_streaming=False), resume_from=tmp_path / 'run' / paused['latest_checkpoint']['path'],
        log_dir=tmp_path / 'logs', run_name='real-preflight')
    events = event_accumulator.EventAccumulator(str(tmp_path / 'logs' / 'real-preflight')).Reload()
    assert [x.step for x in events.Scalars('train/teacher_waveform')] == [1, 2, 3, 4]
    assert [x.step for x in events.Scalars('evaluation/sentinel/beginning/teacher_mel')] == [0, 2, 4]
    status = json.loads((tmp_path / 'run' / 'status.json').read_text())
    assert status['state'] == 'gate_failed' and status['step'] == 4


def test_audio_export_keeps_full_amplitude_and_exact_scored_sample_counts(tmp_path):
    sf = pytest.importorskip('soundfile')
    from audiovae_student.preflight_distillation import _save_audio
    trainer = engine()
    items = crops('diagnostic')
    trainer.model.freeze_normalization_statistics()
    manifest = _save_audio(trainer, {'diagnostic': items}, tmp_path)
    assert len(manifest) == 2
    for row, crop in zip(manifest, items):
        audio, sample_rate = sf.read(tmp_path / row['teacher'], dtype='float32')
        assert sample_rate == 48000 and len(audio) == crop.valid_scored_samples
        assert sf.info(tmp_path / row['teacher']).subtype == 'FLOAT'
        torch.testing.assert_close(torch.from_numpy(audio), crop.teacher_audio[0, 0, crop.scored_slice], rtol=0, atol=0)
        assert sf.info(tmp_path / row['student']).frames == crop.valid_scored_samples
