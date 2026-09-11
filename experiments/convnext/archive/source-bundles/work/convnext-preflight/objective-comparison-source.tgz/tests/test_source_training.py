"""Small CPU checks for one-pass continuation, not trained audio quality."""
from copy import deepcopy
from dataclasses import asdict, replace

import pytest
import torch

from audiovae_student.corpus_training import CorpusTrainingConfig
from audiovae_student.losses import WarmupLossConfig
from audiovae_student.model import StudentConfig, StudentDecoder
from audiovae_student.optimizers import build_optimizer_bundle
from audiovae_student.source_training import SourceTrainingConfig, initialize_bootstrap, plain, train_sources
from audiovae_student.teacher import CHECKPOINT_SHA256
from audiovae_student.training import _restore_rng, _rng_state
from test_source_corpus import make_rows, open_corpus


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model():
    return StudentDecoder(StudentConfig(hidden_channels=4, expansion_channels=8, head_channels=8))


def settings():
    return SourceTrainingConfig(total_steps=3, batch_size=2, scored_frames=2,
                                optimizer="adamw", checkpoint_interval=2)


def losses():
    return WarmupLossConfig(teacher_fft_sizes=(32, 64), reference_fft_sizes_16k=(16,))


def corpus_at(directory):
    rows, counts = make_rows(directory / "audio", lengths=(3840, 3840, 1920), dev=True)
    return open_corpus(directory, rows, counts)


def assert_nested_same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_same(a, b)
    else:
        assert left == right


def test_true_batch_one_pass_changes_only_student_and_validates_once(tmp_path, monkeypatch):
    corpus = corpus_at(tmp_path)
    student = model()
    original = deepcopy(student.state_dict())
    import audiovae_student.source_training as training
    calls = []
    actual = training.evaluate
    def count(*args):
        calls.append(True)
        return actual(*args)
    monkeypatch.setattr(training, "evaluate", count)
    result = train_sources(student, corpus, tmp_path / "run", config=settings(), loss_config=losses())
    assert result["state"] == "completed" and result["step"] == 3
    assert result["remaining_unique_segments"] == 0
    assert result["unique_scored_hours_seen"] == 7680 / 16000 / 3600
    assert calls == [True]
    assert any(not torch.equal(value, student.state_dict()[key]) for key, value in original.items())
    assert all(not p.requires_grad and p.grad is None for p in corpus._teacher.parameters())
    # Resuming an already completed run does not repeat final validation.
    train_sources(model(), corpus, tmp_path / "run", config=settings(), loss_config=losses(),
                  resume_from=tmp_path / "run/latest.pt")
    assert calls == [True]
    corpus.close()


def test_interruption_resumes_identical_model_optimizer_sampler_and_rng(tmp_path):
    corpus = corpus_at(tmp_path)
    initial = model()
    initial_rng = _rng_state()
    train_sources(deepcopy(initial), corpus, tmp_path / "whole", config=settings(), loss_config=losses())
    _restore_rng(initial_rng)
    interrupted = train_sources(deepcopy(initial), corpus, tmp_path / "split", config=settings(),
                                loss_config=losses(), stop_after_updates=1)
    assert interrupted["state"] == "paused" and interrupted["validation"] == {}
    train_sources(model(), corpus, tmp_path / "split", config=settings(), loss_config=losses(),
                  resume_from=tmp_path / "split/latest.pt")
    a = torch.load(tmp_path / "whole/latest.pt", weights_only=True)
    b = torch.load(tmp_path / "split/latest.pt", weights_only=True)
    for name in ("model", "optimizer", "sampler", "rng", "validation"):
        assert_nested_same(a[name], b[name])
    corpus.close()


def test_resume_rejects_step_sampler_disagreement(tmp_path):
    corpus = corpus_at(tmp_path)
    train_sources(model(), corpus, tmp_path / "run", config=settings(), loss_config=losses(), stop_after_updates=1)
    path = tmp_path / "run/latest.pt"
    state = torch.load(path, weights_only=True)
    state["step"] = 2
    torch.save(state, path)
    with pytest.raises(ValueError, match="segment consumption disagree"):
        train_sources(model(), corpus, tmp_path / "run", config=settings(), loss_config=losses(), resume_from=path)
    corpus.close()


def test_exhaustion_rejects_before_teacher_work_or_updates(tmp_path):
    corpus = corpus_at(tmp_path)
    with pytest.raises(ValueError, match="Insufficient unique"):
        train_sources(model(), corpus, tmp_path / "run", config=replace(settings(), total_steps=4),
                      loss_config=losses())
    assert corpus._teacher.model.encoded_lengths == []
    assert not (tmp_path / "run/latest.pt").exists()
    corpus.close()


def test_bootstrap_import_preserves_step_optimizer_and_teacher_identity(tmp_path):
    student = model()
    config = settings()
    opt = build_optimizer_bundle(student, optimizer="adamw", lr=config.learning_rate,
                                 weight_decay=config.weight_decay, betas=config.betas)
    old = CorpusTrainingConfig(total_steps=1, optimizer="adamw")
    path = tmp_path / "bootstrap.pt"
    state = {"identity": {"kind": "cached_corpus_reconstruction_warmup",
                          "teacher": {"checkpoint_sha256": CHECKPOINT_SHA256},
                          "model_config": plain(student.config.to_dict()),
                          "loss_config": plain(asdict(losses())),
                          "training_config": plain(asdict(old)), "data_fingerprint": "prior-corpus"},
             "model": student.state_dict(), "optimizer": opt.state_dict(), "rng": _rng_state(), "step": 1}
    torch.save(state, path)
    fresh = model()
    fresh_opt = build_optimizer_bundle(fresh, optimizer="adamw", lr=config.learning_rate,
                                       weight_decay=config.weight_decay, betas=config.betas)
    step, lineage = initialize_bootstrap(path, fresh, fresh_opt, config, losses())
    assert step == 1 and lineage["previous_effective_batch"] == 8
    assert_nested_same(student.state_dict(), fresh.state_dict())
    assert_nested_same(opt.state_dict(), fresh_opt.state_dict())
    state["identity"]["teacher"]["checkpoint_sha256"] = "0" * 64
    torch.save(state, path)
    with pytest.raises(ValueError, match="pinned original teacher"):
        initialize_bootstrap(path, fresh, fresh_opt, config, losses())
