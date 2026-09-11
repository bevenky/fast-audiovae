"""State-preserving continuation and fail-closed exposure accounting, CPU only."""
from copy import deepcopy
from dataclasses import replace
import fcntl
import json

import pytest
import torch

from audiovae_student.cache import sample_training_crop
from audiovae_student.discriminators import AudioDiscriminators, DiscriminatorConfig
from audiovae_student.model import StudentConfig, StudentDecoder
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.recipe_v2 import RecipeV2Config, RecipeV2Engine
from audiovae_student.recipe_v2_continuation import (
    extend_total_budget, load_committed_parent, observe_training_signal, run_recipe_v2_continuation)
from audiovae_student.restart_data import file_sha
from audiovae_student.training import _restore_rng, _rng_state
from test_recipe_v2_pilot import fixture, run


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def tiny_discriminators():
    return AudioDiscriminators(DiscriminatorConfig(periods=(2,),
        mpd_channels=(2, 2, 4, 4, 4), fft_sizes=(64,), mrd_channels=2))


def parent_fixture(path, *, pause=3):
    data = fixture(path)
    run(path / "parent", data, max_updates=pause)
    checkpoint = path / "parent/latest.pt"
    payload, parent = load_committed_parent(checkpoint, file_sha(checkpoint), path / "parent",
                                          data["plan"], device="cpu")
    return data, payload, parent


def engine_from(state):
    engine = RecipeV2Engine(StudentDecoder(StudentConfig(**state["model_config"])),
        recipe=RecipeV2Config(**state["recipe"]), discriminators=tiny_discriminators())
    engine.load_state_dict(deepcopy(state))
    return engine


def test_budget_migration_preserves_exact_next_gan_update_and_rng(tmp_path):
    data, payload, _ = parent_fixture(tmp_path)
    original = payload["engine"]
    before = state_fingerprint(original)
    migrated = extend_total_budget(original, 6)
    assert state_fingerprint(original) == before
    normalized = deepcopy(migrated)
    normalized["recipe"]["total_steps"] = original["recipe"]["total_steps"]
    normalized["config"]["total_steps"] = original["config"]["total_steps"]
    assert state_fingerprint(normalized) == before
    left, right = engine_from(original), engine_from(migrated)
    window = data["plan"]["windows"][payload["sampler"]["cursor"]]
    crop = sample_training_crop(data["corpus"].get(window.source_id), window.start_frame,
                                window.scored_frames, 29)
    _restore_rng(payload["rng"])
    lm = left.train_step([crop])
    left_rng = _rng_state()
    _restore_rng(payload["rng"])
    rm = right.train_step([crop])
    assert state_fingerprint(left_rng) == state_fingerprint(_rng_state())
    assert lm == rm
    ls, rs = left.state_dict(), right.state_dict()
    for container in ("recipe", "config"):
        rs[container]["total_steps"] = ls[container]["total_steps"]
    assert state_fingerprint(ls) == state_fingerprint(rs)
    assert right.discriminator_updates == 2
    assert right.calibration == original["calibration"]


@pytest.mark.parametrize("damage", ["cosine", "shrink", "before_warmup", "missing_calibration", "config_budget"])
def test_budget_extension_rejects_schedule_or_stage_changes(tmp_path, damage):
    _, payload, _ = parent_fixture(tmp_path)
    state, total = deepcopy(payload["engine"]), 6
    if damage == "cosine":
        state["config"]["learning_rate_schedule"] = "cosine"
    elif damage == "shrink":
        total = state["recipe"]["total_steps"] - 1
    elif damage == "before_warmup":
        state["step"] = 0
    elif damage == "missing_calibration":
        state["calibration"] = None
    else:
        state["config"]["total_steps"] += 1
    with pytest.raises(ValueError):
        extend_total_budget(state, total)


@pytest.mark.parametrize("damage", ["inflight", "journal", "metrics", "manifest", "calibration", "plan"])
def test_parent_validation_rejects_drift_or_uncertain_exposure(tmp_path, damage):
    data, _, _ = parent_fixture(tmp_path)
    checkpoint = tmp_path / "parent/latest.pt"
    expected = file_sha(checkpoint)
    if damage == "inflight":
        (tmp_path / "parent/inflight.json").write_text('{}')
    elif damage in ("journal", "metrics"):
        name = "exposure.jsonl" if damage == "journal" else "metrics.jsonl"
        with (tmp_path / "parent" / name).open("a") as handle:
            handle.write('{"step":4}\n')
    elif damage == "manifest":
        (tmp_path / "parent/run.json").write_text('{}')
    elif damage == "calibration":
        (tmp_path / "parent/calibration.json").write_text('{"calibration":{}}')
    else:
        data["plan"]["metadata"]["tampered"] = True
    with pytest.raises(ValueError):
        load_committed_parent(checkpoint, expected, tmp_path / "parent", data["plan"], device="cpu")


def continuation_fixture(path):
    from audiovae_student.continuation_data import (plan_fixed_window_continuation,
        write_continuation_plan, load_continuation_plan)

    data, payload, parent = parent_fixture(path, pause=2)
    from audiovae_student import continuation_data as cd
    catalog = dict(rows=(), counts={}, windows=(), reserved=(), conditions={},
        identity={"publication_sha256": "a" * 64,
                  "files_sha256": {name: "b" * 64 for name in cd._SUPPLEMENT_FILES}})
    body = cd._catalog_body(catalog)
    catalog["identity"] = {**body, "identity_sha256": cd.digest(body)}
    # The no-supplement control must preserve the exact remaining parent batches.
    plan = plan_fixed_window_continuation(data["plan"], payload["sampler"], parent=parent,
        calibration_windows=data["calibration_windows"], calibration_rows=data["calibration_rows"],
        calibration_counts=data["calibration_counts"], supplement=catalog,
        target_expressive_fraction=0, reserved_rows=data["heldout_rows"])
    write_continuation_plan(plan, path / "continuation-plan")
    plan = load_continuation_plan(path / "continuation-plan")
    args = dict(corpus=data["corpus"], plan=plan, heldout_crops=data["heldout_crops"],
        heldout_rows=data["heldout_rows"], data_identity=data["data_identity"],
        parent_checkpoint=path / "parent/latest.pt", parent_checkpoint_sha256=parent["checkpoint_sha256"],
        parent_run_dir=path / "parent", parent_plan=data["plan"], device="cpu",
        checkpoint_interval=2, min_free_bytes=0, supplement=catalog,
        expected_plan_identity_sha256=plan["identity"]["identity_sha256"])
    return data, payload, args


def continue_run(path, args, *, resume=False, max_updates=None):
    return run_recipe_v2_continuation(output_dir=path, **args, discriminators=tiny_discriminators(),
        resume_from=path / "latest.pt" if resume else None, max_updates=max_updates)


def test_uninterrupted_and_split_continuation_preserve_engine_rng_and_cursor(tmp_path):
    data, parent, args = continuation_fixture(tmp_path)
    before_files = {p.name: file_sha(p) for p in (tmp_path / "parent").iterdir() if p.is_file()}
    complete = continue_run(tmp_path / "whole", args)
    partial = continue_run(tmp_path / "split", args, max_updates=1)
    assert partial["step"] == 3 and partial["consumed_windows"] == 1
    resumed = continue_run(tmp_path / "split", args, resume=True)
    assert complete["step"] == resumed["step"] == 4
    left = torch.load(tmp_path / "whole/latest.pt", weights_only=True)
    right = torch.load(tmp_path / "split/latest.pt", weights_only=True)
    for key in ("engine", "rng", "sampler", "latest_evaluation", "teacher_coverage", "journal_sha256"):
        assert state_fingerprint(left[key]) == state_fingerprint(right[key]), key
    assert left["identity"]["global_start_step"] == 2
    assert left["identity"]["segment_window_count"] == 2
    assert left["engine"]["calibration"] == parent["engine"]["calibration"]
    records = [json.loads(s) for s in (tmp_path / "whole/exposure.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records] == [3, 4]
    assert [r["cursor"] for r in records] == [1, 2]
    assert sorted(p.name for p in (tmp_path / "whole").glob("evaluation-step*.json")) == ['evaluation-step000004.json']
    assert before_files == {p.name: file_sha(p) for p in (tmp_path / "parent").iterdir() if p.is_file()}
    # Every fresh child retained the existing normalization, with no rereads of
    # calibration source windows beyond the four accesses made by its parent.
    assert data["corpus"].access.count("cal-a") == 4


@pytest.mark.parametrize("damage", ["inflight", "journal", "metrics", "coverage", "coverage_bin", "parent_coverage", "heldout", "plan", "evaluation"])
def test_continuation_resume_rejects_drift_and_uncheckpointed_work(tmp_path, damage):
    _, _, args = continuation_fixture(tmp_path)
    path = tmp_path / "child"
    continue_run(path, args, max_updates=None if damage == "evaluation" else 1)
    if damage == "inflight":
        (path / "inflight.json").write_text('{}')
    elif damage in ("journal", "metrics"):
        with (path / ("exposure.jsonl" if damage == "journal" else "metrics.jsonl")).open("a") as handle:
            handle.write('{"step":4}\n')
    elif damage in ("coverage", "coverage_bin", "parent_coverage"):
        saved = torch.load(path / "latest.pt", weights_only=True)
        if damage == "parent_coverage":
            saved["parent_teacher_coverage"]["valid_samples"] += 1
        else:
            saved["teacher_coverage"]["valid_samples" if damage == "coverage" else "exact_zero_samples"] += 1
        torch.save(saved, path / "latest.pt")
    elif damage == "heldout":
        args["heldout_crops"][0].teacher_audio[..., 0] += .1
    elif damage == "plan":
        args["plan"]["metadata"]["tampered"] = True
    else:
        (path / "evaluation-step000004.json").write_text('{}')
    with pytest.raises(ValueError):
        continue_run(path, args, resume=True)


def test_live_pause_saves_committed_state_before_another_update(tmp_path, monkeypatch):
    _, _, args = continuation_fixture(tmp_path)
    original = RecipeV2Engine.train_step
    path = tmp_path / "paused"
    def request_after_update(self, crops):
        result = original(self, crops)
        (path / "pause.request.json").write_text('{"action":"pause"}')
        return result
    monkeypatch.setattr(RecipeV2Engine, "train_step", request_after_update)
    result = continue_run(path, args)
    assert result["state"] == "paused_by_request" and result["step"] == 3
    saved = torch.load(path / "latest.pt", weights_only=True)
    assert saved["engine"]["step"] == 3 and saved["sampler"]["cursor"] == 1
    assert not (path / "inflight.json").exists()
    monkeypatch.setattr(RecipeV2Engine, "train_step", original)
    (path / "pause.request.json").unlink()
    resumed = continue_run(path, args, resume=True)
    assert resumed["step"] == 4


def test_failure_after_optimizer_keeps_inflight_and_rejects_replay(tmp_path, monkeypatch):
    _, _, args = continuation_fixture(tmp_path)
    original = RecipeV2Engine.train_step
    def fail(self, crops):
        original(self, crops)
        raise RuntimeError("injected after update")
    monkeypatch.setattr(RecipeV2Engine, "train_step", fail)
    path = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="after update"):
        continue_run(path, args)
    assert json.loads((path / "inflight.json").read_text())["step"] == 3
    assert torch.load(path / "latest.pt", weights_only=True)["engine"]["step"] == 2
    with pytest.raises(ValueError, match="in-flight"):
        continue_run(path, args, resume=True)


def test_active_parent_cannot_be_forked_while_training(tmp_path):
    _, _, args = continuation_fixture(tmp_path)
    with (tmp_path / ".parent.runner.lock").open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="parent run is still active"):
            continue_run(tmp_path / "child", args)
    assert not (tmp_path / "child").exists()


def test_invalid_output_cannot_create_even_a_lock_inside_parent(tmp_path):
    _, _, args = continuation_fixture(tmp_path)
    before = {p.name: file_sha(p) for p in (tmp_path / "parent").iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="outside"):
        continue_run(tmp_path / "parent/child", args)
    assert before == {p.name: file_sha(p) for p in (tmp_path / "parent").iterdir() if p.is_file()}


@pytest.mark.parametrize("optimizer", ["adamw", pytest.param("muon_adamw",
    marks=pytest.mark.skipif(not callable(getattr(torch.optim, "Muon", None)),
                            reason="Native torch.optim.Muon unavailable"))])
def test_nonempty_supplement_extends_budget_and_preserves_partial_final_batch(tmp_path, optimizer):
    from audiovae_student import continuation_data as cd
    from audiovae_student.comparison_data import write_comparison_plan, load_comparison_plan
    from audiovae_student.recipe_v2_pilot import run_recipe_v2
    from audiovae_student.restart_data import PilotWindow
    from test_representative_pilot import Corpus
    from test_restart_data import row, counts, ledger_for

    data = fixture(tmp_path)
    train = [row("speech-" + str(i), samples=10*640) for i in range(4)]
    original = dict(rows=train, windows=[PilotWindow(r.source_id, start, 5, 3200, r.dataset, r.language)
        for start in (0, 5) for r in train], counts=counts(train), reserved=data["heldout_rows"],
        excluded=(), ledger=ledger_for(train + data["calibration_rows"]), seed=47,
        metadata={"minimum_output_samples":9120,"minimum_input_samples":3040})
    write_comparison_plan(original, tmp_path / "batched-plan")
    data["plan"] = load_comparison_plan(tmp_path / "batched-plan")
    data["corpus"] = Corpus(train + data["calibration_rows"])
    torch.manual_seed(991)
    run_recipe_v2(output_dir=tmp_path / "parent", **data,
        recipe=RecipeV2Config(total_steps=4, reconstruction_warmup_steps=2,
            learning_rate_warmup_steps=1, perceptual_ramp_steps=2, optimizer=optimizer),
        model_config=StudentConfig(hidden_channels=4, expansion_channels=8, head_channels=8,
            dilations=(1,) if optimizer == "adamw" else (1,) * 10,
            layer_scale_init=.1, normalization_mode="masked_batch_norm",
            adapter_mode="raw_repeat_phase_bias"), device="cpu", batch_size=2,
        checkpoint_interval=2, min_free_bytes=0, discriminators=tiny_discriminators(), max_updates=2)
    checkpoint = tmp_path / "parent/latest.pt"
    payload, parent = load_committed_parent(checkpoint, file_sha(checkpoint), tmp_path / "parent",
                                           data["plan"], device="cpu")
    event = replace(row("new-event", samples=5*640, dataset="emogator"), license="Apache-2.0")
    catalog = dict(rows=(event,), counts=counts([event]), reserved=(),
        windows=(PilotWindow(event.source_id, 0, 5, 3200, event.dataset, event.language,
                             condition="emotional_nonverbal"),),
        conditions={event.source_id:"emotional_nonverbal"},
        identity={"publication_sha256":"a"*64,
            "files_sha256":{name:"b"*64 for name in cd._SUPPLEMENT_FILES}})
    body = cd._catalog_body(catalog)
    catalog["identity"] = {**body,"identity_sha256":cd.digest(body)}
    data["corpus"].records.update(Corpus([event]).records)
    plan = cd.plan_fixed_window_continuation(data["plan"], payload["sampler"], parent=parent,
        calibration_windows=data["calibration_windows"], calibration_rows=data["calibration_rows"],
        calibration_counts=data["calibration_counts"], supplement=catalog, reserved_rows=data["heldout_rows"])
    cd.write_continuation_plan(plan, tmp_path / "child-plan")
    args = dict(corpus=data["corpus"], plan=cd.load_continuation_plan(tmp_path / "child-plan"),
        heldout_crops=data["heldout_crops"], heldout_rows=data["heldout_rows"],
        data_identity=data["data_identity"], parent_checkpoint=checkpoint,
        parent_checkpoint_sha256=parent["checkpoint_sha256"], parent_run_dir=tmp_path / "parent",
        parent_plan=data["plan"], supplement=catalog, device="cpu", checkpoint_interval=2, min_free_bytes=0)
    args["expected_plan_identity_sha256"] = args["plan"]["identity"]["identity_sha256"]
    whole = continue_run(tmp_path / "whole", args)
    continue_run(tmp_path / "split", args, max_updates=2)
    split = continue_run(tmp_path / "split", args, resume=True)
    assert whole["step"] == split["step"] == 5
    assert split["remaining_windows"] == 0 and split["consumed_windows"] == 5
    assert split["teacher_coverage"]["valid_samples"] == 5 * 9600
    assert split["unique_scored_hours"] == 9 * 9600 / 172_800_000
    left = torch.load(tmp_path / "whole/latest.pt", weights_only=True)
    right = torch.load(tmp_path / "split/latest.pt", weights_only=True)
    for key in ("engine", "sampler", "rng", "latest_evaluation", "journal_sha256", "teacher_coverage"):
        assert state_fingerprint(left[key]) == state_fingerprint(right[key]), key
    metrics = [json.loads(s) for s in (tmp_path / "split/metrics.jsonl").read_text().splitlines()]
    assert [m["examples"] for m in metrics] == [2, 2, 1]
    assert [m["step"] for m in metrics] == [3, 4, 5]
    assert left["identity"]["segment_window_count"] == 5
    assert left["engine"]["calibration"] == payload["engine"]["calibration"]


def test_detached_signal_observer_keeps_exact_gan_state_and_rng(tmp_path):
    data, payload, _ = parent_fixture(tmp_path)
    plain, observed = engine_from(payload["engine"]), engine_from(payload["engine"])
    window = data["plan"]["windows"][payload["sampler"]["cursor"]]
    crop = sample_training_crop(data["corpus"].get(window.source_id), window.start_frame,
                                window.scored_frames, 29)
    # An extreme value in unscored history must not enter the teacher report.
    crop.teacher_audio[..., 0] = 3
    target_before = crop.teacher_audio.clone()
    _restore_rng(payload["rng"])
    lm = plain.train_step([crop])
    left_rng = _rng_state()
    _restore_rng(payload["rng"])
    with observe_training_signal(observed.model, [crop]) as signal:
        rm = observed.train_step([crop])
    assert lm == rm
    assert state_fingerprint(plain.state_dict()) == state_fingerprint(observed.state_dict())
    assert state_fingerprint(left_rng) == state_fingerprint(_rng_state())
    assert torch.equal(crop.teacher_audio, target_before)
    assert signal["training_signal/teacher_peak_abs"] == float(target_before[..., crop.scored_slice].abs().max())
    assert signal["training_signal/teacher_full_scale_samples"] == 0
    assert signal["training_signal/scored_samples"] == crop.valid_scored_samples
    assert not observed.model._forward_hooks


def test_signal_observer_hook_removed_on_failure(tmp_path):
    _, payload, _ = parent_fixture(tmp_path)
    engine = engine_from(payload["engine"])
    with pytest.raises(RuntimeError, match="injected"):
        with observe_training_signal(engine.model, ()):
            raise RuntimeError("injected")
    assert not engine.model._forward_hooks


def test_reviewed_plan_identity_mismatch_fails_before_run_creation(tmp_path):
    _, _, args = continuation_fixture(tmp_path)
    args["expected_plan_identity_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="reviewed handoff identity"):
        continue_run(tmp_path / "child", args)
    assert not (tmp_path / "child").exists()
