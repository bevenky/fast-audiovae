from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.training import _rng_state
from native_chain import (adamw_analytic_decomposition, additive_localization, direction_report, intercept_step,
    native_chain_once, select_probes)
from test_diagnose_objectives_updates import prepared, deterministic


def test_direction_sign_distinguishes_raw_and_native_transformation():
    metric = {"peak": (torch.tensor([2.]),)}
    raw = direction_report(metric, (torch.tensor([-3.]),))
    native = direction_report(metric, (torch.tensor([.5]),))
    assert raw["metrics"]["peak"]["first_order_change"] == -6
    assert raw["metrics"]["peak"]["effect"] == "decrease"
    assert native["metrics"]["peak"]["first_order_change"] == 1
    assert native["metrics"]["peak"]["effect"] == "increase"


def test_adamw_algebra_reconstructs_native_step_and_exposes_momentum_reversal():
    p = torch.nn.Parameter(torch.tensor([.4, -.2]))
    optimizer = torch.optim.AdamW([{"params": [p], "param_names": ["p"]}],
        lr=.01, betas=(.9, .99), eps=1e-8, weight_decay=.1, foreach=False)
    p.grad = torch.tensor([-2., -1.]); optimizer.step()
    before = {"p": p.detach().clone()}
    gradient = torch.tensor([.01, .02]); p.grad = gradient.clone(); optimizer.step()
    delta = (p.detach() - before["p"],)
    metric = {"peak": (torch.tensor([1., 1.]),)}
    optimizer_before = state_fingerprint(optimizer.state_dict())
    report = adamw_analytic_decomposition((("p", p),), (gradient,), delta, before, optimizer, metric)
    assert report["reconstruction"]["passed"]
    assert report["directions"]["raw_gradient"]["metrics"]["peak"]["effect"] == "decrease"
    assert report["directions"]["first_moment_without_preconditioner"]["metrics"]["peak"]["effect"] == "increase"
    assert report["directions"]["native_adaptive_term"]["metrics"]["peak"]["effect"] == "increase"
    assert state_fingerprint(optimizer.state_dict()) == optimizer_before
    assert report["additional_optimizer_steps"] == report["additional_model_forwards"] == 0
    with pytest.raises(RuntimeError, match="reconstruct"):
        adamw_analytic_decomposition((("p", p),), (gradient,), (delta[0] + .1,), before, optimizer, metric)


def test_fixed_actual_divisor_can_reverse_direction_without_momentum():
    p = torch.nn.Parameter(torch.tensor([.4, -.2]))
    optimizer = torch.optim.AdamW([{"params": [p], "param_names": ["p"]}],
        lr=.01, betas=(0., .9), eps=1e-8, weight_decay=0.)
    gradient = torch.ones(2)
    optimizer.state[p] = {"step": torch.tensor(1.), "exp_avg": gradient.clone(),
                          "exp_avg_sq": torch.tensor([1000., .1])}
    before = {"p": p.detach().clone()}
    divisor = (optimizer.state[p]["exp_avg_sq"].double() / .1).sqrt() + 1e-8
    with torch.no_grad():
        p.copy_((before["p"].double() - .01 * gradient.double() / divisor).float())
    report = adamw_analytic_decomposition((("p", p),), (gradient,), (p.detach() - before["p"],),
        before, optimizer, {"peak": (torch.tensor([2., -1.]),)})
    assert report["directions"]["raw_gradient"]["metrics"]["peak"]["effect"] == "decrease"
    assert report["directions"]["first_moment_without_preconditioner"]["metrics"]["peak"]["effect"] == "decrease"
    assert report["directions"]["current_gradient_with_actual_divisor"]["metrics"]["peak"]["effect"] == "increase"
    assert report["reconstruction"]["passed"]


def test_localization_is_exclusive_additive_and_separates_route_from_layer():
    named = (("blocks.9.expand.weight", torch.nn.Parameter(torch.zeros(1))),
             ("blocks.9.norm.weight", torch.nn.Parameter(torch.zeros(1))),
             ("output.weight", torch.nn.Parameter(torch.zeros(1))))
    metric = {"peak": (torch.tensor([2.]), torch.tensor([-3.]), torch.tensor([1.]))}
    raw = (torch.tensor([-1.]), torch.tensor([1.]), torch.tensor([-2.]))
    native = (torch.tensor([2.]), torch.tensor([.5]), torch.tensor([-.5]))
    manifest = [{"optimizer": route, "parameters": [{"name": name, "shape": [1], "dtype": "torch.float32"}]}
                for (name, _), route in zip(named, ("muon", "adamw", "adamw"))]
    reports = {"raw_negative_postclip_gradient": direction_report(metric, raw),
               "native_parameter_delta": direction_report(metric, native)}
    result = additive_localization(named, metric, raw, native, manifest, reports)
    groups = result["directions"]["native_parameter_delta"]["optimizer_routes"]["groups"]
    assert groups["muon"]["first_order_metric_change"]["peak"] == 4
    assert groups["adamw"]["first_order_metric_change"]["peak"] == -2
    families = result["directions"]["native_parameter_delta"]["parameter_families"]["groups"]
    assert families["norm"]["parameter_names"] == ["blocks.9.norm.weight"]
    assert families["blocks.9"]["parameter_names"] == ["blocks.9.expand.weight"]
    assert all(check["passed"] for direction in result["directions"].values()
               for partition in direction.values() for check in partition["additivity_checks"].values())
    with pytest.raises(ValueError, match="partition"):
        additive_localization(named, metric, raw, native, manifest + manifest[:1], reports)


def test_interceptor_preserves_native_operation_and_restores_on_failure():
    p = torch.nn.Parameter(torch.tensor([1.]))
    p.grad = torch.tensor([2.])
    class Bundle:
        def step(self):
            with torch.no_grad():
                p.add_(p.grad, alpha=-.1)
    bundle = Bundle()
    captured = {}
    with pytest.raises(RuntimeError, match="Only one"):
        with intercept_step(bundle, (("p", p),), captured):
            bundle.step()
            bundle.step()
    assert "step" not in bundle.__dict__
    assert float(p.detach()) == pytest.approx(.8)
    assert captured["calls"] == 1 and float(captured["gradient"][0]) == 2
    bundle.step()
    assert float(p.detach()) == pytest.approx(.6)


def test_one_native_step_restores_all_state_and_preserves_quiet_metrics(prepared):
    engine, crops = prepared
    probes = [replace(crops[0], source_id="encoded_zero", teacher_audio=torch.zeros_like(crops[0].teacher_audio)), crops[0]]
    before, rng = state_fingerprint(engine.state_dict()), state_fingerprint(_rng_state())
    old_grad = next(engine.model.parameters()).grad
    old_value = old_grad.clone()
    report = native_chain_once(engine, crops, probes)
    assert report["generator_updates"] == report["discriminator_updates"] == 1
    assert report["training_metrics"]["step"] == 5
    denoms = report["before"]["denominators"]
    assert denoms["teacher_quiet_residual_mse"] == denoms["encoded_zero_quiet_mse"] + denoms["natural_quiet_residual_mse"]
    assert denoms["natural_quiet_residual_mse"] == report["before"]["rows"][1]["quiet_samples"]
    assert report["before"]["metrics"]["natural_quiet_residual_mse"] == pytest.approx(report["before"]["rows"][1]["teacher_quiet_residual_mse"], rel=1e-6)
    assert report["before"]["denominators"] == report["after"]["denominators"]
    assert report["raw_negative_postclip_gradient"]["direction_norm"] == pytest.approx(1., abs=1e-5)
    assert report["native_parameter_delta"]["direction_norm"] > 0
    assert report["state_restored"] and report["retained_updates"] == 0
    assert report["adamw_analytic_decomposition"]["reconstruction"]["passed"]
    localization = report["additive_first_order_localization"]
    assert localization["no_additional_model_forwards_or_optimizer_steps"]
    assert all(check["passed"] for direction in localization["directions"].values()
               for partition in direction.values() for check in partition["additivity_checks"].values())
    assert state_fingerprint(engine.state_dict()) == before
    assert state_fingerprint(_rng_state()) == rng
    assert next(engine.model.parameters()).grad is old_grad
    assert torch.equal(old_grad, old_value)
    assert "step" not in engine.optimizer.__dict__


def test_native_step_failure_still_restores_optimizer_and_engine(prepared):
    engine, crops = prepared
    engine.recipe = replace(engine.recipe, total_steps=engine.step)
    engine.config = replace(engine.config, total_steps=engine.step)
    probes = [replace(crops[0], source_id="encoded_zero"), crops[0]]
    before, rng = state_fingerprint(engine.state_dict()), state_fingerprint(_rng_state())
    original = engine.train_step
    def failure(batch):
        original(batch)
        raise RuntimeError("injected post-update failure")
    engine.train_step = failure
    with pytest.raises(RuntimeError, match="injected"):
        native_chain_once(engine, crops, probes)
    del engine.train_step
    assert state_fingerprint(engine.state_dict()) == before
    assert state_fingerprint(_rng_state()) == rng
    assert "step" not in engine.optimizer.__dict__


def test_budget_extension_keeps_rate_and_restores_saved_budget(prepared):
    engine, crops = prepared
    engine.recipe = replace(engine.recipe, total_steps=engine.step)
    engine.config = replace(engine.config, total_steps=engine.step)
    before = state_fingerprint(engine.state_dict())
    rate = engine.learning_rate()
    report = native_chain_once(engine, crops, [replace(crops[0], source_id="encoded_zero"), crops[0]])
    assert report["disposable_budget_extension"]["total_steps_during_disposable_probe"] == 5
    assert report["training_metrics"]["learning_rate"] == rate
    assert engine.recipe.total_steps == engine.config.total_steps == engine.step == 4
    assert state_fingerprint(engine.state_dict()) == before


def test_probe_selection_preserves_order_and_rejects_duplicates():
    crops = [SimpleNamespace(source_id="encoded_zero" if i == 0 else str(i), start_frame=i) for i in range(10)]
    selection = [{"source_id": c.source_id, "start_frame": c.start_frame} for c in reversed(crops)]
    assert select_probes(crops, selection) == list(reversed(crops))
    with pytest.raises(ValueError, match="unique"):
        select_probes(crops, selection[:-1] + [selection[0]])
