"""Parameter routing and native-optimizer state qualification."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from audiovae_student.model import StudentConfig, StudentDecoder
from audiovae_student.optimizers import (
    build_optimizer_bundle,
    native_muon_available,
    partition_student_parameters,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_student():
    return StudentDecoder(StudentConfig(hidden_channels=4, expansion_channels=8, head_channels=8))


def test_hybrid_selects_only_twenty_hidden_linear_weights():
    model = small_student()
    muon, adamw = partition_student_parameters(model)
    expected = {f"blocks.{i}.{projection}.weight" for i in range(10)
                for projection in ("expand", "project")}
    assert muon.optimizer == "muon" and adamw.optimizer == "adamw"
    assert set(muon.names) == expected
    assert len(muon.parameters) == 20
    assert all(parameter.ndim == 2 for parameter in muon.parameters)
    assert set(muon.names).isdisjoint(adamw.names)
    assert set(muon.names) | set(adamw.names) == {name for name, parameter in model.named_parameters()
                                               if parameter.requires_grad}
    routed = [id(parameter) for group in (muon, adamw) for parameter in group.parameters]
    assert len(set(routed)) == len(routed) == len(list(model.parameters()))
    assert all(any(token in name for token in (".expand.weight", ".project.weight"))
               for name in muon.names)
    assert "adapter.weight" in adamw.names
    assert "blocks.0.depthwise.conv.weight" in adamw.names
    assert "blocks.0.norm.weight" in adamw.names
    assert "blocks.0.scale" in adamw.names
    assert "affine.scale" in adamw.names
    assert "head.conv.weight" in adamw.names
    assert "output.weight" in adamw.names
    assert "activation.weight" in adamw.names


def test_frozen_teacher_never_enters_student_optimizer():
    model = small_student()
    model.teacher = nn.Linear(64, 64).requires_grad_(False)
    groups = partition_student_parameters(model)
    assert all(not name.startswith("teacher.") for group in groups for name in group.names)
    assert not {id(p) for p in model.teacher.parameters()} & {
        id(p) for group in groups for p in group.parameters}
    model.teacher.requires_grad_(True)
    with pytest.raises(ValueError, match="Pass only the student"):
        partition_student_parameters(model)


def test_hybrid_rejects_aliased_hidden_weights():
    model = small_student()
    model.blocks[1].expand.weight = model.blocks[0].expand.weight
    with pytest.raises(ValueError, match="Aliased"):
        partition_student_parameters(model)


def test_hybrid_does_not_select_conv1d_by_dimension():
    model = small_student()
    model.blocks[0].expand = nn.Conv1d(4, 8, 1)
    with pytest.raises(ValueError, match="two-dimensional Linear"):
        partition_student_parameters(model)


def test_hybrid_requires_all_ten_blocks_and_trainable_hidden_weights():
    model = StudentDecoder(StudentConfig(hidden_channels=4, expansion_channels=8,
                                         head_channels=8, dilations=(1,)))
    with pytest.raises(ValueError, match="all ten blocks"):
        partition_student_parameters(model)
    model = small_student()
    model.blocks[0].expand.weight.requires_grad_(False)
    with pytest.raises(ValueError, match="twenty"):
        partition_student_parameters(model)


def test_missing_native_muon_fails_without_fallback(monkeypatch):
    monkeypatch.delattr(torch.optim, "Muon", raising=False)
    assert not native_muon_available()
    with pytest.raises(RuntimeError, match="No fallback was used"):
        build_optimizer_bundle(small_student(), optimizer="muon_adamw")
    assert set(build_optimizer_bundle(small_student()).optimizers) == {"adamw"}


def test_adamw_bundle_rejects_name_or_member_mismatch():
    bundle = build_optimizer_bundle(small_student())
    state = bundle.state_dict()
    changed = deepcopy(state)
    changed["group_manifest"][0]["parameters"][0]["name"] = "wrong.weight"
    with pytest.raises(ValueError, match="parameter-name groups"):
        bundle.load_state_dict(changed)
    changed = deepcopy(state)
    changed["optimizers"]["other"] = changed["optimizers"].pop("adamw")
    with pytest.raises(ValueError, match="members"):
        bundle.load_state_dict(changed)
    changed = deepcopy(state)
    changed["optimizers"]["adamw"]["param_groups"][0]["param_names"].reverse()
    with pytest.raises(ValueError, match="ordered parameter names"):
        bundle.load_state_dict(changed)


def test_manifest_fingerprint_is_stable_and_distinguishes_routes():
    first = build_optimizer_bundle(small_student())
    second = build_optimizer_bundle(small_student())
    assert first.group_fingerprint == second.group_fingerprint
    first_copy = first.group_manifest
    first_copy[0]["parameters"][0]["name"] = "mutated"
    assert first.group_manifest == second.group_manifest
    model = small_student()
    model.activation.weight.requires_grad_(False)
    assert build_optimizer_bundle(model).group_fingerprint != first.group_fingerprint


@pytest.mark.skipif(not native_muon_available(), reason="Native torch.optim.Muon is unavailable")
def test_native_hybrid_updates_both_groups_and_resumes_identically(tmp_path):
    torch.manual_seed(17)
    model = small_student()
    bundle = build_optimizer_bundle(model, optimizer="muon_adamw", lr=1e-3)
    assert type(bundle.optimizers["muon"]) is torch.optim.Muon
    assert type(bundle.optimizers["adamw"]) is torch.optim.AdamW
    assert bundle.optimizers["muon"].param_groups[0]["adjust_lr_fn"] == "match_rms_adamw"
    before = deepcopy(model.state_dict())
    for parameter in model.parameters():
        parameter.grad = torch.linspace(-0.5, 0.5, parameter.numel()).reshape_as(parameter)
    bundle.step()
    assert not torch.equal(model.blocks[0].expand.weight, before["blocks.0.expand.weight"])
    assert not torch.equal(model.adapter.weight, before["adapter.weight"])
    assert bundle.optimizers["muon"].state
    assert bundle.optimizers["adamw"].state
    checkpoint = tmp_path / "optimizers.pt"
    torch.save({"model": model.state_dict(), "optimizer": bundle.state_dict()}, checkpoint)
    loaded = torch.load(checkpoint, weights_only=True)
    resumed = small_student()
    resumed.load_state_dict(loaded["model"])
    resumed_bundle = build_optimizer_bundle(resumed, optimizer="muon_adamw", lr=1e-3)
    resumed_bundle.load_state_dict(loaded["optimizer"])
    for current_model, current_bundle in ((model, bundle), (resumed, resumed_bundle)):
        current_bundle.zero_grad()
        for parameter in current_model.parameters():
            parameter.grad = torch.full_like(parameter, 0.25)
        current_bundle.step()
    assert all(torch.equal(value, resumed.state_dict()[name]) for name, value in model.state_dict().items())
    assert set(bundle.state_dict()["optimizers"]) == {"muon", "adamw"}
