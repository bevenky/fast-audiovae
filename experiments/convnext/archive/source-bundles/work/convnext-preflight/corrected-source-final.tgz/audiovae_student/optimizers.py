"""Explicit student parameter routing to native PyTorch optimizers.

Muon updates only the twenty ConvNeXt hidden Linear weight matrices. The
remainder stays on AdamW. This module contains no Muon implementation and
never substitutes another optimizer when native torch.optim.Muon is absent.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import inspect
import json
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class NamedParameterGroup:
    optimizer: str
    names: tuple[str, ...]
    parameters: tuple[nn.Parameter, ...]


def partition_student_parameters(model: nn.Module, *, optimizer: str = "muon_adamw"
                                 ) -> tuple[NamedParameterGroup, ...]:
    """Partition each trainable student parameter exactly once.

    AdamW mode supports generic preflight fixtures. Hybrid mode deliberately
    accepts only the complete StudentDecoder and never discovers matrices by
    dimensionality. Frozen attached parameters are excluded; unexpected
    trainable modules and aliased parameters are rejected.
    """
    if optimizer not in {"adamw", "muon_adamw"}:
        raise ValueError("optimizer must be 'adamw' or 'muon_adamw'")
    named = [(name, parameter) for name, parameter in model.named_parameters(remove_duplicate=False)
             if parameter.requires_grad]
    if not named:
        raise ValueError("The student must have trainable parameters")
    if len({id(parameter) for _, parameter in named}) != len(named):
        raise ValueError("Aliased trainable parameters require an explicit optimizer policy")
    if optimizer == "adamw":
        return (NamedParameterGroup("adamw", tuple(name for name, _ in named),
                                     tuple(parameter for _, parameter in named)),)

    from .model import StudentDecoder

    if not isinstance(model, StudentDecoder) or len(model.blocks) != 10:
        raise ValueError("Muon routing requires a standalone StudentDecoder with all ten blocks")
    allowed_roots = {"adapter", "stem", "stem_norm", "blocks", "affine", "head", "activation", "output"}
    if any(name.split(".", 1)[0] not in allowed_roots for name, _ in named):
        raise ValueError("Pass only the student; unexpected trainable modules must not enter its optimizer")
    expected_names = {f"blocks.{index}.{projection}.weight" for index in range(10)
                      for projection in ("expand", "project")}
    muon_names: list[str] = []
    muon_parameters: list[nn.Parameter] = []
    adam_names: list[str] = []
    adam_parameters: list[nn.Parameter] = []
    for name, parameter in named:
        if name in expected_names:
            module = model.get_submodule(name.rsplit(".", 1)[0])
            if not isinstance(module, nn.Linear) or parameter is not module.weight or parameter.ndim != 2:
                raise ValueError(f"Muon target {name} must be a native two-dimensional Linear weight")
            muon_names.append(name)
            muon_parameters.append(parameter)
        else:
            adam_names.append(name)
            adam_parameters.append(parameter)
    if set(muon_names) != expected_names:
        raise ValueError("All twenty hidden Linear weights must remain trainable for this Muon experiment")
    if not adam_parameters:
        raise ValueError("The student must retain its complementary AdamW parameter group")
    return (NamedParameterGroup("muon", tuple(muon_names), tuple(muon_parameters)),
            NamedParameterGroup("adamw", tuple(adam_names), tuple(adam_parameters)))


def native_muon_available() -> bool:
    return callable(getattr(torch.optim, "Muon", None))


class OptimizerBundle:
    """The small optimizer interface used by reconstruction preflight.

    This is not a drop-in torch.optim.Optimizer subclass for arbitrary LR
    schedulers or GradScaler. Those future integrations must handle each
    optimizer explicitly and checkpoint their own scheduler/scaler state.
    """

    def __init__(self, groups: tuple[NamedParameterGroup, ...],
                 optimizers: dict[str, torch.optim.Optimizer]):
        self.optimizers = optimizers
        self._manifest = [{"optimizer": group.optimizer,
                           "parameters": [{"name": name, "shape": list(parameter.shape),
                                           "dtype": str(parameter.dtype)}
                                          for name, parameter in zip(group.names, group.parameters)]}
                          for group in groups]
        encoded = json.dumps(self._manifest, sort_keys=True, separators=(",", ":"))
        self.group_fingerprint = hashlib.sha256(encoded.encode()).hexdigest()

    @property
    def group_manifest(self) -> list[dict[str, Any]]:
        return deepcopy(self._manifest)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers.values():
            optimizer.step()

    def state_dict(self) -> dict[str, Any]:
        return {"format_version": 1, "group_manifest": self.group_manifest,
                "group_fingerprint": self.group_fingerprint,
                "optimizers": {name: optimizer.state_dict()
                               for name, optimizer in self.optimizers.items()}}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (state.get("format_version") != 1
                or state.get("group_fingerprint") != self.group_fingerprint
                or state.get("group_manifest") != self._manifest):
            raise ValueError("Optimizer checkpoint parameter-name groups do not match this student")
        saved = state.get("optimizers", {})
        if set(saved) != set(self.optimizers):
            raise ValueError("Optimizer checkpoint members do not match the requested optimizer")
        # Check every group before loading either optimizer. Native optimizer
        # loading otherwise matches positional IDs without checking names.
        for name, optimizer in self.optimizers.items():
            actual_groups = optimizer.param_groups
            saved_groups = saved[name].get("param_groups", [])
            if len(actual_groups) != len(saved_groups):
                raise ValueError("Optimizer checkpoint group count does not match")
            for current, previous in zip(actual_groups, saved_groups):
                if (current["param_names"] != previous.get("param_names")
                        or len(current["params"]) != len(previous.get("params", []))):
                    raise ValueError("Optimizer checkpoint ordered parameter names do not match")
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(saved[name])


def build_optimizer_bundle(model: nn.Module, *, optimizer: str = "adamw",
                           lr: float = 2e-4, weight_decay: float = 0.01,
                           betas: tuple[float, float] = (0.9, 0.999),
                           muon_momentum: float = 0.95, muon_ns_steps: int = 5,
                           muon_adjust_lr_fn: str = "match_rms_adamw") -> OptimizerBundle:
    groups = partition_student_parameters(model, optimizer=optimizer)
    if optimizer == "muon_adamw":
        if not native_muon_available():
            raise RuntimeError("Muon was requested, but this PyTorch build has no native torch.optim.Muon. "
                               "Install a compatible PyTorch build or explicitly select optimizer='adamw'. "
                               "No fallback was used.")
        if "adjust_lr_fn" not in inspect.signature(torch.optim.Muon).parameters:
            raise RuntimeError("Native torch.optim.Muon must support adjust_lr_fn='match_rms_adamw'; "
                               "this PyTorch API is incompatible. No fallback was used.")
    instances = {}
    for group in groups:
        parameters = [{"params": list(group.parameters), "param_names": list(group.names)}]
        if group.optimizer == "muon":
            instances["muon"] = torch.optim.Muon(parameters, lr=lr, weight_decay=weight_decay,
                                                  momentum=muon_momentum, nesterov=True,
                                                  ns_steps=muon_ns_steps,
                                                  adjust_lr_fn=muon_adjust_lr_fn)
        else:
            instances["adamw"] = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay,
                                                    betas=betas)
    return OptimizerBundle(groups, instances)
