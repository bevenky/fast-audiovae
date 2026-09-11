"""Isolated matrix-optimizer comparison in the decoder's existing WN coordinates.

Only the nine residual pointwise ``weight_v`` tensors use matrix updates. Each
native [out, in, 1] tensor is viewed independently as [out, in]; no parameter is
replaced, no layer is merged with another, and no effective-weight rewrite occurs.
The remaining 81 tensors use the original torch AdamW implementation. Startup
constraints are deliberately outside this module and must check the final full
90-tensor proposal. See optimizer-comparison-v1/optimizer-sources.md for formulas.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

import torch
from torch.optim import Optimizer


VERSION = "audiovae2_recovery_optimizers_v1"
METHODS = ("adamw", "muon", "normuon", "shampoo")
MATRIX_NAMES = tuple(f"model.{s}.block.{u}.block.3.weight_v"
                     for s in (3, 4, 5) for u in (2, 3, 4))
NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
DOUBLE_STATE = {"gradient_ema", "factor_left", "factor_right", "graft_second",
                "inverse_left", "inverse_right"}


def _positive(value, name, *, zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (value < 0 if zero else value <= 0)):
        raise ValueError(f"Invalid {name}")
    return float(value)


def _beta(value, name):
    value = _positive(value, name, zero=True)
    if value >= 1:
        raise ValueError(f"{name} must be below one")
    return value


def _json_group(group):
    return {k: copy.deepcopy(v) for k, v in group.items() if k != "params"}


def _named(model, expected_widths):
    named = list(model.group_named_parameters())
    if (len(named) != 90 or len({n for n, _ in named}) != 90
            or len({id(p) for _, p in named}) != 90
            or any(not p.requires_grad or p.dtype != torch.float32 for _, p in named)
            or {id(p) for _, p in named} != {id(p) for p in model.parameters() if p.requires_grad}):
        raise ValueError("Require exactly the original 90 unique trainable FP32 group tensors")
    lookup = dict(named)
    if len(expected_widths) != 3:
        raise ValueError("Require three declared stage widths")
    for stage, width in zip((3, 4, 5), expected_widths):
        if type(width) is not int or width <= 0:
            raise ValueError("Invalid declared stage width")
        for unit in (2, 3, 4):
            name = f"model.{stage}.block.{unit}.block.3.weight_v"
            if name not in lookup or tuple(lookup[name].shape) != (width, width, 1):
                raise ValueError(f"Unexpected pointwise matrix: {name}")
            module = model.decoder.get_submodule(name.rsplit(".", 1)[0])
            if (not isinstance(module, torch.nn.Conv1d) or module.groups != 1
                    or tuple(module.kernel_size) != (1,) or module.weight_v is not lookup[name]):
                raise ValueError("Matrix allowlist must resolve to original pointwise WN parameters")
    return named


def matrix_parameter_manifest(model, *, expected_widths=(384, 256, 128)):
    named = _named(model, expected_widths)
    return [{"name": n, "canonical_index": i, "native_shape": list(p.shape),
             "matrix_shape": list(p.shape[:2]), "row_axis": "output_channel",
             "column_axis": "input_channel", "coordinate": "weight_v"}
            for i, (n, p) in enumerate(named) if n in MATRIX_NAMES]


def _hyperparameters(method, matrix_lr, adam_lr, momentum, neuron_beta2,
                     ns_steps, rms_target, shampoo_beta1, shampoo_beta2,
                     shampoo_damping, precondition_frequency, start_preconditioning_step):
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}")
    if method == "adamw" and matrix_lr is None:
        matrix_lr = adam_lr
    matrix_lr = _positive(matrix_lr, "matrix_lr")
    if _positive(adam_lr, "adam_lr") != 3e-5:
        raise ValueError("The 81-parameter Adam recipe must retain lr=3e-5")
    if type(ns_steps) is not int or ns_steps <= 0:
        raise ValueError("Invalid NS iteration count")
    if (type(precondition_frequency) is not int or precondition_frequency < 1
            or type(start_preconditioning_step) is not int
            or start_preconditioning_step < precondition_frequency
            or start_preconditioning_step % precondition_frequency):
        raise ValueError("Start must be a positive multiple of the refresh interval")
    return {
        "method": method,
        "adam": {"lr": adam_lr, "betas": [0.9, 0.99], "eps": 1e-8, "weight_decay": 0.0},
        "matrix": {"lr": matrix_lr, "momentum": _beta(momentum, "momentum"),
            "nesterov": False, "ns_steps": ns_steps, "ns_coefficients": list(NS_COEFFICIENTS),
            "ns_epsilon": 1e-7, "orthogonalization_dtype": "float32",
            "rms_target": _positive(rms_target, "rms_target"),
            "neuron_beta2": _beta(neuron_beta2, "neuron_beta2"), "neuron_epsilon": 1e-8,
            "shampoo_beta1": _beta(shampoo_beta1, "shampoo_beta1"),
            "shampoo_beta2": _beta(shampoo_beta2, "shampoo_beta2"),
            "shampoo_damping": _positive(shampoo_damping, "shampoo_damping"),
            "shampoo_exponent": -0.25, "shampoo_dtype": "float64",
            "graft_beta2": 0.99, "graft_epsilon": 1e-8, "bias_correction": True,
            "precondition_frequency": precondition_frequency,
            "start_preconditioning_step": start_preconditioning_step,
            "weight_decay": 0.0},
    }


def newton_schulz(matrix, *, steps=5, epsilon=1e-7):
    """Independent two-dimensional FP32 NS5 polynomial, never batched flattening."""
    if matrix.ndim != 2 or not torch.isfinite(matrix).all():
        raise ValueError("NS requires one finite 2D matrix")
    x = matrix.detach().float().clone()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (torch.linalg.vector_norm(x) + epsilon)
    a, b, c = NS_COEFFICIENTS
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.T if transposed else x


def _rms_scale(matrix, target):
    norm = torch.linalg.vector_norm(matrix)
    if not torch.isfinite(norm):
        raise RuntimeError("Nonfinite orthogonalized update")
    return matrix if norm.item() == 0 else matrix * (target * math.sqrt(matrix.numel()) / norm)


def _inverse_fourth(matrix, damping):
    # Explicit fixed damping; no adaptive jitter, hidden diagonal fallback or clipping.
    a = (matrix + matrix.T) * 0.5
    a = a + damping * torch.eye(a.shape[0], dtype=a.dtype, device=a.device)
    values, vectors = torch.linalg.eigh(a)
    if not torch.isfinite(values).all() or values.min().item() <= 0:
        raise RuntimeError("Shampoo factor is not numerically positive definite")
    result = (vectors * values.pow(-0.25).unsqueeze(0)) @ vectors.T
    if not torch.isfinite(result).all():
        raise RuntimeError("Nonfinite Shampoo inverse root")
    return result


@torch.no_grad()
def matrix_direction(gradient, state, method, config):
    """Advance one matrix's genuine state once and return its pre-LR direction.

    This public helper supports small rectangular formula tests. The model factory
    permits only the nine authenticated square pointwise tensors.
    """
    if (method not in METHODS[1:] or gradient.ndim != 2 or gradient.dtype != torch.float32
            or not torch.isfinite(gradient).all()):
        raise ValueError("Require one finite FP32 matrix and a matrix method")
    step = state.get("step", 0) + 1
    if method in ("muon", "normuon"):
        if not state:
            state["momentum"] = torch.zeros_like(gradient)
            if method == "normuon":
                state["variance_neuron"] = torch.zeros_like(gradient[:, :1])
        state["momentum"].lerp_(gradient, 1 - config["momentum"])
        direction = newton_schulz(state["momentum"], steps=config["ns_steps"],
                                 epsilon=config["ns_epsilon"])
        if method == "normuon":
            variance = state["variance_neuron"]
            variance.lerp_(direction.square().mean(dim=1, keepdim=True),
                           1 - config["neuron_beta2"])
            direction = direction / (variance.sqrt() + config["neuron_epsilon"])
        direction = _rms_scale(direction, config["rms_target"])
    else:
        g = gradient.double()
        if not state:
            m, n = g.shape
            state.update(gradient_ema=torch.zeros_like(g), graft_second=torch.zeros_like(g),
                factor_left=g.new_zeros(m, m), factor_right=g.new_zeros(n, n),
                inverse_left=g.new_empty(0, 0), inverse_right=g.new_empty(0, 0), last_refresh=0)
        b1, b2, gb = config["shampoo_beta1"], config["shampoo_beta2"], config["graft_beta2"]
        state["gradient_ema"].lerp_(g, 1 - b1)
        state["graft_second"].lerp_(g.square(), 1 - gb)
        state["factor_left"].lerp_(g @ g.T, 1 - b2)
        state["factor_right"].lerp_(g.T @ g, 1 - b2)
        filtered = state["gradient_ema"] / (1 - b1**step)
        graft = filtered / ((state["graft_second"] / (1 - gb**step)).sqrt() + config["graft_epsilon"])
        if step < config["start_preconditioning_step"]:
            direction = graft
        else:
            if step % config["precondition_frequency"] == 0:
                state["inverse_left"] = _inverse_fourth(state["factor_left"] / (1 - b2**step), config["shampoo_damping"])
                state["inverse_right"] = _inverse_fourth(state["factor_right"] / (1 - b2**step), config["shampoo_damping"])
                state["last_refresh"] = step
            direction = state["inverse_left"] @ filtered @ state["inverse_right"]
            norm, graft_norm = torch.linalg.vector_norm(direction), torch.linalg.vector_norm(graft)
            if norm.item() == 0:
                if graft_norm.item() != 0:
                    raise RuntimeError("Zero Shampoo direction cannot receive a nonzero graft")
            else:
                direction = direction * (graft_norm / norm)
    if not torch.isfinite(direction).all():
        raise RuntimeError("Nonfinite matrix direction")
    state["step"] = step
    return direction


def _bind(optimizer, named, config):
    optimizer._recovery_params = tuple(p for _, p in named)
    optimizer._recovery_config = copy.deepcopy(config)
    optimizer._recovery_config_seal = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    optimizer._recovery_groups = [_json_group(g) for g in optimizer.param_groups]
    optimizer._recovery_group_ids = [tuple(id(p) for p in g["params"]) for g in optimizer.param_groups]


class RecoveryMixedOptimizer(Optimizer):
    """Native torch AdamW for 81 tensors, explicit matrix updates for nine tensors."""
    def __init__(self, named_parameters, config):
        named = list(named_parameters)
        if config["method"] not in METHODS[1:]:
            raise ValueError("Mixed optimizer requires a matrix method")
        matrix_names = set(config["matrix_names"])
        adam_params = [p for n, p in named if n not in matrix_names]
        matrix_params = [p for n, p in named if n in matrix_names]
        if len(named) != 90 or len(adam_params) != 81 or len(matrix_params) != 9:
            raise ValueError("Require disjoint 81/9 optimizer partition")
        adam = config["adam"]
        self._adam = torch.optim.AdamW(adam_params, lr=adam["lr"], betas=tuple(adam["betas"]),
                                      eps=adam["eps"], weight_decay=adam["weight_decay"])
        adam_group = dict(self._adam.param_groups[0], algorithm="adamw")
        matrix_group = dict(params=matrix_params, algorithm=config["method"], **config["matrix"])
        super().__init__([adam_group, matrix_group], {})
        self._adam.param_groups = [self.param_groups[0]]
        self._adam.state = self.state
        _bind(self, named, config)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        params = self._recovery_params
        _assert_binding(self, params)
        if any(p.grad is None or p.grad.is_sparse or p.grad.shape != p.shape
               or p.grad.dtype != p.dtype or not torch.isfinite(p.grad).all() for p in params):
            raise RuntimeError("All 90 ordinary gradients must exist and be finite dense FP32 tensors")
        previous = 0 if not self.state else _step(next(iter(self.state.values()))["step"])
        assert_optimizer_state(self, params, previous)
        # Matrix arithmetic is prepared before torch Adam mutates its parameters.
        # The surrounding retention transaction owns rollback for any hard failure.
        proposals = []
        config = self._recovery_config["matrix"]
        for p in self.param_groups[1]["params"]:
            direction = matrix_direction(p.grad.reshape(p.shape[0], p.shape[1]),
                                         self.state[p], self._recovery_config["method"], config)
            proposals.append((p, direction.to(dtype=p.dtype).reshape_as(p)))
        self._adam.step()
        for p, direction in proposals:
            p.add_(direction, alpha=-config["lr"])
        assert_optimizer_state(self, params, previous + 1)
        return loss

    def state_dict(self):
        result = super().state_dict()
        result["recovery_config"] = optimizer_config(self)
        return result

    def _load_native(self, state_dict):
        super().load_state_dict(state_dict)
        # Optimizer.load_state_dict casts floating states to parameter dtype. Recover
        # FP64 Shampoo statistics from serialized originals, not the rounded cast.
        for group, saved_group in zip(self.param_groups, state_dict["param_groups"], strict=True):
            if group["algorithm"] == "shampoo":
                for p, sid in zip(group["params"], saved_group["params"], strict=True):
                    for key, value in state_dict["state"].get(sid, {}).items():
                        if key in DOUBLE_STATE:
                            if not isinstance(value, torch.Tensor) or value.dtype != torch.float64:
                                raise ValueError("Serialized Shampoo statistics must remain FP64")
                            self.state[p][key] = value.detach().to(device=p.device, dtype=torch.float64).clone()
        self._adam.param_groups = [self.param_groups[0]]
        self._adam.state = self.state

    def load_state_dict(self, state_dict):
        _assert_binding(self, self._recovery_params)
        if state_dict.get("recovery_config") != self._recovery_config:
            raise ValueError("Optimizer checkpoint configuration or matrix identity differs")
        if [_json_group(g) for g in state_dict["param_groups"]] != self._recovery_groups:
            raise ValueError("Optimizer checkpoint group recipe differs")
        old = copy.deepcopy(super().state_dict())
        if ([g["params"] for g in state_dict["param_groups"]]
                != [g["params"] for g in old["param_groups"]]):
            raise ValueError("Serialized canonical parameter-state mapping changed")
        try:
            self._load_native(state_dict)
            step = 0 if not self.state else _step(next(iter(self.state.values()))["step"])
            assert_optimizer_state(self, self._recovery_params, step)
        except BaseException:
            self._load_native(old)
            raise


def build_optimizer(model, method, *, matrix_lr=None, adam_lr=3e-5, momentum=0.95,
                    neuron_beta2=0.95, ns_steps=5, rms_target=0.2,
                    shampoo_beta1=0.9, shampoo_beta2=0.99, shampoo_damping=1e-12,
                    precondition_frequency=10, start_preconditioning_step=10,
                    expected_widths=(384, 256, 128)):
    named = _named(model, expected_widths)
    cfg = _hyperparameters(method, matrix_lr, adam_lr, momentum, neuron_beta2,
        ns_steps, rms_target, shampoo_beta1, shampoo_beta2, shampoo_damping,
        precondition_frequency, start_preconditioning_step)
    cfg.update(version=VERSION, canonical_names=[n for n, _ in named],
        matrix_names=[n for n, _ in named if n in MATRIX_NAMES],
        matrix_manifest=matrix_parameter_manifest(model, expected_widths=expected_widths),
        canonical_shapes=[list(p.shape) for _, p in named],
        tensor_count=90, matrix_tensor_count=0 if method == "adamw" else 9,
        parameterization="unchanged legacy weight_norm g/v; matrix optimizer acts on v",
        no_inference_operations_added=True)
    if method == "adamw":
        if cfg["matrix"]["lr"] == adam_lr:
            adam_parameters = [p for _, p in named]
        else:
            adam_parameters = [
                {"params": [p for n, p in named if n not in MATRIX_NAMES], "lr": adam_lr},
                {"params": [p for n, p in named if n in MATRIX_NAMES], "lr": cfg["matrix"]["lr"]},
            ]
        optimizer = torch.optim.AdamW(adam_parameters, lr=adam_lr,
            betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0)
        _bind(optimizer, named, cfg)
    else:
        optimizer = RecoveryMixedOptimizer(named, cfg)
    assert_optimizer_state(optimizer, [p for _, p in named], 0)
    return optimizer


def _step(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or not torch.isfinite(value).all():
            raise ValueError("Invalid optimizer step tensor")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or int(value) != value:
        raise ValueError("Invalid optimizer counter")
    return int(value)


def _assert_binding(optimizer, params):
    if not hasattr(optimizer, "_recovery_config"):
        raise ValueError("Optimizer lacks the factory's sealed recipe")
    if hashlib.sha256(json.dumps(optimizer._recovery_config, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest() != optimizer._recovery_config_seal:
        raise ValueError("Optimizer's sealed configuration changed")
    if tuple(map(id, params)) != tuple(map(id, optimizer._recovery_params)):
        raise ValueError("Canonical 90-parameter ordering changed")
    if ([tuple(id(p) for p in g["params"]) for g in optimizer.param_groups] != optimizer._recovery_group_ids
            or [_json_group(g) for g in optimizer.param_groups] != optimizer._recovery_groups):
        raise ValueError("Optimizer group membership or immutable recipe changed")
    if (len(params) != 90 or len({id(p) for p in params}) != 90
            or any(not p.requires_grad or p.dtype != torch.float32 or not torch.isfinite(p).all() for p in params)
            or [list(p.shape) for p in params] != optimizer._recovery_config["canonical_shapes"]):
        raise ValueError("Native parameter contract changed")


def optimizer_config(optimizer):
    _assert_binding(optimizer, optimizer._recovery_params)
    return copy.deepcopy(optimizer._recovery_config)


def _tensor_state(value, shape, dtype, device, name):
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
            or value.dtype != dtype or value.device != device or not torch.isfinite(value).all()):
        raise ValueError(f"Invalid optimizer tensor state {name}")


def assert_optimizer_state(optimizer, params, expected_step):
    """Fail on topology, recipe, counter, state-shape/dtype or finite-state drift."""
    params = list(params)
    if type(expected_step) is not int or expected_step < 0:
        raise ValueError("Expected step must be a nonnegative integer")
    _assert_binding(optimizer, params)
    if expected_step == 0:
        if optimizer.state:
            raise ValueError("Fresh optimizer state must be truly empty")
        return {"step": 0, "states": 0, "finite": True, "canonical_tensors": 90}
    if set(optimizer.state) != set(params):
        raise ValueError("Optimizer state must cover every native parameter exactly")
    matrix_ids = set() if not isinstance(optimizer, RecoveryMixedOptimizer) else {id(p) for p in optimizer.param_groups[1]["params"]}
    method = optimizer._recovery_config["method"]
    cfg = optimizer._recovery_config["matrix"]
    for p in params:
        state = optimizer.state[p]
        if "step" not in state or _step(state["step"]) != expected_step:
            raise ValueError("Optimizer state counter differs from completed updates")
        if id(p) not in matrix_ids:
            if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("Unexpected torch AdamW state schema")
            for key in ("exp_avg", "exp_avg_sq"):
                _tensor_state(state[key], p.shape, p.dtype, p.device, key)
            if (state["exp_avg_sq"] < 0).any():
                raise ValueError("Negative Adam second moment")
        elif method in ("muon", "normuon"):
            keys = {"step", "momentum"} | ({"variance_neuron"} if method == "normuon" else set())
            if set(state) != keys:
                raise ValueError("Unexpected Muon state schema")
            _tensor_state(state["momentum"], p.shape[:2], torch.float32, p.device, "momentum")
            if method == "normuon":
                _tensor_state(state["variance_neuron"], (p.shape[0], 1), torch.float32, p.device, "variance_neuron")
                if (state["variance_neuron"] < 0).any():
                    raise ValueError("Negative neuron second moment")
        else:
            if set(state) != DOUBLE_STATE | {"step", "last_refresh"}:
                raise ValueError("Unexpected Shampoo state schema")
            m, n = p.shape[:2]
            refresh = (expected_step // cfg["precondition_frequency"] * cfg["precondition_frequency"]
                       if expected_step >= cfg["start_preconditioning_step"] else 0)
            if type(state["last_refresh"]) is not int or state["last_refresh"] != refresh:
                raise ValueError("Shampoo inverse-root refresh ledger differs")
            for key, shape in (("gradient_ema", (m, n)), ("graft_second", (m, n)),
                    ("factor_left", (m, m)), ("factor_right", (n, n)),
                    ("inverse_left", (m, m) if refresh else (0, 0)),
                    ("inverse_right", (n, n) if refresh else (0, 0))):
                _tensor_state(state[key], shape, torch.float64, p.device, key)
            if (state["graft_second"] < 0).any():
                raise ValueError("Negative graft second moment")
    return {"step": expected_step, "states": 90, "finite": True,
            "matrix_states": len(matrix_ids), "canonical_tensors": 90}
