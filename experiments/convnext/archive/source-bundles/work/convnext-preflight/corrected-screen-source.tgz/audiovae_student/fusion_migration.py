"""Strict parent restoration followed by explicitly identified fusion forks.

This module does not launch training or reset experiment clocks. Candidate
checkpoints use a different format so changed functions cannot masquerade as
an exact RecipeV2 resume. A separate runner owns data exposure, fresh-MRD
preparation, global RNG restoration, saving and any experimental resume.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import inspect

import torch
from torch import nn

from .discriminators import AudioDiscriminators, DiscriminatorConfig, ResolutionDiscriminator
from .fusion_architecture import FusionArchitectureConfig, FusionStudentDecoder
from .fusion_objectives import ComplexAudioDiscriminators, ShortTimeReconstructionV2
from .model import StudentConfig, StudentDecoder
from .objective_comparison import state_fingerprint
from .optimizers import NamedParameterGroup, OptimizerBundle, build_optimizer_bundle, partition_student_parameters
from .recipe_v2 import RecipeV2Config, RecipeV2Engine


FUSION_VARIANTS = ("control", "tanh", "filter", "zero_padding", "short_mel", "fresh_magnitude", "complex")


class FusionRecipeV2Engine(RecipeV2Engine):
    """Unchanged training step with an explicitly experimental state format."""

    def state_dict(self):
        if not hasattr(self, "fusion_migration"):
            raise ValueError("Complete parent restoration and migration before saving a fusion experiment")
        state = super().state_dict()
        state.update(format_version="fusion_recipe_v1", fusion_variant=self.fusion_migration["variant"],
                     fusion_migration=deepcopy(self.fusion_migration))
        return state

    def load_state_dict(self, state):
        raise ValueError("Fusion checkpoints require an explicit experimental resume; use build_fusion_engine for a parent fork")


def _finite_state(value):
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise ValueError("Parent state contains a nonfinite tensor")
    elif isinstance(value, dict):
        for child in value.values():
            _finite_state(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _finite_state(child)


def _adamw_like(source, group):
    # Newer AdamW stores its fixed decoupled-weight-decay choice in defaults
    # even when its constructor does not accept that keyword. Preserve every
    # actual group option and pass only declared constructor options.
    options = deepcopy(source.defaults)
    constructor_keys = set(inspect.signature(torch.optim.AdamW).parameters)
    unsupported = set(options) - constructor_keys
    if unsupported - {"decoupled_weight_decay"}:
        raise ValueError("Unrecognized AdamW defaults require an explicit migration policy")
    if "decoupled_weight_decay" in unsupported and options["decoupled_weight_decay"] is not True:
        raise ValueError("Expected AdamW's decoupled weight decay")
    return torch.optim.AdamW([group], **{key: value for key, value in options.items() if key in constructor_keys})


def _rebind_student_optimizer(engine, replacement):
    old_bundle = engine.optimizer
    previous = deepcopy(old_bundle.state_dict())
    filtering = replacement.output_filter is not None
    parameter = replacement.output_filter.conv.weight if filtering else None
    if parameter is not None:
        parameter.requires_grad_(False)
    try:
        groups = partition_student_parameters(replacement, optimizer=engine.config.optimizer)
        rebound = build_optimizer_bundle(replacement, optimizer=engine.config.optimizer,
            lr=engine.config.learning_rate, weight_decay=engine.config.weight_decay)
        rebound.load_state_dict(previous)
    finally:
        if parameter is not None:
            parameter.requires_grad_(True)
    if filtering:
        original_adam = rebound.optimizers["adamw"]
        if len(original_adam.param_groups) != 1:
            raise ValueError("Filter migration requires one original complementary AdamW group")
        group = deepcopy({key: value for key, value in original_adam.param_groups[0].items()
                          if key not in {"params", "param_names"}})
        group.update(params=[parameter], param_names=["output_filter.conv.weight"])
        filter_optimizer = _adamw_like(original_adam, group)
        groups += (NamedParameterGroup("filter_adamw", ("output_filter.conv.weight",), (parameter,)),)
        rebound = OptimizerBundle(groups, {**rebound.optimizers, "filter_adamw": filter_optimizer})
    for name, original in old_bundle.optimizers.items():
        if state_fingerprint(original.state_dict()) != state_fingerprint(rebound.optimizers[name].state_dict()):
            raise RuntimeError("Original student optimizer tensors or parameter order changed")
    engine.model, engine.optimizer = replacement, rebound


def _fresh_magnitude(existing):
    replacement = deepcopy(existing)
    cfg = existing.config
    replacement.resolutions = nn.ModuleList([
        ResolutionDiscriminator(size, cfg.mrd_channels, cfg.log_epsilon) for size in cfg.fft_sizes
    ]).to(next(existing.parameters()).device)
    replacement.resolutions.train(existing.training)
    return replacement


def _replace_spectral_heads(engine, replacement):
    old_bank, old_optimizer = engine.discriminators, engine.discriminator_optimizer
    old_named, new_named = dict(old_bank.named_parameters()), dict(replacement.named_parameters())
    period_names = [name for name in old_named if name.startswith("periods.")]
    if set(period_names) != {name for name in new_named if name.startswith("periods.")}:
        raise ValueError("Fresh spectral migration changed period discriminator parameter names")
    for name in period_names:
        if not torch.equal(old_named[name], new_named[name]):
            raise ValueError("Fresh spectral migration changed trained period discriminator weights")
    if len(old_optimizer.param_groups) != 1:
        raise ValueError("Fresh spectral migration requires the original single AdamW group")
    group = deepcopy({key: value for key, value in old_optimizer.param_groups[0].items()
                      if key not in {"params", "param_names"}})
    group.update(params=list(new_named.values()), param_names=list(new_named))
    new_optimizer = _adamw_like(old_optimizer, group)
    # Source order was already strictly restored through the unchanged parent
    # bank. Resolve it to actual parameter names before migrating moments.
    for name in period_names:
        source_parameter = old_named[name]
        if source_parameter in old_optimizer.state:
            new_optimizer.state[new_named[name]] = deepcopy(old_optimizer.state[source_parameter])
    original_moments = {name: old_optimizer.state.get(old_named[name], {}) for name in period_names}
    restored_moments = {name: new_optimizer.state.get(new_named[name], {}) for name in period_names}
    if state_fingerprint(original_moments) != state_fingerprint(restored_moments):
        raise RuntimeError("Period discriminator optimizer moments changed")
    if any(parameter in new_optimizer.state for name, parameter in new_named.items()
           if name.startswith("resolutions.")):
        raise RuntimeError("Fresh spectral heads unexpectedly inherited old optimizer state")
    engine.discriminators, engine.discriminator_optimizer = replacement, new_optimizer
    return {"preserved_period_parameter_count": len(period_names),
            "preserved_period_optimizer_state_sha256": state_fingerprint(restored_moments),
            "fresh_spectral_parameter_count": sum(name.startswith("resolutions.") for name in new_named),
            "fresh_spectral_optimizer_state": "empty"}


def build_fusion_engine(parent_engine_state: dict, variant: str, device="cpu"):
    """Restore an unchanged RecipeV2 parent, then apply one declared candidate.

    Initialization of throwaway base layers preserves the caller's CPU RNG.
    Architecture/short-mel candidates consume no global RNG. Fresh spectral
    candidates consume caller-seeded RNG only for their new MRD parameters.
    Crop RNG, global training step, calibration and old optimizer tensors are
    preserved; no parameter update or teacher forward occurs here.
    """
    if variant not in FUSION_VARIANTS:
        raise ValueError("Unknown fusion variant")
    if not isinstance(parent_engine_state, dict) or parent_engine_state.get("format_version") != "recipe_v2":
        raise ValueError("Fusion forks require an original strict RecipeV2 parent state")
    if any(key.startswith("fusion_") for key in parent_engine_state):
        raise ValueError("An experimental state cannot be relabeled as an original RecipeV2 parent")
    _finite_state(parent_engine_state)
    parent_sha = state_fingerprint(parent_engine_state)
    # Optimizer.load_state_dict may otherwise alias same-device source tensors.
    # Isolate the entire parent state before any restore or later training.
    state = deepcopy(parent_engine_state)
    recipe = RecipeV2Config(**state["recipe"])
    target_device = torch.device(device)
    if target_device.type not in {"cpu", "cuda"}:
        raise ValueError("Fusion training candidates support CPU or CUDA only")
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = StudentDecoder(StudentConfig(**state["model_config"]))
        discriminators = AudioDiscriminators(DiscriminatorConfig(**state["discriminator_config"]))
        engine = FusionRecipeV2Engine(model.to(target_device), recipe=recipe,
                                      discriminators=discriminators.to(target_device))
        # Qualified base call is intentional: all original strict checks must
        # pass before changing any architecture, objective or optimizer group.
        RecipeV2Engine.load_state_dict(engine, state)
    architecture = FusionArchitectureConfig(terminal_tanh=variant == "tanh",
        causal_output_filter=variant == "filter", zero_startup_padding=variant == "zero_padding")
    if variant in {"tanh", "filter", "zero_padding"}:
        _rebind_student_optimizer(engine, FusionStudentDecoder.from_decoder(engine.model, architecture))
    if variant == "short_mel":
        engine.reconstruction = ShortTimeReconstructionV2().to(target_device)
    discriminator_migration = None
    if variant in {"fresh_magnitude", "complex"}:
        replacement = (_fresh_magnitude(engine.discriminators) if variant == "fresh_magnitude"
                       else ComplexAudioDiscriminators.from_existing(engine.discriminators))
        discriminator_migration = _replace_spectral_heads(engine, replacement)
    elif (state_fingerprint(engine.discriminators.state_dict()) != state_fingerprint(state["discriminators"])
          or state_fingerprint(engine.discriminator_optimizer.state_dict()) != state_fingerprint(state["discriminator_optimizer"])):
        raise RuntimeError("Migration changed an unchanged discriminator or its optimizer")
    current = engine.model.state_dict()
    preserved = {name: current[name] for name in state["model"]}
    if state_fingerprint(preserved) != state_fingerprint(state["model"]):
        raise RuntimeError("Migration changed original student parameters or calibration buffers")
    for name, original in state["optimizer"]["optimizers"].items():
        if state_fingerprint(engine.optimizer.optimizers[name].state_dict()) != state_fingerprint(original):
            raise RuntimeError("Migration changed original student optimizer state")
    if (engine.step != state["step"] or engine.discriminator_updates != state["discriminator_updates"]
            or state_fingerprint(engine.calibration) != state_fingerprint(state["calibration"])
            or state_fingerprint(engine.balancer.state_dict()) != state_fingerprint(state["balancer"])
            or not torch.equal(engine.crop_generator.get_state(), state["crop_rng"].cpu())):
        raise RuntimeError("Migration changed an existing training clock, calibration, balance or crop RNG")
    if state_fingerprint(parent_engine_state) != parent_sha:
        raise RuntimeError("Source parent state was mutated")
    receipt = {
        "format_version": 1, "variant": variant, "parent_engine_state_sha256": parent_sha,
        "parent_step": engine.step, "architecture": architecture.to_dict(),
        "required_context_latent_frames": (engine.model.required_context_latent_frames
            if isinstance(engine.model, FusionStudentDecoder) else (engine.model.config.history_frames + 3) // 4),
        "reconstruction_config": asdict(engine.reconstruction.config),
        "discriminator_config": asdict(engine.discriminators.config),
        "original_student_state_preserved": True,
        "original_student_state_sha256": state_fingerprint(preserved),
        "original_student_optimizers_preserved": True,
        "calibration_preserved": True, "crop_rng_preserved": True, "balancer_preserved": True,
        "training_step_preserved": True, "discriminator_updates_preserved": True,
        "new_student_optimizer_groups": sorted(set(engine.optimizer.optimizers) - set(state["optimizer"]["optimizers"])),
        "discriminator_state_policy": ("trained_periods_preserved_fresh_spectral_heads"
            if discriminator_migration is not None else "all_discriminators_and_moments_preserved"),
        "discriminator_migration": discriminator_migration,
        "production_export_qualified": False,
    }
    engine.fusion_migration = deepcopy(receipt)
    return engine, receipt
