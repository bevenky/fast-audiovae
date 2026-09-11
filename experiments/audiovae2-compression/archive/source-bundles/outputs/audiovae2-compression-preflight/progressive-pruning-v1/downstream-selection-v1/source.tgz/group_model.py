"""Teacher-initialized replacement of AudioVAE2 decoder stages 2 through 4.

This experiment keeps the complete group's external feature coordinates and
causal geometry. Only its two internal channel boundaries are narrowed. The
teacher module supplied by the caller is never modified or retained by reference.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from collections.abc import Iterable, Mapping

import torch
from torch import Tensor, nn


GROUP_INDICES = (3, 4, 5)
GROUP_PREFIXES = tuple(
    f"{collection}.{index}."
    for collection in ("model", "sr_cond_model")
    for index in GROUP_INDICES
)


def _legacy_weight_hook(module: nn.Module):
    hooks = [
        hook for hook in module._forward_pre_hooks.values()
        if getattr(hook, "name", None) == "weight"
        and hasattr(hook, "compute_weight") and hasattr(hook, "dim")
    ]
    if len(hooks) > 1:
        raise ValueError("Multiple weight-normalization hooks on one convolution")
    return hooks[0] if hooks else None


def effective_weight(module: nn.Module) -> Tensor:
    """Return current effective weights, including after an optimizer/state load.

    Legacy weight_norm leaves a cached nonleaf ``weight`` attribute which can be
    stale between forwards. Its hook computes the correct value from g and v.
    """
    hook = _legacy_weight_hook(module)
    return hook.compute_weight(module) if hook is not None else module.weight


@torch.no_grad()
def assign_effective_weight(
    module: nn.Module, weight: Tensor, bias: Tensor | None = None
) -> None:
    """Install a selected effective matrix without retaining the old WN scale.

    Legacy ConvTranspose weight_norm uses dim=0 (input-channel rows), unlike
    its output bias. Read the actual hook rather than assuming an output axis.
    Zero rows get a harmless unit direction and zero scale, avoiding 0/0.
    If ``bias`` is omitted the existing bias is left untouched.
    """
    old = effective_weight(module)
    if weight.shape != old.shape or not torch.isfinite(weight).all():
        raise ValueError("Effective weight has an invalid shape or nonfinite values")
    if bias is not None:
        if module.bias is None or bias.shape != module.bias.shape or not torch.isfinite(bias).all():
            raise ValueError("Bias has an invalid shape or nonfinite values")
    hook = _legacy_weight_hook(module)
    if hook is None:
        if hasattr(module, "parametrizations") and "weight" in module.parametrizations:
            raise ValueError("This teacher-copy implementation expects legacy weight_norm")
        module.weight.copy_(weight)
    else:
        v = weight.to(module.weight_v).clone()
        dim = hook.dim
        if dim == -1:
            g = torch.linalg.vector_norm(v).reshape_as(module.weight_g)
            if g.item() == 0:
                v.reshape(-1)[0] = 1
        else:
            axes = tuple(axis for axis in range(v.ndim) if axis != dim)
            g = torch.linalg.vector_norm(v, dim=axes, keepdim=True)
            rows = v.movedim(dim, 0).reshape(v.shape[dim], -1)
            zero = g.reshape(-1) == 0
            if zero.any():
                # movedim/reshape may allocate; explicitly move the repaired
                # contiguous rows back to the original tensor geometry.
                rows = rows.clone()
                rows[zero, 0] = 1
                shape = v.movedim(dim, 0).shape
                v = rows.reshape(shape).movedim(0, dim).contiguous()
        if not torch.isfinite(g).all():
            raise ValueError("Effective weight norm is nonfinite")
        module.weight_v.copy_(v)
        module.weight_g.copy_(g.reshape_as(module.weight_g))
        module.weight = hook.compute_weight(module)
    if bias is not None:
        module.bias.copy_(bias)


def clone_teacher(teacher_decoder: nn.Module) -> nn.Module:
    """Independent decoder copy, including legacy WN's nonleaf cache tensors."""
    memo = {}
    for module in teacher_decoder.modules():
        if _legacy_weight_hook(module) is not None:
            cached = module.__dict__.get("weight")
            if isinstance(cached, Tensor):
                memo[id(cached)] = effective_weight(module).detach().clone()
    result = copy.deepcopy(teacher_decoder, memo)
    result.requires_grad_(False).eval()
    return result


def _indices(values: Iterable[int] | Tensor, width: int, name: str) -> Tensor:
    if isinstance(values, Tensor):
        if values.ndim != 1 or values.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must be a one-dimensional integer selection")
        index = values.detach().to(device="cpu", dtype=torch.long).clone()
    else:
        values = list(values)
        if any(isinstance(x, bool) or not isinstance(x, int) for x in values):
            raise ValueError(f"{name} must contain integer channel indices")
        index = torch.tensor(values, dtype=torch.long)
    if not 0 < index.numel() <= width:
        raise ValueError(f"{name} must select at least one and at most {width} channels")
    if index.min().item() < 0 or index.max().item() >= width:
        raise ValueError(f"{name} contains an out-of-range channel")
    if index.unique().numel() != index.numel():
        raise ValueError(f"{name} contains duplicate channels")
    return index


def _validate_decoder(decoder: nn.Module) -> None:
    if not hasattr(decoder, "model") or len(decoder.model) != 11:
        raise ValueError("Expected the depthwise AudioVAE2 decoder with two stem layers and six stages")
    for index in range(2, 8):
        stage = decoder.model[index]
        if not hasattr(stage, "block") or len(stage.block) != 5:
            raise ValueError("Every stage must have an upsampler and exactly three residual units; noise is unsupported")
        up = stage.block[1]
        if not isinstance(up, nn.ConvTranspose1d) or up.groups != 1:
            raise ValueError("Expected an ordinary causal transposed convolution")
        if up.kernel_size != (2 * up.stride[0],):
            raise ValueError("Unexpected upsampling kernel geometry")
        if getattr(stage, "input_channels", None) != up.in_channels:
            raise ValueError("Stage input_channels disagrees with its upsampler")
        for unit, dilation in zip(stage.block[2:], (1, 3, 9)):
            if not hasattr(unit, "block") or len(unit.block) != 4:
                raise ValueError("Expected Snake/depthwise/Snake/pointwise residual units")
            depth, point = unit.block[1], unit.block[3]
            width = up.out_channels
            if not isinstance(depth, nn.Conv1d) or (
                depth.groups, depth.in_channels, depth.out_channels,
                depth.kernel_size, depth.dilation
            ) != (width, width, width, (7,), (dilation,)):
                raise ValueError("Unexpected residual depthwise geometry or dilation")
            if not isinstance(point, nn.Conv1d) or (
                point.groups, point.in_channels, point.out_channels, point.kernel_size
            ) != (1, width, width, (1,)):
                raise ValueError("Unexpected residual pointwise geometry")
    if decoder.sr_bin_boundaries is not None:
        if len(decoder.sr_cond_model) != len(decoder.model):
            raise ValueError("Conditioning and decoder module lists must align")
        for index in range(2, 8):
            cond = decoder.sr_cond_model[index]
            if cond.cond_type != "scale_bias" or not isinstance(cond.out_layer, nn.Identity):
                raise ValueError("Expected the pinned scale_bias conditioning with identity out_layer")
            width = decoder.model[index].input_channels
            if cond.scale_embed.weight.shape != cond.bias_embed.weight.shape or cond.scale_embed.embedding_dim != width:
                raise ValueError("Conditioning width differs from its stage input")


def _sample_rate_index(decoder: nn.Module, x: Tensor, sr_cond, default: int):
    if decoder.sr_bin_boundaries is None:
        return None
    if sr_cond is None:
        sr_cond = torch.tensor([default], device=x.device, dtype=torch.int32)
    elif not isinstance(sr_cond, Tensor):
        sr_cond = torch.as_tensor(sr_cond, device=x.device, dtype=torch.int32)
    sr_cond = sr_cond.to(device=x.device)
    if sr_cond.ndim == 0:
        sr_cond = sr_cond.reshape(1)
    if sr_cond.ndim != 1 or sr_cond.numel() not in (1, x.shape[0]):
        raise ValueError("Sample rate must be a scalar, singleton or one value per batch item")
    return decoder.get_sr_idx(sr_cond)


def _run(decoder: nn.Module, x: Tensor, start: int, stop: int, sr_idx) -> Tensor:
    for index in range(start, stop):
        if decoder.sr_bin_boundaries is not None:
            cond = decoder.sr_cond_model[index]
            if cond is not None:
                x = cond(x, sr_idx)
        x = decoder.model[index](x)
    return x


def teacher_trace(
    teacher_decoder: nn.Module, z: Tensor, sr_cond=None, *, output_sample_rate: int = 48000
) -> dict[str, Tensor]:
    """Run the unmodified teacher, exposing raw boundaries without hooks.

    This helper does not change grad mode. Target callers should use no_grad;
    differentiable diagnostics may deliberately leave it enabled.
    """
    sr_idx = _sample_rate_index(teacher_decoder, z, sr_cond, output_sample_rate)
    x = _run(teacher_decoder, z, 0, 3, sr_idx)
    result = {"group_input": x, "stage1_output": x}
    for index, key in ((3, "stage2_output"), (4, "stage3_output"), (5, "group_output")):
        x = _run(teacher_decoder, x, index, index + 1, sr_idx)
        result[key] = x
    result["stage4_output"] = result["group_output"]
    x = _run(teacher_decoder, x, 6, 7, sr_idx)
    result["stage5_output"] = x
    x = _run(teacher_decoder, x, 7, 8, sr_idx)
    result["stage6_output"] = x
    x = _run(teacher_decoder, x, 8, 10, sr_idx)
    result["pre_tanh"] = x
    result["waveform"] = _run(teacher_decoder, x, 10, len(teacher_decoder.model), sr_idx)
    return result


@torch.no_grad()
def _copy_snake(source: nn.Module, target: nn.Module, index: Tensor) -> None:
    selected = source.alpha.index_select(1, index.to(source.alpha.device))
    if selected.shape != target.alpha.shape:
        raise ValueError("Snake parameter shape disagrees with the selected channels")
    target.alpha.copy_(selected)


@torch.no_grad()
def _copy_conv(source: nn.Module, target: nn.Module, ins: Tensor, outs: Tensor) -> None:
    weight = effective_weight(source).detach()
    ins, outs = ins.to(weight.device), outs.to(weight.device)
    if isinstance(source, nn.ConvTranspose1d):
        weight = weight.index_select(0, ins).index_select(1, outs)
    elif source.groups == source.in_channels == source.out_channels:
        if not torch.equal(ins, outs):
            raise ValueError("Depthwise input and output coordinate selections must match")
        weight = weight.index_select(0, outs)
    elif source.groups == 1:
        weight = weight.index_select(0, outs).index_select(1, ins)
    else:
        raise ValueError("Unsupported grouped convolution")
    bias = None if source.bias is None else source.bias.index_select(0, outs)
    if (source.bias is None) != (target.bias is None):
        raise ValueError("Bias presence differs between teacher and student")
    assign_effective_weight(target, weight, bias)


def _narrow_stage(source: nn.Module, ins: Tensor, outs: Tensor) -> nn.Module:
    up = source.block[1]
    target = type(source)(
        input_dim=ins.numel(), output_dim=outs.numel(), stride=up.stride[0],
        groups=outs.numel(), use_noise_block=False,
    ).to(device=up.weight_v.device, dtype=up.weight_v.dtype)
    _copy_snake(source.block[0], target.block[0], ins)
    _copy_conv(up, target.block[1], ins, outs)
    for src_unit, dst_unit in zip(source.block[2:], target.block[2:]):
        _copy_snake(src_unit.block[0], dst_unit.block[0], outs)
        _copy_conv(src_unit.block[1], dst_unit.block[1], outs, outs)
        _copy_snake(src_unit.block[2], dst_unit.block[2], outs)
        _copy_conv(src_unit.block[3], dst_unit.block[3], outs, outs)
    return target


def _narrow_condition(source: nn.Module, index: Tensor) -> nn.Module:
    target = copy.deepcopy(source)
    for name in ("scale_embed", "bias_embed"):
        embedding = getattr(source, name)
        selected = embedding.weight.detach().index_select(1, index.to(embedding.weight.device)).clone()
        setattr(target, name, nn.Embedding.from_pretrained(selected, freeze=False))
    return target


class CompressedDecoderGroup(nn.Module):
    """Independent teacher decoder with trainable stages 2–4 only."""

    def __init__(self, decoder: nn.Module, selections: Mapping[str, list[int]], output_sample_rate: int):
        super().__init__()
        self.decoder = decoder
        self.selections = copy.deepcopy(dict(selections))
        self.output_sample_rate = int(output_sample_rate)
        self.decoder.requires_grad_(False)
        for index in GROUP_INDICES:
            self.decoder.model[index].requires_grad_(True)
            if self.decoder.sr_bin_boundaries is not None:
                self.decoder.sr_cond_model[index].requires_grad_(True)
        self.eval()

    def train(self, mode: bool = True):
        # All frozen modules remain in inference mode even during fitting.
        super().train(False)
        self.training = mode
        for index in GROUP_INDICES:
            self.decoder.model[index].train(mode)
            if self.decoder.sr_bin_boundaries is not None:
                self.decoder.sr_cond_model[index].train(mode)
        return self

    def prefix_from_latents(self, z: Tensor, sr_cond=None) -> Tensor:
        idx = _sample_rate_index(self.decoder, z, sr_cond, self.output_sample_rate)
        return _run(self.decoder, z, 0, 3, idx)

    def group_from_input(self, x: Tensor, sr_cond=None) -> Tensor:
        idx = _sample_rate_index(self.decoder, x, sr_cond, self.output_sample_rate)
        return _run(self.decoder, x, 3, 6, idx)

    def suffix_from_group(self, h: Tensor, sr_cond=None) -> Tensor:
        # Frozen parameters do not imply no_grad: waveform loss must reach h.
        idx = _sample_rate_index(self.decoder, h, sr_cond, self.output_sample_rate)
        return _run(self.decoder, h, 6, len(self.decoder.model), idx)

    def forward_from_latents(self, z: Tensor, sr_cond=None) -> dict[str, Tensor]:
        return teacher_trace(self.decoder, z, sr_cond, output_sample_rate=self.output_sample_rate)

    def forward_latents(self, z: Tensor, sr_cond=None) -> tuple[Tensor, Tensor]:
        result = self.forward_from_latents(z, sr_cond)
        return result["waveform"], result["group_output"]

    def forward(self, z: Tensor, sr_cond=None) -> Tensor:
        return self.forward_latents(z, sr_cond)[0]

    forward_group = group_from_input
    forward_suffix = suffix_from_group
    decode = forward

    def group_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [(name, param) for name, param in self.decoder.named_parameters() if name.startswith(GROUP_PREFIXES)]

    named_trainable_group_parameters = group_named_parameters

    def trainable_group_parameters(self) -> list[nn.Parameter]:
        return [param for _, param in self.group_named_parameters()]

    def group_state_dict(self) -> OrderedDict[str, Tensor]:
        """Detached group-only tensors; clone before retaining across updates."""
        return OrderedDict((name, tensor.detach()) for name, tensor in self.decoder.state_dict().items() if name.startswith(GROUP_PREFIXES))

    @torch.no_grad()
    def load_group_state_dict(self, state: Mapping[str, Tensor]) -> None:
        expected = self.group_state_dict()
        if set(state) != set(expected):
            raise ValueError("Group checkpoint keys do not match this architecture")
        for name, tensor in state.items():
            if not isinstance(tensor, Tensor) or tensor.shape != expected[name].shape or tensor.dtype != expected[name].dtype:
                raise ValueError(f"Invalid group checkpoint tensor: {name}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Nonfinite group checkpoint tensor: {name}")
        for name, tensor in state.items():
            expected[name].copy_(tensor)


def build_student(
    teacher_decoder: nn.Module, stage2_indices: Iterable[int] | Tensor,
    stage3_indices: Iterable[int] | Tensor, *, output_sample_rate: int = 48000,
) -> CompressedDecoderGroup:
    """Select internal coordinates jointly; identity selections make a control.

    The caller must select indices using fitting data only. They are recorded in
    ``student.selections``. No channel ranking or model fitting occurs here.
    """
    _validate_decoder(teacher_decoder)
    index2 = _indices(stage2_indices, teacher_decoder.model[3].block[1].out_channels, "stage2_indices")
    index3 = _indices(stage3_indices, teacher_decoder.model[4].block[1].out_channels, "stage3_indices")
    outer_in = torch.arange(teacher_decoder.model[3].input_channels)
    outer_out = torch.arange(teacher_decoder.model[5].block[1].out_channels)
    decoder = clone_teacher(teacher_decoder)
    # New module constructors initialize discarded random weights on CPU. Do not
    # let this deterministic migration consume the caller's training RNG stream.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        for index, ins, outs in ((3, outer_in, index2), (4, index2, index3), (5, index3, outer_out)):
            source = teacher_decoder.model[index]
            identity = torch.equal(ins, torch.arange(source.input_channels)) and torch.equal(outs, torch.arange(source.block[1].out_channels))
            if not identity:
                decoder.model[index] = _narrow_stage(source, ins, outs)
                if decoder.sr_bin_boundaries is not None:
                    decoder.sr_cond_model[index] = _narrow_condition(teacher_decoder.sr_cond_model[index], ins)
    return CompressedDecoderGroup(
        decoder, {"stage2_indices": index2.tolist(), "stage3_indices": index3.tolist()}, output_sample_rate,
    )


capture_teacher_boundaries = teacher_trace


def trainable_group_parameters(student: CompressedDecoderGroup) -> list[nn.Parameter]:
    return student.trainable_group_parameters()


def named_trainable_group_parameters(student: CompressedDecoderGroup) -> list[tuple[str, nn.Parameter]]:
    return student.group_named_parameters()
