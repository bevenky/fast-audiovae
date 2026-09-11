"""Independent, nested width cuts with inherited recovered effective weights.

Original channel IDs are provenance, not a claim that adapted channels still
equal teacher activations. The unchanged original teacher remains the target.
This module neither ranks channels from data nor loads/migrates Adam moments.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence

import torch

import group_model as gm


WIDTH_SCHEDULE = ((512, 256), (384, 256), (384, 192), (256, 192), (256, 128))
KEYS = ("stage2_indices", "stage3_indices")
OPTIMIZER_POLICY = {
    "name": "AdamW", "lr": 3e-5, "betas": (0.9, 0.99), "eps": 1e-8,
    "weight_decay": 0.0, "state_at_each_cut": "fresh",
    "reason": "Effective-weight slicing changes the weight-normalization coordinates; moments are not sliced.",
}


def _selection(selection, original_widths):
    if not isinstance(selection, Mapping) or any(key not in selection for key in KEYS):
        raise ValueError("Selection requires stage2_indices and stage3_indices in original coordinates")
    return {key: gm._indices(selection[key], width, key).tolist()
            for key, width in zip(KEYS, original_widths)}


def validate_schedule(steps, final_selection=None, *, original_widths=(512, 256), widths=WIDTH_SCHEDULE):
    """Return detached selections after checking the fixed one-boundary cuts.

    Explicit alternate widths support small CPU fixtures, not adaptive ranking.
    Unchanged boundaries retain their exact coordinate order across a cut.
    """
    widths = tuple(tuple(x) for x in widths)
    if not widths or widths[0] != tuple(original_widths) or len(steps) != len(widths):
        raise ValueError("Schedule length or original widths disagree")
    result = [_selection(step, original_widths) for step in steps]
    if result[0] != {key: list(range(n)) for key, n in zip(KEYS, original_widths)}:
        raise ValueError("The initial schedule entry must be the original ordered full-width teacher")
    for index, (step, width) in enumerate(zip(result, widths)):
        if tuple(len(step[key]) for key in KEYS) != width:
            raise ValueError(f"Wrong widths at schedule entry {index}")
        if index:
            _relative_selection(result[index - 1], step)
    if final_selection is not None and result[-1] != _selection(final_selection, original_widths):
        raise ValueError("Final selection differs from the preserved candidate coordinates")
    return result


def make_schedule(final_stage2_ids, final_stage3_ids, *, stage2_order, stage3_order,
                  widths=WIDTH_SCHEDULE):
    """Use caller-supplied fitting-only rankings for the intermediate supersets.

    The complete orders identify the 384/192 supersets; the final selections
    keep their supplied order and must be contained in these supersets. No
    fallback selection, calibration or ranking is performed here.
    """
    widths = tuple(tuple(x) for x in widths)
    if len(widths) != 5:
        raise ValueError("Expected the five-entry gradual schedule")
    original = widths[0]
    orders = [gm._indices(order, width, "ranking").tolist()
              for order, width in zip((stage2_order, stage3_order), original)]
    if any(len(order) != width for order, width in zip(orders, original)):
        raise ValueError("Each ranking must contain every original channel exactly once")
    final = _selection(dict(zip(KEYS, (final_stage2_ids, final_stage3_ids))), original)
    middle = {KEYS[0]: sorted(orders[0][:widths[1][0]]),
              KEYS[1]: sorted(orders[1][:widths[2][1]])}
    steps = [dict(zip(KEYS, (list(range(original[0])), list(range(original[1]))))),
             {KEYS[0]: middle[KEYS[0]], KEYS[1]: list(range(original[1]))},
             middle,
             {KEYS[0]: final[KEYS[0]], KEYS[1]: middle[KEYS[1]]},
             final]
    return validate_schedule(steps, final, original_widths=original, widths=widths)


def _relative_selection(previous, following):
    relative, cuts = {}, 0
    for key in KEYS:
        before, after = previous[key], following[key]
        positions = {channel: index for index, channel in enumerate(before)}
        if len(after) > len(before) or not set(after).issubset(positions):
            raise ValueError(f"{key} is not a nested narrowing of the recovered model")
        if len(after) == len(before):
            if after != before:
                raise ValueError("An unchanged channel boundary must retain its ordering")
        else:
            cuts += 1
        relative[key] = [positions[channel] for channel in after]
    if cuts != 1:
        raise ValueError("Each progressive cut must narrow exactly one channel boundary")
    return relative


def _metadata(model, selection, original_widths, outer_widths, history):
    # These ordinary Python values add no modules, operations or state tensors.
    model.selections = copy.deepcopy(selection)
    model.progressive_selection = copy.deepcopy(selection)
    model.progressive_provenance = {
        "format": "progressive_group_v1", "original_widths": list(original_widths),
        "outer_widths": list(outer_widths), "selections": copy.deepcopy(history),
        "optimizer_policy": copy.deepcopy(OPTIMIZER_POLICY),
        "target": "original unmodified teacher, never the previous compressed model",
    }
    return model


def initialize_from_teacher(teacher_decoder, selection, *, output_sample_rate=48000):
    """Clone the original decoder, or initialize the first cut from its weights.

    An identity selection is the no-pruning control. The caller authenticates
    the original teacher checkpoint/source and seals the schedule separately.
    """
    gm._validate_decoder(teacher_decoder)
    original = tuple(teacher_decoder.model[i].block[1].out_channels for i in (3, 4))
    selected = _selection(selection, original)
    outer = (teacher_decoder.model[3].input_channels, teacher_decoder.model[5].block[1].out_channels)
    model = gm.build_student(teacher_decoder, selected[KEYS[0]], selected[KEYS[1]],
                             output_sample_rate=output_sample_rate)
    return _metadata(model, selected, original, outer, [selected])


def migrate(previous_model, next_selection):
    """Create an independent next cut from the recovered model, with no update.

    Original IDs are translated to the previous decoder's physical positions.
    build_student couples both ends of each changed boundary, including Snake,
    depthwise/pointwise weights and sample-rate conditioning. Effective WN
    tensors are sliced and reparameterized, rather than raw g/v or Adam state.
    """
    provenance = getattr(previous_model, "progressive_provenance", None)
    if not isinstance(provenance, Mapping) or provenance.get("format") != "progressive_group_v1":
        raise ValueError("Previous model lacks original-coordinate progressive provenance")
    original = tuple(provenance["original_widths"])
    previous = _selection(previous_model.progressive_selection, original)
    if previous_model.selections != previous or provenance["selections"][-1] != previous:
        raise ValueError("Previous model's selection provenance is inconsistent")
    gm._validate_decoder(previous_model.decoder)
    observed = tuple(previous_model.decoder.model[i].block[1].out_channels for i in (3, 4))
    if observed != tuple(len(previous[key]) for key in KEYS):
        raise ValueError("Previous decoder widths differ from recorded original channel IDs")
    outer = (previous_model.decoder.model[3].input_channels,
             previous_model.decoder.model[5].block[1].out_channels)
    if list(outer) != provenance["outer_widths"]:
        raise ValueError("The fixed external group interface changed")
    selected = _selection(next_selection, original)
    relative = _relative_selection(previous, selected)
    model = gm.build_student(previous_model.decoder, relative[KEYS[0]], relative[KEYS[1]],
                             output_sample_rate=previous_model.output_sample_rate)
    return _metadata(model, selected, original, outer, [*provenance["selections"], selected])


def fresh_optimizer(model, lr=3e-5):
    """New AdamW parameters/moments after a cut; the caller preserves the old one."""
    if not isinstance(lr, (int, float)) or isinstance(lr, bool) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Learning rate must be positive and finite")
    expected = model.trainable_group_parameters()
    if not expected or any(not parameter.requires_grad for parameter in expected):
        raise ValueError("All group parameters must remain trainable")
    if {id(p) for p in model.parameters() if p.requires_grad} != {id(p) for p in expected}:
        raise ValueError("Only the existing group and its conditioning may be trainable")
    return torch.optim.AdamW(expected, lr=lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0)
