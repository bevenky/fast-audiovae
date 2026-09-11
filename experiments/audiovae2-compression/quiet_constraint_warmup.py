"""Warm Q's gradient-bearing forward without changing its quiet constraints.

This deliberately permits execution-cache warming, but never backward, parameter
updates, changed acceptance tolerances, or a cached success after failed parity.
"""
from __future__ import annotations

import math

import torch

import quiet_projected_update as q

VERSION = 'audiovae2_quiet_constraint_warmup_v1'


def _shape_key(model, entry):
    x = entry['group_input']
    if (not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[0] != 1
            or x.dtype != torch.float32 or x.requires_grad):
        raise ValueError('Warmup requires detached singleton FP32 group inputs')
    # A cache belongs to this model and execution geometry, never another cut.
    return (id(model), tuple(x.shape), tuple(x.stride()), str(x.device), str(x.dtype),
            tuple(m.training for m in model.modules()),
            tuple(p.requires_grad for p in model.parameters()))


def _capture(model, entry, *, grad):
    with torch.set_grad_enabled(grad):
        prediction = model.suffix_from_group(model.group_from_input(entry['group_input']))
        if grad and not prediction.requires_grad:
            raise RuntimeError('Quiet warmup did not exercise the gradient-bearing suffix')
        if not bool(torch.isfinite(prediction).all()):
            raise RuntimeError('Nonfinite quiet warmup prediction')
        rows = q.window_excesses(prediction, entry)
        signature = tuple((r['cohort'], r['window_index'],
                           tuple(float(v.detach()) for v in r['values']), r['passed'])
                          for r in rows)
        if not all(math.isfinite(v) for row in signature for v in row[2]):
            raise RuntimeError('Nonfinite quiet warmup constraint')
        # Neither a graph nor a mutable cached weight may escape this capture.
        return prediction.detach().clone(), signature


def _compare(left, right):
    a, sa = left
    b, sb = right
    same_geometry = a.shape == b.shape and a.dtype == b.dtype and a.device == b.device
    wave_exact = same_geometry and torch.equal(a, b)
    max_abs = float((a.double() - b.double()).abs().max()) if same_geometry and a.numel() else None
    return {'waveform_exact': wave_exact, 'constraints_exact': sa == sb,
            'waveform_max_abs': max_abs,
            'passed': wave_exact and sa == sb}


def warm_quiet_entries(model, teacher, entries, warmed_shapes, *, optimizer=None):
    """Warm new shapes, then check EVERY entry by grad/no-grad/repeated grad.

    ``entries`` are Q's existing fixed teacher-window dictionaries. Three
    enable-grad forwards warm a representative of each new shape. Subsequent
    verification includes other inputs sharing that shape, not just its first
    representative. Already-cached shapes skip warming but still verify supplied
    entries. Only complete success publishes new keys into ``warmed_shapes``.

    The existing diagnostic guard restores RNG, modes, requires-grad and gradient
    slots and rejects/restores raw model or optimizer mutation. JIT executor and
    legacy weight-normalization forward caches intentionally are not rewound.
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError('Inference mode cannot warm a gradient-bearing forward')
    entries = list(entries)
    keys = [_shape_key(model, entry) for entry in entries]
    pending = {}
    for key, entry in zip(keys, entries):
        if key not in warmed_shapes:
            pending.setdefault(key, entry)
    report = {'version': VERSION, 'entries_checked': len(entries),
              'new_shapes': len(pending), 'warmup_grad_forwards': 3 * len(pending),
              'verification_forwards': 3 * len(entries), 'backwards': 0,
              'cold_comparisons': [], 'entries': []}
    with q.replay.diagnostic_state_guard(model, teacher, optimizer):
        for key, entry in pending.items():
            first = _capture(model, entry, grad=True)
            second = _capture(model, entry, grad=True)
            third = _capture(model, entry, grad=True)
            report['cold_comparisons'].append({
                'shape': list(key[1]), 'first_vs_third': _compare(first, third),
                'second_vs_third': _compare(second, third)})
            del first, second, third
        for index, entry in enumerate(entries):
            first = _capture(model, entry, grad=True)
            no_grad = _capture(model, entry, grad=False)
            repeat = _capture(model, entry, grad=True)
            comparisons = {'entry_index': index,
                           'grad_vs_no_grad': _compare(first, no_grad),
                           'grad_vs_repeated_grad': _compare(first, repeat)}
            report['entries'].append(comparisons)
            if not all(comparisons[k]['passed'] for k in ('grad_vs_no_grad', 'grad_vs_repeated_grad')):
                raise RuntimeError(f'Quiet warmup parity failed for entry {index}: {comparisons}')
            del first, no_grad, repeat
    # The guard can raise on exit; never mark a mutated or failed shape as warm.
    warmed_shapes.update(pending)
    report['preserved'] = True
    report['passed'] = True
    return report
