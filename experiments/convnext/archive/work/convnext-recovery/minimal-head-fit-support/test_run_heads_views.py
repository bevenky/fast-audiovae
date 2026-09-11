"""CPU smoke tests of the runner's actual view/replay definitions.

Load only these definitions from its AST so the pure CPU checks do not import
the remote-only context module or create an experiment directory.
"""
import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F


SOURCE = Path(__file__).parents[1] / "architecture-experiments/run_heads.py"
names = {"transform", "HeadContract", "OutputView", "features", "validation_score"}
nodes = [node for node in ast.parse(SOURCE.read_text()).body
         if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
namespace = {"__name__": __name__, "torch": torch, "nn": nn, "F": F,
             "dataclass": dataclass, "bounded_waveform": lambda x: x.clamp(-1, 1)}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)


class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Conv1d(3, 480, 1, bias=False)
        self.config = SimpleNamespace()

    @staticmethod
    def _waveform(value):
        return value.transpose(1, 2).reshape(value.shape[0], 1, -1)

    def forward(self, x):
        return self._waveform(self.output(x))

    def initial_state(self):
        return 0

    def forward_stream(self, x, state):
        return self(x), state + x.shape[-1]


@pytest.mark.parametrize("mode", ["raw", "clamp", "tanh"])
def test_wrapper_batch_functional_stream_and_hook_accounting(mode):
    base = Base().eval()
    x = torch.randn(1, 3, 7)
    view = namespace["OutputView"](base, "fixture", mode)
    observed = []
    hook = view.register_forward_hook(lambda module, args, value: observed.append(value))
    expected = namespace["transform"](base(x), mode)
    actual = view(x)
    hook.remove()
    assert len(observed) == 1 and torch.equal(actual, expected)
    state = view.initial_state()
    pieces = []
    for a, b in ((0, 2), (2, 6), (6, 7)):
        y, state = view.forward_stream(x[..., a:b], state)
        pieces.append(y)
    assert state == 7 and torch.cat(pieces, -1).shape == expected.shape
    # A different CPU Conv1d width can change FP32 accumulation order. Use the
    # existing streaming tolerance, while requiring exact counts separately.
    torch.testing.assert_close(torch.cat(pieces, -1), expected, rtol=0, atol=2e-6)
    assert set(view.state_dict()) == {"base.output.weight"}
    assert not base.training


def test_feature_replay_is_exact_and_readonly():
    base = Base().eval()
    x = torch.randn(1, 3, 5)
    weight = base.output.weight.detach().clone()
    rng = torch.random.get_rng_state().clone()
    h, audio = namespace["features"](base, x)
    assert torch.equal(h, x) and torch.equal(audio, base(x))
    assert torch.equal(weight, base.output.weight)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert not base.output._forward_pre_hooks and not base.training


def test_validation_score_retains_all_masked_samples():
    base = Base().eval()
    h = torch.randn(1, 3, 5)
    target = base(h).detach() * .9
    valid = torch.zeros_like(target, dtype=torch.bool)
    valid[..., 19:-17] = True
    quiet = valid.clone()
    quiet[..., 600:] = False
    bank = [{"h": h, "target": target, "valid": valid, "quiet": quiet}]
    with torch.no_grad():
        result = namespace["validation_score"](base.output.weight[..., 0], "raw", bank, "cpu")
        error = base(h).double() - target.double()
    assert result["valid_samples"] == int(valid.sum())
    assert result["quiet_samples"] == int(quiet.sum())
    assert result["mse"] == float(error[valid].square().mean())
    assert result["quiet_mse"] == float(error[quiet].square().mean())
