"""Equivalent FP32 transpose forms for targeted numerical diagnosis."""
import types


def apply_paired_projection(model, stages=None):
    """One dot product covers current and preceding input for each output phase.

    The V3 form rounds two independent channel reductions, then sums them.
    This form submits both contributions as one channel reduction. Weights and
    bias remain literal FP32; activated-input histories remain identical.
    """
    import torch

    def decode(self, x, history):
        length = x.shape[-1]
        channels, stride = self.weight.shape[1], self.stride
        if length == 0:
            return x.new_empty((1, channels, 0)), history.clone()
        previous = torch.cat((history, x[..., :-1]), dim=-1)
        features = torch.cat((x[0], previous[0]), dim=0)
        projection = torch.mm(self._paired_weight, features)
        y = projection.reshape(channels, stride, length).permute(0, 2, 1)
        y = y.reshape(1, channels, length*stride) + self.bias[None, :, None]
        return y.contiguous(), x[..., -1:].clone()

    count = 0
    for index, stage in enumerate(model.stages):
        if stages is not None and index not in stages:
            continue
        module = stage.transpose
        weight, stride = module.weight, module.stride
        assert weight.dtype == torch.float32 and weight.shape[-1] == 2*stride
        current = weight[..., :stride].permute(1, 2, 0)
        previous = weight[..., stride:].permute(1, 2, 0)
        packed = torch.cat((current, previous), dim=-1).reshape(weight.shape[1]*stride, 2*weight.shape[0]).contiguous()
        module.register_buffer('_paired_weight', packed, persistent=False)
        module.decode = types.MethodType(decode, module)
        count += 1
    return count


def verify_cpu_algebra():
    import torch
    from types import SimpleNamespace
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(408)
    rows = []
    for stride in (2, 5, 6, 8):
        layer = torch.nn.Module()
        layer.stride = stride
        layer.register_buffer('weight', torch.randn(3, 2, 2*stride, generator=generator)*.1)
        layer.register_buffer('bias', torch.tensor([.25, -.125]))
        model = SimpleNamespace(stages=[SimpleNamespace(transpose=layer)])
        assert apply_paired_projection(model) == 1
        for length in (0, 1, 2, 5):
            x = torch.randn(1, 3, length, generator=generator)
            history = torch.randn(1, 3, 1, generator=generator)
            actual, state = layer.decode(x, history)
            if length:
                reference = torch.nn.functional.conv_transpose1d(torch.cat((history, x), -1), layer.weight, layer.bias, stride=stride)
                reference = reference[..., stride:stride+length*stride]
                torch.testing.assert_close(actual, reference, atol=2e-7, rtol=2e-6)
                assert torch.equal(state, x[..., -1:])
                error = (actual-reference).abs().max().item()
            else:
                assert actual.shape == (1, 2, 0) and torch.equal(state, history)
                error = 0.
            rows.append(dict(stride=stride, frames=length, max_abs=error))
    return rows


if __name__ == '__main__':
    import json
    print(json.dumps(verify_cpu_algebra(), indent=2))
