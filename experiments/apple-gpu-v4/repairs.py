"""Controlled channel-order and first-upsampler fallback interventions."""
import types


def apply_interleaved_projection(model):
    import torch

    def decode(self, x, history):
        length = x.shape[-1]
        channels, stride = self.weight.shape[1], self.stride
        if length == 0:
            return x.new_empty((1, channels, 0)), history.clone()
        previous = torch.cat((history, x[..., :-1]), dim=-1)
        # Adjacent reduction coordinates hold one channel's two time taps.
        features = torch.stack((x[0], previous[0]), dim=1).reshape(2*x.shape[1], length)
        p = torch.mm(self._interleaved_weight, features)
        y = p.reshape(channels, stride, length).permute(0, 2, 1).reshape(1, channels, length*stride)
        return (y+self.bias[None, :, None]).contiguous(), x[..., -1:].clone()

    count = 0
    for stage in model.stages:
        module = stage.transpose
        weight, stride = module.weight, module.stride
        assert weight.dtype == torch.float32 and weight.shape[-1] == 2*stride
        current = weight[..., :stride].permute(1, 2, 0)
        previous = weight[..., stride:].permute(1, 2, 0)
        packed = torch.stack((current, previous), dim=-1).reshape(weight.shape[1]*stride, 2*weight.shape[0]).contiguous()
        module.register_buffer('_interleaved_weight', packed, persistent=False)
        module.decode = types.MethodType(decode, module)
        count += 1
    return count


def apply(model, variant):
    import candidates
    if variant == 'hybrid':
        import torch
        def decode(self, x, history):
            length = x.shape[-1]
            channels, stride = self.weight.shape[1], self.stride
            if length == 0:
                return x.new_empty((1,channels,0)), history.clone()
            joined = torch.cat((history,x),dim=-1)
            p = torch.mm(self._v3_weight,joined[0]).reshape(channels,2*stride,length+1)
            y = (p[:,:stride,1:]+p[:,stride:,:-1]).permute(0,2,1).reshape(1,channels,length*stride)
            return (y+self.bias[None,:,None]).contiguous(), x[...,-1:].clone()
        for stage in model.stages[1:]:
            module = stage.transpose; weight = module.weight
            packed = weight.permute(1,2,0).reshape(weight.shape[1]*weight.shape[2],weight.shape[0]).contiguous()
            module.register_buffer('_v3_weight',packed,persistent=False)
            module.decode = types.MethodType(decode,module)
        assert candidates.apply_paired_projection(model,stages={0}) == 1
        return 6
    if variant == 'interleaved':
        return apply_interleaved_projection(model)
    if variant == 'retain_first':
        return candidates.apply_paired_projection(model, stages={1,2,3,4,5})
    raise ValueError(variant)


def verify_cpu_algebra():
    import torch
    from types import SimpleNamespace
    torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(511)
    rows = []
    for stride in (2,5,6,8):
        layer = torch.nn.Module(); layer.stride = stride
        layer.register_buffer('weight', torch.randn(3,2,2*stride,generator=gen)*.1)
        layer.register_buffer('bias', torch.tensor([.25,-.125]))
        apply_interleaved_projection(SimpleNamespace(stages=[SimpleNamespace(transpose=layer)]))
        for length in (0,1,2,5):
            x = torch.randn(1,3,length,generator=gen); h = torch.randn(1,3,1,generator=gen)
            y, state = layer.decode(x,h)
            if length:
                ref = torch.nn.functional.conv_transpose1d(torch.cat((h,x),-1),layer.weight,layer.bias,stride=stride)
                ref = ref[...,stride:stride+length*stride]
                torch.testing.assert_close(y,ref,atol=2e-7,rtol=2e-6)
                assert torch.equal(state,x[...,-1:])
                err = (y-ref).abs().max().item()
            else:
                assert y.shape==(1,2,0) and torch.equal(state,h); err=0.
            rows.append(dict(stride=stride,frames=length,max_abs=err))
    return rows


if __name__ == '__main__':
    import json
    print(json.dumps(verify_cpu_algebra(),indent=2))
