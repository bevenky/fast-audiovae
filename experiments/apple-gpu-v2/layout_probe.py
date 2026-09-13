from experiment import *
torch.set_num_threads(1);torch.set_num_interop_threads(1)
r={}
with torch.inference_mode():
 m=build('before_compile'); z=torch.from_numpy(cases()['English'][...,:1].copy()).to('mps');y,s=m.decode(z,{})
 for k,v in s.items():
  if not(v.dtype==torch.float32 and v.device==z.device and v.layout==torch.strided and tuple(v.shape)==m.state_shapes[k] and v.is_contiguous()):
   r[k]=dict(shape=list(v.shape),expected=list(m.state_shapes[k]),strides=list(v.stride()),dtype=str(v.dtype),device=str(v.device),layout=str(v.layout),contiguous=v.is_contiguous())
 torch.mps.synchronize()
print(json.dumps(r,indent=2))
