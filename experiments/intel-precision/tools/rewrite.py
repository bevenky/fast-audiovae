"""Derive mixed matrix precision variants from the exact accepted Intel graph."""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper
SOURCE_SHA='1ddb6dcc2b0c3ccea90d309f6ebec10eb12e844fb6320cd2d525dbfd01d4cf29'
MATRIX_DOMAIN='fast.audiovae.precision.matrix.experimental'
STAGE_DOMAIN='fast.audiovae.precision.stage.experimental'
UP_DOMAIN='fast.audiovae.precision.upsample.experimental'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def attrs(n):return {a.name:helper.get_attribute_value(a) for a in n.attribute}
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 assert sha(a.source)==SOURCE_SHA
 source=onnx.load(a.source);assert not source.functions
 constants={x.name:x for x in source.graph.initializer}
 manifest={'source':str(a.source.resolve()),'source_sha256':SOURCE_SHA,'variants':{},'gpu_used':False}
 for name,mode,min_dim in [('int8_all',8,0),('int8_large',8,128),('fp16_all',16,0)]:
  m=copy.deepcopy(source);nodes=[];changed=[]
  for n in m.graph.node:
   at=attrs(n)
   is_matrix=(not n.domain and n.op_type=='MatMul') or (n.domain=='venky.audio.intel.decoder.experimental' and n.op_type=='IntelPlainMatMulF32')
   if is_matrix:
    assert len(n.input)==2 and n.input[0] in constants and len(n.output)==1,n.name
    w=numpy_helper.to_array(constants[n.input[0]]);assert w.dtype==np.float32 and w.ndim==2 and np.isfinite(w).all()
    rows,cols=w.shape
    if min(rows,cols)>=min_dim:
     nodes.append(helper.make_node('PrecisionMatMulF32',list(n.input),list(n.output),name=n.name,domain=MATRIX_DOMAIN,native_abi=1,M=rows,K=cols,precision_mode=mode,backend=1,shards=2))
     changed.append({'node':n.name,'kind':'standalone','weight_shape':[rows,cols]});continue
   elif n.domain=='fast.audiovae.stage.experimental' and n.op_type=='StageStackF32':
    if at['channels']>=min_dim:
     new=copy.deepcopy(n);new.domain=STAGE_DOMAIN;new.attribute.append(helper.make_attribute('precision_mode',mode));nodes.append(new)
     changed.append({'node':n.name,'kind':'three_residual_matrices','channels':at['channels']});continue
   elif n.domain=='fast.audiovae.upsample.experimental' and n.op_type=='UpsampleStageF32':
    assert min_dim<=128
    new=copy.deepcopy(n);new.domain=UP_DOMAIN;new.attribute.append(helper.make_attribute('precision_mode',mode));nodes.append(new)
    changed.append({'node':n.name,'kind':'two_projections_three_residual_matrices','channels':128});continue
   nodes.append(copy.deepcopy(n))
  assert sum(v['kind']=='standalone' for v in changed)==(17 if not min_dim else 14),(name,changed)
  assert sum(v['kind']=='three_residual_matrices' for v in changed)==(3 if not min_dim else 1)
  assert sum(v['kind']=='two_projections_three_residual_matrices' for v in changed)==1
  del m.graph.node[:];m.graph.node.extend(nodes)
  for domain in (MATRIX_DOMAIN,STAGE_DOMAIN,UP_DOMAIN):m.opset_import.append(helper.make_opsetid(domain,1))
  assert len(m.graph.initializer)==len(source.graph.initializer)
  assert all(x.SerializeToString()==y.SerializeToString() for x,y in zip(m.graph.initializer,source.graph.initializer))
  onnx.checker.check_model(m,check_custom_domain=False)
  a.output_dir.mkdir(parents=True,exist_ok=True);target=a.output_dir/(name+'.onnx');assert not target.exists()
  onnx.save(m,target)
  manifest['variants'][name]={'path':str(target.resolve()),'sha256':sha(target),'precision_mode':mode,'changes':changed,'nodes':len(nodes),'constants_unchanged':True,'scope':'Matrix operands only; FP32 accumulation/output for FP16, INT32 accumulation then FP32 output for INT8. Snake, depthwise, bias/residual and final waveform convolution retain FP32.'}
 (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
if __name__=='__main__':main()
