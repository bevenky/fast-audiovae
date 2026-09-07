"""Explicit allowlist-only FP32 W@X rewrite; preserves original graph/weights."""
import copy,hashlib,json
from pathlib import Path
import onnx
from onnx import helper as h,TensorProto as TP
from .elementwise import copy_external_data
PIN='25cad99a6840855ade0a49871197f48ee0e1d317';DOMAIN='venky.audio.cpu.aocl.rows'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rewrite(source,output,nodes,row_tiles):
 source=Path(source).resolve();output=Path(output).resolve()
 if source==output:raise ValueError('Source must be preserved')
 if not 1<=row_tiles<=64:raise ValueError('row_tiles outside1..64')
 if len(nodes)!=len(set(nodes)) or not nodes:raise ValueError('Nonempty unique node allowlist required')
 model=onnx.load(str(source),load_external_data=False)
 if model.functions or model.graph.sparse_initializer or any(a.type in (onnx.AttributeProto.GRAPH,onnx.AttributeProto.GRAPHS) for n in model.graph.node for a in n.attribute):raise ValueError('Flat dense graph required')
 inits={t.name:t for t in model.graph.initializer};overridable={v.name for v in model.graph.input}
 info={v.name:v for v in [*model.graph.value_info,*model.graph.input,*model.graph.output]}
 selected=set(nodes);found=set();records=[]
 for node in model.graph.node:
  if node.name not in selected:continue
  if node.name in found:raise ValueError('Duplicate selected node name')
  found.add(node.name)
  if node.domain not in ('','ai.onnx') or node.op_type!='MatMul' or len(node.input)!=2 or len(node.output)!=1 or node.attribute:raise ValueError('Selected node is not ordinary attribute-free MatMul: '+node.name)
  w=inits.get(node.input[0]);x=info.get(node.input[1])
  if w is None or w.name in overridable or w.data_type!=TP.FLOAT or len(w.dims)!=2 or min(w.dims)<=0 or x is None:raise ValueError('Immutable FP32 W and annotated X required: '+node.name)
  m,k=w.dims;t=x.type.tensor_type;d=t.shape.dim
  if t.elem_type!=TP.FLOAT or len(d)!=3 or not d[1].HasField('dim_value') or d[1].dim_value!=k:raise ValueError('Expected FP32 X[B,K,T]: '+node.name)
  replacement=h.make_node('PackedRowsMatMulF32',list(node.input),list(node.output),name=node.name,domain=DOMAIN,rows=m,inner=k,row_tiles=row_tiles,native_abi=1,aocl_pin=PIN)
  records.append({'node':node.name,'weight':w.name,'M':m,'K':k,'row_tiles':row_tiles});node.CopyFrom(replacement)
 if found!=selected:raise ValueError('Allowlist names not found: '+repr(selected-found))
 if any(o.domain==DOMAIN for o in model.opset_import):raise ValueError('Source already imports AOCL domain')
 model.opset_import.append(h.make_opsetid(DOMAIN,1));output.parent.mkdir(parents=True,exist_ok=True)
 external=copy_external_data(model,source.parent,output.parent)
 data=model.SerializeToString()
 if output.exists() and output.read_bytes()!=data:raise ValueError('Refusing to overwrite different graph')
 output.write_bytes(data);onnx.checker.check_model(str(output),full_check=False)
 result={'source':str(source),'source_sha256':sha(source),'output':str(output),'output_sha256':sha(output),'allowlist':nodes,'records':records,'external_data':external,'pin':PIN,'weights':'Unmodified','scope':'Graph rewrite only; no inference'}
 output.with_suffix(output.suffix+'.aocl.json').write_text(json.dumps(result,indent=2)+'\n');return result

SELECTED_NODES = ['ncc_up_1_split_matmul_current_node', 'ncc_up_1_split_matmul_previous_unshifted_node', 'node_conv1d_3__bct_mm', 'node_conv1d_5__bct_mm', 'node_conv1d_7__bct_mm', 'ncc_up_2_split_matmul_current_node', 'ncc_up_2_split_matmul_previous_unshifted_node']
