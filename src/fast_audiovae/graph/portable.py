"""Switch audited custom-op domains/guards without rebuilding model arithmetic.

Requires the actual portable CPU library and refuses unsupported ISA/policy.
Preserves tensor protobufs and external-data files byte-for-byte; no inference.
"""
from __future__ import annotations
import argparse
import copy
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto as TP
OLD='venky.audio.cpu';DOMAIN='venky.audio.cpu.portable'
ALLOWED={'SnakeF32':3,'CausalDW7F32':3,'CausalDW7SnakeF32':5,'PhaseSumBiasInterleaveF32':3,'CombinedPhaseSumBiasInterleaveF32':2}


def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def capabilities(library):
    p=Path(library).resolve();lib=ctypes.CDLL(str(p))
    for name in ('ncc_abi_version','ncc_compiled_tile','ncc_selected_backend'):
        fn=getattr(lib,name);fn.argtypes=[];fn.restype=ctypes.c_uint32
    for name in ('ncc_backend_available','ncc_vector_sine_available'):
        fn=getattr(lib,name);fn.argtypes=[ctypes.c_int32];fn.restype=ctypes.c_int32
    lib.ncc_capabilities.argtypes=[];lib.ncc_capabilities.restype=ctypes.c_uint64
    lib.ncc_portable_build_id.argtypes=[];lib.ncc_portable_build_id.restype=ctypes.c_char_p
    lib.ncc_snake_math_name.argtypes=[ctypes.c_int32];lib.ncc_snake_math_name.restype=ctypes.c_char_p
    if lib.ncc_abi_version()!=1:raise ValueError('Native ABI must be1')
    return lib,{'library':str(p),'sha256':sha256(p),'abi':1,'tile':int(lib.ncc_compiled_tile()),
                'capability_bits':int(lib.ncc_capabilities()),'selected_backend':int(lib.ncc_selected_backend()),
                'sine_math':lib.ncc_snake_math_name(0).decode(),'build_id':lib.ncc_portable_build_id().decode()}


def convert(source,output,library):
    source=Path(source).resolve();output=Path(output).resolve()
    if source==output:raise ValueError('Preserve the source model with a different output path')
    lib,caps=capabilities(library)
    model=onnx.load(str(source),load_external_data=False)
    if model.functions or model.graph.sparse_initializer or any(a.type in (onnx.AttributeProto.GRAPH,onnx.AttributeProto.GRAPHS) for n in model.graph.node for a in n.attribute):
        raise ValueError('Only flat dense graphs are supported')
    init={t.name:t for t in model.graph.initializer};inputs={v.name for v in model.graph.input}
    counts={};renamed=0
    def constant(name,shape):
        if name in inputs or name not in init:raise ValueError('Custom coefficients must be non-overridable initializers: '+name)
        t=init[name]
        if t.data_type!=TP.FLOAT or tuple(t.dims)!=tuple(shape):raise ValueError('Coefficient dtype/shape mismatch: '+name)
        a=numpy_helper.to_array(copy.deepcopy(t),base_dir=str(source.parent))
        if not np.isfinite(a).all():raise ValueError('Nonfinite custom coefficient: '+name)
    for node in model.graph.node:
        if node.domain not in (OLD,DOMAIN):continue
        if node.op_type not in ALLOWED or len(node.input)!=ALLOWED[node.op_type] or len(node.output)!=1:
            raise ValueError('Unsupported custom node: '+node.name)
        a={x.name:helper.get_attribute_value(x) for x in node.attribute}
        if len(a)!=len(node.attribute):raise ValueError('Duplicate node attributes')
        co=a.get('channels');base={'channels','native_abi','row_batches'}
        if type(co)!=int or not 0<co<=2**31-1 or a.get('native_abi')!=1 or type(a.get('row_batches'))!=int or a['row_batches']<0:
            raise ValueError('Invalid common custom attributes: '+node.name)
        phase='Phase' in node.op_type;snake='Snake' in node.op_type
        if phase:
            required=base|{'stride','previous_shift'}
            if a.get('stride') not in (2,5,6,8) or a.get('previous_shift')!=1:raise ValueError('Invalid phase attributes')
            constant(node.input[-1],(co,))
        else:
            required=base|{'backend'}
            backend=a.get('backend')
            if type(backend)!=int or not lib.ncc_backend_available(backend):raise ValueError('CPU backend unavailable')
            if node.op_type!='SnakeF32':
                required.add('dilation');d=a.get('dilation')
                if type(d)!=int or not 0<d<=(2**31-1)//6:raise ValueError('Invalid dilation')
                constant(node.input[1],(co,1,7));constant(node.input[2],(co,))
            if snake:
                guard='require_vforce' if node.domain==OLD else 'require_vector_sine'
                required.add(guard)
                if a.get(guard)!=1 or not lib.ncc_vector_sine_available(backend):raise ValueError('Explicit accurate vector-sine policy/backend required')
                constant(node.input[-2],(co,));constant(node.input[-1],(co,))
                if node.domain==OLD:
                    next(x for x in node.attribute if x.name==guard).name='require_vector_sine';renamed+=1
        if set(a)!=required:raise ValueError('Unexpected/missing custom attributes on '+node.name+': '+repr(set(a)^required))
        node.domain=DOMAIN;counts[node.op_type]=counts.get(node.op_type,0)+1
    if not counts:raise ValueError('Source has no supported custom nodes')
    for o in model.opset_import:
        if o.domain in (OLD,DOMAIN) and o.version!=1:raise ValueError('Custom opset must be1')
    others=[o for o in model.opset_import if o.domain not in (OLD,DOMAIN)]
    del model.opset_import[:];model.opset_import.extend(others);model.opset_import.append(helper.make_opsetid(DOMAIN,1))
    output.parent.mkdir(parents=True,exist_ok=True)
    external={}
    # The audited graphs keep external data only in dense initializers.
    for node in model.graph.node:
        for attr in node.attribute:
            ts=[attr.t] if attr.type==onnx.AttributeProto.TENSOR else list(attr.tensors) if attr.type==onnx.AttributeProto.TENSORS else []
            if any(t.data_location==TP.EXTERNAL for t in ts):raise ValueError('External Constant tensor unsupported')
    for t in model.graph.initializer:
        if t.data_location!=TP.EXTERNAL:continue
        fields={v.key:v.value for v in t.external_data};location=fields.get('location','');p=Path(location)
        if not location or p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe external data path')
        src=(source.parent/p).resolve();dst=(output.parent/p).resolve()
        if not src.is_file():raise FileNotFoundError(src)
        offset=int(fields.get('offset',0));length=int(fields.get('length',src.stat().st_size-offset))
        if offset<0 or length<0 or offset+length>src.stat().st_size:raise ValueError('External data bounds invalid')
        if location in external:continue
        digest=sha256(src);dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():
            if sha256(dst)!=digest:raise ValueError('Refusing to overwrite different external weights')
        else:shutil.copy2(src,dst)
        external[location]={'sha256':digest,'bytes':src.stat().st_size,'source':str(src),'copy':str(dst)}
    data=model.SerializeToString()
    if output.exists() and output.read_bytes()!=data:raise ValueError('Refusing to overwrite a different derived graph')
    output.write_bytes(data);onnx.checker.check_model(str(output),full_check=False)
    result={'source':str(source),'source_sha256':sha256(source),'output':str(output),'output_sha256':sha256(output),
            'domain':DOMAIN,'opset':1,'counts':counts,'renamed_snake_guards':renamed,'capabilities':caps,'external_data':external,
            'policy':'Only custom domain and sine guard renamed; tensors and all numerical nodes preserved; no model execution'}
    output.with_suffix(output.suffix+'.portable.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
