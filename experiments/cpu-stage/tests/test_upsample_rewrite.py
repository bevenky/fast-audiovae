"""Static upsample rewrite tests. No native loading, sessions, or inference."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('upsample_rewrite',ROOT/'upsample/rewrite.py')
rewrite_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(rewrite_module)


def fixture(time='T', batch=1):
    tensors=[]
    def tensor(name, dims):
        values=np.zeros(dims,np.float32)
        values.reshape(-1)[0]=-0.0
        tensors.append(numpy_helper.from_array(values,name));return name
    wc=tensor('wc',(256,256));wp=tensor('wp',(256,256));pb=tensor('phase_bias',(128,))
    constants=[]
    dims=([128,1,7],[128],[128],[128],[128],[128],[128,128],[128])
    for unit in range(3):
        constants.extend(tensor(f'u{unit}_{i}',shape) for i,shape in enumerate(dims))
    nodes=[helper.make_node('MatMul',[wc,'x'],['current'],name='current_mm'),
        helper.make_node('MatMul',[wp,'x'],['previous'],name='previous_mm'),
        helper.make_node('PhaseSumBiasInterleaveF32',['current','previous',pb],['phase'],name='phase',
            domain=rewrite_module.NATIVE,native_abi=1,channels=128,stride=2,previous_shift=1,row_batches=0),
        helper.make_node('StageStackF32',['phase',*constants],['y'],name='sp_stage_c128',
            domain=rewrite_module.STAGE,native_abi=1,channels=128,tile_time=128,segments=2,backend=5,matrix_mode=0,matrix_isa=512)]
    outtime=2*time if type(time) is int else 'twice_'+time
    model=helper.make_model(helper.make_graph(nodes,'upsampling_fixture',
        [helper.make_tensor_value_info('x',TensorProto.FLOAT,[batch,256,time])],
        [helper.make_tensor_value_info('y',TensorProto.FLOAT,[batch,128,outtime])],tensors),
        opset_imports=[helper.make_opsetid('',20),helper.make_opsetid(rewrite_module.NATIVE,1),
                      helper.make_opsetid(rewrite_module.STAGE,1)],ir_version=10)
    return model


class UpsampleRewriteTests(unittest.TestCase):
    def apply(self,model,**options):
        onnx.checker.check_model(model)
        result,report=rewrite_module.rewrite_model(model,**options)
        onnx.checker.check_model(result)
        return result,report

    def test_tiles_modes_and_input_contract(self):
        original=fixture();before=original.SerializeToString()
        for q in (64,128,256):
            for mode in (0,1):
                result,report=self.apply(original,tile_time=q,segments=4,projection_mode=mode)
                self.assertEqual(report['count'],1);self.assertEqual(len(result.graph.node),1)
                n=result.graph.node[0]
                self.assertEqual(n.op_type,'UpsampleStageF32');self.assertEqual(n.domain,rewrite_module.DOMAIN)
                self.assertEqual(list(n.input[:4]),['x','wc','wp','phase_bias'])
                self.assertEqual(list(n.input[4:]),list(original.graph.node[-1].input[1:]))
                self.assertEqual(list(n.output),['y']);self.assertEqual(len(n.input),28)
                a=rewrite_module.attrs(n)
                self.assertEqual((a['projection_mode'],a['projection_isa']),(mode,0 if mode else 512))
                self.assertEqual((a['backend'],a['matrix_mode'],a['matrix_isa']),(5,0,512))
                self.assertEqual([v.SerializeToString() for v in result.graph.initializer],
                                 [v.SerializeToString() for v in original.graph.initializer])
                self.assertEqual(original.SerializeToString(),before)

    def test_static_zero_and_symbolic_batch_shapes(self):
        for b,t in ((1,0),(1,17),('B','T')):
            result,_=self.apply(fixture(t,b));self.assertEqual(len(result.graph.node),1)

    def test_order_follows_phase_operands(self):
        model=fixture();model.graph.node[2].input[0]='previous';model.graph.node[2].input[1]='current'
        result,_=self.apply(model)
        self.assertEqual(list(result.graph.node[0].input[1:3]),['wp','wc'])

    def test_unselected_nodes_and_graph_outputs_preserved(self):
        model=fixture();extra=helper.make_node('Identity',['y'],['outside'],name='unselected')
        model.graph.node.append(extra)
        model.graph.output.append(helper.make_tensor_value_info('outside',TensorProto.FLOAT,[1,128,'twice_T']))
        result,_=self.apply(model)
        self.assertEqual(result.graph.node[-1].SerializeToString(),extra.SerializeToString())
        self.assertEqual([v.SerializeToString() for v in result.graph.output],
                         [v.SerializeToString() for v in model.graph.output])

    def test_intermediate_outputs_and_fanout_rejected(self):
        for value,c in (('current',256),('previous',256),('phase',128)):
            model=fixture();model.graph.output.append(helper.make_tensor_value_info(value,TensorProto.FLOAT,[1,c,'T']))
            with self.assertRaisesRegex(ValueError,'external consumer|graph output'):self.apply(model)
            model=fixture();model.graph.node.append(helper.make_node('Identity',[value],['leak'],name='leak'))
            with self.assertRaisesRegex(ValueError,'external consumer|graph output'):self.apply(model)

    def test_coefficients_must_be_finite_immutable_fp32_correct_shape(self):
        for name in ('wc','wp','phase_bias','u0_0','u1_6','u2_7'):
            for kind in ('nan','overridable','dtype','shape'):
                model=fixture();t=next(v for v in model.graph.initializer if v.name==name)
                if kind=='nan':
                    value=numpy_helper.to_array(t).copy();value.reshape(-1)[0]=np.nan;t.CopyFrom(numpy_helper.from_array(value,name))
                elif kind=='overridable':model.graph.input.append(helper.make_tensor_value_info(name,TensorProto.FLOAT,list(t.dims)))
                elif kind=='dtype':t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float16),name))
                else:
                    value=numpy_helper.to_array(t).reshape(-1);t.CopyFrom(numpy_helper.from_array(value,name))
                    if list(t.dims)==[128]:t.dims[0]=127;t.raw_data=t.raw_data[:-4]
                with self.subTest(name=name,kind=kind),self.assertRaises(ValueError):self.apply(model)

    def test_phase_and_stage_contracts_rejected(self):
        cases=[(2,'stride',5),(2,'previous_shift',0),(2,'native_abi',2),(2,'row_batches',-1),
               (3,'matrix_mode',1),(3,'matrix_isa',0),(3,'backend',4),(3,'segments',0),(3,'tile_time',65)]
        for index,name,value in cases:
            model=fixture()
            next(a for a in model.graph.node[index].attribute if a.name==name).i=value
            with self.subTest(name=name,value=value),self.assertRaises(ValueError):self.apply(model)

    def test_projection_kind_and_shared_input_required(self):
        model=fixture();model.graph.node[0].op_type='Gemm'
        with self.assertRaises(ValueError):self.apply(model)
        model=fixture();model.graph.input.append(helper.make_tensor_value_info('other',TensorProto.FLOAT,[1,256,'T']))
        model.graph.node[1].input[1]='other'
        with self.assertRaisesRegex(ValueError,'same activation'):self.apply(model)
        model=fixture();model.graph.node[2].input[1]='current'
        with self.assertRaisesRegex(ValueError,'distinct projections'):self.apply(model)

    def test_identity_reshape_aliases_are_proved_and_removed(self):
        model=fixture();shape=numpy_helper.from_array(np.array([0,0,-1],np.int64),'identity_shape')
        model.graph.initializer.append(shape)
        model.graph.node.insert(2,helper.make_node('Reshape',['current','identity_shape'],['current_view'],name='current_view'))
        model.graph.node[3].input[0]='current_view'
        model.graph.node.insert(4,helper.make_node('Reshape',['phase','identity_shape'],['phase_view'],name='phase_view'))
        model.graph.node[5].input[0]='phase_view'
        result,report=self.apply(model)
        self.assertEqual(len(result.graph.node),1);self.assertEqual(len(report['changes'][0]['removed_nodes']),6)
        for malformed in (np.array([0,128,-1],np.int64),np.array([0,-1,128],np.int64)):
            bad=copy.deepcopy(model);bad.graph.initializer[-1].CopyFrom(numpy_helper.from_array(malformed,'identity_shape'))
            with self.assertRaises(ValueError):self.apply(bad)

    def test_producer_shape_overrides_forged_value_info(self):
        model=fixture();model.graph.input[0].type.tensor_type.shape.dim[1].dim_value=128
        model.graph.value_info.append(helper.make_tensor_value_info('current',TensorProto.FLOAT,[1,256,'T']))
        with self.assertRaisesRegex(ValueError,'matrix shape|input'):self.apply(model)
        model=fixture();model.graph.value_info.append(helper.make_tensor_value_info('phase',TensorProto.FLOAT,[1,64,'twice_T']))
        with self.assertRaisesRegex(ValueError,'contradict'):self.apply(model)

    def test_dynamic_shape_program_alias(self):
        model=fixture(batch='B');model.graph.initializer.append(numpy_helper.from_array(np.array([0],np.int64),'axes'))
        model.graph.node.insert(0,helper.make_node('Shape',['x'],['x_shape'],name='get_shape'))
        model.graph.node.insert(3,helper.make_node('Reshape',['current','x_shape'],['current_view'],name='view'))
        model.graph.node[4].input[0]='current_view'
        result,_=self.apply(model)
        self.assertEqual(len(result.graph.node),2)
        self.assertEqual(result.graph.node[0].name,'get_shape')

    def test_parameter_and_collision_rejections(self):
        for options in ({'tile_time':65},{'tile_time':True},{'segments':0},{'segments':65},
                        {'projection_mode':2},{'projection_isa':0},{'projection_mode':1,'projection_isa':512}):
            with self.subTest(options=options),self.assertRaises(ValueError):self.apply(fixture(),**options)
        model=fixture();model.graph.node[0].name='upsample_stage_c128'
        with self.assertRaisesRegex(ValueError,'collision'):self.apply(model)
        result,_=self.apply(fixture())
        with self.assertRaisesRegex(ValueError,'Already'):self.apply(result)

    def test_file_audit_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source.onnx';output=root/'result.onnx'
            onnx.save_model(fixture(),source);before=source.read_bytes()
            report=rewrite_module.rewrite(source,output)
            self.assertEqual(source.read_bytes(),before)
            self.assertEqual(report['source_sha256'],hashlib.sha256(before).hexdigest())
            self.assertEqual(report['output_sha256'],rewrite_module.sha(output))
            self.assertTrue(output.with_suffix('.upsample.json').is_file())
            with self.assertRaises(ValueError):rewrite_module.rewrite(source,source)
            with self.assertRaises(ValueError):rewrite_module.rewrite(source,output)

    def test_external_coefficients_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source.onnx';output=root/'out/result.onnx'
            onnx.save_model(fixture(),source,save_as_external_data=True,all_tensors_to_one_file=True,location='coefficients.data',size_threshold=0)
            before=onnx.load(source,load_external_data=False)
            original_bytes=(root/'coefficients.data').read_bytes()
            report=rewrite_module.rewrite(source,output)
            after=onnx.load(output,load_external_data=False)
            self.assertEqual([v.SerializeToString() for v in before.graph.initializer],
                             [v.SerializeToString() for v in after.graph.initializer])
            self.assertEqual((output.parent/'coefficients.data').read_bytes(),original_bytes)
            self.assertTrue(report['external_files'])


if __name__=='__main__':unittest.main()
