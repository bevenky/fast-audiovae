"""Static stage rewrite safety tests. No runtime sessions or kernel execution."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import onnx
from onnx import helper,TensorProto
ROOT = Path(__file__).resolve().parents[1]
def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
fixture=load('apple_stage_fixture',ROOT/'check_stage.py').fixture
rewrite=load('apple_stage_rewrite',ROOT/'rewrite.py').rewrite

class RewriteTests(unittest.TestCase):
    def source(self):
        m,_=fixture(32)
        m.graph.input[0].type.tensor_type.shape.dim[0].dim_value=1
        last=copy.deepcopy(m.graph.output[-1]);last.type.tensor_type.shape.dim[0].dim_value=1
        del m.graph.output[:];m.graph.output.append(last)
        return m
    def apply(self,m,**kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'source.onnx';dst=Path(tmp)/'candidate.onnx';onnx.save_model(m,src)
            report=rewrite(src,dst,channels=(32,),**kwargs)
            return onnx.load(dst),report
    def test_all_tiles_preserve_constants_and_graph_contract(self):
        m=self.source()
        for q in (64,128,256):
            result,report=self.apply(m,tile_time=q)
            self.assertEqual(report['count'],1);self.assertEqual(len(result.graph.node),1)
            self.assertEqual(list(result.graph.node[0].input)[0],'x')
            self.assertEqual(result.graph.node[0].op_type,'StageStackF32')
            self.assertEqual(result.ir_version,m.ir_version)
            self.assertEqual([t.SerializeToString() for t in result.graph.initializer[:len(m.graph.initializer)]],
                             [t.SerializeToString() for t in m.graph.initializer])
    def test_exposed_intermediate_rejected(self):
        m=self.source();m.graph.output.append(helper.make_tensor_value_info('u0_output',TensorProto.FLOAT,[1,32,'T']))
        with self.assertRaises(ValueError):self.apply(m)
    def test_external_consumer_rejected(self):
        m=self.source();m.graph.node.append(helper.make_node('Identity',['u1_post'],['external'],name='fanout'))
        with self.assertRaises(ValueError):self.apply(m)
    def test_wrong_residual_input_rejected(self):
        m=self.source()
        for n in m.graph.node:
            if n.name=='u1_residual':n.input[0]='x'
        with self.assertRaises(ValueError):self.apply(m)
    def test_reordered_dilations_rejected(self):
        m=self.source()
        for n in m.graph.node:
            if n.name=='u2_dwpost':
                for a in n.attribute:
                    if a.name=='dilation':a.i=3
        with self.assertRaises(ValueError):self.apply(m)
    def test_bias_operand_order_rejected(self):
        m=self.source()
        for n in m.graph.node:
            if n.name=='u0_residual':n.input.reverse()
        with self.assertRaises(ValueError):self.apply(m)
    def test_reshape_with_real_permutation_not_accepted(self):
        m=self.source()
        m.graph.node.insert(0,helper.make_node('Transpose',['x'],['permuted'],name='transpose',perm=[0,2,1]))
        m.graph.node[1].input[0]='permuted'
        with self.assertRaises(ValueError):self.apply(m)
    def test_invalid_experimental_parameters_rejected(self):
        for options in ({'tile_time':65},{'segments':0},{'backend':0},{'matrix_mode':3},{'matrix_isa':0}):
            with self.assertRaises(ValueError):self.apply(self.source(),**options)

if __name__=='__main__':unittest.main()
