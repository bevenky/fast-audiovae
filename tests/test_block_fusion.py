"""Checked experimental graph fusion tests; no model checkpoints or inference."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from fast_audiovae.graph.block_fusion import DOMAINS, rewrite, rewrite_model


def fixture(domain=DOMAINS[0], dynamic_target=False):
    c=3; shape=[1,c,'T']
    arrays={'w':np.ones((c,1,7),np.float32), 'b':np.zeros(c,np.float32),
            'ap':np.ones(c,np.float32), 'rp':np.ones(c,np.float32),
            'aq':np.ones(c,np.float32)*.7, 'rq':np.ones(c,np.float32)*.8,
            'pw':np.eye(c,dtype=np.float32), 'pb':np.arange(c,dtype=np.float32).reshape(1,c,1),
            'target':np.array([1,c,-1],np.int64)}
    attr=dict(channels=c,native_abi=1,row_batches=0,backend=0)
    attr['require_vforce' if domain==DOMAINS[0] else 'require_vector_sine']=1
    nodes=[helper.make_node('Reshape',['x','target'],['view'],name='input_view'),
           helper.make_node('SnakeF32',['view','ap','rp'],['pre'],name='pre',domain=domain,**attr)]
    if dynamic_target:
        nodes.insert(0,helper.make_node('Shape',['x'],['dynamic_target'],name='shape'))
    nodes += [helper.make_node('Reshape',['pre','dynamic_target' if dynamic_target else 'target'],['alias'],name='alias'),
              helper.make_node('CausalDW7SnakeF32',['alias','w','b','aq','rq'],['post'],name='post',domain=domain,dilation=9,**attr),
              helper.make_node('MatMul',['pw','post'],['product'],name='pointwise'),
              helper.make_node('Add',['product','pb'],['biased'],name='bias'),
              helper.make_node('Add',['x','biased'],['y'],name='skip')]
    vi=lambda n:helper.make_tensor_value_info(n,TensorProto.FLOAT,shape)
    graph=helper.make_graph(nodes,'block_fixture',[vi('x')],[vi('y')],
        [numpy_helper.from_array(v,n) for n,v in arrays.items()],
        value_info=[vi(n) for n in ['view','pre','alias','post','biased']])
    return helper.make_model(graph,ir_version=10,opset_imports=[helper.make_opsetid('',20),helper.make_opsetid(domain,1)])


class BlockFusion(unittest.TestCase):
    def test_modes_and_byte_preservation(self):
        for domain in DOMAINS:
            for mode,counts in [('chain',(1,0)),('adds',(0,1)),('both',(1,1)),('none',(0,0))]:
                m=fixture(domain,True);before=m.SerializeToString()
                out,a=rewrite_model(m,variant=mode,expected_chains=counts[0],expected_adds=counts[1])
                self.assertEqual(a['counts'],dict(zip(('chain','adds'),counts)))
                self.assertEqual(m.SerializeToString(),before)
                self.assertEqual([t.SerializeToString() for t in m.graph.initializer],
                                 [t.SerializeToString() for t in out.graph.initializer[:len(m.graph.initializer)]])

    def test_backend_policy_and_required_manifest(self):
        for domain,backends in [(DOMAINS[0],(0,2)),(DOMAINS[1],(0,4,5))]:
            for backend in backends:
                out,a=rewrite_model(fixture(domain),variant='both',backend=backend)
                self.assertIn('SnakeDW7SnakeF32',a['required_operators'])
                self.assertIn('BiasResidualF32',a['required_operators'])
                for n in out.graph.node:
                    if n.domain==domain:
                        self.assertEqual(next(x.i for x in n.attribute if x.name=='backend'),backend)
        for backend in (1,3,True,6):
            with self.assertRaises(ValueError):rewrite_model(fixture(DOMAINS[1]),backend=backend)
        with self.assertRaises(ValueError):rewrite_model(fixture(),backend=5)

    def test_fanout_outputs_unknown_attributes_and_coefficients_rejected(self):
        for mutation in ('fanout','output','unknown','policy','overridable','coefficient','dilation'):
            m=fixture();ns={n.name:n for n in m.graph.node}
            if mutation=='fanout':m.graph.node.append(helper.make_node('Identity',['pre'],['extra']))
            if mutation=='output':m.graph.output.append(helper.make_tensor_value_info('pre',TensorProto.FLOAT,[1,3,'T']))
            if mutation=='unknown':ns['post'].attribute.append(helper.make_attribute('extra',1))
            if mutation=='policy':next(a for a in ns['pre'].attribute if a.name=='row_batches').i=1
            if mutation=='overridable':m.graph.input.append(helper.make_tensor_value_info('ap',TensorProto.FLOAT,[3]))
            if mutation=='coefficient':next(t for t in m.graph.initializer if t.name=='ap').dims[0]=1
            if mutation=='dilation':next(a for a in ns['post'].attribute if a.name=='dilation').i=2
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                rewrite_model(m,variant='chain',expected_chains=1)

    def test_shape_program_proof_rejects_lying_annotations(self):
        m=fixture(dynamic_target=True)
        shape=next(n for n in m.graph.node if n.name=='shape')
        # Target comes from an unrelated graph input, despite all annotations
        # claiming that time is the same symbol. No equality may be inferred.
        m.graph.input.append(helper.make_tensor_value_info('other',TensorProto.FLOAT,[1,3,'T']))
        shape.input[0]='other'
        with self.assertRaises(ValueError):rewrite_model(m,variant='chain',expected_chains=1)
        m=fixture();t=next(t for t in m.graph.initializer if t.name=='target')
        t.CopyFrom(numpy_helper.from_array(np.array([1,1,-1],np.int64),'target'))
        with self.assertRaises(ValueError):rewrite_model(m,variant='chain',expected_chains=1)

    def test_invalid_shape_view_is_not_removed_with_alias(self):
        m=fixture(dynamic_target=True)
        m.graph.initializer.append(numpy_helper.from_array(np.array([0],np.int64),'bad_axis'))
        shape=next(n for n in m.graph.node if n.name=='shape')
        shape.output[0]='shape_before_bad_squeeze'
        m.graph.node.insert(1,helper.make_node('Squeeze',['shape_before_bad_squeeze','bad_axis'],['dynamic_target']))
        # Shape(x) has length three, so axis 0 cannot be squeezed. Matching
        # flattened shape values must not suppress that original error.
        with self.assertRaises(ValueError):rewrite_model(m,variant='chain',expected_chains=1)

    def test_activation_reshape_rejects_invalid_rank_and_zero_inference(self):
        for mutation in ('rank2','zero_inference'):
            m=fixture()
            # Only the alias target is malformed, leaving the pre-Snake shape
            # valid. Flattening an invalid target must not make it removable.
            shape=np.array([[1,3,-1]],np.int64) if mutation=='rank2' else np.array([0,3,-1],np.int64)
            m.graph.initializer.append(numpy_helper.from_array(shape,'bad_alias_target'))
            alias=next(n for n in m.graph.node if n.name=='alias')
            alias.input[1]='bad_alias_target'
            if mutation=='zero_inference':alias.attribute.append(helper.make_attribute('allowzero',1))
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                rewrite_model(m,variant='chain',expected_chains=1)

        m=fixture(dynamic_target=True)
        m.graph.initializer.append(numpy_helper.from_array(np.array([0,-1],np.int64),'bad_shape_target'))
        shape=next(n for n in m.graph.node if n.name=='shape')
        shape.output[0]='shape_before_bad_reshape'
        m.graph.node.insert(1,helper.make_node('Reshape',['shape_before_bad_reshape','bad_shape_target'],['dynamic_target'],allowzero=1))
        with self.assertRaises(ValueError):rewrite_model(m,variant='chain',expected_chains=1)

    def test_add_order_broadcast_fanout_and_unknown_attrs_rejected(self):
        for mutation in ('swapfinal','swapbias','rank1bias','fanout','shape','unknown'):
            m=fixture();ns={n.name:n for n in m.graph.node}
            if mutation=='swapfinal':ns['skip'].input.reverse()
            if mutation=='swapbias':ns['bias'].input.reverse()
            if mutation=='rank1bias':
                t=next(t for t in m.graph.initializer if t.name=='pb');del t.dims[:];t.dims.append(3)
            if mutation=='fanout':m.graph.node.append(helper.make_node('Identity',['biased'],['extra']))
            if mutation=='shape':m.graph.output[0].type.tensor_type.shape.dim[1].dim_value=4
            if mutation=='unknown':ns['bias'].attribute.append(helper.make_attribute('extra',1))
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                rewrite_model(m,variant='adds',expected_adds=1)

    def test_external_files_and_original_tensor_protobufs_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);source=p/'source.onnx';out=p/'candidate/model.onnx'
            onnx.save_model(fixture(),source,save_as_external_data=True,all_tensors_to_one_file=True,
                            location='weights.bin',size_threshold=0)
            original=onnx.load(source,load_external_data=False)
            original_data=(p/'weights.bin').read_bytes();original_source=source.read_bytes()
            a=rewrite(source,out,variant='both',expected_chains=1,expected_adds=1)
            self.assertEqual((out.parent/'weights.bin').read_bytes(),original_data)
            self.assertEqual(source.read_bytes(),original_source)
            derived=onnx.load(out,load_external_data=False)
            self.assertEqual([t.SerializeToString() for t in original.graph.initializer],
                [t.SerializeToString() for t in derived.graph.initializer[:len(original.graph.initializer)]])
            self.assertEqual(a['counts'],{'chain':1,'adds':1})
            with self.assertRaises(ValueError):rewrite(source,source,variant='none')


if __name__=='__main__':unittest.main()
