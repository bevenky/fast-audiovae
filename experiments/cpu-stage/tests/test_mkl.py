"""Static MKL adapter checks. Optional pinned-model rewrite performs no inference."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto

ROOT=Path(__file__).resolve().parents[1]
MKL=ROOT/'mkl'
sys.path.insert(0,str(MKL))

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

prepare=load('mkl_prepare_test',MKL/'prepare_decoder.py')
build=load('mkl_build_test',MKL/'build_decoder.py')
REFERENCE=json.loads((MKL/'reference-hashes.json').read_text())


class MKLStaticTests(unittest.TestCase):
    def test_native_source_and_original_rewriter_remain_identical(self):
        for name,key in [('intel_decoder.cpp','native_cpp_sha256'),('prepare_decoder.py','original_rewriter_sha256')]:
            self.assertEqual(hashlib.sha256((MKL/name).read_bytes()).hexdigest(),REFERENCE[key])
        self.assertEqual(prepare.PIN,REFERENCE['native_source_sha256'])
        self.assertEqual(sum(prepare.SIGNATURES.values()),12)
        self.assertEqual(len(prepare.SIGNATURES),5)

    def test_wrong_original_hash_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.onnx';out=root/'out.onnx'
            source.write_bytes(b'not the pinned original')
            args=['prepare_decoder.py','--source',str(source),'--output',str(out),'--threads','2']
            with patch.object(sys,'argv',args),self.assertRaisesRegex(ValueError,'exact accepted default Linux graph'):
                prepare.main()
            self.assertFalse(out.exists())

    def test_thread_budget_remains_one_or_two(self):
        args=['prepare_decoder.py','--source','original.onnx','--output','new.onnx','--threads','4']
        with patch.object(sys,'argv',args),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            prepare.main()

    def test_matrix_helper_rejects_overridable_and_wrong_batch(self):
        prepare.b.libraries()
        w=numpy_helper.from_array(np.eye(3,dtype=np.float32),'w')
        x=helper.make_tensor_value_info('x',TensorProto.FLOAT,[1,3,'T'])
        y=helper.make_tensor_value_info('y',TensorProto.FLOAT,[1,3,'T'])
        model=helper.make_model(helper.make_graph([helper.make_node('MatMul',['w','x'],['y'],name='mm')],
            'fixture',[x],[y],[w]))
        self.assertEqual(prepare.b.matrix_nodes(model)[0][3],(3,3,'T'))
        mutable=copy.deepcopy(model);mutable.graph.input.append(helper.make_tensor_value_info('w',TensorProto.FLOAT,[3,3]))
        with self.assertRaises(ValueError):prepare.b.matrix_nodes(mutable)
        model.graph.input[0].type.tensor_type.shape.dim[0].dim_value=2
        with self.assertRaises(ValueError):prepare.b.matrix_nodes(model)

    def test_dependency_pins_select_sequential_cpu_libraries(self):
        pins=json.loads((MKL/'dependency-pins.json').read_text())
        self.assertEqual(pins['version'],'2026.1.0')
        self.assertEqual(pins['direct_link_libraries'],['libmkl_intel_lp64.so.3','libmkl_sequential.so.3','libmkl_core.so.3'])
        self.assertEqual(len(pins['cpu_library_sha256']),11)
        self.assertFalse(pins['gpu_runtime'])
        self.assertFalse(pins['dependency_download_or_install'])
        self.assertNotIn('libmkl_intel_thread.so.3',pins['cpu_library_sha256'])

    def test_build_dependency_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'library.so';p.write_bytes(b'wrong library')
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                build.verified_files(directory,{'library.so':'0'*64})

    @unittest.skipUnless(os.environ.get('FAST_AUDIOVAE_NATIVE_SOURCE'),
                         'Set FAST_AUDIOVAE_NATIVE_SOURCE to the pinned original native ONNX for graph reproduction')
    def test_pinned_graph_reproduces_measured_one_and_two_thread_artifacts(self):
        source=Path(os.environ['FAST_AUDIOVAE_NATIVE_SOURCE']).resolve()
        self.assertEqual(prepare.b.sha(source),REFERENCE['native_source_sha256'])
        model=onnx.load(source,load_external_data=False)
        initializers=[t.SerializeToString() for t in model.graph.initializer]
        with tempfile.TemporaryDirectory() as directory:
            for threads in (1,2):
                output=Path(directory)/f'decoder_{threads}t.onnx'
                args=['prepare_decoder.py','--source',str(source),'--output',str(output),'--threads',str(threads)]
                with patch.object(sys,'argv',args),contextlib.redirect_stdout(io.StringIO()):prepare.main()
                self.assertEqual(prepare.b.sha(output),REFERENCE['derivative_sha256_by_threads'][str(threads)])
                report=json.loads(output.with_suffix('.json').read_text())
                self.assertEqual(report['nodes'],REFERENCE['selected_nodes'])
                actual=onnx.load(output,load_external_data=False)
                self.assertEqual(initializers,[t.SerializeToString() for t in actual.graph.initializer])
                self.assertEqual(len([n for n in actual.graph.node if n.op_type=='IntelPlainMatMulF32']),12)
                onnx.checker.check_model(str(output))
        self.assertEqual(prepare.b.sha(source),REFERENCE['native_source_sha256'])


if __name__=='__main__':unittest.main()
