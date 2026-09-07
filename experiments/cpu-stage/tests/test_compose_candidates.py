"""Source-only composition safety checks, with tiny synthetic ONNX graphs."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import onnx
from onnx import helper
from fast_audiovae.graph.block_fusion import rewrite, rewrite_model

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('composer', HERE / 'compose_candidates.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
spec = importlib.util.spec_from_file_location('fixture', HERE.parents[1] / 'tests/test_block_fusion.py')
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)


class Composition(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'source.onnx'
        self.base = self.root / 'base.onnx'
        self.model = f.fixture('venky.audio.cpu.portable', dynamic_target=True)
        # Independent early matrix plus identity coefficients enables cumulative
        # MKL and residual-region tests without large pretrained tensors.
        first = helper.make_node('MatMul', ['pw', 'input'], ['x'], name='early')
        self.model.graph.node.insert(0, first)
        self.model.graph.input[0].name = 'input'
        onnx.save_model(self.model, self.source)
        rewrite(self.source, self.base, variant='both', backend=5,
                expected_chains=1, expected_adds=1)

    def tearDown(self):
        self.temp.cleanup()

    def derivative(self, kind):
        model = copy.deepcopy(self.model)
        names = {n.name: n for n in model.graph.node}
        if kind == 'mkl':
            replacement = helper.make_node('IntelPlainMatMulF32', ['pw', 'input'], ['x'],
                name='early', domain=c.KINDS[kind][0], M=3, K=3, N=0, mode=0, shards=2)
            removed = {'early'}
            records = [{'name': 'early', 'M': 3, 'K': 3, 'weights': 'pw'}]
        else:
            proven, proof = rewrite_model(model, variant='both')
            add = next(r for r in proof['changes'] if r['kind'] == 'adds')
            alias = next(t for t in proven.graph.initializer if t.name == add['node'] + '_bias')
            model.graph.initializer.append(copy.deepcopy(alias))
            if kind == 'matrix':
                removed = {'pointwise', 'bias', 'skip'}
                replacement = helper.make_node('PointwiseBiasResidualF32', ['pw', 'post', alias.name, 'x'],
                    ['y'], name='matrix', domain=c.KINDS[kind][0], channels=3)
                records = [{'node':'matrix','matmul':'pointwise','bias_add':'bias','residual_add':'skip',
                            'bias_source':'pb','bias_alias':alias.name}]
            else:
                removed = {'input_view', 'pre', 'alias', 'post', 'pointwise', 'bias', 'skip'}
                replacement = helper.make_node('StageStackF32', ['x','w','b','ap','rp','aq','rq','pw',alias.name],
                    ['y'], name='stage', domain=c.KINDS[kind][0], channels=3)
                records = [{'node':'stage','removed_nodes':sorted(removed), 'unit_outputs':['y'],
                            'unit_constants':[['w','b','ap','rp','aq','rq','pw',alias.name]]}]
        anchor = max(i for i,n in enumerate(model.graph.node) if n.name in removed)
        nodes = [replacement if i == anchor else n for i,n in enumerate(model.graph.node)
                 if n.name not in removed or i == anchor]
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        model.opset_import.append(helper.make_opsetid(c.KINDS[kind][0],1))
        path = self.root / (kind + '.onnx')
        audit = path.with_suffix(c.KINDS[kind][2])
        report = {'source_sha256': c.sha(self.source), 'count': len(records)}
        report['nodes' if kind == 'mkl' else 'changes'] = records
        self.save_derivative(model,path,audit,report)
        return kind,path,audit

    def save_derivative(self, model, path, audit, report):
        onnx.save_model(model,path)
        report['output_sha256'] = c.sha(path)
        audit.write_text(json.dumps(report))

    def compose(self, *candidates):
        return c.compose(self.source,self.base,self.root/'out.onnx',candidates)

    def test_matrix_replaces_complete_already_fused_add_region(self):
        r = self.compose(self.derivative('matrix'))
        self.assertEqual(r['counts'], {'matrix':1})
        self.assertEqual(len(r['regions'][0]['removed_base_nodes']),2)
        self.assertTrue(r['base_reproved_from_original'])
        self.assertFalse(r['inference_executed'])
        original = onnx.load(self.source)
        output = onnx.load(self.root/'out.onnx')
        self.assertEqual([c.blob(t) for t in original.graph.initializer],
                         [c.blob(t) for t in output.graph.initializer[:len(original.graph.initializer)]])

    def test_stage_absorbs_whole_chain_and_add_regions(self):
        r = self.compose(self.derivative('stage'))
        self.assertEqual(r['counts'], {'stage':1})
        self.assertEqual(len(r['regions'][0]['removed_base_nodes']),4)

    def test_mkl_and_matrix_compose_cumulatively(self):
        r = self.compose(self.derivative('mkl'),self.derivative('matrix'))
        self.assertEqual(r['counts'], {'mkl':1,'matrix':1})

    def test_mkl_and_stage_compose_cumulatively(self):
        r = self.compose(self.derivative('mkl'),self.derivative('stage'))
        self.assertEqual(r['counts'], {'mkl':1,'stage':1})

    def test_overlapping_stage_and_matrix_rejected(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.compose(self.derivative('stage'),self.derivative('matrix'))
        self.assertFalse((self.root/'out.onnx').exists())

    def test_wrong_source_or_output_binding_rejected(self):
        candidate = self.derivative('mkl')
        report = json.loads(candidate[2].read_text())
        for field in ('source_sha256','output_sha256'):
            bad = dict(report);bad[field] = '0'*64
            candidate[2].write_text(json.dumps(bad))
            with self.subTest(field=field),self.assertRaises(ValueError):
                self.compose(candidate)

    def test_coefficient_change_rejected_even_if_audit_rehashed(self):
        candidate = self.derivative('mkl')
        model = onnx.load(candidate[1]);report = json.loads(candidate[2].read_text())
        tensor = next(t for t in model.graph.initializer if t.name=='pw')
        tensor.raw_data = b'\0' * len(tensor.raw_data)
        self.save_derivative(model,candidate[1],candidate[2],report)
        with self.assertRaisesRegex(ValueError,'initializer protobuf'):
            self.compose(candidate)

    def test_unselected_node_mutation_rejected(self):
        candidate = self.derivative('mkl')
        model = onnx.load(candidate[1]);report = json.loads(candidate[2].read_text())
        next(n for n in model.graph.node if n.name=='pre').attribute[0].i = 1
        self.save_derivative(model,candidate[1],candidate[2],report)
        with self.assertRaisesRegex(ValueError,'Unselected derivative node changed'):
            self.compose(candidate)

    def test_extra_unrecorded_native_chain_pass_rejected(self):
        candidate = self.derivative('matrix')
        model = onnx.load(candidate[1]);report = json.loads(candidate[2].read_text())
        next(n for n in model.graph.node if n.name=='pre').name = 'unexpected'
        self.save_derivative(model,candidate[1],candidate[2],report)
        with self.assertRaisesRegex(ValueError,'unaudited'):
            self.compose(candidate)

    def test_base_reproof_rejects_rehashed_backend_change(self):
        candidate = self.derivative('mkl')
        model = onnx.load(self.base)
        node = next(n for n in model.graph.node if n.op_type=='SnakeDW7SnakeF32')
        next(a for a in node.attribute if a.name=='backend').i = 4
        audit = self.base.with_suffix('.block-fusion.json')
        report = json.loads(audit.read_text())
        self.save_derivative(model,self.base,audit,report)
        with self.assertRaisesRegex(ValueError,'fresh original-source block proof'):
            self.compose(candidate)

    def test_alias_bytes_changed_rejected(self):
        candidate = self.derivative('matrix')
        model = onnx.load(candidate[1]);report = json.loads(candidate[2].read_text())
        tensor = model.graph.initializer[-1];tensor.raw_data = b'\0'*len(tensor.raw_data)
        self.save_derivative(model,candidate[1],candidate[2],report)
        with self.assertRaisesRegex(ValueError,'Alias coefficient bytes'):
            self.compose(candidate)

    def test_output_overwrite_rejected(self):
        candidate = self.derivative('mkl')
        self.compose(candidate)
        before = (self.root/'out.onnx').read_bytes()
        with self.assertRaisesRegex(ValueError,'new, separate'):
            self.compose(candidate)
        self.assertEqual(before,(self.root/'out.onnx').read_bytes())

    def test_boundary_fanout_rejected(self):
        # A reported region excludes the bias output despite an external user.
        model = copy.deepcopy(self.model)
        model.graph.node.append(helper.make_node('Identity',['biased'],['extra'],name='extra'))
        index = c.graph_index(model)
        candidate = self.derivative('matrix')
        derivative = onnx.load(candidate[1]);replacement = next(n for n in derivative.graph.node if n.name=='matrix')
        with self.assertRaisesRegex(ValueError,'boundary/fanout'):
            c.check_boundary(index,{'pointwise','bias','skip'},replacement,c.graph_index(derivative)['initializers'])

    def test_external_tensor_file_corruption_rejected(self):
        # A separate derivative directory allows corruption without changing the
        # original, while its ONNX protobuf and audit hashes remain unchanged.
        onnx.save_model(self.model,self.source,save_as_external_data=True,
                        all_tensors_to_one_file=True,location='weights.bin',size_threshold=0)
        self.model = onnx.load(self.source,load_external_data=False)
        rewrite(self.source,self.base,variant='both',backend=5,expected_chains=1,expected_adds=1)
        candidate = self.derivative('mkl')
        separate = self.root/'separate';separate.mkdir()
        new_model = separate/candidate[1].name;new_audit = separate/candidate[2].name
        new_model.write_bytes(candidate[1].read_bytes());new_audit.write_bytes(candidate[2].read_bytes())
        data = bytearray((self.root/'weights.bin').read_bytes());data[0] ^= 1
        (separate/'weights.bin').write_bytes(data)
        with self.assertRaisesRegex(ValueError,'external coefficient files'):
            self.compose((candidate[0],new_model,new_audit))


if __name__ == '__main__':
    unittest.main()
