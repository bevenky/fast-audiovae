"""Regression for legacy WN weights surviving as computed ONNX graph values."""
from pathlib import Path
import sys

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "convnext"))
import export_preflight as export
import group_model as gm
from test_group_model import TinyDecoder


@pytest.fixture(autouse=True)
def cpu_policy():
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    torch.manual_seed(19)
    yield
    torch.set_num_threads(threads)
    torch.set_rng_state(rng)


def candidate():
    student = gm.build_student(TinyDecoder().eval().requires_grad_(False), list(range(16)), list(range(8)))
    student.requires_grad_(False).eval()
    return student


def test_export_copy_preserves_effective_weights_and_unfolded_reference_including_zero_rows():
    student = candidate()
    conv = student.decoder.model[3].block[1]
    weight = gm.effective_weight(conv).detach().clone()
    weight[0] = 0
    gm.assign_effective_weight(conv, weight)
    before = {name:value.clone() for name,value in student.state_dict().items()}
    rng = torch.get_rng_state().clone()
    folded, receipt = export.folded_export_copy(student)
    assert receipt["effective_weights_bitwise_equal"] and receipt["folded_convolutions"] == 45
    assert torch.equal(rng, torch.get_rng_state())
    original_modules = dict(student.decoder.named_modules())
    for name, module in folded.student.decoder.named_modules():
        assert not hasattr(module,"weight_g") and not hasattr(module,"weight_v")
        if isinstance(module,(torch.nn.Conv1d,torch.nn.ConvTranspose1d)):
            assert torch.equal(module.weight,gm.effective_weight(original_modules[name]))
            assert module.weight.data_ptr() != gm.effective_weight(original_modules[name]).data_ptr()
    with torch.no_grad():
        for frames in (2,5):
            z = torch.randn(1,8,frames)
            for expected, actual in zip(student.forward_latents(z), folded(z)):
                torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    for name,value in student.state_dict().items():
        assert torch.equal(value,before[name])
    assert hasattr(conv,"weight_g") and hasattr(conv,"weight_v")


def test_computed_wn_graph_is_reproduced_then_folded_graph_has_real_dimensions_and_cpu_parity(tmp_path,monkeypatch):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    student = candidate()
    reference = export.ExportDecoder(student).eval()
    shapes = tuple(tuple(gm.effective_weight(student.decoder.model[i].block[1]).shape) for i in range(2,8))
    monkeypatch.setattr(export,"EXPECTED_UPSAMPLING_WEIGHTS",shapes)
    z = torch.randn(1,8,2)

    def save(model,path,constant_folding):
        with torch.no_grad():
            torch.onnx.export(model,(z,),path,input_names=["latents"],output_names=["waveform","group_output"],
                dynamic_axes={"latents":{2:"frames"},"waveform":{2:"samples"},"group_output":{2:"group_frames"}},
                opset_version=18,dynamo=False,do_constant_folding=constant_folding,export_params=True,
                keep_initializers_as_inputs=False,training=torch.onnx.TrainingMode.EVAL)
        graph = onnx.load(str(path))
        onnx.checker.check_model(graph)
        return graph

    # Disabling optional folding reproduces the real2.14 computed-WN graph even
    # on older Torch versions whose exporter happened to fold these nodes.
    computed = save(reference,tmp_path/"computed.onnx",False)
    initializers = {tensor.name for tensor in computed.graph.initializer}
    assert any(node.input[1] not in initializers for node in computed.graph.node if node.op_type=="ConvTranspose")
    with pytest.raises(ValueError,match="expected smaller weight/stride"):
        export.graph_dimensions(computed)
    folded,_ = export.folded_export_copy(student)
    graph = save(folded,tmp_path/"folded.onnx",True)
    assert [row["weight_shape"] for row in export.graph_dimensions(graph)] == [list(shape) for shape in shapes]
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(tmp_path/"folded.onnx"),sess_options=options,providers=["CPUExecutionProvider"])
    assert session.get_providers()==["CPUExecutionProvider"]
    with torch.no_grad():
        for frames in (2,5):
            value = torch.randn(1,8,frames)
            expected = reference(value)
            actual = session.run(None,{"latents":value.numpy()})
            for name,t,p in zip(("waveform","group_output"),expected,actual):
                assert export.compare(t.numpy(),p,tuple(t.shape),name)["passed"]
