import json
from pathlib import Path

import pytest
onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper

from semanticnn.invariants import MarginInvariant
from semanticnn.regions import BoxRegion
from semanticnn.verification import VerificationConfig, verify_models


def _make_single_gemm(path: Path, weight, bias):
    input_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [len(weight[0])])
    output_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [len(weight)])

    w_init = helper.make_tensor("W", TensorProto.FLOAT, dims=[len(weight), len(weight[0])], vals=[v for row in weight for v in row])
    b_init = helper.make_tensor("B", TensorProto.FLOAT, dims=[len(bias)], vals=bias)

    gemm = helper.make_node("Gemm", inputs=["x", "W", "B"], outputs=["y"], transB=1)
    graph = helper.make_graph([gemm], "g", [input_info], [output_info], [w_init, b_init])
    model = helper.make_model(graph, producer_name="test")
    onnx.save(model, path)


def test_margin_pass_with_identical_models(tmp_path):
    ref = tmp_path / "ref.onnx"
    cand = tmp_path / "cand.onnx"
    _make_single_gemm(ref, [[2.0, 0.0], [0.0, 1.0]], [0.0, 0.0])
    _make_single_gemm(cand, [[2.0, 0.0], [0.0, 1.0]], [0.0, 0.0])

    region = BoxRegion(lower=[1.0, 0.0], upper=[1.0, 0.0])
    inv = MarginInvariant(label=0, kappa=0.5)

    res = verify_models(str(ref), str(cand), region, inv, VerificationConfig(samples=32, seed=1))
    assert res.status == "PASS"
    assert res.certificate.bounds["drift_inf_eta"] == 0.0


def test_fail_detected_by_sampling(tmp_path):
    ref = tmp_path / "ref.onnx"
    cand = tmp_path / "cand.onnx"
    _make_single_gemm(ref, [[2.0, 0.0], [0.0, 1.0]], [0.0, 0.0])
    _make_single_gemm(cand, [[-2.0, 0.0], [0.0, 1.0]], [0.0, 0.0])

    region = BoxRegion(lower=[1.0, 0.0], upper=[1.0, 0.0])
    inv = MarginInvariant(label=0, kappa=0.1)

    res = verify_models(str(ref), str(cand), region, inv, VerificationConfig(samples=8, seed=0))
    assert res.status == "FAIL"
    assert res.counterexamples


def test_region_json_loader(tmp_path):
    region_file = tmp_path / "region.json"
    region_file.write_text(json.dumps({"lower": [-1, -2], "upper": [1, 2]}))
    region = BoxRegion.from_json(region_file)
    assert list(region.lower) == [-1.0, -2.0]
    assert list(region.upper) == [1.0, 2.0]
