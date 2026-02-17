from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


@dataclass(frozen=True)
class Layer:
    weight: np.ndarray
    bias: np.ndarray
    relu_after: bool = False


@dataclass(frozen=True)
class FeedForwardIR:
    input_name: str
    output_name: str
    layers: list[Layer]

    @property
    def input_dim(self) -> int:
        return int(self.layers[0].weight.shape[1])

    @property
    def output_dim(self) -> int:
        return int(self.layers[-1].weight.shape[0])

    def eval(self, x: np.ndarray) -> np.ndarray:
        y = x.astype(np.float64)
        for layer in self.layers:
            y = layer.weight @ y + layer.bias
            if layer.relu_after:
                y = np.maximum(y, 0.0)
        return y


def _initializer_map(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {init.name: numpy_helper.to_array(init).astype(np.float64) for init in model.graph.initializer}


def _gemm_to_affine(node, init_map: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    a, b, c = node.input[:3]
    if a in init_map:
        raise ValueError("Unsupported Gemm: constant input matrix used as activation.")
    w = init_map[b]
    bias = init_map[c]

    trans_b = 0
    for attr in node.attribute:
        if attr.name == "transB":
            trans_b = int(attr.i)

    if trans_b == 0:
        w = w.T
    return w.astype(np.float64), bias.astype(np.float64).reshape(-1)


def load_feedforward_onnx(path: str | Path) -> FeedForwardIR:
    model = onnx.load(str(path))
    graph = model.graph
    init_map = _initializer_map(model)

    layers: list[Layer] = []
    pending_affine: tuple[np.ndarray, np.ndarray] | None = None

    for node in graph.node:
        if node.op_type == "Gemm":
            pending_affine = _gemm_to_affine(node, init_map)
            layers.append(Layer(weight=pending_affine[0], bias=pending_affine[1], relu_after=False))
            pending_affine = None
        elif node.op_type == "Relu":
            if not layers:
                raise ValueError("Relu encountered before affine layer.")
            prev = layers[-1]
            layers[-1] = Layer(weight=prev.weight, bias=prev.bias, relu_after=True)
        else:
            raise ValueError(f"Unsupported ONNX operator in MVP: {node.op_type}")

    if not layers:
        raise ValueError("No supported layers found in ONNX model.")

    input_name = graph.input[0].name
    output_name = graph.output[0].name
    return FeedForwardIR(input_name=input_name, output_name=output_name, layers=layers)
