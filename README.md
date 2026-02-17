# SemanticNN

SemanticNN is a transformation-aware verification toolkit focused on proving semantic invariants between a reference ONNX model `f` and transformed model `f_T`.

## Current MVP (implemented)

- ONNX ingestion for affine/ReLU-style feed-forward graphs.
- Hyper-rectangle region specification.
- SDNet-style drift bounds via interval abstraction.
- Certified margin-preservation rule from bounded drift.
- PASS / FAIL / INCONCLUSIVE results with machine-readable certificate output.
- CLI workflow with optional falsification-by-sampling.

## Install

```bash
poetry install
```

## Verify

```bash
poetry run semanticnn verify \
  --ref ref.onnx \
  --cand cand.onnx \
  --region region.json \
  --label 0 \
  --kappa 0.0 \
  --out certificate.json
```

Region file format:

```json
{
  "lower": [-1.0, -1.0],
  "upper": [ 1.0,  1.0]
}
```
