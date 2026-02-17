# SemanticNN System Specification

## 1. Purpose and Scope

SemanticNN is a transformation-aware verification platform for neural-network deployment pipelines. It verifies whether **task-level semantic invariants** are preserved when a reference model `f` is transformed into `f_T` (e.g., quantization, pruning, distillation, compilation, retraining).

Primary objective:
- Replace strict functional equivalence checks with certified checks of **semantic invariance** over an input region `R`.

Out of scope (v1):
- End-to-end training infrastructure.
- Hardware kernel benchmarking.
- Full support for all neural operators.

---

## 2. Research-to-System Mapping

Core concepts from the research are represented as first-class platform entities:

- **Transformation** `T`: deterministic or stochastic procedure that maps `f -> f_T`.
- **Invariant** `I(f, x)`: task-level boolean property over model output semantics.
- **Relation** `R_T(x) = (f(x), f_T(x))`: relational semantics between original/transformed models.
- **SDNet** `Δ_T(x) = f_T(x) - f(x)`: semantic drift network used for certified drift bounds.
- **Certified checker**: sound abstraction engine that proves/disproves invariant preservation over `R`.

---

## 3. Product Requirements

### 3.1 Functional Requirements

1. **Model Pair Ingestion**
   - Import two models with aligned input/output schema.
   - Validate shape, operator-domain, and output-head compatibility.

2. **Transformation Metadata**
   - Register transformation type and parameters (e.g., bit-width, sparsity target, distillation temperature).
   - Preserve provenance: model hash, transformation config, environment hash.

3. **Invariant Definition API**
   - Built-in invariants:
     - Classification margin preservation.
     - Top-1 label preservation.
     - Pairwise ranking preservation.
     - Regression envelope (`|y_T - y| <= eps`).
   - Custom invariants via a typed DSL and Python callback interface.

4. **Region Specification**
   - Support hyper-rectangles, sampled datasets, and named safety regions.
   - Optional partitioning for refinement.

5. **SDNet Construction**
   - Build shared-graph relational representation for `f` and `f_T`.
   - Compute `Δ_T` and expose symbolic cancellation statistics.

6. **Certified Verification Engine**
   - Sound upper/lower bounds on SDNet outputs with:
     - Interval Bound Propagation (IBP) baseline.
     - Optional branch-and-bound refinement.
   - Prove sufficient conditions for invariant preservation or return unresolved status.

7. **Counterexample Search (Best-Effort)**
   - If proof fails, attempt adversarial relational search for invariant violations.
   - Mark as *empirical* evidence, not a proof.

8. **Reporting**
   - Machine-readable certificates (JSON).
   - Human-readable report: assumptions, bound tightness, pass/fail/inconclusive, and violating examples.

### 3.2 Non-Functional Requirements

- **Soundness first**: no false proofs under declared assumptions.
- **Reproducibility**: deterministic runs under fixed seeds and versions.
- **Extensibility**: pluggable abstract domains and invariant modules.
- **Traceability**: full audit trail from model artifact to verification result.
- **Usability**: single CLI command for common workflows.

### 3.3 Model Format Strategy (Why ONNX First)

The MVP targets ONNX as the primary interchange format for `f` and `f_T`.

Rationale:
- **Transformation realism:** Many deployment-time transformations of interest (quantization, graph rewriting, hardware compilation) are already expressed on exported/intermediate graphs rather than raw training graphs.
- **Framework neutrality:** ONNX decouples verification from a specific training stack (PyTorch, TensorFlow, JAX), which aligns with transformation-centric verification goals.
- **Deterministic graph semantics:** Static graph structure and standardized operators simplify relational alignment and SDNet construction compared with dynamic eager execution graphs.
- **Toolchain interoperability:** Existing deployment pipelines and compilers (e.g., ONNX Runtime/TensorRT flows) often consume ONNX directly, so verification can be placed near deployment.

Why not PyTorch-only in v1:
- PyTorch eager mode is convenient for research iteration but introduces framework-specific semantics and dynamic control-flow complexity that can obscure sound relational analysis.
- Supporting only PyTorch would constrain applicability to one ecosystem and weaken reproducibility across deployment backends.

Planned approach:
- **v1:** ONNX-first ingestion and verification.
- **v1.x:** Add a `torch.export`/TorchScript-to-IR adapter path that normalizes to the same internal graph IR.
- **v2:** First-class multi-frontend ingestion (PyTorch + ONNX) with shared proof obligations and identical certificate format.

---

## 4. System Architecture

## 4.1 High-Level Components

1. **Artifact Manager**
   - Stores models, transformation metadata, datasets/regions, and run manifests.

2. **IR & Graph Alignment Layer**
   - Converts source models to a normalized intermediate representation.
   - Aligns common subgraphs between `f` and `f_T`.

3. **SDNet Builder**
   - Generates drift graph `Δ_T` from aligned IR.
   - Enables symbolic cancellation where computationally shared.

4. **Invariant Engine**
   - Encodes invariant obligations into bound constraints over `f`, `f_T`, or `Δ_T`.

5. **Abstract Verification Core**
   - Domain backends (IBP initially).
   - Refinement scheduler (branch-and-bound).
   - Soundness guards and assumption tracking.

6. **Falsification Engine**
   - Gradient-based / search-based relation violation probing.

7. **CLI + Python API + Reporter**
   - Entry points for automation and experimentation.

## 4.2 Data Flow

1. Load `f`, `f_T`, transformation metadata, and region `R`.
2. Validate compatibility and build aligned relational IR.
3. Construct SDNet `Δ_T`.
4. Encode invariant obligations.
5. Run abstract bound analysis (+ refinement).
6. Decide:
   - **PASS (certified)**: invariant preserved under assumptions.
   - **FAIL (witness found)**: concrete violation.
   - **INCONCLUSIVE**: bound too loose / budget exhausted.
7. Emit report + certificate + diagnostics.

---

## 5. Mathematical Verification Contracts

## 5.1 Core Contract

Given region `R` and invariant `I`, prove:

`forall x in R: I(f, x) => I(f_T, x)`

If invariant is Lipschitz in outputs with constant `L`, and we certify:

`||Δ_T(x)||_inf <= eta` for all `x in R`,

then invariants with slack `delta` are preserved when:

`L * eta <= delta`.

## 5.2 Margin-Specific Contract

For label `y` and margin `m_f(x, y) = f_y(x) - max_{j!=y} f_j(x)`, prove:

If `min_{x in R} m_f(x, y) >= kappa + 2*eta` and `||Δ_T(x)||_inf <= eta`, then

`m_{f_T}(x, y) >= kappa`.

The system shall provide this as a built-in proof rule.

---

## 6. Interfaces

## 6.1 CLI (v1)

- `semanticnn verify --ref model.onnx --cand model_q.onnx --invariant margin --region region.json --out report.json`
- `semanticnn inspect-sdnet --ref ... --cand ... --region ...`
- `semanticnn falsify --ref ... --cand ... --invariant ...`

## 6.2 Python API (v1)

- `load_model(path) -> ModelArtifact`
- `define_region(spec) -> Region`
- `margin_invariant(label, kappa)`
- `verify(ref, cand, invariant, region, config) -> VerificationResult`

Result object fields:
- `status: PASS | FAIL | INCONCLUSIVE`
- `proof_certificate`
- `assumptions`
- `bound_summary`
- `counterexamples[]`

---

## 7. Data and File Specifications

- `models/`: immutable model artifacts + hashes.
- `transforms/`: transformation manifests.
- `regions/`: input-region specs.
- `runs/<timestamp>/`:
  - `config.yaml`
  - `certificate.json`
  - `report.md`
  - `counterexamples.json`
  - `metrics.json`

Certificate schema (minimum):
- model hashes (`ref_hash`, `cand_hash`)
- invariant id + parameters
- region id + normalized bounds
- abstract domain + refinement settings
- certified bound(s)
- theorem/rule applied
- verdict
- reproducibility metadata (seed, versions)

---

## 8. Verification Algorithm (Reference Pipeline)

1. Normalize and align model graphs.
2. Construct SDNet with shared-node cancellation.
3. Initialize abstract domain over region `R`.
4. Propagate bounds through relational graph.
5. Evaluate invariant obligations.
6. If undecided and budget remains, refine partitions.
7. Return certified verdict or inconclusive.
8. Optionally run falsification to seek concrete violations.

---

## 9. Safety and Assurance Requirements

- Explicit assumption ledger in every verdict.
- Separate **certified** vs **empirical** evidence channels.
- No silent fallback from sound to unsound methods.
- Any unsound acceleration must be opt-in and clearly labeled.

---

## 10. Performance Targets (Initial)

- Small/medium MLP/CNN ONNX models (<10M params).
- End-to-end verification under 10 minutes for baseline IBP on moderate regions.
- Refinement budget configurable by wall-clock and node count.

---

## 11. MVP Roadmap

### Phase 1: Foundations
- ONNX ingestion + compatibility checks.
- SDNet construction for affine + ReLU networks.
- IBP domain backend.
- Margin invariant checker.
- CLI `verify` and JSON certificate.

### Phase 1.x: Frontend Expansion
- PyTorch export adapter (`torch.export`/TorchScript) lowered into the same normalized IR.
- Cross-frontend conformance checks (PyTorch-exported vs native ONNX graph parity).

### Phase 2: Tightening & Diagnostics
- Shared-graph symbolic cancellation metrics.
- Branch-and-bound refinement.
- Counterexample search.
- Rich reports and visualization hooks.

### Phase 3: Generalization
- Ranking/regression invariants.
- Additional domains (zonotope/linear relaxations).
- Broader operator support and transformation templates.

---

## 12. Risks and Mitigations

- **Loose bounds -> inconclusive outcomes**
  - Mitigation: refinement, cancellation-aware IR, domain upgrades.

- **Operator mismatch across transformed models**
  - Mitigation: canonical IR conversion and operator capability matrix.

- **Spec ambiguity for custom invariants**
  - Mitigation: typed invariant DSL + validation tests.

- **Scalability limits on large models**
  - Mitigation: region partitioning, modular verification, approximation tiers.

---

## 13. Definition of Done (MVP)

An MVP release is complete when:

1. A user can verify margin preservation for an ONNX model pair on a box region.
2. The tool emits a machine-checkable certificate with assumptions and bounds.
3. Verdicts are correctly labeled PASS/FAIL/INCONCLUSIVE.
4. At least one benchmark suite demonstrates end-to-end reproducibility.
