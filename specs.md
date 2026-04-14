
---

# 📜 specs.md

```markdown
# inspect-otel Specification

## Purpose

Provide a standardized OpenTelemetry-based telemetry system for Inspect AI evaluation pipelines.

---

## Core Concepts

## Run (Eval)

A "run" corresponds to a single eval executed via `eval_set()`.

Each eval is independently traced.

Attributes:
- eval.run_id (eval.id)
- eval.name
- eval.type
- eval.dataset
- eval.dataset.version
- eval.experiment_id
- llm.model

---
Add Eval Set Semantics
## eval_set() Semantics

inspect-otel assumes:

- `eval_set()` executes multiple evals
- Each eval is treated as an independent run
- Runs are grouped logically by execution time (not explicitly parented)

This allows:
- per-eval comparison
- regression tracking across evals
- dataset-level analysis

### Sample

Represents a single evaluation datapoint.

Attributes:
- eval.sample_id
- eval.score
- eval.expected
- eval.passed

---

### LLM Call

Represents a model invocation.

Attributes:
- llm.prompt
- llm.completion
- llm.tokens.prompt
- llm.tokens.completion
- llm.latency_ms

---

### Tool Call

Represents external tool usage.

Attributes:
- tool.name
- tool.input
- tool.output

---

## Span Hierarchy

eval_set (implicit)
├── inspect.run (per eval)
│ ├── inspect.sample
│ │ ├── llm.call
│ │ ├── tool.call

## Configuration Schema

```yaml
telemetry:
  service_name: string
  otlp_endpoint: string

  sampling:
    rate: float (0.0 - 1.0)

  logging:
    failure_only: bool

  async: bool

Performance Requirements
Must support 100k+ samples
Async logging required
Sampling supported
Compatibility
OpenTelemetry compliant
Works with:
Phoenix
LangSmith (via adapter)
Jaeger / Tempo / Grafana
Extensibility
Additional exporters
Custom span attributes
Dataset syncing modules
Future Enhancements
Embedding-based failure clustering
Regression detection
Dataset version diffing

# 🚀 How to Publish

```bash
pip install build
python -m build
pip install twine
twine upload dist/*