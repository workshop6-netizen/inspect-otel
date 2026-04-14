# inspect-otel

OpenTelemetry integration for Inspect AI evaluation workflows.

## Overview

inspect-otel provides a standardised telemetry layer for LLM evaluation systems built with [Inspect AI](https://inspect.aisi.org.uk). It exports traces to any OpenTelemetry-compatible backend:

- Arize Phoenix (`http://localhost:6006/v1/traces`)
- Grafana / Jaeger / Tempo
- LangSmith (via a custom OTLP exporter)

## Features

- OpenTelemetry-native tracing via the `inspect_ai.hooks.Hooks` interface
- Per-task root spans and per-sample child spans, fully nested
- Decorators for tracing LLM and tool calls in standalone code
- Sampling support (TraceIdRatioBased) for large-scale evals
- Async batching (BatchSpanProcessor) or synchronous mode
- Failure-only logging mode (skip successfully-scored samples)
- 100 % test coverage

---

## Installation

```bash
pip install inspect-otel
```

## Quick Start

### 1. Config file (`config.yaml`)

```yaml
telemetry:
  service_name: inspect-evals
  otlp_endpoint: http://localhost:6006/v1/traces

  sampling:
    rate: 0.2

  logging:
    failure_only: true

  async: true
```

### 2. Register hooks and run evals

```python
from inspect_ai import eval_set
from inspect_otel import setup_hooks

setup_hooks(config_path="config.yaml")

eval_set(evals=[my_eval], model="gpt-4")
```

Or build the manager manually for full control:

```python
from inspect_ai import eval_set
from inspect_ai.hooks import hooks
from inspect_otel.config import load_config, build_manager
from inspect_otel.logger import OpenTelemetryLogger

config = load_config("config.yaml")
manager = build_manager(config)

@hooks(name="opentelemetry", description="OTel tracing")
class _OTelHooks(OpenTelemetryLogger):
    def __init__(self) -> None:
        super().__init__(manager)

eval_set(evals=[my_eval], model="gpt-4")
```

---

## Auto-Instrumentation Decorators

```python
from inspect_otel import trace_llm, trace_tool

@trace_llm(model="gpt-4")
def call_model(prompt: str) -> str:
    return llm.generate(prompt)

@trace_tool("web_search")
def search_tool(query: str) -> str:
    return search(query)
```

Both decorators record `llm.model` / `tool.name`, latency, exceptions, and set appropriate span status.

---

## Span Hierarchy

```
inspect.run          <- one per task  (on_task_start / on_task_end)
  +-- inspect.sample <- one per sample (on_sample_end)
```

### Key Span Attributes

| Span | Attribute | Source |
|---|---|---|
| `inspect.run` | `eval.run_id` | `TaskStart.run_id` |
| `inspect.run` | `eval.name` | `EvalSpec.task` |
| `inspect.run` | `llm.model` | `EvalSpec.model` |
| `inspect.run` | `eval.dataset` | `EvalSpec.dataset.name` |
| `inspect.run` | `eval.experiment_id` | `EvalSpec.metadata["experiment_id"]` |
| `inspect.sample` | `eval.sample_id` | `EvalSample.id` |
| `inspect.sample` | `eval.score` | `Score.as_float()` |
| `inspect.sample` | `llm.latency_ms` | `EvalSample.total_time * 1000` |
| `inspect.sample` | `llm.model` | first key of `EvalSample.model_usage` |

---

## Configuration Reference

```yaml
telemetry:
  service_name: string        # OTel service.name resource attribute
  otlp_endpoint: string|null  # OTLP/HTTP endpoint (null = no export)

  sampling:
    rate: float               # 0.0-1.0, default 1.0

  logging:
    failure_only: bool        # default false

  async: bool                 # true = BatchSpanProcessor (default)
```

---

## Phoenix Integration

```bash
pip install arize-phoenix
python -m phoenix.server.main
```

Set `otlp_endpoint: http://localhost:6006/v1/traces`.

---

## Development

```bash
pip install -e ".[dev]"
pytest --cov=inspect_otel
ruff check inspect_otel/ tests/
```

---

## Publish

```bash
pip install build twine
python -m build
twine upload dist/*
```

---

## Roadmap

- [ ] LangSmith exporter adapter
- [ ] Dataset sync / version diffing
- [ ] CLI tooling
- [ ] Streamlit dashboard
- [ ] Embedding-based failure clustering
- [ ] Regression detection across eval versions
