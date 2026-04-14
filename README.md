# inspect-otel

OpenTelemetry integration for Inspect AI evaluation workflows.

## Overview

inspect-otel provides a standardised telemetry layer for LLM evaluation systems built with [Inspect AI](https://inspect.aisi.org.uk). It exports traces to any OpenTelemetry-compatible backend:

- Arize Phoenix (`http://localhost:6006/v1/traces`)
- Grafana / Jaeger / Tempo
- Any OTLP/HTTP-compatible collector

## Features

- OpenTelemetry-native tracing via the `inspect_ai.hooks.Hooks` interface
- Per-task root spans, per-sample child spans, **per-LLM-call and per-tool-call grandchild spans**, fully nested
- `contextvars`-based span context propagation — concurrent samples do not bleed context into one another
- Multiple scorer support — every scorer score is emitted as a separate `eval.score.<scorer>` attribute
- `eval.dataset.version` and `eval.experiment_id` span attributes for experiment tracking
- Graceful shutdown via `on_eval_set_end` and `provider.shutdown()`
- Decorators for tracing LLM and tool calls in standalone code
- Sampling support (TraceIdRatioBased) for large-scale evals
- Async batching (BatchSpanProcessor) or synchronous mode
- Failure-only logging mode (skip successfully-scored samples)
- 100% test coverage (136 tests)

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
    rate: 1.0

  logging:
    failure_only: false

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
inspect.run              <- one per task  (on_task_start / on_task_end)
  +-- inspect.sample     <- one per sample (on_sample_start ... on_sample_end)
        +-- llm.call     <- one per model API call (on_model_usage)
        +-- tool.call    <- one per tool invocation (on_sample_event / ToolEvent)
```

Span context is propagated via `contextvars.ContextVar` so that concurrent samples running in parallel asyncio tasks never mix their parent references.

### Key Span Attributes

#### `inspect.run`

| Attribute | Source |
|---|---|
| `eval.run_id` | `TaskStart.run_id` |
| `eval.name` | `EvalSpec.task` |
| `eval.type` | always `"benchmark"` |
| `llm.model` | `EvalSpec.model` |
| `eval.dataset` | `EvalSpec.dataset.name` |
| `eval.dataset.version` | `EvalSpec.dataset.version` |
| `eval.experiment_id` | `EvalSpec.metadata["experiment_id"]` |

#### `inspect.sample`

| Attribute | Source |
|---|---|
| `eval.sample_id` | `EvalSample.id` |
| `eval.score` | first non-None score (legacy convenience key) |
| `eval.score.<scorer>` | per-scorer value from `EvalSample.scores` |
| `llm.latency_ms` | `EvalSample.total_time * 1000` |
| `llm.model` | first key of `EvalSample.model_usage` |
| `eval.input.*` | stringified inputs |
| `eval.output.*` | stringified outputs |
| `eval.expected.*` | stringified expected values |

#### `llm.call`

| Attribute | Source |
|---|---|
| `llm.model` | `ModelUsageData.model_name` |
| `llm.input_tokens` | `ModelUsage.input_tokens` |
| `llm.output_tokens` | `ModelUsage.output_tokens` |
| `llm.total_tokens` | `ModelUsage.total_tokens` |
| `llm.latency_ms` | `ModelUsageData.call_duration * 1000` |
| `llm.retries` | `ModelUsageData.retries` (only when > 0) |

#### `tool.call`

| Attribute | Source |
|---|---|
| `tool.name` | `ToolEvent.function` |
| `tool.input.<arg>` | `ToolEvent.arguments` (each key-value pair) |

---

## Configuration Reference

```yaml
telemetry:
  service_name: string        # OTel service.name resource attribute
  otlp_endpoint: string|null  # OTLP/HTTP endpoint (null = no export)

  sampling:
    rate: float               # 0.0-1.0, default 1.0

  logging:
    failure_only: bool        # default false -- only log failed samples

  async: bool                 # true = BatchSpanProcessor (default)
                              # false = SimpleSpanProcessor (blocks on each span)
```

---

## Graceful Shutdown

inspect-otel registers an `on_eval_set_end` hook that calls `manager.shutdown()`,
which in turn calls `provider.shutdown()` on every backend. This flushes any
buffered spans and cleanly terminates exporter connections at the end of the
eval set -- no `atexit` handler registration required.

---

## Multiple Scorers

When a sample has multiple scorers (e.g. both `accuracy` and `f1`), each score
is emitted as a separate attribute:

```
eval.score.accuracy = 1.0
eval.score.f1       = 0.87
eval.score          = 1.0   # first non-None value, legacy key
```

---

## Phoenix Integration

```bash
pip install arize-phoenix
python -m phoenix.server.main
```

Set `otlp_endpoint: http://localhost:6006/v1/traces` and open
`http://localhost:6006` in your browser.

See [example.md](example.md) for a full walkthrough.

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
