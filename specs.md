# inspect-otel Specification

## Purpose

Provide a standardised OpenTelemetry-based telemetry layer for Inspect AI evaluation
pipelines, enabling traces, spans, and attributes to be exported to any OTLP-compatible
backend (Phoenix, Jaeger, Tempo, etc.).

---

## Core Concepts

### Run (Task)

A *run* maps 1-to-1 with a single Inspect AI task execution. It is opened by
`on_task_start` and closed by `on_task_end`.

**Span name:** `inspect.run`

**Attributes:**

| Key | Type | Description |
|---|---|---|
| `eval.run_id` | string | `TaskStart.run_id` |
| `eval.name` | string | `EvalSpec.task` name |
| `eval.type` | string | Always `"benchmark"` |
| `llm.model` | string | Model identifier from `EvalSpec.model` |
| `eval.dataset` | string | `EvalSpec.dataset.name` (if present) |
| `eval.dataset.version` | string | `EvalSpec.dataset.version` (if present) |
| `eval.experiment_id` | string | `EvalSpec.metadata["experiment_id"]` (if present) |

---

### Sample

A *sample* maps to one evaluation data-point within a task. It is opened by
`on_sample_start` (creating a span that stays open), and finalized with all
scores and metadata by `on_sample_end`.

**Span name:** `inspect.sample`

**Attributes:**

| Key | Type | Description |
|---|---|---|
| `eval.sample_id` | string | `EvalSample.id` |
| `eval.score` | float | First non-None scorer value (legacy convenience) |
| `eval.score.<scorer>` | float | Per-scorer value (one attribute per scorer) |
| `eval.input.input` | string | Stringified sample input |
| `eval.output.output` | string | Stringified sample output |
| `eval.expected.target` | string | Stringified target / expected value |
| `llm.latency_ms` | float | `EvalSample.total_time * 1000` |
| `llm.model` | string | First model name from `EvalSample.model_usage` |

**Span status:**

- `OK` when the first non-None score is > 0
- `ERROR` when a score is present but <= 0
- `UNSET` when no score is available

---

### LLM Call

An *llm.call* span is emitted for every cached-miss model API invocation via the
`on_model_usage` hook. It is a child of the active `inspect.sample` span (via
`contextvars.ContextVar` context propagation) or of `inspect.run` if no sample
context is set.

**Span name:** `llm.call`

**Attributes:**

| Key | Type | Description |
|---|---|---|
| `llm.model` | string | `ModelUsageData.model_name` |
| `llm.input_tokens` | int | `ModelUsage.input_tokens` |
| `llm.output_tokens` | int | `ModelUsage.output_tokens` |
| `llm.total_tokens` | int | `ModelUsage.total_tokens` |
| `llm.latency_ms` | float | `ModelUsageData.call_duration * 1000` |
| `llm.retries` | int | `ModelUsageData.retries` (only when > 0) |

**Span status:** Always `OK`.

---

### Tool Call

A *tool.call* span is emitted for every `ToolEvent` received via the
`on_sample_event` hook. Non-`ToolEvent` events are silently ignored.

**Span name:** `tool.call`

**Attributes:**

| Key | Type | Description |
|---|---|---|
| `tool.name` | string | `ToolEvent.function` |
| `tool.input.<arg>` | string | Each key-value pair from `ToolEvent.arguments` |

---

## Span Hierarchy

```
inspect.run              (on_task_start -> on_task_end)
  +-- inspect.sample     (on_sample_start -> on_sample_end)
        +-- llm.call     (on_model_usage)
        +-- tool.call    (on_sample_event[ToolEvent])
```

---

## Context Propagation

Each `on_sample_start` call sets a `contextvars.ContextVar` (`_current_otel_ctx`)
to an OTel `Context` carrying the newly opened `inspect.sample` span as the
current span.

Because Inspect AI runs samples as independent `asyncio` tasks, each task
inherits a copy of the context at creation time. Setting the ContextVar inside
`on_sample_start` therefore scopes the value to the sample's task only;
concurrent samples do not share context.

`on_model_usage` and `on_sample_event` read this ContextVar to parent `llm.call`
and `tool.call` spans correctly under the right sample.

---

## Multiple Scorer Support

All scorer results in `EvalSample.scores` are iterated and emitted as
individual span attributes (`eval.score.<scorer_name>`). If `as_float()` raises
`TypeError` or `ValueError`, the scorer's value is stored as `None` and omitted
from the span. The first non-None value is also emitted under the legacy key
`eval.score` for backwards compatibility.

---

## Failure-Only Logging

When `failure_only: true` is configured:

- Samples with a first non-None score > 0 are considered *passing*.
- For passing samples, `log_sample()` returns early **without** calling
  `span.end()` on the sample span opened by `start_sample()`.
- Because the OTel SDK only exports a span when `end()` is called, the span is
  silently dropped without appearing in the backend.
- `llm.call` and `tool.call` child spans are emitted regardless (they belong to
  a global in-process tracer context and are independent of `failure_only`).

---

## Graceful Shutdown

`on_eval_set_end` is called by Inspect AI when the entire eval set finishes.
The hook calls `manager.shutdown()`, which fans out to `backend.shutdown()` on
every registered backend. This calls `TracerProvider.shutdown()`, which:

1. Flushes all spans buffered in `BatchSpanProcessor` to the exporter.
2. Terminates exporter connections cleanly.
3. Prevents further spans from being recorded.

No `atexit` handler is required.

---

## Configuration Schema

```yaml
telemetry:
  service_name: string        # OTel service.name resource attribute
  otlp_endpoint: string|null  # OTLP/HTTP endpoint

  sampling:
    rate: float               # 0.0-1.0, default 1.0 (TraceIdRatioBased)

  logging:
    failure_only: bool        # default false

  async: bool                 # true = BatchSpanProcessor (default)
                              # false = SimpleSpanProcessor
```

---

## Performance Requirements

- Must support 100k+ samples per eval set.
- `BatchSpanProcessor` is the default; use `async: false` only for debugging.
- Sampling is applied at the `TracerProvider` level via a `ParentBased(TraceIdRatioBased(rate))` sampler.

---

## Compatibility

- OpenTelemetry SDK >= 1.25 (OTLP/HTTP protobuf exporter)
- inspect-ai >= 0.3.70 (`inspect_ai.hooks.Hooks` API)
- Python >= 3.10

Tested backends:
- Arize Phoenix (via OTLP/HTTP)
- Jaeger / Tempo / Grafana

---

## Extensibility

- Additional backends: subclass `OpenTelemetryBackend` and pass to `TelemetryManager`.
- Custom span attributes: override `on_task_start` / `on_sample_end` in a subclass of `OpenTelemetryLogger`.
- Multiple exporters simultaneously: pass a list of backends to `TelemetryManager`.
