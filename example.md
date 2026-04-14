# End-to-End Example: Inspect AI + inspect-otel + Arize Phoenix

This walkthrough shows how to run an Inspect AI benchmark, collect
OpenTelemetry traces with inspect-otel, and explore them in the Arize Phoenix
UI — from scratch.

---

## 1. Prerequisites

Python 3.10+ and `pip` are required.

```bash
python --version   # 3.10 or newer
```

---

## 2. Install Dependencies

```bash
# Core libraries
pip install inspect-ai>=0.3.70 inspect-otel

# Arize Phoenix (local OTLP collector + UI)
pip install arize-phoenix
```

---

## 3. Start Arize Phoenix

Open a **separate terminal** and run:

```bash
python -m phoenix.server.main
```

Phoenix starts two services:

| Service | URL | Purpose |
|---|---|---|
| Web UI | `http://localhost:6006` | Trace browser |
| OTLP/HTTP | `http://localhost:6006/v1/traces` | Span ingestion endpoint |

Leave this terminal running. Navigate to `http://localhost:6006` in your
browser — you will see an empty Projects view until traces arrive.

---

## 4. Create the Config File

Create `config.yaml` in your project directory:

```yaml
telemetry:
  service_name: math-benchmark
  otlp_endpoint: http://localhost:6006/v1/traces

  sampling:
    rate: 1.0           # collect every span

  logging:
    failure_only: false # log all samples (pass and fail)

  async: true           # use BatchSpanProcessor for throughput
```

---

## 5. Create the Benchmark

Create `benchmark.py`:

```python
"""Simple arithmetic benchmark wired to inspect-otel."""

from inspect_ai import eval_set, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import chain_of_thought, generate
from inspect_otel import setup_hooks

# ------------------------------------------------------------------
# 1. Register OpenTelemetry hooks BEFORE calling eval_set()
# ------------------------------------------------------------------
setup_hooks(config_path="config.yaml")

# ------------------------------------------------------------------
# 2. Define the benchmark dataset inline
# ------------------------------------------------------------------
SAMPLES = [
    Sample(input="What is 12 + 7?",    target="19"),
    Sample(input="What is 144 / 12?",  target="12"),
    Sample(input="What is 8 * 9?",     target="72"),
    Sample(input="What is 100 - 37?",  target="63"),
    Sample(input="What is 2 ** 10?",   target="1024"),
    Sample(input="What is sqrt(81)?",  target="9"),
    Sample(input="What is 17 % 5?",    target="2"),
    Sample(input="What is 3! ?",       target="6"),
]


# ------------------------------------------------------------------
# 3. Define the task
# ------------------------------------------------------------------
@task
def math_qa():
    return dict(
        dataset=SAMPLES,
        solver=[chain_of_thought(), generate()],
        scorer=match(),
    )


# ------------------------------------------------------------------
# 4. Run the eval set
# ------------------------------------------------------------------
if __name__ == "__main__":
    eval_set(
        evals=[math_qa()],
        model="openai/gpt-4o-mini",  # swap for your preferred model
    )
```

---

## 6. Set Your API Key

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="..."

# Azure OpenAI
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://..."
```

---

## 7. Run the Benchmark

```bash
python benchmark.py
```

inspect-ai will fire hooks in this order:

1. `on_task_start` → inspect-otel opens an `inspect.run` span.
2. For every sample: `on_sample_start` → opens an `inspect.sample` span and sets `ContextVar`.
3. For every model API call: `on_model_usage` → emits an `llm.call` span with token counts and latency.
4. For every tool invocation (if tools are used): `on_sample_event` → emits a `tool.call` span.
5. `on_sample_end` → finalises the `inspect.sample` span with scores and sets span status.
6. `on_task_end` → closes `inspect.run`, flushes spans to Phoenix.
7. `on_eval_set_end` → calls `provider.shutdown()` for graceful teardown.

---

## 8. Explore Traces in Phoenix

Open `http://localhost:6006` in your browser.

### 8.1 Projects

Phoenix automatically groups spans by `service.name`. After the benchmark
finishes you will see a **math-benchmark** project.

### 8.2 Trace List

Click the project to see the list of traces. Each row represents one
`inspect.run` root span (one per task execution).

### 8.3 Span Details

Click a trace to open the waterfall view:

```
inspect.run  [task duration]
  +-- inspect.sample  [sample duration]    sample_id=1, eval.score=1.0
        +-- llm.call  [API latency]        model=gpt-4o-mini, tokens=87
        +-- llm.call  [API latency]        model=gpt-4o-mini, tokens=92
```

Each `inspect.sample` span carries:

| Attribute | Example Value | Description |
|---|---|---|
| `eval.sample_id` | `1` | Which sample was evaluated |
| `eval.score.accuracy` | `1.0` | Per-scorer score |
| `eval.score` | `1.0` | Legacy key (first non-None value) |
| `eval.input.input` | `"What is 12 + 7?"` | The question asked |
| `eval.output.output` | `"19"` | The model's answer |
| `eval.expected.target` | `"19"` | The correct answer |
| `llm.latency_ms` | `1234.5` | Total sample wall-clock time |

Each `llm.call` span carries:

| Attribute | Example Value | Description |
|---|---|---|
| `llm.model` | `gpt-4o-mini` | Model name |
| `llm.input_tokens` | `42` | Prompt tokens |
| `llm.output_tokens` | `18` | Completion tokens |
| `llm.total_tokens` | `60` | Total tokens |
| `llm.latency_ms` | `823.1` | Per-API-call latency |

### 8.4 Filtering and Querying

Use Phoenix's built-in filters to find:

- **Failing samples:** filter `eval.score = 0`
- **High-latency calls:** sort by `llm.latency_ms` descending
- **Specific experiments:** filter `eval.experiment_id = "my-exp"`
- **Specific dataset version:** filter `eval.dataset.version = "v2"`

---

## 9. Tracking Experiments

Tag each run with an experiment ID so you can compare multiple runs in Phoenix side-by-side. Pass it via `EvalSpec.metadata`:

```python
@task
def math_qa():
    return dict(
        dataset=SAMPLES,
        solver=[chain_of_thought(), generate()],
        scorer=match(),
        metadata={"experiment_id": "exp-001"},
    )
```

inspect-otel reads `EvalSpec.metadata["experiment_id"]` in `on_task_start` and
records it as `eval.experiment_id` on the `inspect.run` span.

---

## 10. Tracking Dataset Versions

If your dataset object has a `version` attribute, it is automatically recorded as `eval.dataset.version`:

```python
from inspect_ai.dataset import MemoryDataset

dataset = MemoryDataset(samples=SAMPLES, name="math-qa", version="v2.0")
```

In Phoenix you can then filter by `eval.dataset.version` to compare v1 vs v2 results side-by-side.

---

## 11. Multiple Scorers

Define multiple scorers to get per-scorer attributes on every sample span:

```python
from inspect_ai.scorer import match, includes

@task
def math_qa_multi_scorer():
    return dict(
        dataset=SAMPLES,
        solver=[chain_of_thought(), generate()],
        scorer=[match(), includes()],
    )
```

inspect-otel emits:

```
eval.score.match    = 1.0
eval.score.includes = 1.0
eval.score          = 1.0   # first non-None, legacy key
```

---

## 12. Failure-Only Mode

For large benchmarks (10k+ samples) where you only care about failures:

```yaml
telemetry:
  logging:
    failure_only: true
```

Samples with a score > 0 are silently dropped before being sent to Phoenix. The `inspect.sample` span is opened by `on_sample_start` but never ended, so it is never exported. `llm.call` and `tool.call` spans are still emitted regardless.

---

## 13. Sampling for Very Large Evals

To collect only a fraction of spans (e.g. 20%):

```yaml
telemetry:
  sampling:
    rate: 0.2
```

The OTel SDK applies a `TraceIdRatioBased(0.2)` sampler, wrapped in
`ParentBased` so child spans are always consistent with their parent.

---

## 14. Using Tools

To see `tool.call` spans, add a tool to the solver chain:

```python
from inspect_ai.tools import web_search
from inspect_ai.solver import use_tools

@task
def math_qa_with_search():
    return dict(
        dataset=SAMPLES,
        solver=[use_tools(web_search()), chain_of_thought(), generate()],
        scorer=match(),
    )
```

Every time the model calls `web_search`, inspect-otel emits a `tool.call` span with:

- `tool.name = "web_search"`
- `tool.input.query = "<the search query>"`

---

## 15. Hook Lifecycle Summary

| Hook | inspect-otel action | OTel span |
|---|---|---|
| `on_task_start` | `manager.start_run()` | Opens `inspect.run` |
| `on_sample_start` | `manager.start_sample()`, sets ContextVar | Opens `inspect.sample` |
| `on_model_usage` | `manager.log_model_call()` | Emits `llm.call` (child of sample) |
| `on_sample_event[ToolEvent]` | `manager.log_tool_event()` | Emits `tool.call` (child of sample) |
| `on_sample_end` | `manager.log_sample()` | Closes `inspect.sample` with scores |
| `on_task_end` | `manager.end_run()`, `manager.flush()` | Closes `inspect.run`, flushes |
| `on_eval_set_end` | `manager.shutdown()` | Calls `provider.shutdown()` |
