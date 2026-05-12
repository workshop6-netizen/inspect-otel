"""OpenTelemetry tracing backend for inspect-otel."""

from __future__ import annotations

import contextvars
import json
import threading
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace import StatusCode

_TRACER_NAME = "inspect-otel"
_SERVICE_NAME_ATTR = "service.name"

# OpenInference semantic convention keys understood by Arize Phoenix
_SPAN_KIND = "openinference.span.kind"
_INPUT_VALUE = "input.value"
_OUTPUT_VALUE = "output.value"
_LLM_MODEL_NAME = "llm.model_name"
_LLM_PROMPT_TOKENS = "llm.token_count.prompt"
_LLM_COMPLETION_TOKENS = "llm.token_count.completion"
_LLM_TOTAL_TOKENS = "llm.token_count.total"

# ContextVar that carries the current OTel context across async boundaries.
# Each asyncio task (sample) runs in its own copy of the context, so setting
# this var in one sample does not bleed into another.
_current_otel_ctx: contextvars.ContextVar[otel_context.Context | None] = (
    contextvars.ContextVar("_current_otel_ctx", default=None)
)


class OpenTelemetryBackend:
    """OpenTelemetry tracing backend.

    Manages a TracerProvider, creates a root span per eval run, and emits
    child spans for each sample, LLM call, and tool event.  Supports async
    batching and failure-only logging mode.
    """

    def __init__(
        self,
        service_name: str,
        otlp_endpoint: str | None,
        sample_rate: float = 1.0,
        failure_only: bool = False,
        async_mode: bool = True,
    ) -> None:
        resource = Resource(attributes={_SERVICE_NAME_ATTR: service_name})
        clamped_rate = min(1.0, max(0.0, sample_rate))
        sampler = ParentBased(TraceIdRatioBased(clamped_rate))
        self._provider = TracerProvider(resource=resource, sampler=sampler)

        if otlp_endpoint:
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            processor: BatchSpanProcessor | SimpleSpanProcessor = (
                BatchSpanProcessor(exporter)
                if async_mode
                else SimpleSpanProcessor(exporter)
            )
            self._provider.add_span_processor(processor)

        self._tracer = self._provider.get_tracer(_TRACER_NAME)
        self._failure_only = failure_only
        self._run_spans: dict[str, trace.Span] = {}
        # Keyed by (eval_id, sample_id); stores open inspect.sample spans.
        self._sample_spans: dict[tuple[str, str], trace.Span] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public interface used by TelemetryManager
    # ------------------------------------------------------------------

    def start_run(self, run_id: str, metadata: dict[str, Any]) -> None:
        """Open a root span for an eval run."""
        span = self._tracer.start_span("inspect.run")
        span.set_attribute(_SPAN_KIND, "CHAIN")
        for key, value in metadata.items():
            if value is not None:
                span.set_attribute(key, str(value))
        with self._lock:
            self._run_spans[run_id] = span

    def start_sample(
        self, run_id: str, sample_id: str, metadata: dict[str, Any]
    ) -> None:
        """Open an ``inspect.sample`` child span and propagate context via ContextVar.

        The span stays open until :meth:`log_sample` closes it.  While open,
        any LLM or tool child spans created in the same asyncio-task context
        will be nested under it automatically.
        """
        with self._lock:
            run_span = self._run_spans.get(run_id)
        if run_span is None:
            return

        parent_ctx = trace.set_span_in_context(run_span)
        span = self._tracer.start_span("inspect.sample", context=parent_ctx)
        span.set_attribute(_SPAN_KIND, "CHAIN")
        span.set_attribute("eval.sample_id", sample_id)
        for key, value in metadata.items():
            if value is not None:
                span.set_attribute(key, str(value))

        with self._lock:
            self._sample_spans[(run_id, sample_id)] = span

        # Propagate the sample span as current OTel context so that child
        # spans (llm.call, tool.call) created in this asyncio task are
        # automatically parented under it.
        _current_otel_ctx.set(trace.set_span_in_context(span))

    def log_model_call(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        latency_ms: float,
    ) -> None:
        """Emit an ``llm.call`` child span with token counts and latency.

        Uses the active OTel context (set by start_sample) so the span is
        automatically nested under the current sample span.
        """
        parent_ctx = _current_otel_ctx.get()
        if parent_ctx is None:
            return

        with self._tracer.start_as_current_span("llm.call", context=parent_ctx) as span:
            span.set_attribute(_SPAN_KIND, "LLM")
            span.set_attribute(_LLM_MODEL_NAME, model_name)
            span.set_attribute(_LLM_PROMPT_TOKENS, input_tokens)
            span.set_attribute(_LLM_COMPLETION_TOKENS, output_tokens)
            span.set_attribute(_LLM_TOTAL_TOKENS, total_tokens)
            span.set_attribute("llm.latency_ms", latency_ms)
            span.set_status(StatusCode.OK)

    def log_tool_event(
        self,
        run_id: str,
        eval_id: str,
        sample_id: str,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> None:
        """Emit a ``tool.call`` child span nested under the active sample span."""
        parent_ctx = _current_otel_ctx.get()
        if parent_ctx is None:
            with self._lock:
                sample_span = self._sample_spans.get((eval_id, sample_id))
            if sample_span is None:
                return
            parent_ctx = trace.set_span_in_context(sample_span)

        with self._tracer.start_as_current_span("tool.call", context=parent_ctx) as span:
            span.set_attribute(_SPAN_KIND, "TOOL")
            span.set_attribute("tool.name", tool_name)
            span.set_attribute(_INPUT_VALUE, json.dumps(tool_input))

    def log_sample(
        self,
        run_id: str,
        sample_id: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        expected: dict[str, Any],
        scores: dict[str, float | None],
        metadata: dict[str, Any],
    ) -> None:
        """Finalise and close the ``inspect.sample`` span opened by :meth:`start_sample`.

        If :meth:`start_sample` was never called, a self-contained span is
        created and closed immediately so no data is lost.
        """
        with self._lock:
            span = self._sample_spans.pop((run_id, sample_id), None)
            run_span = self._run_spans.get(run_id)

        # Determine a representative float score (first non-None entry).
        score: float | None = next(
            (v for v in scores.values() if v is not None), None
        )
        passed = score is not None and score > 0

        if self._failure_only and passed:
            # Do NOT call span.end() — OTel only exports on end(); silently
            # dropping an unended span avoids emitting it to the backend.
            return

        if span is None:
            # Fallback: no start_sample call — emit a self-contained span.
            if run_span is None:
                return
            parent_ctx = trace.set_span_in_context(run_span)
            span = self._tracer.start_span("inspect.sample", context=parent_ctx)
            span.set_attribute(_SPAN_KIND, "CHAIN")
            span.set_attribute("eval.sample_id", str(sample_id))

        # Phoenix renders input.value / output.value as the primary I/O display.
        input_text = next(iter(inputs.values()), None)
        output_text = next(iter(outputs.values()), None)
        if input_text is not None:
            span.set_attribute(_INPUT_VALUE, str(input_text))
        if output_text is not None:
            span.set_attribute(_OUTPUT_VALUE, str(output_text))

        # Keep full dicts as structured attributes for completeness.
        _set_prefixed_attributes(span, "eval.input", inputs)
        _set_prefixed_attributes(span, "eval.output", outputs)
        _set_prefixed_attributes(span, "eval.expected", expected)

        for metric, value in scores.items():
            if value is not None:
                span.set_attribute(f"eval.score.{metric}", float(value))
        if score is not None:
            span.set_attribute("eval.score", float(score))

        for key, value in metadata.items():
            if value is not None:
                span.set_attribute(key, str(value))

        if passed:
            span.set_status(StatusCode.OK)
        elif score is not None:
            span.set_status(StatusCode.ERROR, "Sample did not pass")

        span.end()

    def end_run(self, run_id: str) -> None:
        """Close the root span for the given run."""
        with self._lock:
            span = self._run_spans.pop(run_id, None)
        if span is not None:
            span.end()

    def flush(self) -> None:
        """Force-flush all pending spans to the exporter."""
        self._provider.force_flush()

    def shutdown(self) -> None:
        """Flush and shut down the underlying :class:`TracerProvider`."""
        self._provider.shutdown()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _set_prefixed_attributes(
    span: trace.Span, prefix: str, data: dict[str, Any]
) -> None:
    for key, value in data.items():
        if value is not None:
            span.set_attribute(f"{prefix}.{key}", str(value))
