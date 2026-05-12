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
_SESSION_ID = "session.id"
_LLM_MODEL_NAME = "llm.model_name"
_LLM_PROMPT_TOKENS = "llm.token_count.prompt"
_LLM_COMPLETION_TOKENS = "llm.token_count.completion"
_LLM_TOTAL_TOKENS = "llm.token_count.total"
_LLM_CACHE_READ_TOKENS = "llm.token_count.prompt_details.cache_read"
_LLM_CACHE_WRITE_TOKENS = "llm.token_count.prompt_details.cache_write"
_LLM_REASONING_TOKENS = "llm.token_count.completion_details.reasoning"
_LLM_FINISH_REASON = "llm.output_messages.0.message.finish_reason"

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
        span.set_attribute(_SESSION_ID, run_id)
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
        span.set_attribute(_SESSION_ID, run_id)
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
        cache_read_tokens: int | None = None,
        cache_write_tokens: int | None = None,
        reasoning_tokens: int | None = None,
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
            if cache_read_tokens is not None:
                span.set_attribute(_LLM_CACHE_READ_TOKENS, cache_read_tokens)
            if cache_write_tokens is not None:
                span.set_attribute(_LLM_CACHE_WRITE_TOKENS, cache_write_tokens)
            if reasoning_tokens is not None:
                span.set_attribute(_LLM_REASONING_TOKENS, reasoning_tokens)
            span.set_status(StatusCode.OK)

    def log_model_cache_call(
        self,
        model_name: str,
        cache_read_tokens: int,
        total_tokens: int,
    ) -> None:
        """Emit an ``llm.cache_hit`` child span for cache-served responses."""
        parent_ctx = _current_otel_ctx.get()
        if parent_ctx is None:
            return

        with self._tracer.start_as_current_span("llm.cache_hit", context=parent_ctx) as span:
            span.set_attribute(_SPAN_KIND, "LLM")
            span.set_attribute(_LLM_MODEL_NAME, model_name)
            span.set_attribute(_LLM_CACHE_READ_TOKENS, cache_read_tokens)
            span.set_attribute(_LLM_TOTAL_TOKENS, total_tokens)
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
        score_details: dict[str, dict[str, Any]],
        messages: list[dict[str, str]],
        metadata: dict[str, Any],
        finish_reason: str | None = None,
        epoch: int | None = None,
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
            span.set_attribute(_SESSION_ID, run_id)
            span.set_attribute("eval.sample_id", str(sample_id))

        # Primary I/O text for Phoenix's span header.
        input_text = next(iter(inputs.values()), None)
        output_text = next(iter(outputs.values()), None)
        if input_text is not None:
            span.set_attribute(_INPUT_VALUE, str(input_text))
        if output_text is not None:
            span.set_attribute(_OUTPUT_VALUE, str(output_text))

        # Structured eval attributes.
        _set_prefixed_attributes(span, "eval.input", inputs)
        _set_prefixed_attributes(span, "eval.output", outputs)
        _set_prefixed_attributes(span, "eval.expected", expected)

        for metric, value in scores.items():
            if value is not None:
                span.set_attribute(f"eval.score.{metric}", float(value))
        if score is not None:
            span.set_attribute("eval.score", float(score))

        if epoch is not None:
            span.set_attribute("eval.epoch", epoch)

        if finish_reason:
            span.set_attribute(_LLM_FINISH_REASON, finish_reason)

        # Full conversation as LLM message attributes (Phoenix chat view).
        _set_message_attributes(span, messages)

        for key, value in metadata.items():
            if value is not None:
                span.set_attribute(key, str(value))

        if passed:
            span.set_status(StatusCode.OK)
        elif score is not None:
            span.set_status(StatusCode.ERROR, "Sample did not pass")

        # EVALUATOR child spans — one per scorer, created before ending parent.
        sample_ctx = trace.set_span_in_context(span)
        for scorer_name, score_value in scores.items():
            details = score_details.get(scorer_name, {})
            self._emit_evaluator_span(
                parent_ctx=sample_ctx,
                scorer_name=scorer_name,
                score_value=score_value,
                label=details.get("label"),
                explanation=details.get("explanation"),
                answer=details.get("answer"),
                input_text=input_text,
            )

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
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit_evaluator_span(
        self,
        parent_ctx: otel_context.Context,
        scorer_name: str,
        score_value: float | None,
        label: str | None,
        explanation: str | None,
        answer: str | None,
        input_text: str | None,
    ) -> None:
        """Emit an EVALUATOR child span for a single scorer result."""
        with self._tracer.start_as_current_span(
            f"eval.{scorer_name}", context=parent_ctx
        ) as span:
            span.set_attribute(_SPAN_KIND, "EVALUATOR")
            span.set_attribute("eval.name", scorer_name)
            if input_text is not None:
                span.set_attribute(_INPUT_VALUE, input_text)
            if score_value is not None:
                span.set_attribute("eval.score", float(score_value))
                span.set_attribute(_OUTPUT_VALUE, str(score_value))
            if label is not None:
                span.set_attribute("eval.label", label)
            if explanation is not None:
                span.set_attribute("eval.explanation", explanation)
            if answer is not None:
                span.set_attribute("eval.answer", answer)
            if score_value is not None and score_value > 0:
                span.set_status(StatusCode.OK)
            elif score_value is not None:
                span.set_status(StatusCode.ERROR, "Did not pass")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _set_prefixed_attributes(
    span: trace.Span, prefix: str, data: dict[str, Any]
) -> None:
    for key, value in data.items():
        if value is not None:
            span.set_attribute(f"{prefix}.{key}", str(value))


def _set_message_attributes(
    span: trace.Span, messages: list[dict[str, str]]
) -> None:
    """Write conversation history using OpenInference flat-indexed message format.

    Input messages (user/system) go to ``llm.input_messages.*``.
    The final assistant message goes to ``llm.output_messages.*``.
    """
    if not messages:
        return

    # Split: last assistant message → output; everything else → input.
    if messages[-1].get("role") == "assistant":
        input_msgs = messages[:-1]
        output_msgs = messages[-1:]
    else:
        input_msgs = messages
        output_msgs = []

    for i, msg in enumerate(input_msgs):
        prefix = f"llm.input_messages.{i}.message"
        span.set_attribute(f"{prefix}.role", msg.get("role", ""))
        if msg.get("content"):
            span.set_attribute(f"{prefix}.content", msg["content"])

    for i, msg in enumerate(output_msgs):
        prefix = f"llm.output_messages.{i}.message"
        span.set_attribute(f"{prefix}.role", msg.get("role", ""))
        if msg.get("content"):
            span.set_attribute(f"{prefix}.content", msg["content"])
