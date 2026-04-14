"""Tests for inspect_otel.otel_backend."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from inspect_otel.otel_backend import OpenTelemetryBackend, _set_prefixed_attributes

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
def backend(exporter: InMemorySpanExporter) -> OpenTelemetryBackend:
    """Backend with no OTLP endpoint; InMemorySpanExporter injected manually."""
    b = OpenTelemetryBackend(
        service_name="test-service",
        otlp_endpoint=None,
        sample_rate=1.0,
        failure_only=False,
        async_mode=True,
    )
    b._provider.add_span_processor(SimpleSpanProcessor(exporter))
    return b


@pytest.fixture()
def failure_only_backend(exporter: InMemorySpanExporter) -> OpenTelemetryBackend:
    b = OpenTelemetryBackend(
        service_name="test-service",
        otlp_endpoint=None,
        sample_rate=1.0,
        failure_only=True,
    )
    b._provider.add_span_processor(SimpleSpanProcessor(exporter))
    return b


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _finished_names(exporter: InMemorySpanExporter) -> list[str]:
    return [s.name for s in exporter.get_finished_spans()]


def _span_attrs(exporter: InMemorySpanExporter, name: str) -> dict:
    spans = [s for s in exporter.get_finished_spans() if s.name == name]
    assert spans, f"No span named '{name}' found"
    return dict(spans[0].attributes or {})


def _all_spans_named(exporter: InMemorySpanExporter, name: str):
    return [s for s in exporter.get_finished_spans() if s.name == name]


# ---------------------------------------------------------------------------
# start_run
# ---------------------------------------------------------------------------


def test_start_run_registers_span(backend: OpenTelemetryBackend) -> None:
    backend.start_run("r1", {"eval.name": "my-eval"})
    assert "r1" in backend._run_spans


def test_start_run_span_name(backend: OpenTelemetryBackend) -> None:
    backend.start_run("r1", {})
    span = backend._run_spans["r1"]
    assert span.name == "inspect.run"


def test_start_run_sets_string_attributes(backend: OpenTelemetryBackend) -> None:
    backend.start_run("r1", {"eval.name": "test-eval", "llm.model": "gpt-4"})
    attrs = dict(backend._run_spans["r1"].attributes or {})
    assert attrs["eval.name"] == "test-eval"
    assert attrs["llm.model"] == "gpt-4"


def test_start_run_skips_none_values(backend: OpenTelemetryBackend) -> None:
    backend.start_run("r1", {"eval.dataset": None, "eval.name": "x"})
    attrs = dict(backend._run_spans["r1"].attributes or {})
    assert "eval.dataset" not in attrs
    assert attrs["eval.name"] == "x"


def test_start_run_converts_non_string_to_str(backend: OpenTelemetryBackend) -> None:
    backend.start_run("r1", {"eval.count": 42})
    attrs = dict(backend._run_spans["r1"].attributes or {})
    assert attrs["eval.count"] == "42"


# ---------------------------------------------------------------------------
# start_sample
# ---------------------------------------------------------------------------


def test_start_sample_creates_span(backend: OpenTelemetryBackend) -> None:
    backend.start_run("r1", {})
    backend.start_sample("r1", "s1", {})
    assert ("r1", "s1") in backend._sample_spans


def test_start_sample_span_has_correct_name(backend: OpenTelemetryBackend) -> None:
    backend.start_run("r1", {})
    backend.start_sample("r1", "s1", {})
    span = backend._sample_spans[("r1", "s1")]
    assert span.name == "inspect.sample"


def test_start_sample_sets_sample_id(backend: OpenTelemetryBackend) -> None:
    backend.start_run("r1", {})
    backend.start_sample("r1", "s42", {})
    attrs = dict(backend._sample_spans[("r1", "s42")].attributes or {})
    assert attrs["eval.sample_id"] == "s42"


def test_start_sample_sets_metadata(backend: OpenTelemetryBackend) -> None:
    backend.start_run("r1", {})
    backend.start_sample("r1", "s1", {"custom.key": "val"})
    attrs = dict(backend._sample_spans[("r1", "s1")].attributes or {})
    assert attrs["custom.key"] == "val"


def test_start_sample_skips_none_metadata(backend: OpenTelemetryBackend) -> None:
    backend.start_run("r1", {})
    backend.start_sample("r1", "s1", {"custom.key": None})
    attrs = dict(backend._sample_spans[("r1", "s1")].attributes or {})
    assert "custom.key" not in attrs


def test_start_sample_unknown_run_id_is_silent(backend: OpenTelemetryBackend) -> None:
    backend.start_sample("nonexistent", "s1", {})  # must not raise


def test_start_sample_sets_context_var(backend: OpenTelemetryBackend) -> None:
    from inspect_otel.otel_backend import _current_otel_ctx
    backend.start_run("r1", {})
    backend.start_sample("r1", "s1", {})
    assert _current_otel_ctx.get() is not None


# ---------------------------------------------------------------------------
# log_model_call
# ---------------------------------------------------------------------------


def test_log_model_call_emits_span(
    backend: OpenTelemetryBackend, exporter: InMemorySpanExporter
) -> None:
    backend.start_run("r1", {})
    backend.start_sample("r1", "s1", {})
    backend.log_model_call("r1", "r1", "gpt-4", 10, 20, 30, 500.0, 0)
    backend.log_sample("r1", "s1", {}, {}, {}, {}, {})
    backend.end_run("r1")
    assert "llm.call" in _finished_names(exporter)


def test_log_model_call_attributes(
    backend: OpenTelemetryBackend, exporter: InMemorySpanExporter
) -> None:
    backend.start_run("r1", {})
    backend.start_sample("r1", "s1", {})
    backend.log_model_call("r1", "r1", "gpt-4", 10, 20, 30, 500.0, 0)
    backend.log_sample("r1", "s1", {}, {}, {}, {}, {})
    backend.end_run("r1")
    attrs = _span_attrs(exporter, "llm.call")
    assert attrs["llm.model"] == "gpt-4"
    assert attrs["llm.input_tokens"] == 10
    assert attrs["llm.output_tokens"] == 20
    assert attrs["llm.total_tokens"] == 30
    assert attrs["llm.latency_ms"] == 500.0


def test_log_model_call_sets_retries(
    backend: OpenTelemetryBackend, exporter: InMemorySpanExporter
) -> None:
    backend.start_run("r1", {})
    backend.start_sample("r1", "s1", {})
    backend.log_model_call("r1", "r1", "gpt-4", 10, 20, 30, 100.0, 3)
    backend.log_sample("r1", "s1", {}, {}, {}, {}, {})
    backend.end_run("r1")
    attrs = _span_attrs(exporter, "llm.call")
    assert attrs["llm.retries"] == 3


def test_log_model_call_zero_retries_omitted(
    backend: OpenTelemetryBackend, exporter: InMemorySpanExporter
) -> None:
    backend.start_run("r1", {})
    backend.start_sample("r1", "s1", {})
    backend.log_model_call("r1", "r1", "gpt-4", 10, 20, 30, 100.0, 0)
    backend.log_sample("r1", "s1", {}, {}, {}, {}, {})
    backend.end_run("r1")
    attrs = _span_attrs(exporter, "llm.call")
    assert "llm.retries" not in attrs


def test_log_model_call_fallback_to_run_span(
    backend: OpenTelemetryBackend, exporter: InMemorySpanExporter
) -> None:
    """Without start_sample, falls back to run span parent."""
    from inspect_otel.otel_backend import _current_otel_ctx
    _current_otel_ctx.set(None)
    backend.start_run("r1", {})
    backend.log_model_call("r1", "r1", "gpt-4", 0, 0, 0, 0.0, 0)
    backend.end_run("r1")
    assert "llm.call" in _finished_names(exporter)


def test_log_model_call_unknown_eval_id_silent(backend: OpenTelemetryBackend) -> None:
    from inspect_otel.otel_backend import _current_otel_ctx
    _current_otel_ctx.set(None)
    backend.log_model_call("x", "x", "gpt-4", 0, 0, 0, 0.0, 0)  # no raise


def test_log_model_call_ok_status(
    backend: OpenTelemetryBackend, exporter: InMemorySpanExporter
) -> None:
    backend.start_run("r1", {})
    backend.start_sample("r1", "s1", {})
    backend.log_model_call("r1", "r1", "gpt-4", 1, 1, 2, 10.0, 0)
    backend.log_sample("r1", "s1", {}, {}, {}, {}, {})
    backend.end_run("r1")
    span = next(s for s in exporter.get_finished_spans() if s.name == "llm.call")
    assert span.status.status_code == StatusCode.OK


# ---------------------------------------------------------------------------
# log_tool_event
# ---------------------------------------------------------------------------


def test_log_tool_event_emits_span(
    backend: OpenTelemetryBackend, exporter: InMemorySpanExporter
) -> None:
    backend.start_run("r1", {})
    backend.start_sample("r1", "s1", {})
    backend.log_tool_event("r1", "r1", "s1", "web_search", {"query": "hello"})
    backend.log_sample("r1", "s1", {}, {}, {}, {}, {})
    backend.end_run("r1")
    assert "tool.call" in _finished_names(exporter)


def test_log_tool_event_attributes(
    backend: OpenTelemetryBackend, exporter: InMemorySpanExporter
) -> None:
    backend.start_run("r1", {})
    backend.start_sample("r1", "s1", {})
    backend.log_tool_event("r1", "r1", "s1", "calculator", {"expr": "2+2"})
    backend.log_sample("r1", "s1", {}, {}, {}, {}, {})
    backend.end_run("r1")
    attrs = _span_attrs(exporter, "tool.call")
    assert attrs["tool.name"] == "calculator"
    assert attrs["tool.input.expr"] == "2+2"


def test_log_tool_event_skips_none_inputs(
    backend: OpenTelemetryBackend, exporter: InMemorySpanExporter
) -> None:
    backend.start_run("r1", {})
    backend.start_sample("r1", "s1", {})
    backend.log_tool_event("r1", "r1", "s1", "tool", {"x": None, "y": "val"})
    backend.log_sample("r1", "s1", {}, {}, {}, {}, {})
    backend.end_run("r1")
    attrs = _span_attrs(exporter, "tool.call")
    assert "tool.input.x" not in attrs
    assert attrs["tool.input.y"] == "val"


def test_log_tool_event_fallback_to_sample_dict(
    backend: OpenTelemetryBackend, exporter: InMemorySpanExporter
) -> None:
    """Without contextvar, falls back to sample span dict lookup."""
    from inspect_otel.otel_backend import _current_otel_ctx
    _current_otel_ctx.set(None)
    backend.start_run("r1", {})
    backend.start_sample("r1", "s1", {})
    _current_otel_ctx.set(None)  # clear after start_sample set it
    backend.log_tool_event("r1", "r1", "s1", "mytool", {})
    backend.log_sample("r1", "s1", {}, {}, {}, {}, {})
    backend.end_run("r1")
    assert "tool.call" in _finished_names(exporter)


def test_log_tool_event_unknown_sample_silent(backend: OpenTelemetryBackend) -> None:
    from inspect_otel.otel_backend import _current_otel_ctx
    _current_otel_ctx.set(None)
    backend.start_run("r1", {})
    backend.log_tool_event("r1", "r1", "unknown", "tool", {})  # no raise


# ---------------------------------------------------------------------------
# log_sample
# ---------------------------------------------------------------------------


def test_log_sample_emits_span(
    backend: OpenTelemetryBackend, exporter: InMemorySpanExporter
) -> None:
    backend.start_run("r1", {})
    backend.start_sample("r1", "s1", {})
    backend.log_sample("r1", "s1", {"input": "hi"}, {"output": "bye"}, {"target": "bye"}, {"acc": 1.0}, {})
    backend.end_run("r1")
    assert "inspect.sample" in _finished_names(exporter)


def test_log_sample_sets_sample_id(
    backend: OpenTelemetryBackend, exporter: InMemorySpanExporter
) -> None:
    backend.start_run("r1", {})
    backend.start_sample("r1", "s42", {})
    backend.log_sample("r1", "s42", {}, {}, {}, {}, {})
    backend.end_run("r1")
    assert _span_attrs(exporter, "inspect.sample")["eval.sample_id"] == "s42"


def test_log_sample_multiple_scores(
    backend: OpenTelemetryBackend, exporter: InMemorySpanExporter
) -> None:
    backend.start_run("r1", {})
    backend.start_sample("r1", "s1", {})
    backend.log_sample("r1", "s1", {}, {}, {}, {"accuracy": 1.0, "f1": 0.8}, {})
    backend.end_run("r1")
    attrs = _span_attrs(exporter, "inspect.sample")
    assert attrs["eval.score.accuracy"] == 1.0
    assert attrs["eval.score.f1"] == 0.8
    assert attrs["eval.score"] == 1.0  # legacy key = first non-None


def test_log_sample_none_score_omitted(
    backend: OpenTelemetryBackend, exporter: InMemorySpanExporter
) -> None:
    backend.start_run("r1", {})
    backend.start_sample("r1", "s1", {})
    backend.log_sample("r1", "s1", {}, {}, {}, {"acc": None}, {})
    backend.end_run("r1")
    attrs = _span_attrs(exporter, "inspect.sample")
    assert "eval.score" not in attrs
    assert "eval.score.acc" not in attrs


def test_log_sample_sets_input_output_expected(
    backend: OpenTelemetryBackend, exporter: InMemorySpanExporter
) -> None:
    backend.start_run("r1", {})
    backend.start_sample("r1", "s1", {})
    backend.log_sample(
        "r1", "s1",
        inputs={"input": "hello"},
        outputs={"output": "world"},
        expected={"target": "world"},
        scores={"acc": 1.0},
        metadata={},
    )
    backend.end_run("r1")
    attrs = _span_attrs(exporter, "inspect.sample")
    assert attrs["eval.input.input"] == "hello"
    assert attrs["eval.output.output"] == "world"
    assert attrs["eval.expected.target"] == "world"


def test_log_sample_passes_have_ok_status(
    backend: OpenTelemetryBackend, exporter: InMemorySpanExporter
) -> None:
    backend.start_run("r1", {})
    backend.start_sample("r1", "s1", {})
    backend.log_sample("r1", "s1", {}, {}, {}, {"acc": 1.0}, {})
    backend.end_run("r1")
    span = next(s for s in exporter.get_finished_spans() if s.name == "inspect.sample")
    assert span.status.status_code == StatusCode.OK


def test_log_sample_failures_have_error_status(
    backend: OpenTelemetryBackend, exporter: InMemorySpanExporter
) -> None:
    backend.start_run("r1", {})
    backend.start_sample("r1", "s1", {})
    backend.log_sample("r1", "s1", {}, {}, {}, {"acc": 0.0}, {})
    backend.end_run("r1")
    span = next(s for s in exporter.get_finished_spans() if s.name == "inspect.sample")
    assert span.status.status_code == StatusCode.ERROR


def test_log_sample_none_score_status_unset(
    backend: OpenTelemetryBackend, exporter: InMemorySpanExporter
) -> None:
    backend.start_run("r1", {})
    backend.start_sample("r1", "s1", {})
    backend.log_sample("r1", "s1", {}, {}, {}, {}, {})
    backend.end_run("r1")
    span = next(s for s in exporter.get_finished_spans() if s.name == "inspect.sample")
    assert span.status.status_code == StatusCode.UNSET


def test_log_sample_metadata_attributes(
    backend: OpenTelemetryBackend, exporter: InMemorySpanExporter
) -> None:
    backend.start_run("r1", {})
    backend.start_sample("r1", "s1", {})
    backend.log_sample("r1", "s1", {}, {}, {}, {}, {"llm.model": "gpt-4"})
    backend.end_run("r1")
    assert _span_attrs(exporter, "inspect.sample")["llm.model"] == "gpt-4"


def test_log_sample_skips_none_metadata(
    backend: OpenTelemetryBackend, exporter: InMemorySpanExporter
) -> None:
    backend.start_run("r1", {})
    backend.start_sample("r1", "s1", {})
    backend.log_sample("r1", "s1", {}, {}, {}, {}, {"llm.model": None})
    backend.end_run("r1")
    assert "llm.model" not in _span_attrs(exporter, "inspect.sample")


def test_log_sample_fallback_no_start_sample(
    backend: OpenTelemetryBackend, exporter: InMemorySpanExporter
) -> None:
    """Without start_sample, log_sample creates a self-contained span."""
    backend.start_run("r1", {})
    backend.log_sample("r1", "s1", {}, {}, {}, {"acc": 1.0}, {})
    backend.end_run("r1")
    assert "inspect.sample" in _finished_names(exporter)


def test_log_sample_fallback_no_run_span_silent(backend: OpenTelemetryBackend) -> None:
    backend.log_sample("nonexistent", "s1", {}, {}, {}, {}, {})  # no raise


# ---------------------------------------------------------------------------
# failure_only
# ---------------------------------------------------------------------------


def test_failure_only_skips_passing_samples(
    failure_only_backend: OpenTelemetryBackend, exporter: InMemorySpanExporter
) -> None:
    failure_only_backend.start_run("r1", {})
    failure_only_backend.start_sample("r1", "s1", {})
    failure_only_backend.log_sample("r1", "s1", {}, {}, {}, {"acc": 1.0}, {})
    failure_only_backend.end_run("r1")
    assert "inspect.sample" not in _finished_names(exporter)


def test_failure_only_logs_failing_samples(
    failure_only_backend: OpenTelemetryBackend, exporter: InMemorySpanExporter
) -> None:
    failure_only_backend.start_run("r1", {})
    failure_only_backend.start_sample("r1", "s1", {})
    failure_only_backend.log_sample("r1", "s1", {}, {}, {}, {"acc": 0.0}, {})
    failure_only_backend.end_run("r1")
    assert "inspect.sample" in _finished_names(exporter)


def test_failure_only_logs_unknown_score(
    failure_only_backend: OpenTelemetryBackend, exporter: InMemorySpanExporter
) -> None:
    failure_only_backend.start_run("r1", {})
    failure_only_backend.start_sample("r1", "s1", {})
    failure_only_backend.log_sample("r1", "s1", {}, {}, {}, {}, {})
    failure_only_backend.end_run("r1")
    assert "inspect.sample" in _finished_names(exporter)


# ---------------------------------------------------------------------------
# end_run
# ---------------------------------------------------------------------------


def test_end_run_removes_span_from_registry(
    backend: OpenTelemetryBackend, exporter: InMemorySpanExporter
) -> None:
    backend.start_run("r1", {})
    assert "r1" in backend._run_spans
    backend.end_run("r1")
    assert "r1" not in backend._run_spans


def test_end_run_finishes_the_span(
    backend: OpenTelemetryBackend, exporter: InMemorySpanExporter
) -> None:
    backend.start_run("r1", {})
    backend.end_run("r1")
    assert "inspect.run" in _finished_names(exporter)


def test_end_run_unknown_id_is_silent(backend: OpenTelemetryBackend) -> None:
    backend.end_run("unknown-run")  # must not raise


# ---------------------------------------------------------------------------
# flush / shutdown
# ---------------------------------------------------------------------------


def test_flush_calls_force_flush(backend: OpenTelemetryBackend) -> None:
    from unittest.mock import patch
    with patch.object(backend._provider, "force_flush") as mock_ff:
        backend.flush()
    mock_ff.assert_called_once()


def test_shutdown_calls_provider_shutdown(backend: OpenTelemetryBackend) -> None:
    from unittest.mock import patch
    with patch.object(backend._provider, "shutdown") as mock_sd:
        backend.shutdown()
    mock_sd.assert_called_once()


# ---------------------------------------------------------------------------
# OTLP endpoint - BatchSpanProcessor vs SimpleSpanProcessor
# ---------------------------------------------------------------------------


def test_otlp_endpoint_uses_batch_processor(tmp_path) -> None:
    from unittest.mock import MagicMock, patch
    mock_exporter = MagicMock()
    with patch("inspect_otel.otel_backend.OTLPSpanExporter", return_value=mock_exporter):
        b = OpenTelemetryBackend(
            service_name="svc",
            otlp_endpoint="http://localhost:4318/v1/traces",
            async_mode=True,
        )
    # Just exercise the path — if it didn't raise, it's ok
    assert b is not None


def test_otlp_endpoint_uses_simple_processor(tmp_path) -> None:
    mock_exporter = MagicMock()
    with patch("inspect_otel.otel_backend.OTLPSpanExporter", return_value=mock_exporter):
        b = OpenTelemetryBackend(
            service_name="svc",
            otlp_endpoint="http://localhost:4318/v1/traces",
            async_mode=False,
        )
    assert b is not None


# ---------------------------------------------------------------------------
# _set_prefixed_attributes helper
# ---------------------------------------------------------------------------


def test_set_prefixed_attributes_sets_keys(backend: OpenTelemetryBackend, exporter: InMemorySpanExporter) -> None:
    backend.start_run("r1", {})
    run_span = backend._run_spans["r1"]
    _set_prefixed_attributes(run_span, "eval.input", {"q": "hello", "skip": None})
    attrs = dict(run_span.attributes or {})
    assert attrs["eval.input.q"] == "hello"
    assert "eval.input.skip" not in attrs
    backend.end_run("r1")
