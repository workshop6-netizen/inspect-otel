"""Tests for inspect_otel.decorators."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from inspect_otel.decorators import trace_llm, trace_tool

# ---------------------------------------------------------------------------
# Fixtures - wire up a real (in-memory) OTel SDK for the test process
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def otel_in_memory():
    """Patch trace.get_tracer in the decorators module to use an in-memory provider.

    Newer OTel SDK versions only allow one global TracerProvider, so we patch
    the module-level reference instead of replacing the global provider.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with patch("inspect_otel.decorators.trace.get_tracer", side_effect=provider.get_tracer):
        yield exporter
    exporter.clear()


def _finished_names(exporter: InMemorySpanExporter) -> list[str]:
    return [s.name for s in exporter.get_finished_spans()]



def _sample_span(exporter: InMemorySpanExporter, name: str):
    spans = [s for s in exporter.get_finished_spans() if s.name == name]
    assert spans, f"No span named '{name}'"
    return spans[0]


# ---------------------------------------------------------------------------
# trace_llm
# ---------------------------------------------------------------------------


def test_trace_llm_creates_span(otel_in_memory: InMemorySpanExporter) -> None:
    @trace_llm(model="gpt-4")
    def my_fn(x: int) -> int:
        return x + 1

    my_fn(1)
    assert "llm.call" in _finished_names(otel_in_memory)


def test_trace_llm_sets_model_attribute(otel_in_memory: InMemorySpanExporter) -> None:
    @trace_llm(model="gpt-4o")
    def my_fn() -> str:
        return "ok"

    my_fn()
    span = _sample_span(otel_in_memory, "llm.call")
    assert dict(span.attributes or {})["llm.model"] == "gpt-4o"


def test_trace_llm_sets_latency_attribute(otel_in_memory: InMemorySpanExporter) -> None:
    @trace_llm(model="gpt-4")
    def my_fn() -> str:
        return "ok"

    my_fn()
    span = _sample_span(otel_in_memory, "llm.call")
    assert "llm.latency_ms" in dict(span.attributes or {})
    assert dict(span.attributes or {})["llm.latency_ms"] >= 0


def test_trace_llm_ok_status_on_success(otel_in_memory: InMemorySpanExporter) -> None:
    @trace_llm(model="gpt-4")
    def my_fn() -> str:
        return "ok"

    my_fn()
    span = _sample_span(otel_in_memory, "llm.call")
    assert span.status.status_code == StatusCode.OK


def test_trace_llm_returns_value(otel_in_memory: InMemorySpanExporter) -> None:
    @trace_llm(model="gpt-4")
    def my_fn(x: int) -> int:
        return x * 2

    assert my_fn(3) == 6


def test_trace_llm_default_model(otel_in_memory: InMemorySpanExporter) -> None:
    @trace_llm()
    def my_fn() -> str:
        return "ok"

    my_fn()
    span = _sample_span(otel_in_memory, "llm.call")
    assert dict(span.attributes or {})["llm.model"] == "unknown"


def test_trace_llm_error_status_on_exception(otel_in_memory: InMemorySpanExporter) -> None:
    @trace_llm(model="gpt-4")
    def my_fn() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        my_fn()

    span = _sample_span(otel_in_memory, "llm.call")
    assert span.status.status_code == StatusCode.ERROR


def test_trace_llm_records_exception_event(otel_in_memory: InMemorySpanExporter) -> None:
    @trace_llm(model="gpt-4")
    def my_fn() -> None:
        raise RuntimeError("test error")

    with pytest.raises(RuntimeError):
        my_fn()

    span = _sample_span(otel_in_memory, "llm.call")
    event_names = [e.name for e in span.events]
    assert "exception" in event_names


def test_trace_llm_reraises_exception(otel_in_memory: InMemorySpanExporter) -> None:
    @trace_llm(model="gpt-4")
    def my_fn() -> None:
        raise TypeError("type err")

    with pytest.raises(TypeError, match="type err"):
        my_fn()


def test_trace_llm_preserves_function_name() -> None:
    @trace_llm(model="gpt-4")
    def my_special_function() -> None:
        pass

    assert my_special_function.__name__ == "my_special_function"


def test_trace_llm_latency_recorded_even_on_exception(
    otel_in_memory: InMemorySpanExporter,
) -> None:
    @trace_llm(model="gpt-4")
    def my_fn() -> None:
        raise ValueError("err")

    with pytest.raises(ValueError):
        my_fn()

    span = _sample_span(otel_in_memory, "llm.call")
    assert "llm.latency_ms" in dict(span.attributes or {})


# ---------------------------------------------------------------------------
# trace_tool
# ---------------------------------------------------------------------------


def test_trace_tool_creates_span(otel_in_memory: InMemorySpanExporter) -> None:
    @trace_tool("web_search")
    def search(q: str) -> str:
        return f"results for {q}"

    search("python")
    assert "tool.call" in _finished_names(otel_in_memory)


def test_trace_tool_sets_name_attribute(otel_in_memory: InMemorySpanExporter) -> None:
    @trace_tool("calculator")
    def calc(x: int) -> int:
        return x + 1

    calc(1)
    span = _sample_span(otel_in_memory, "tool.call")
    assert dict(span.attributes or {})["tool.name"] == "calculator"


def test_trace_tool_returns_value(otel_in_memory: InMemorySpanExporter) -> None:
    @trace_tool("add")
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5


def test_trace_tool_error_status_on_exception(otel_in_memory: InMemorySpanExporter) -> None:
    @trace_tool("bad_tool")
    def bad() -> None:
        raise RuntimeError("tool err")

    with pytest.raises(RuntimeError):
        bad()

    span = _sample_span(otel_in_memory, "tool.call")
    assert span.status.status_code == StatusCode.ERROR


def test_trace_tool_records_exception_event(otel_in_memory: InMemorySpanExporter) -> None:
    @trace_tool("bad_tool")
    def bad() -> None:
        raise ValueError("oops")

    with pytest.raises(ValueError):
        bad()

    span = _sample_span(otel_in_memory, "tool.call")
    event_names = [e.name for e in span.events]
    assert "exception" in event_names


def test_trace_tool_reraises_exception(otel_in_memory: InMemorySpanExporter) -> None:
    @trace_tool("t")
    def t() -> None:
        raise KeyError("missing")

    with pytest.raises(KeyError):
        t()


def test_trace_tool_preserves_function_name() -> None:
    @trace_tool("my_tool")
    def my_important_tool() -> None:
        pass

    assert my_important_tool.__name__ == "my_important_tool"
