"""Decorators for tracing LLM and tool calls with OpenTelemetry."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any, TypeVar

from opentelemetry import trace
from opentelemetry.trace import StatusCode

F = TypeVar("F", bound=Callable[..., Any])

_TRACER_NAME = "inspect-otel"


def trace_llm(model: str = "unknown") -> Callable[[F], F]:
    """Wrap a function in an ``llm.call`` OpenTelemetry span.

    Records ``llm.model`` and ``llm.latency_ms`` attributes.  On exception,
    the span records the exception and sets an ERROR status before re-raising.

    Example::

        @trace_llm(model="gpt-4")
        def call_model(prompt: str) -> str:
            return llm.generate(prompt)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = trace.get_tracer(_TRACER_NAME)
            with tracer.start_as_current_span("llm.call") as span:
                span.set_attribute("llm.model", model)
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    span.set_status(StatusCode.OK)
                    return result
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(StatusCode.ERROR, str(exc))
                    raise
                finally:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    span.set_attribute("llm.latency_ms", elapsed_ms)

        return wrapper  # type: ignore[return-value]

    return decorator


def trace_tool(name: str) -> Callable[[F], F]:
    """Wrap a function in a ``tool.call`` OpenTelemetry span.

    Records ``tool.name`` as an attribute.  On exception, records the
    exception and sets an ERROR status before re-raising.

    Example::

        @trace_tool("web_search")
        def search(query: str) -> str:
            return search_api(query)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = trace.get_tracer(_TRACER_NAME)
            with tracer.start_as_current_span("tool.call") as span:
                span.set_attribute("tool.name", name)
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(StatusCode.ERROR, str(exc))
                    raise

        return wrapper  # type: ignore[return-value]

    return decorator
