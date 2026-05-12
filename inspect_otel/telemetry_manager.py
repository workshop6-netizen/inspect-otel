"""Fan-out manager that delegates to one or more telemetry backends."""

from __future__ import annotations

from typing import Any

from .otel_backend import OpenTelemetryBackend


class TelemetryManager:
    """Delegates telemetry calls to a list of :class:`OpenTelemetryBackend` instances.

    Using a list of backends allows multiple exporters to receive the same
    data simultaneously (e.g. both Phoenix and Jaeger).
    """

    def __init__(self, backends: list[OpenTelemetryBackend]) -> None:
        self._backends = backends

    def start_run(self, run_id: str, metadata: dict[str, Any]) -> None:
        """Signal the start of an eval run to all backends."""
        for backend in self._backends:
            backend.start_run(run_id, metadata)

    def start_sample(
        self, run_id: str, sample_id: str, metadata: dict[str, Any]
    ) -> None:
        """Open a sample span in all backends."""
        for backend in self._backends:
            backend.start_sample(run_id, sample_id, metadata)

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
        invocation_parameters: dict[str, Any] | None = None,
    ) -> None:
        """Emit an ``llm.call`` span to all backends."""
        for backend in self._backends:
            backend.log_model_call(
                model_name=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                reasoning_tokens=reasoning_tokens,
                invocation_parameters=invocation_parameters,
            )

    def log_model_cache_call(
        self,
        model_name: str,
        cache_read_tokens: int,
        total_tokens: int,
    ) -> None:
        """Emit an ``llm.cache_hit`` span to all backends."""
        for backend in self._backends:
            backend.log_model_cache_call(
                model_name=model_name,
                cache_read_tokens=cache_read_tokens,
                total_tokens=total_tokens,
            )

    def log_tool_event(
        self,
        run_id: str,
        eval_id: str,
        sample_id: str,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> None:
        """Emit a ``tool.call`` span to all backends."""
        for backend in self._backends:
            backend.log_tool_event(
                run_id=run_id,
                eval_id=eval_id,
                sample_id=sample_id,
                tool_name=tool_name,
                tool_input=tool_input,
            )

    def log_run_summary(self, run_id: str, log: Any) -> None:
        """Write aggregate metrics from TaskEnd.log to the run span in all backends."""
        for backend in self._backends:
            backend.log_run_summary(run_id, log)

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
        metadata_json: str | None = None,
        token_usage: dict[str, int] | None = None,
        error: str | None = None,
    ) -> None:
        """Log one evaluated sample to all backends."""
        for backend in self._backends:
            backend.log_sample(
                run_id=run_id,
                sample_id=sample_id,
                inputs=inputs,
                outputs=outputs,
                expected=expected,
                scores=scores,
                score_details=score_details,
                messages=messages,
                metadata=metadata,
                finish_reason=finish_reason,
                epoch=epoch,
                metadata_json=metadata_json,
                token_usage=token_usage,
                error=error,
            )

    def end_run(self, run_id: str) -> None:
        """Signal the end of an eval run to all backends."""
        for backend in self._backends:
            backend.end_run(run_id)

    def flush(self) -> None:
        """Flush all pending spans in all backends."""
        for backend in self._backends:
            backend.flush()

    def shutdown(self) -> None:
        """Flush and shut down all backends."""
        for backend in self._backends:
            backend.shutdown()
