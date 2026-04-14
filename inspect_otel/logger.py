"""Inspect AI hooks implementation that exports to OpenTelemetry."""

from __future__ import annotations

from typing import Any

from inspect_ai.event._tool import ToolEvent
from inspect_ai.hooks import (
    EvalSetEnd,
    Hooks,
    ModelUsageData,
    SampleEnd,
    SampleEvent,
    SampleStart,
    TaskEnd,
    TaskStart,
)

from .telemetry_manager import TelemetryManager


class OpenTelemetryLogger(Hooks):
    """Inspect AI :class:`~inspect_ai.hooks.Hooks` subclass that emits spans.

    Attach this to Inspect's hook registry via the :func:`setup_hooks`
    convenience function, or register it manually with the
    ``@hooks(name, description)`` decorator from ``inspect_ai.hooks``.

    Example (via setup_hooks)::

        from inspect_otel import setup_hooks
        from inspect_ai import eval_set

        setup_hooks(config_path="config.yaml")
        eval_set(evals=[my_eval], model="gpt-4")
    """

    def __init__(self, manager: TelemetryManager) -> None:
        self._manager = manager

    # ------------------------------------------------------------------
    # Hooks interface
    # ------------------------------------------------------------------

    async def on_task_start(self, data: TaskStart) -> None:  # type: ignore[override]
        """Open an OTel span for the incoming task / eval run."""
        spec = data.spec
        metadata: dict[str, Any] = {
            "eval.run_id": data.run_id,
            "eval.name": spec.task,
            "eval.type": "benchmark",
            "llm.model": spec.model,
        }
        dataset = spec.dataset
        if dataset and dataset.name:
            metadata["eval.dataset"] = dataset.name
        if dataset and getattr(dataset, "version", None):
            metadata["eval.dataset.version"] = str(dataset.version)
        if spec.metadata:
            experiment_id = spec.metadata.get("experiment_id")
            if experiment_id is not None:
                metadata["eval.experiment_id"] = str(experiment_id)
        self._manager.start_run(data.eval_id, metadata)

    async def on_sample_start(self, data: SampleStart) -> None:  # type: ignore[override]
        """Open an ``inspect.sample`` span before the sample runs."""
        self._manager.start_sample(
            run_id=data.eval_id,
            sample_id=data.sample_id,
            metadata={},
        )

    async def on_model_usage(self, data: ModelUsageData) -> None:  # type: ignore[override]
        """Emit an ``llm.call`` child span with token counts and latency."""
        if data.eval_id is None:
            return
        usage = data.usage
        self._manager.log_model_call(
            run_id=data.run_id or data.eval_id,
            eval_id=data.eval_id,
            model_name=data.model_name,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            latency_ms=data.call_duration * 1000.0,
            retries=data.retries,
        )

    async def on_sample_event(self, data: SampleEvent) -> None:  # type: ignore[override]
        """Emit a ``tool.call`` child span for every :class:`~inspect_ai.event.ToolEvent`."""
        if not isinstance(data.event, ToolEvent):
            return
        event = data.event
        self._manager.log_tool_event(
            run_id=data.run_id,
            eval_id=data.eval_id,
            sample_id=data.sample_id,
            tool_name=event.function,
            tool_input=event.arguments,
        )

    async def on_task_end(self, data: TaskEnd) -> None:  # type: ignore[override]
        """Close the span and flush pending telemetry."""
        self._manager.end_run(data.eval_id)
        self._manager.flush()

    async def on_sample_end(self, data: SampleEnd) -> None:  # type: ignore[override]
        """Emit a child span with per-sample telemetry."""
        sample = data.sample

        scores: dict[str, float | None] = {}
        for scorer_name, score_obj in (sample.scores or {}).items():
            try:
                scores[scorer_name] = score_obj.as_float()
            except (TypeError, ValueError):
                scores[scorer_name] = None

        latency_ms: float = 0.0
        if sample.total_time is not None:
            latency_ms = sample.total_time * 1000.0

        model_name: str | None = None
        if sample.model_usage:
            model_name = next(iter(sample.model_usage.keys()), None)

        self._manager.log_sample(
            run_id=data.eval_id,
            sample_id=str(sample.id),
            inputs={"input": str(sample.input)},
            outputs={"output": str(sample.output)},
            expected={"target": str(sample.target)},
            scores=scores,
            metadata={
                "llm.latency_ms": latency_ms,
                "llm.model": model_name,
            },
        )

    async def on_eval_set_end(self, data: EvalSetEnd) -> None:  # type: ignore[override]
        """Gracefully shut down all backends when the eval set finishes."""
        self._manager.shutdown()
