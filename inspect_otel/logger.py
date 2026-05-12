"""Inspect AI hooks implementation that exports to OpenTelemetry."""

from __future__ import annotations

import json
from typing import Any

from inspect_ai.hooks import (
    EvalSetEnd,
    Hooks,
    ModelCacheUsageData,
    ModelUsageData,
    SampleEnd,
    SampleScoring,
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
        from inspect_ai import eval

        setup_hooks(config_path="config.yaml")
        eval(tasks=[my_task()], model="gpt-4")
    """

    def __init__(self, manager: TelemetryManager) -> None:
        self._manager = manager
        self._current_invocation_params: dict[str, Any] = {}

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
        if spec.tags:
            metadata["eval.tags"] = ",".join(spec.tags)
            metadata["tag.tags"] = list(spec.tags)  # list — Phoenix filter bar
        if spec.metadata:
            experiment_id = spec.metadata.get("experiment_id")
            if experiment_id is not None:
                metadata["eval.experiment_id"] = str(experiment_id)

        # EvalConfig: epochs and limits.
        eval_config = getattr(spec, "config", None)
        epochs = getattr(eval_config, "epochs", None)
        if epochs is not None:
            metadata["eval.epochs"] = epochs

        # Solver and scorer names.
        if spec.solver:
            metadata["eval.solver"] = spec.solver
        scorers = getattr(spec, "scorers", None)
        if scorers:
            metadata["eval.scorers"] = ",".join(
                s.name for s in scorers if getattr(s, "name", None)
            )

        # Git revision for reproducibility.
        revision = getattr(spec, "revision", None)
        if revision:
            if getattr(revision, "origin", None):
                metadata["eval.revision.origin"] = revision.origin
            if getattr(revision, "commit", None):
                metadata["eval.revision.commit"] = revision.commit
            if getattr(revision, "dirty", None) is not None:
                metadata["eval.revision.dirty"] = str(revision.dirty)

        # Package versions (key deps for reproducibility).
        packages = getattr(spec, "packages", None)
        if packages:
            metadata["eval.packages"] = json.dumps(packages)

        # GenerateConfig (model_generate_config) for the Parameters panel.
        gen_config = getattr(spec, "model_generate_config", None)
        self._current_invocation_params = _extract_invocation_params(gen_config)
        if self._current_invocation_params:
            metadata["llm.invocation_parameters"] = json.dumps(self._current_invocation_params)

        self._manager.start_run(data.eval_id, metadata)

    async def on_sample_start(self, data: SampleStart) -> None:  # type: ignore[override]
        """Open an ``inspect.sample`` span before the sample runs."""
        summary = data.summary
        metadata: dict[str, Any] = {
            "eval.sample.id": str(summary.id),
            "eval.epoch": summary.epoch,
        }
        self._manager.start_sample(
            run_id=data.eval_id,
            sample_id=data.sample_id,
            metadata=metadata,
        )

    async def on_model_usage(self, data: ModelUsageData) -> None:  # type: ignore[override]
        """Emit an ``llm.call`` child span with token counts and latency."""
        usage = data.usage
        self._manager.log_model_call(
            model_name=data.model_name,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            latency_ms=data.call_duration * 1000.0,
            cache_read_tokens=usage.input_tokens_cache_read,
            cache_write_tokens=usage.input_tokens_cache_write,
            reasoning_tokens=usage.reasoning_tokens,
            invocation_parameters=self._current_invocation_params,
        )

    async def on_model_cache_usage(self, data: ModelCacheUsageData) -> None:  # type: ignore[override]
        """Emit an ``llm.cache_hit`` span when a cached response is served."""
        usage = data.usage
        self._manager.log_model_cache_call(
            model_name=data.model_name,
            cache_read_tokens=usage.input_tokens_cache_read or usage.input_tokens,
            total_tokens=usage.total_tokens,
        )

    async def on_sample_scoring(self, data: SampleScoring) -> None:  # type: ignore[override]
        """Open an ``inspect.scoring`` child span to demarcate solver vs. scorer time."""
        self._manager.start_scoring_span(
            run_id=data.eval_id,
            sample_id=data.sample_id,
        )

    async def on_task_end(self, data: TaskEnd) -> None:  # type: ignore[override]
        """Add aggregate metrics to the run span, then close it and flush."""
        log = getattr(data, "log", None)
        if log is not None:
            self._manager.log_run_summary(data.eval_id, log)
        self._manager.end_run(data.eval_id)
        self._manager.flush()

    async def on_sample_end(self, data: SampleEnd) -> None:  # type: ignore[override]
        """Emit a child span with per-sample telemetry."""
        sample = data.sample

        # Numeric scores and per-scorer detail (label, explanation, answer).
        scores: dict[str, float | None] = {}
        score_details: dict[str, dict[str, Any]] = {}
        for scorer_name, score_obj in (sample.scores or {}).items():
            scores[scorer_name] = _score_to_float(score_obj)
            score_details[scorer_name] = {
                "label": str(score_obj.value) if score_obj.value is not None else None,
                "explanation": score_obj.explanation,
                "answer": score_obj.answer,
            }

        latency_ms: float = 0.0
        if sample.total_time is not None:
            latency_ms = sample.total_time * 1000.0

        model_name: str | None = None
        if sample.model_usage:
            model_name = next(iter(sample.model_usage.keys()), None)

        output_text = _extract_output_text(sample.output)
        finish_reason = _extract_finish_reason(sample.output)
        messages = _extract_messages(sample.messages)
        sample_token_usage = _aggregate_sample_tokens(sample.model_usage)

        metadata_json: str | None = None
        if sample.metadata:
            try:
                metadata_json = json.dumps(sample.metadata)
            except (TypeError, ValueError):
                metadata_json = str(sample.metadata)

        error: str | None = getattr(sample, "error", None)

        self._manager.log_sample(
            run_id=data.eval_id,
            sample_id=data.sample_id,
            inputs={"input": str(sample.input)},
            outputs={"output": output_text},
            expected={"target": str(sample.target)},
            scores=scores,
            score_details=score_details,
            messages=messages,
            metadata={
                "llm.latency_ms": latency_ms,
                "llm.model": model_name,
            },
            finish_reason=finish_reason,
            epoch=sample.epoch,
            metadata_json=metadata_json,
            token_usage=sample_token_usage,
            error=error,
        )

    async def on_eval_set_end(self, data: EvalSetEnd) -> None:  # type: ignore[override]
        """Gracefully shut down all backends when the eval set finishes."""
        self._manager.shutdown()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

# Categorical score values used by Inspect AI's built-in scorers.
_CATEGORICAL_SCORES: dict[str, float] = {"C": 1.0, "I": 0.0, "P": 0.5}


def _score_to_float(score_obj: Any) -> float | None:
    """Convert an Inspect AI Score to a float, handling categorical values."""
    try:
        return score_obj.as_float()
    except (TypeError, ValueError):
        pass
    val = score_obj.value
    if isinstance(val, str):
        return _CATEGORICAL_SCORES.get(val.upper())
    return None


def _extract_output_text(output: Any) -> str:
    """Extract the completion text from a ModelOutput or return str(output)."""
    completion = getattr(output, "completion", None)
    if isinstance(completion, str) and completion:
        return completion
    return str(output)


def _extract_finish_reason(output: Any) -> str | None:
    """Extract the stop/finish reason from a ModelOutput."""
    choices = getattr(output, "choices", None)
    if choices:
        return getattr(choices[0], "stop_reason", None)
    return None


def _extract_messages(chat_messages: list[Any]) -> list[dict[str, str]]:
    """Convert Inspect AI ChatMessage list to simple role/content dicts."""
    result: list[dict[str, str]] = []
    for msg in chat_messages:
        content = msg.content
        if isinstance(content, list):
            # Content parts — join text parts into a single string.
            text = " ".join(
                p.text if hasattr(p, "text") else str(p)
                for p in content
            )
        else:
            text = str(content) if content is not None else ""
        result.append({"role": str(msg.role), "content": text})
    return result


def _extract_invocation_params(config: Any) -> dict[str, Any]:
    """Extract GenerateConfig fields that Phoenix renders in its Parameters panel."""
    if config is None:
        return {}
    params: dict[str, Any] = {}
    for field in (
        "temperature", "max_tokens", "top_p", "top_k", "seed",
        "frequency_penalty", "presence_penalty", "num_choices",
    ):
        value = getattr(config, field, None)
        if value is not None:
            params[field] = value
    stop_seqs = getattr(config, "stop_seqs", None)
    if stop_seqs:
        params["stop_seqs"] = stop_seqs
    return params


def _aggregate_sample_tokens(model_usage: Any) -> dict[str, int]:
    """Sum token counts across all models used within one sample."""
    if not model_usage:
        return {}
    total_input = total_output = total_tokens = 0
    for usage in model_usage.values():
        total_input += getattr(usage, "input_tokens", 0) or 0
        total_output += getattr(usage, "output_tokens", 0) or 0
        total_tokens += getattr(usage, "total_tokens", 0) or 0
    return {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_tokens,
    }
