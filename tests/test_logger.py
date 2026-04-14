"""Tests for inspect_otel.logger (OpenTelemetryLogger hooks)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from inspect_otel.logger import OpenTelemetryLogger
from inspect_otel.telemetry_manager import TelemetryManager

# ---------------------------------------------------------------------------
# Helpers - build minimal inspect_ai hook data mocks
# ---------------------------------------------------------------------------


def _make_task_start(
    eval_id: str = "eval-001",
    run_id: str = "run-001",
    task_name: str = "my_task",
    model: str = "gpt-4",
    dataset_name: str | None = "test_dataset",
    dataset_version: str | None = None,
    experiment_id: str | None = None,
) -> MagicMock:
    spec = MagicMock()
    spec.task = task_name
    spec.model = model
    spec.dataset = MagicMock()
    spec.dataset.name = dataset_name
    spec.dataset.version = dataset_version
    spec.metadata = {"experiment_id": experiment_id} if experiment_id else {}
    data = MagicMock()
    data.eval_id = eval_id
    data.run_id = run_id
    data.spec = spec
    return data


def _make_task_end(eval_id: str = "eval-001") -> MagicMock:
    data = MagicMock()
    data.eval_id = eval_id
    return data


def _make_sample_start(
    eval_id: str = "eval-001",
    run_id: str = "run-001",
    sample_id: str = "s1",
) -> MagicMock:
    data = MagicMock()
    data.eval_id = eval_id
    data.run_id = run_id
    data.sample_id = sample_id
    return data


def _make_score(float_value: float | None) -> MagicMock:
    score = MagicMock()
    if float_value is None:
        score.as_float.side_effect = TypeError("no value")
    else:
        score.as_float.return_value = float_value
    return score


def _make_sample_end(
    eval_id: str = "eval-001",
    sample_id: int | str = "s1",
    scores: dict | None = None,
    total_time: float | None = 0.5,
    model_name: str | None = "gpt-4",
) -> MagicMock:
    sample = MagicMock()
    sample.id = sample_id
    sample.input = "What is 2+2?"
    sample.output = MagicMock()
    sample.target = "4"
    sample.total_time = total_time
    if scores is not None:
        sample.scores = scores
    else:
        sample.scores = {"accuracy": _make_score(1.0)}
    if model_name:
        sample.model_usage = {model_name: MagicMock()}
    else:
        sample.model_usage = {}

    data = MagicMock()
    data.eval_id = eval_id
    data.sample = sample
    return data


def _make_model_usage(
    eval_id: str = "eval-001",
    run_id: str = "run-001",
    model_name: str = "gpt-4",
    input_tokens: int = 10,
    output_tokens: int = 20,
    total_tokens: int = 30,
    call_duration: float = 0.5,
    retries: int = 0,
) -> MagicMock:
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.total_tokens = total_tokens
    data = MagicMock()
    data.eval_id = eval_id
    data.run_id = run_id
    data.model_name = model_name
    data.usage = usage
    data.call_duration = call_duration
    data.retries = retries
    return data


def _make_sample_event_tool(
    eval_id: str = "eval-001",
    run_id: str = "run-001",
    sample_id: str = "s1",
    function: str = "web_search",
    arguments: dict | None = None,
) -> MagicMock:
    from inspect_ai.event._tool import ToolEvent
    event = MagicMock(spec=ToolEvent)
    event.function = function
    event.arguments = arguments or {"query": "test"}
    data = MagicMock()
    data.eval_id = eval_id
    data.run_id = run_id
    data.sample_id = sample_id
    data.event = event
    return data


def _make_sample_event_non_tool(
    eval_id: str = "eval-001",
    run_id: str = "run-001",
    sample_id: str = "s1",
) -> MagicMock:
    """A SampleEvent whose event is NOT a ToolEvent."""
    data = MagicMock()
    data.eval_id = eval_id
    data.run_id = run_id
    data.sample_id = sample_id
    data.event = MagicMock()  # generic, not ToolEvent
    return data


def _make_eval_set_end() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_manager() -> MagicMock:
    return MagicMock(spec=TelemetryManager)


@pytest.fixture()
def logger(mock_manager: MagicMock) -> OpenTelemetryLogger:
    return OpenTelemetryLogger(mock_manager)


# ---------------------------------------------------------------------------
# on_task_start
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_task_start_calls_start_run(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_task_start(eval_id="e1")
    await logger.on_task_start(data)
    mock_manager.start_run.assert_called_once()
    assert mock_manager.start_run.call_args[0][0] == "e1"


@pytest.mark.asyncio
async def test_on_task_start_passes_eval_name(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_task_start(task_name="benchmark_v2")
    await logger.on_task_start(data)
    metadata = mock_manager.start_run.call_args[0][1]
    assert metadata["eval.name"] == "benchmark_v2"


@pytest.mark.asyncio
async def test_on_task_start_passes_model(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_task_start(model="claude-3")
    await logger.on_task_start(data)
    metadata = mock_manager.start_run.call_args[0][1]
    assert metadata["llm.model"] == "claude-3"


@pytest.mark.asyncio
async def test_on_task_start_passes_dataset(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_task_start(dataset_name="my_dataset")
    await logger.on_task_start(data)
    metadata = mock_manager.start_run.call_args[0][1]
    assert metadata["eval.dataset"] == "my_dataset"


@pytest.mark.asyncio
async def test_on_task_start_omits_dataset_when_none(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_task_start(dataset_name=None)
    await logger.on_task_start(data)
    metadata = mock_manager.start_run.call_args[0][1]
    assert "eval.dataset" not in metadata


@pytest.mark.asyncio
async def test_on_task_start_passes_experiment_id(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_task_start(experiment_id="exp_042")
    await logger.on_task_start(data)
    metadata = mock_manager.start_run.call_args[0][1]
    assert metadata["eval.experiment_id"] == "exp_042"


@pytest.mark.asyncio
async def test_on_task_start_omits_experiment_id_when_absent(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_task_start(experiment_id=None)
    await logger.on_task_start(data)
    metadata = mock_manager.start_run.call_args[0][1]
    assert "eval.experiment_id" not in metadata


@pytest.mark.asyncio
async def test_on_task_start_omits_dataset_when_name_empty(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_task_start(dataset_name="")
    await logger.on_task_start(data)
    metadata = mock_manager.start_run.call_args[0][1]
    assert "eval.dataset" not in metadata


@pytest.mark.asyncio
async def test_on_task_start_passes_dataset_version(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_task_start(dataset_name="ds", dataset_version="v2")
    await logger.on_task_start(data)
    metadata = mock_manager.start_run.call_args[0][1]
    assert metadata["eval.dataset.version"] == "v2"


@pytest.mark.asyncio
async def test_on_task_start_omits_dataset_version_when_none(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_task_start(dataset_name="ds", dataset_version=None)
    await logger.on_task_start(data)
    metadata = mock_manager.start_run.call_args[0][1]
    assert "eval.dataset.version" not in metadata


# ---------------------------------------------------------------------------
# on_sample_start
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_sample_start_calls_start_sample(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_sample_start(eval_id="e1", sample_id="s1")
    await logger.on_sample_start(data)
    mock_manager.start_sample.assert_called_once_with(
        run_id="e1", sample_id="s1", metadata={}
    )


# ---------------------------------------------------------------------------
# on_model_usage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_model_usage_calls_log_model_call(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_model_usage()
    await logger.on_model_usage(data)
    mock_manager.log_model_call.assert_called_once()


@pytest.mark.asyncio
async def test_on_model_usage_passes_token_counts(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_model_usage(input_tokens=5, output_tokens=15, total_tokens=20)
    await logger.on_model_usage(data)
    kwargs = mock_manager.log_model_call.call_args[1]
    assert kwargs["input_tokens"] == 5
    assert kwargs["output_tokens"] == 15
    assert kwargs["total_tokens"] == 20


@pytest.mark.asyncio
async def test_on_model_usage_passes_latency(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_model_usage(call_duration=1.5)
    await logger.on_model_usage(data)
    kwargs = mock_manager.log_model_call.call_args[1]
    assert kwargs["latency_ms"] == pytest.approx(1500.0)


@pytest.mark.asyncio
async def test_on_model_usage_passes_retries(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_model_usage(retries=2)
    await logger.on_model_usage(data)
    kwargs = mock_manager.log_model_call.call_args[1]
    assert kwargs["retries"] == 2


@pytest.mark.asyncio
async def test_on_model_usage_skips_when_eval_id_none(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_model_usage()
    data.eval_id = None
    await logger.on_model_usage(data)
    mock_manager.log_model_call.assert_not_called()


# ---------------------------------------------------------------------------
# on_sample_event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_sample_event_tool_calls_log_tool_event(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_sample_event_tool()
    await logger.on_sample_event(data)
    mock_manager.log_tool_event.assert_called_once()


@pytest.mark.asyncio
async def test_on_sample_event_passes_tool_name(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_sample_event_tool(function="calculator")
    await logger.on_sample_event(data)
    kwargs = mock_manager.log_tool_event.call_args[1]
    assert kwargs["tool_name"] == "calculator"


@pytest.mark.asyncio
async def test_on_sample_event_passes_tool_input(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_sample_event_tool(arguments={"expr": "1+1"})
    await logger.on_sample_event(data)
    kwargs = mock_manager.log_tool_event.call_args[1]
    assert kwargs["tool_input"] == {"expr": "1+1"}


@pytest.mark.asyncio
async def test_on_sample_event_non_tool_does_nothing(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_sample_event_non_tool()
    await logger.on_sample_event(data)
    mock_manager.log_tool_event.assert_not_called()


# ---------------------------------------------------------------------------
# on_task_end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_task_end_calls_end_run(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_task_end(eval_id="e1")
    await logger.on_task_end(data)
    mock_manager.end_run.assert_called_once_with("e1")


@pytest.mark.asyncio
async def test_on_task_end_calls_flush(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_task_end()
    await logger.on_task_end(data)
    mock_manager.flush.assert_called_once()


# ---------------------------------------------------------------------------
# on_sample_end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_sample_end_calls_log_sample(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_sample_end()
    await logger.on_sample_end(data)
    mock_manager.log_sample.assert_called_once()


@pytest.mark.asyncio
async def test_on_sample_end_passes_eval_id(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_sample_end(eval_id="e99")
    await logger.on_sample_end(data)
    kwargs = mock_manager.log_sample.call_args[1]
    assert kwargs["run_id"] == "e99"


@pytest.mark.asyncio
async def test_on_sample_end_passes_sample_id(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_sample_end(sample_id="s42")
    await logger.on_sample_end(data)
    kwargs = mock_manager.log_sample.call_args[1]
    assert kwargs["sample_id"] == "s42"


@pytest.mark.asyncio
async def test_on_sample_end_passes_multiple_scores(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_sample_end(scores={
        "accuracy": _make_score(1.0),
        "f1": _make_score(0.8),
    })
    await logger.on_sample_end(data)
    kwargs = mock_manager.log_sample.call_args[1]
    assert kwargs["scores"]["accuracy"] == 1.0
    assert kwargs["scores"]["f1"] == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_on_sample_end_none_score_when_no_scores(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_sample_end(scores={})
    await logger.on_sample_end(data)
    kwargs = mock_manager.log_sample.call_args[1]
    assert kwargs["scores"] == {}


@pytest.mark.asyncio
async def test_on_sample_end_handles_score_type_error(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_sample_end(scores={"acc": _make_score(None)})
    await logger.on_sample_end(data)
    kwargs = mock_manager.log_sample.call_args[1]
    assert kwargs["scores"]["acc"] is None


@pytest.mark.asyncio
async def test_on_sample_end_latency_ms(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_sample_end(total_time=1.5)
    await logger.on_sample_end(data)
    kwargs = mock_manager.log_sample.call_args[1]
    assert kwargs["metadata"]["llm.latency_ms"] == pytest.approx(1500.0)


@pytest.mark.asyncio
async def test_on_sample_end_zero_latency_when_none(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_sample_end(total_time=None)
    await logger.on_sample_end(data)
    kwargs = mock_manager.log_sample.call_args[1]
    assert kwargs["metadata"]["llm.latency_ms"] == 0.0


@pytest.mark.asyncio
async def test_on_sample_end_passes_model_name(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_sample_end(model_name="claude-3")
    await logger.on_sample_end(data)
    kwargs = mock_manager.log_sample.call_args[1]
    assert kwargs["metadata"]["llm.model"] == "claude-3"


@pytest.mark.asyncio
async def test_on_sample_end_no_model_usage(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_sample_end(model_name=None)
    await logger.on_sample_end(data)
    kwargs = mock_manager.log_sample.call_args[1]
    assert kwargs["metadata"]["llm.model"] is None


# ---------------------------------------------------------------------------
# on_eval_set_end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_eval_set_end_calls_shutdown(
    logger: OpenTelemetryLogger, mock_manager: MagicMock
) -> None:
    data = _make_eval_set_end()
    await logger.on_eval_set_end(data)
    mock_manager.shutdown.assert_called_once()
