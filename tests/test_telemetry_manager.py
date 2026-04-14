"""Tests for inspect_otel.telemetry_manager."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from inspect_otel.telemetry_manager import TelemetryManager


@pytest.fixture()
def mock_backend() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def manager(mock_backend: MagicMock) -> TelemetryManager:
    return TelemetryManager([mock_backend])


def test_start_run_delegates(manager: TelemetryManager, mock_backend: MagicMock) -> None:
    manager.start_run("r1", {"k": "v"})
    mock_backend.start_run.assert_called_once_with("r1", {"k": "v"})


def test_start_sample_delegates(manager: TelemetryManager, mock_backend: MagicMock) -> None:
    manager.start_sample("e1", "s1", {"x": "y"})
    mock_backend.start_sample.assert_called_once_with("e1", "s1", {"x": "y"})


def test_log_model_call_delegates(manager: TelemetryManager, mock_backend: MagicMock) -> None:
    manager.log_model_call(
        run_id="r1", eval_id="e1", model_name="gpt-4",
        input_tokens=10, output_tokens=20, total_tokens=30,
        latency_ms=500.0, retries=0,
    )
    mock_backend.log_model_call.assert_called_once_with(
        run_id="r1", eval_id="e1", model_name="gpt-4",
        input_tokens=10, output_tokens=20, total_tokens=30,
        latency_ms=500.0, retries=0,
    )


def test_log_tool_event_delegates(manager: TelemetryManager, mock_backend: MagicMock) -> None:
    manager.log_tool_event(
        run_id="r1", eval_id="e1", sample_id="s1",
        tool_name="calc", tool_input={"expr": "1+1"},
    )
    mock_backend.log_tool_event.assert_called_once_with(
        run_id="r1", eval_id="e1", sample_id="s1",
        tool_name="calc", tool_input={"expr": "1+1"},
    )


def test_log_sample_delegates(manager: TelemetryManager, mock_backend: MagicMock) -> None:
    manager.log_sample(
        run_id="r1",
        sample_id="s1",
        inputs={"input": "hi"},
        outputs={"output": "bye"},
        expected={"target": "bye"},
        scores={"acc": 1.0},
        metadata={"llm.model": "gpt-4"},
    )
    mock_backend.log_sample.assert_called_once_with(
        run_id="r1",
        sample_id="s1",
        inputs={"input": "hi"},
        outputs={"output": "bye"},
        expected={"target": "bye"},
        scores={"acc": 1.0},
        metadata={"llm.model": "gpt-4"},
    )


def test_end_run_delegates(manager: TelemetryManager, mock_backend: MagicMock) -> None:
    manager.end_run("r1")
    mock_backend.end_run.assert_called_once_with("r1")


def test_flush_delegates(manager: TelemetryManager, mock_backend: MagicMock) -> None:
    manager.flush()
    mock_backend.flush.assert_called_once()


def test_shutdown_delegates(manager: TelemetryManager, mock_backend: MagicMock) -> None:
    manager.shutdown()
    mock_backend.shutdown.assert_called_once()


def test_multiple_backends_start_run() -> None:
    b1, b2 = MagicMock(), MagicMock()
    mgr = TelemetryManager([b1, b2])
    mgr.start_run("r1", {})
    b1.start_run.assert_called_once_with("r1", {})
    b2.start_run.assert_called_once_with("r1", {})


def test_multiple_backends_log_sample() -> None:
    b1, b2 = MagicMock(), MagicMock()
    mgr = TelemetryManager([b1, b2])
    mgr.log_sample("r1", "s1", {}, {}, {}, {}, {})
    b1.log_sample.assert_called_once()
    b2.log_sample.assert_called_once()


def test_multiple_backends_end_run() -> None:
    b1, b2 = MagicMock(), MagicMock()
    mgr = TelemetryManager([b1, b2])
    mgr.end_run("r1")
    b1.end_run.assert_called_once_with("r1")
    b2.end_run.assert_called_once_with("r1")


def test_multiple_backends_shutdown() -> None:
    b1, b2 = MagicMock(), MagicMock()
    mgr = TelemetryManager([b1, b2])
    mgr.shutdown()
    b1.shutdown.assert_called_once()
    b2.shutdown.assert_called_once()
