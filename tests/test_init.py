"""Tests for inspect_otel.__init__ (public API and setup_hooks)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import inspect_otel
from inspect_otel import (
    OpenTelemetryLogger,
    build_manager,
    load_config,
    setup_hooks,
    trace_llm,
    trace_tool,
)

# ---------------------------------------------------------------------------
# Public API symbols
# ---------------------------------------------------------------------------


def test_all_exports_are_importable() -> None:
    for name in inspect_otel.__all__:
        assert hasattr(inspect_otel, name), f"Missing export: {name}"


def test_trace_llm_is_callable() -> None:
    assert callable(trace_llm)


def test_trace_tool_is_callable() -> None:
    assert callable(trace_tool)


def test_load_config_is_callable() -> None:
    assert callable(load_config)


def test_build_manager_is_callable() -> None:
    assert callable(build_manager)


def test_setup_hooks_is_callable() -> None:
    assert callable(setup_hooks)


# ---------------------------------------------------------------------------
# setup_hooks
# ---------------------------------------------------------------------------


def test_setup_hooks_raises_without_config_or_path() -> None:
    with pytest.raises(ValueError, match="Either 'config' or 'config_path' must be provided"):
        setup_hooks()


def test_setup_hooks_with_config_dict(tmp_path: Path) -> None:
    config = {
        "telemetry": {
            "service_name": "test",
            "otlp_endpoint": None,
        }
    }
    registered_classes: list = []

    def fake_register(name: str, description: str):
        def decorator(cls):
            registered_classes.append(cls)
            return cls
        return decorator

    with patch("inspect_otel.OpenTelemetryBackend") as mock_backend, \
         patch("inspect_ai.hooks.hooks", fake_register):
        mock_backend.return_value = MagicMock()
        setup_hooks(config=config)

    assert len(registered_classes) == 1


def test_setup_hooks_with_config_path(tmp_path: Path) -> None:
    config_content = (
        "telemetry:\n  service_name: test\n  otlp_endpoint: null\n"
    )
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text(config_content, encoding="utf-8")

    registered_classes: list = []

    def fake_register(name: str, description: str):
        def decorator(cls):
            registered_classes.append(cls)
            return cls
        return decorator

    with patch("inspect_otel.OpenTelemetryBackend") as mock_backend, \
         patch("inspect_ai.hooks.hooks", fake_register):
        mock_backend.return_value = MagicMock()
        setup_hooks(config_path=config_file)

    assert len(registered_classes) == 1


def test_setup_hooks_registered_class_is_otel_logger_subclass(tmp_path: Path) -> None:
    config = {"telemetry": {"service_name": "test", "otlp_endpoint": None}}
    captured: list = []

    def fake_register(name: str, description: str):
        def decorator(cls):
            captured.append(cls)
            return cls
        return decorator

    with patch("inspect_otel.OpenTelemetryBackend") as mock_backend, \
         patch("inspect_ai.hooks.hooks", fake_register):
        mock_backend.return_value = MagicMock()
        setup_hooks(config=config)

    assert issubclass(captured[0], OpenTelemetryLogger)


def test_setup_hooks_inner_class_instantiates_with_manager(tmp_path: Path) -> None:
    config = {"telemetry": {"service_name": "test", "otlp_endpoint": None}}
    captured_cls: list = []

    def fake_register(name: str, description: str):
        def decorator(cls):
            captured_cls.append(cls)
            return cls
        return decorator

    mock_backend_instance = MagicMock()
    with patch("inspect_otel.OpenTelemetryBackend", return_value=mock_backend_instance), \
         patch("inspect_ai.hooks.hooks", fake_register):
        setup_hooks(config=config)

    # Instantiate the registered class and check it is an OpenTelemetryLogger
    instance = captured_cls[0]()
    assert isinstance(instance, OpenTelemetryLogger)
