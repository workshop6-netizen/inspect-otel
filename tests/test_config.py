"""Tests for inspect_otel.config."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from inspect_otel.config import build_manager, load_config
from inspect_otel.telemetry_manager import TelemetryManager

# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


@pytest.fixture()
def config_file(tmp_path: Path) -> Path:
    content = textwrap.dedent("""\
        telemetry:
          service_name: test-service
          otlp_endpoint: http://localhost:4318/v1/traces
          sampling:
            rate: 0.5
          logging:
            failure_only: true
          async: false
    """)
    p = tmp_path / "config.yaml"
    p.write_text(content, encoding="utf-8")
    return p


def test_load_config_returns_dict(config_file: Path) -> None:
    cfg = load_config(config_file)
    assert isinstance(cfg, dict)


def test_load_config_parses_service_name(config_file: Path) -> None:
    cfg = load_config(config_file)
    assert cfg["telemetry"]["service_name"] == "test-service"


def test_load_config_parses_endpoint(config_file: Path) -> None:
    cfg = load_config(config_file)
    assert cfg["telemetry"]["otlp_endpoint"] == "http://localhost:4318/v1/traces"


def test_load_config_parses_sampling_rate(config_file: Path) -> None:
    cfg = load_config(config_file)
    assert cfg["telemetry"]["sampling"]["rate"] == 0.5


def test_load_config_parses_failure_only(config_file: Path) -> None:
    cfg = load_config(config_file)
    assert cfg["telemetry"]["logging"]["failure_only"] is True


def test_load_config_parses_async_flag(config_file: Path) -> None:
    cfg = load_config(config_file)
    assert cfg["telemetry"]["async"] is False


def test_load_config_accepts_path_object(config_file: Path) -> None:
    cfg = load_config(config_file)
    assert "telemetry" in cfg


def test_load_config_accepts_string_path(config_file: Path) -> None:
    cfg = load_config(str(config_file))
    assert "telemetry" in cfg


def test_load_config_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


# ---------------------------------------------------------------------------
# build_manager
# ---------------------------------------------------------------------------


_MINIMAL_CONFIG: dict = {"telemetry": {}}

_FULL_CONFIG: dict = {
    "telemetry": {
        "service_name": "my-service",
        "otlp_endpoint": None,
        "sampling": {"rate": 0.3},
        "logging": {"failure_only": True},
        "async": False,
    }
}


def test_build_manager_returns_telemetry_manager() -> None:
    mgr = build_manager(_MINIMAL_CONFIG)
    assert isinstance(mgr, TelemetryManager)


def test_build_manager_defaults_service_name() -> None:
    """service_name defaults to 'inspect-ai' when absent."""
    with patch("inspect_otel.config.OpenTelemetryBackend") as mock_cls:
        mock_cls.return_value = MagicMock()
        build_manager(_MINIMAL_CONFIG)
    _, kwargs = mock_cls.call_args
    assert kwargs["service_name"] == "inspect-ai"


def test_build_manager_defaults_sample_rate() -> None:
    with patch("inspect_otel.config.OpenTelemetryBackend") as mock_cls:
        mock_cls.return_value = MagicMock()
        build_manager(_MINIMAL_CONFIG)
    _, kwargs = mock_cls.call_args
    assert kwargs["sample_rate"] == 1.0


def test_build_manager_defaults_failure_only_false() -> None:
    with patch("inspect_otel.config.OpenTelemetryBackend") as mock_cls:
        mock_cls.return_value = MagicMock()
        build_manager(_MINIMAL_CONFIG)
    _, kwargs = mock_cls.call_args
    assert kwargs["failure_only"] is False


def test_build_manager_defaults_async_true() -> None:
    with patch("inspect_otel.config.OpenTelemetryBackend") as mock_cls:
        mock_cls.return_value = MagicMock()
        build_manager(_MINIMAL_CONFIG)
    _, kwargs = mock_cls.call_args
    assert kwargs["async_mode"] is True


def test_build_manager_forwards_full_config() -> None:
    with patch("inspect_otel.config.OpenTelemetryBackend") as mock_cls:
        mock_cls.return_value = MagicMock()
        build_manager(_FULL_CONFIG)
    _, kwargs = mock_cls.call_args
    assert kwargs["service_name"] == "my-service"
    assert kwargs["sample_rate"] == 0.3
    assert kwargs["failure_only"] is True
    assert kwargs["async_mode"] is False
