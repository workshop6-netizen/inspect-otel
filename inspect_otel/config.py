"""Configuration loading and manager factory for inspect-otel."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .otel_backend import OpenTelemetryBackend
from .telemetry_manager import TelemetryManager


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load telemetry configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.
    """
    resolved = Path(path).resolve()
    with resolved.open("r", encoding="utf-8") as f:
        result: dict[str, Any] = yaml.safe_load(f)
    return result


def build_manager(config: dict[str, Any]) -> TelemetryManager:
    """Construct a :class:`TelemetryManager` from a loaded config dict.

    The ``telemetry`` section supports the following keys:

    .. code-block:: yaml

        telemetry:
          service_name: my-service     # default: "inspect-ai"
          otlp_endpoint: http://...    # default: None (no export)
          experiment_id: "exp-001"     # surfaced as eval.experiment_id
          dataset_version: "v1.2"      # surfaced as eval.dataset.version
          sampling:
            rate: 1.0                  # 0.0-1.0, default 1.0
          logging:
            failure_only: false        # only log failed samples
          async: true                  # BatchSpanProcessor vs Simple

    Args:
        config: Configuration dict as returned by :func:`load_config`.

    Returns:
        A fully initialised :class:`TelemetryManager`.
    """
    t = config["telemetry"]
    backend = OpenTelemetryBackend(
        service_name=t.get("service_name", "inspect-ai"),
        otlp_endpoint=t.get("otlp_endpoint"),
        sample_rate=t.get("sampling", {}).get("rate", 1.0),
        failure_only=t.get("logging", {}).get("failure_only", False),
        async_mode=t.get("async", True),
    )
    return TelemetryManager([backend])
