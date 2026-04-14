"""inspect-otel: OpenTelemetry integration for Inspect AI evaluation workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import build_manager, load_config
from .decorators import trace_llm, trace_tool
from .logger import OpenTelemetryLogger
from .otel_backend import OpenTelemetryBackend
from .telemetry_manager import TelemetryManager

if TYPE_CHECKING:
    import os

__all__ = [
    "OpenTelemetryBackend",
    "OpenTelemetryLogger",
    "TelemetryManager",
    "build_manager",
    "load_config",
    "setup_hooks",
    "trace_llm",
    "trace_tool",
]


def setup_hooks(
    config: dict[str, Any] | None = None,
    config_path: str | os.PathLike[str] | None = None,
) -> None:
    """Register OpenTelemetry hooks with Inspect AI.

    Either ``config`` or ``config_path`` must be supplied.

    Args:
        config: A pre-loaded configuration dictionary.
        config_path: Path to a YAML configuration file.

    Example::

        from inspect_otel import setup_hooks
        from inspect_ai import eval_set

        setup_hooks(config_path="config.yaml")
        eval_set(evals=[my_eval], model="gpt-4")
    """
    if config is None:
        if config_path is None:
            raise ValueError("Either 'config' or 'config_path' must be provided.")
        config = load_config(config_path)
    manager = build_manager(config)

    from inspect_ai.hooks import hooks as _register

    @_register(
        name="inspect-otel",
        description="OpenTelemetry tracing for Inspect AI evaluation workflows.",
    )
    class _OTelHooks(OpenTelemetryLogger):
        def __init__(self) -> None:
            super().__init__(manager)
