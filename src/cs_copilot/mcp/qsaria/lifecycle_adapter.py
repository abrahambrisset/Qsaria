"""FastMCP adapter for Qsaria lifecycle and reporting operations."""

from __future__ import annotations

import asyncio
import inspect
import logging
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict

from cs_copilot.mcp.errors import MCPToolError
from cs_copilot.mcp.tool_adapter import ToolSpec, _coerce_return_value, _resolve_annotations

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QsariaLifecycleSpec(ToolSpec):
    """A model-free experiment lifecycle operation."""


def is_qsaria_lifecycle_spec(spec: ToolSpec) -> bool:
    """Return whether ``spec`` uses the lifecycle facade adapter."""

    return isinstance(spec, QsariaLifecycleSpec)


def build_qsaria_lifecycle_tool(
    spec: QsariaLifecycleSpec,
    facade: Any,
) -> Callable[..., Any]:
    """Wrap a lifecycle facade method without a generic session manifest.

    Lifecycle methods own their atomic experiment writes.  Avoiding the
    generic MCP manifest prevents bootstrap and catalog-style reads from
    creating an unrelated implicit session.
    """

    if not isinstance(spec, QsariaLifecycleSpec):
        raise TypeError("build_qsaria_lifecycle_tool requires a QsariaLifecycleSpec")

    bound_method = getattr(facade, spec.method)
    signature = inspect.signature(bound_method)
    resolved = _resolve_annotations(bound_method)
    public_params = [
        parameter.replace(annotation=resolved.get(name, parameter.annotation))
        for name, parameter in signature.parameters.items()
    ]
    return_annotation = resolved.get("return", signature.return_annotation)
    public_signature = inspect.Signature(
        public_params,
        return_annotation=return_annotation,
    )
    public_annotations: Dict[str, Any] = {
        parameter.name: parameter.annotation
        for parameter in public_params
        if parameter.annotation is not inspect.Parameter.empty
    }
    if return_annotation is not inspect.Signature.empty:
        public_annotations["return"] = return_annotation

    async def _invoke(**kwargs: Any) -> Any:
        try:
            if inspect.iscoroutinefunction(bound_method):
                result = await bound_method(**kwargs)
            else:
                result = await asyncio.to_thread(bound_method, **kwargs)
            return _coerce_return_value(result)
        except MCPToolError:
            raise
        except Exception as exc:  # noqa: BLE001 - stable MCP error envelope
            logger.error(
                "Qsaria lifecycle tool %s failed: %s\n%s",
                spec.mcp_name,
                exc,
                traceback.format_exc(),
            )
            raise MCPToolError(f"{spec.mcp_name} failed: {exc}") from exc

    _invoke.__name__ = spec.mcp_name
    _invoke.__qualname__ = spec.mcp_name
    _invoke.__doc__ = spec.summary or (bound_method.__doc__ or "").strip()
    _invoke.__signature__ = public_signature  # type: ignore[attr-defined]
    _invoke.__annotations__ = public_annotations
    _invoke.__wrapped__ = bound_method  # type: ignore[attr-defined]
    return _invoke


__all__ = [
    "QsariaLifecycleSpec",
    "build_qsaria_lifecycle_tool",
    "is_qsaria_lifecycle_spec",
]
