"""Model Context Protocol (MCP) server subpackage for ml_ecosystem_snapshots.

Provides modular components for JSON-RPC 2.0 stdio transport (`protocol`),
tool schema declarations and dispatch (`registry`), thread-safe snapshot
caching (`cache`), and domain-specific verification tools (`tools`).
"""

from __future__ import annotations

from ml_ecosystem_snapshots.mcp.cache import SnapshotRepository
from ml_ecosystem_snapshots.mcp.models import (
    AntiPatternExplanationResult,
    CodeBlockCheckResult,
    CodeBlockFinding,
    ConceptTranslationResult,
    DomainOpCheckResult,
    HallucinationCheckResult,
    IrCheckResult,
    IsaCheckResult,
)
from ml_ecosystem_snapshots.mcp.protocol import (
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcResponse,
    run_stdio_loop,
)
from ml_ecosystem_snapshots.mcp.registry import McpToolSpec, dispatch_tool_call

__all__ = [
    "AntiPatternExplanationResult",
    "CodeBlockCheckResult",
    "CodeBlockFinding",
    "ConceptTranslationResult",
    "DomainOpCheckResult",
    "HallucinationCheckResult",
    "IrCheckResult",
    "IsaCheckResult",
    "JsonRpcError",
    "JsonRpcRequest",
    "JsonRpcResponse",
    "McpToolSpec",
    "SnapshotRepository",
    "dispatch_tool_call",
    "run_stdio_loop",
]
