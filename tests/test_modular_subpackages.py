"""Unit tests for the modular mcp/, introspection/, and serialization/ subpackages."""

from __future__ import annotations

import io
import json
from typing import Any, Dict

from ml_ecosystem_snapshots.introspection import (
    GhostInspector,
    extract_accepted_kwargs_from_ast,
    preload_griffe_cache,
    sanitize_param_default,
    sanitize_type_str,
    unwrap_decorated_target,
)
from ml_ecosystem_snapshots.mcp import (
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcResponse,
    McpToolSpec,
    SnapshotRepository,
    dispatch_tool_call,
    run_stdio_loop,
)
from ml_ecosystem_snapshots.mcp.tools import (
    CONCEPT_ARG_MAPPINGS,
    check_code_block,
    check_hallucination,
    check_mlir_op,
    check_ptx_instruction,
    check_rdna_instruction,
    check_sass_instruction,
    check_stablehlo_op,
    explain_anti_pattern,
    get_api_signature,
    search_apis,
    translate_concept_arguments,
)
from ml_ecosystem_snapshots.serialization import (
    hydrate_ghost_ref,
    migrate_ghost_ref_v2,
)


def test_mcp_protocol_models_and_stdio_loop() -> None:
    """Verify JsonRpcRequest, JsonRpcResponse, JsonRpcError models and run_stdio_loop execution."""
    req = JsonRpcRequest(id=1, method="tools/list", params={})
    assert req.jsonrpc == "2.0"
    assert req.id == 1

    err = JsonRpcError(code=-32601, message="Not found")
    resp = JsonRpcResponse(id=1, error=err)
    assert resp.error is not None
    assert resp.error.code == -32601

    spec = McpToolSpec(
        name="sample",
        description="Sample tool",
        input_schema={"type": "object"},
    )
    assert spec.name == "sample"

    in_buf = io.StringIO(
        chr(10).join(["", '{"jsonrpc": "2.0", "id": 1, "method": "ping"}', "not_json"])
    )
    out_buf = io.StringIO()

    def handler(msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle test JSON-RPC request.

        Args:
            msg: Parsed JSON-RPC message dictionary.

        Returns:
            JSON-RPC response dictionary.
        """
        return {"jsonrpc": "2.0", "id": msg.get("id"), "result": {"ok": True}}

    run_stdio_loop(handler, input_stream=in_buf, output_stream=out_buf)
    lines = [line for line in out_buf.getvalue().splitlines() if line.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["result"]["ok"] is True
    assert json.loads(lines[1])["error"]["code"] == -32700


def test_mcp_cache_snapshot_repository() -> None:
    """Verify SnapshotRepository preload, get, and clear operations."""
    backing: Dict[str, Dict[str, Any]] = {}
    repo = SnapshotRepository(backing_store=backing)
    repo.preload("custom_fw", {"categories": {"UTIL": []}}, version="1.0")
    assert "custom_fw@1.0" in backing
    repo.preload("custom_fw", {"categories": {"UTIL": []}})
    assert "custom_fw" in backing
    snap = repo.get("nvidia_sass")
    assert "categories" in snap
    repo.clear()
    assert len(backing) == 0


def test_mcp_registry_and_tools_reexports() -> None:
    """Verify dispatch_tool_call and mcp.tools re-exports operate deterministically."""
    unknown_resp = dispatch_tool_call(99, "nonexistent_tool", {})
    assert unknown_resp["error"]["code"] == -32601

    valid_resp = dispatch_tool_call(100, "check_sass_instruction", {"mnemonic": "FFMA"})
    assert "result" in valid_resp

    assert isinstance(CONCEPT_ARG_MAPPINGS, dict)
    assert callable(check_code_block)
    assert callable(check_hallucination)
    assert callable(check_mlir_op)
    assert callable(check_ptx_instruction)
    assert callable(check_rdna_instruction)
    assert callable(check_sass_instruction)
    assert callable(check_stablehlo_op)
    assert callable(explain_anti_pattern)
    assert callable(get_api_signature)
    assert callable(search_apis)
    assert callable(translate_concept_arguments)


def test_introspection_and_serialization_subpackages() -> None:
    """Verify introspection and serialization subpackage helpers."""

    def sample_kw(**kwargs: Any) -> Any:
        """Sample function accessing kwargs['alpha'].

        Args:
            kwargs: Variadic keyword arguments.

        Returns:
            Extracted alpha value.
        """
        return kwargs.get("alpha")

    assert extract_accepted_kwargs_from_ast(sample_kw) == ["alpha"]
    assert sanitize_param_default(42) == ("42", None, False)
    assert sanitize_type_str("Optional[int]") == "int | None"
    assert unwrap_decorated_target(sample_kw) is sample_kw
    preload_griffe_cache(frameworks=[])
    assert GhostInspector is not None

    raw = {"name": "op", "api_path": "pkg.op", "kind": "function", "params": []}
    hydrated = hydrate_ghost_ref(raw)
    assert hydrated.api_path == "pkg.op"
    migrated = migrate_ghost_ref_v2(raw)
    assert migrated.schema_version == "2.0.0"
