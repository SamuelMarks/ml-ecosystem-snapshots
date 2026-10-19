"""JSON-RPC 2.0 transport and envelope models for the MCP stdio server."""

from __future__ import annotations

import json
import sys
from typing import Any, Callable, Dict, Optional, TextIO, Union
from pydantic import BaseModel, Field


class JsonRpcError(BaseModel):
    """Typed JSON-RPC 2.0 error payload."""

    code: int
    message: str


class JsonRpcRequest(BaseModel):
    """Typed JSON-RPC 2.0 request envelope."""

    jsonrpc: str = "2.0"
    id: Optional[Union[int, str]] = None
    method: str = ""
    params: Dict[str, Any] = Field(default_factory=dict)


class JsonRpcResponse(BaseModel):
    """Typed JSON-RPC 2.0 response envelope."""

    jsonrpc: str = "2.0"
    id: Optional[Union[int, str]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[JsonRpcError] = None


def run_stdio_loop(
    message_handler: Callable[[Dict[str, Any]], Dict[str, Any]],
    input_stream: Optional[TextIO] = None,
    output_stream: Optional[TextIO] = None,
) -> None:
    """Run the JSON-RPC 2.0 stdio event loop, dispatching parsed requests to `message_handler`.

    Args:
        message_handler: Callable accepting a parsed JSON-RPC request dictionary and returning a response dictionary.
        input_stream: Optional input text stream (defaults to `sys.stdin`).
        output_stream: Optional output text stream (defaults to `sys.stdout`).
    """
    in_s = input_stream or sys.stdin
    out_s = output_stream or sys.stdout

    for line in in_s:
        line_clean = line.strip()
        if not line_clean:
            continue
        try:
            req = json.loads(line_clean)
            resp = message_handler(req)
            out_s.write(json.dumps(resp) + "\n")
            out_s.flush()
        except Exception as e:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {e}"},
            }
            out_s.write(json.dumps(err_resp) + "\n")
            out_s.flush()


__all__ = ["JsonRpcError", "JsonRpcRequest", "JsonRpcResponse", "run_stdio_loop"]
