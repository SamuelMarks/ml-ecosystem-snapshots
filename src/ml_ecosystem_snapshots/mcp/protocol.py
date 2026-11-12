"""JSON-RPC 2.0 transport and envelope models for the MCP stdio server."""

from __future__ import annotations

import json
import sys
import typing

import pydantic

from ml_ecosystem_snapshots.exceptions import SnapshotError
from ml_ecosystem_snapshots.models import JsonValue


class JsonRpcError(pydantic.BaseModel):
    """Typed JSON-RPC 2.0 error payload."""

    model_config = pydantic.ConfigDict(extra="forbid")

    code: int
    message: str


class JsonRpcContentItem(pydantic.BaseModel):
    """Typed MCP content block item within `JsonRpcResult`."""

    model_config = pydantic.ConfigDict(extra="forbid")

    type: str = "text"
    text: str = ""


class JsonRpcResult(pydantic.BaseModel):
    """Typed JSON-RPC 2.0 result payload for `tools/list` and `tools/call`."""

    model_config = pydantic.ConfigDict(extra="allow")

    tools: list[dict[str, JsonValue]] | None = None
    content: list[JsonRpcContentItem] | None = None


class JsonRpcRequest(pydantic.BaseModel):
    """Typed JSON-RPC 2.0 request envelope."""

    model_config = pydantic.ConfigDict(extra="allow")

    jsonrpc: str = "2.0"
    id: int | str | None = None
    method: str = ""
    params: dict[str, JsonValue] = pydantic.Field(default_factory=dict)


class JsonRpcResponse(pydantic.BaseModel):
    """Typed JSON-RPC 2.0 response envelope."""

    model_config = pydantic.ConfigDict(extra="forbid")

    jsonrpc: str = "2.0"
    id: int | str | None = None
    result: JsonRpcResult | None = None
    error: JsonRpcError | None = None

    def to_wire_dict(self) -> dict[str, JsonValue]:
        """Serialize response to a JSON-RPC 2.0 wire dictionary preserving `id` while omitting unset `result`/`error`.

        Returns:
            JSON-RPC 2.0 wire dictionary.
        """
        payload = self.model_dump(exclude_none=True)
        payload["id"] = self.id
        return payload


def run_stdio_loop(
    message_handler: typing.Callable[
        [typing.Mapping[str, JsonValue]],
        JsonRpcResponse | typing.Mapping[str, JsonValue],
    ],
    input_stream: typing.TextIO | None = None,
    output_stream: typing.TextIO | None = None,
) -> None:
    """Run the JSON-RPC 2.0 stdio event loop, dispatching parsed requests to `message_handler`.

    Args:
        message_handler: Callable accepting a parsed JSON-RPC request dictionary and returning a `JsonRpcResponse` or dictionary.
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
            serialized = resp.to_wire_dict() if isinstance(resp, JsonRpcResponse) else dict(resp)
            nl = chr(10)
            out_s.write(json.dumps(serialized) + nl)
            out_s.flush()
        except (
            SnapshotError,
            ImportError,
            AttributeError,
            TypeError,
            ValueError,
            LookupError,
            OSError,
            RuntimeError,
            SyntaxError,
            pydantic.ValidationError,
        ) as e:
            nl = chr(10)
            err_resp = JsonRpcResponse(
                id=None,
                error=JsonRpcError(code=-32700, message=f"Parse error: {e}"),
            )
            out_s.write(json.dumps(err_resp.to_wire_dict()) + nl)
            out_s.flush()


__all__ = [
    "JsonRpcContentItem",
    "JsonRpcError",
    "JsonRpcRequest",
    "JsonRpcResponse",
    "JsonRpcResult",
    "run_stdio_loop",
]
