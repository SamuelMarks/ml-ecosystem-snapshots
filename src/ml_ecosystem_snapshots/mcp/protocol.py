"""JSON-RPC 2.0 transport and envelope models for the MCP stdio server."""

from __future__ import annotations

import json
import sys
import typing

import pydantic

from ml_ecosystem_snapshots.models import JsonValue


class JsonRpcError(pydantic.BaseModel):
    """Typed JSON-RPC 2.0 error payload."""

    code: int
    message: str


class JsonRpcRequest(pydantic.BaseModel):
    """Typed JSON-RPC 2.0 request envelope."""

    jsonrpc: str = "2.0"
    id: int | str | None = None
    method: str = ""
    params: dict[str, JsonValue] = pydantic.Field(default_factory=dict)


class JsonRpcResponse(pydantic.BaseModel):
    """Typed JSON-RPC 2.0 response envelope."""

    jsonrpc: str = "2.0"
    id: int | str | None = None
    result: dict[str, JsonValue] | None = None
    error: JsonRpcError | None = None


class JsonRpcResultDict(dict[str, object]):
    """Typed dictionary wrapper for JSON-RPC 2.0 result payloads."""

    @typing.overload
    def __getitem__(self, key: typing.Literal["tools"]) -> list[dict[str, str]]: ...

    @typing.overload
    def __getitem__(self, key: typing.Literal["content"]) -> list[dict[str, str]]: ...

    @typing.overload
    def __getitem__(self, key: str) -> object: ...

    def __getitem__(self, key: str) -> object:
        """Retrieve a JSON-RPC result field by dictionary key.

        Args:
            key: Field name.

        Returns:
            Field value.
        """
        return super().__getitem__(key)


class JsonRpcResponseDict(dict[str, object]):
    """Typed dictionary wrapper for JSON-RPC 2.0 responses."""

    @typing.overload
    def __getitem__(self, key: typing.Literal["result"]) -> JsonRpcResultDict: ...

    @typing.overload
    def __getitem__(self, key: typing.Literal["error"]) -> dict[str, str]: ...

    @typing.overload
    def __getitem__(self, key: typing.Literal["jsonrpc"]) -> str: ...

    @typing.overload
    def __getitem__(self, key: typing.Literal["id"]) -> str | int | None: ...

    @typing.overload
    def __getitem__(self, key: str) -> object: ...

    def __getitem__(self, key: str) -> object:
        """Retrieve a JSON-RPC response field by dictionary key.

        Args:
            key: Field name.

        Returns:
            Field value.
        """
        return super().__getitem__(key)


def run_stdio_loop(
    message_handler: typing.Callable[
        [typing.Mapping[str, object]], typing.Mapping[str, object]
    ],
    input_stream: typing.TextIO | None = None,
    output_stream: typing.TextIO | None = None,
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


__all__ = [
    "JsonRpcError",
    "JsonRpcRequest",
    "JsonRpcResponse",
    "JsonRpcResponseDict",
    "JsonRpcResultDict",
    "run_stdio_loop",
]
