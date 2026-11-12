"""Deterministic MCP tool specification model and dictionary dispatch registry."""

from __future__ import annotations

import json
import typing

import pydantic
import typing_extensions

from ml_ecosystem_snapshots.mcp.cache import SnapshotRepository
from ml_ecosystem_snapshots.mcp.models import DomainOpCheckResult
from ml_ecosystem_snapshots.mcp.protocol import (
    JsonRpcContentItem,
    JsonRpcError,
    JsonRpcResponse,
    JsonRpcResult,
)
from ml_ecosystem_snapshots.mcp.schemas import (
    DomainOpToolRequest,
    get_mcp_tools_list,
)
from ml_ecosystem_snapshots.mcp.tools import (
    check_array_api_op,
    check_aten_op,
    check_chex_op,
    check_code_block,
    check_cpp_op,
    check_flash_attention_op,
    check_grain_op,
    check_hallucination,
    check_metal_op,
    check_mlir_op,
    check_nccl_op,
    check_onnx_op,
    check_optax_op,
    check_orbax_op,
    check_ptx_instruction,
    check_rdna_instruction,
    check_safetensors_op,
    check_sass_instruction,
    check_scipy_op,
    check_stablehlo_op,
    check_torchaudio_op,
    check_torchvision_op,
    check_wasm_instruction,
    check_webgl_op,
    check_wgsl_op,
    explain_anti_pattern,
    get_api_signature,
    search_apis,
    translate_concept_arguments,
)
from ml_ecosystem_snapshots.models import JsonValue

ToolHandlerResult: typing_extensions.TypeAlias = "pydantic.BaseModel | dict[str, JsonValue] | list[str] | None"
ToolHandlerFn: typing_extensions.TypeAlias = "typing.Callable[[dict[str, JsonValue]], ToolHandlerResult]"


class McpToolSpec(pydantic.BaseModel):
    """Pydantic model describing an MCP tool declaration and input schema."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    input_schema: dict[str, JsonValue]
    handler: (
        typing.Callable[
            [dict[str, JsonValue]],
            pydantic.BaseModel | dict[str, JsonValue] | list[str] | None,
        ]
        | None
    ) = None


def _as_str_list(val: JsonValue) -> list[str] | None:
    """Coerce a JSON value into `list[str] | None`.

    Args:
        val: Raw JSON value.

    Returns:
        List of strings if `val` is a list, otherwise None.
    """
    if isinstance(val, list):
        return [str(x) for x in val]
    return None


def _as_int_or_none(val: JsonValue) -> int | None:
    """Coerce a JSON value into `int | None`.

    Args:
        val: Raw JSON value.

    Returns:
        Integer value or None.
    """
    if isinstance(val, int) and not isinstance(val, bool):
        return val
    return None


def _as_str_or_none(val: JsonValue) -> str | None:
    """Coerce a JSON value into `str | None`.

    Args:
        val: Raw JSON value.

    Returns:
        String value or None.
    """
    return str(val) if val is not None else None


def _as_dict_or_none(val: JsonValue) -> dict[str, JsonValue] | None:
    """Coerce a JSON value into `dict[str, JsonValue] | None`.

    Args:
        val: Raw JSON value.

    Returns:
        Dictionary or None.
    """
    return dict(val) if isinstance(val, dict) else None


def _as_str_dict_or_list(val: JsonValue) -> list[str] | dict[str, str] | None:
    """Coerce a JSON value into `list[str] | dict[str, str] | None`.

    Args:
        val: Raw JSON value.

    Returns:
        Coerced list/dict of strings or `None`.
    """
    if isinstance(val, list):
        return [str(x) for x in val]
    if isinstance(val, dict):
        return {str(k): str(v) for k, v in val.items()}
    return None


def _as_rank_dict_or_list(val: JsonValue) -> list[int | str] | dict[str, int | str] | None:
    """Coerce a JSON value into `list[int | str] | dict[str, int | str] | None`.

    Args:
        val: Raw JSON value.

    Returns:
        Coerced rank constraint list/dict or `None`.
    """
    if isinstance(val, list):
        return [x if isinstance(x, int) and not isinstance(x, bool) else str(x) for x in val]
    if isinstance(val, dict):
        return {str(k): (v if isinstance(v, int) and not isinstance(v, bool) else str(v)) for k, v in val.items()}
    return None


def _as_shape_list(val: JsonValue) -> list[typing.Sequence[int | str]] | None:
    """Coerce a JSON value into `list[Sequence[int | str]] | None`.

    Args:
        val: Raw JSON value.

    Returns:
        Coerced list of shape sequences or `None`.
    """
    if not isinstance(val, list):
        return None
    out: list[typing.Sequence[int | str]] = []
    for item in val:
        if isinstance(item, list):
            out.append([d if isinstance(d, int) and not isinstance(d, bool) else str(d) for d in item])
    return out


def _as_shape_dict(val: JsonValue) -> dict[str, typing.Sequence[int | str]] | None:
    """Coerce a JSON value into `dict[str, Sequence[int | str]] | None`.

    Args:
        val: Raw JSON value.

    Returns:
        Coerced mapping of shape sequences or `None`.
    """
    if not isinstance(val, dict):
        return None
    out: dict[str, typing.Sequence[int | str]] = {}
    for k, item in val.items():
        if isinstance(item, list):
            out[str(k)] = [d if isinstance(d, int) and not isinstance(d, bool) else str(d) for d in item]
    return out


class _SimpleDomainVerifierProto(typing.Protocol):
    """Protocol for a standard MCP domain operation verifier callable."""

    def __call__(
        self,
        op_name: str,
        inputs_count: int | None = None,
        attributes: list[str] | None = None,
    ) -> DomainOpCheckResult | dict[str, JsonValue]:
        """Validate a domain operation.

        Args:
            op_name: Operation name.
            inputs_count: Optional input count.
            attributes: Optional attribute names.

        Returns:
            Validation result model or dictionary.
        """
        ...


def _make_domain_op_handler(fn: _SimpleDomainVerifierProto) -> ToolHandlerFn:
    """Create a Pydantic-validated MCP tool handler for a standard domain op verifier.

    Args:
        fn: Target domain operation verifier callable.

    Returns:
        Typed `ToolHandlerFn` accepting raw JSON-RPC arguments.
    """

    def _handler(a: dict[str, JsonValue]) -> ToolHandlerResult:
        """Validate request arguments via `DomainOpToolRequest` and invoke `fn`.

        Args:
            a: Raw JSON-RPC tool arguments dictionary.

        Returns:
            Domain op check result.
        """
        req = DomainOpToolRequest(
            op_name=str(a["op_name"]),
            inputs_count=_as_int_or_none(a.get("inputs_count")),
            attributes=_as_str_list(a.get("attributes")),
        )
        return fn(
            req.op_name,
            inputs_count=req.inputs_count,
            attributes=req.attributes,
        )

    return _handler


def _build_dispatch_table(repository: SnapshotRepository | None = None) -> dict[str, ToolHandlerFn]:
    """Build the dictionary mapping MCP tool names to handler callables without importing `mcp_server`.

    Args:
        repository: Optional injected `SnapshotRepository` instance.

    Returns:
        Dictionary mapping tool name string to handler function.
    """
    simple_domain_tools: dict[str, _SimpleDomainVerifierProto] = {
        "check_wgsl_op": check_wgsl_op,
        "check_webgl_op": check_webgl_op,
        "check_cpp_op": check_cpp_op,
        "check_array_api_op": check_array_api_op,
        "check_scipy_op": check_scipy_op,
        "check_torchvision_op": check_torchvision_op,
        "check_torchaudio_op": check_torchaudio_op,
        "check_safetensors_op": check_safetensors_op,
        "check_aten_op": check_aten_op,
        "check_nccl_op": check_nccl_op,
        "check_flash_attention_op": check_flash_attention_op,
        "check_chex_op": check_chex_op,
        "check_grain_op": check_grain_op,
        "check_orbax_op": check_orbax_op,
        "check_optax_op": check_optax_op,
    }

    table: dict[str, ToolHandlerFn] = {
        "get_api_signature": lambda a: get_api_signature(
            str(a["framework"]),
            str(a["api_path"]),
            version=_as_str_or_none(a.get("version")),
            repository=repository,
        ),
        "search_apis": lambda a: search_apis(
            str(a["framework"]),
            str(a["query"]),
            _as_int_or_none(a.get("limit")) or 10,
            version=_as_str_or_none(a.get("version")),
            repository=repository,
        ),
        "check_hallucination": lambda a: check_hallucination(
            str(a["framework"]),
            str(a["api_path"]),
            _as_str_list(a.get("kwargs")),
            args_count=_as_int_or_none(a.get("args_count")),
            strict_kwargs=bool(a.get("strict_kwargs", True)),
            kwarg_values=_as_dict_or_none(a.get("kwarg_values")),
            version=_as_str_or_none(a.get("version")),
            repository=repository,
            arg_dtypes=_as_str_dict_or_list(a.get("arg_dtypes")),
            kwarg_dtypes={str(k): str(v) for k, v in a["kwarg_dtypes"].items()}
            if isinstance(a.get("kwarg_dtypes"), dict) and isinstance(a["kwarg_dtypes"], dict)
            else None,
            arg_ranks=_as_rank_dict_or_list(a.get("arg_ranks")),
            kwarg_ranks={
                str(k): (v if isinstance(v, int) and not isinstance(v, bool) else str(v))
                for k, v in a["kwarg_ranks"].items()
            }
            if isinstance(a.get("kwarg_ranks"), dict) and isinstance(a["kwarg_ranks"], dict)
            else None,
            arg_shapes=_as_shape_list(a.get("arg_shapes")),
            kwarg_shapes=_as_shape_dict(a.get("kwarg_shapes")),
            strict_c_extensions=bool(a.get("strict_c_extensions", False)),
        ),
        "explain_anti_pattern": lambda a: explain_anti_pattern(
            str(a["framework"]),
            str(a["api_path"]),
            str(a["hallucinated_argument"]),
            passed_value=a.get("passed_value"),
            repository=repository,
        ),
        "check_sass_instruction": lambda a: check_sass_instruction(
            str(a["mnemonic"]),
            _as_str_list(a.get("operands")),
            _as_str_list(a.get("modifiers")),
            _as_str_or_none(a.get("sm_arch")),
            control_codes=_as_dict_or_none(a.get("control_codes")),
            repository=repository,
        ),
        "check_rdna_instruction": lambda a: check_rdna_instruction(
            str(a["mnemonic"]),
            _as_str_list(a.get("operands")),
            _as_str_or_none(a.get("encoding")),
            _as_str_or_none(a.get("gfx_arch")),
            modifiers=_as_str_list(a.get("modifiers")),
            wave_size=_as_int_or_none(a.get("wave_size")),
            repository=repository,
        ),
        "check_ptx_instruction": lambda a: check_ptx_instruction(
            str(a["mnemonic"]),
            types=_as_str_list(a.get("types")),
            operands=_as_str_list(a.get("operands")),
            state_space=_as_str_or_none(a.get("state_space")),
            scope=_as_str_or_none(a.get("scope")),
            vector_width=_as_str_or_none(a.get("vector_width")),
            sm_arch=_as_str_or_none(a.get("sm_arch")),
        ),
        "check_mlir_op": lambda a: check_mlir_op(
            str(a["op_name"]),
            _as_int_or_none(a.get("operands_count")),
            _as_str_list(a.get("attributes")),
            _as_str_list(a.get("operand_types")),
            result_types=_as_str_list(a.get("result_types")),
            structured_attributes=_as_dict_or_none(a.get("structured_attributes")),
            regions=_as_dict_or_none(a.get("regions")),
            repository=repository,
        ),
        "check_stablehlo_op": lambda a: check_stablehlo_op(
            str(a["op_name"]),
            _as_int_or_none(a.get("operands_count")),
            _as_str_list(a.get("attributes")),
            _as_str_list(a.get("operand_types")),
            result_types=_as_str_list(a.get("result_types")),
            structured_attributes=_as_dict_or_none(a.get("structured_attributes")),
            regions=_as_dict_or_none(a.get("regions")),
            repository=repository,
        ),
        "check_code_block": lambda a: check_code_block(
            str(a["code"]),
            framework=_as_str_or_none(a.get("framework")),
            sm_arch=_as_str_or_none(a.get("sm_arch")),
            gfx_arch=_as_str_or_none(a.get("gfx_arch")),
            repository=repository,
        ),
        "translate_concept_arguments": lambda a: translate_concept_arguments(
            str(a["concept"]),
            str(a["source_framework"]),
            str(a["target_framework"]),
            _as_dict_or_none(a.get("source_kwargs")) or {},
        ),
        "check_onnx_op": lambda a: check_onnx_op(
            str(a["op_name"]),
            domain=str(a.get("domain", "")),
            inputs_count=_as_int_or_none(a.get("inputs_count")),
            attributes=_as_str_list(a.get("attributes")),
        ),
        "check_metal_op": lambda a: check_metal_op(
            str(a["op_name"]),
            address_space=_as_str_or_none(a.get("address_space")),
            inputs_count=_as_int_or_none(a.get("inputs_count")),
            attributes=_as_str_list(a.get("attributes")),
        ),
        "check_wasm_instruction": lambda a: check_wasm_instruction(
            str(a["mnemonic"]),
            operands_count=_as_int_or_none(a.get("operands_count")),
            attributes=_as_str_list(a.get("attributes")),
        ),
    }

    for tool_name, fn in simple_domain_tools.items():
        table[tool_name] = _make_domain_op_handler(fn)

    return table


def dispatch_tool_call(
    msg_id: str | int | None,
    tool_name: str | None,
    args: dict[str, JsonValue],
    repository: SnapshotRepository | None = None,
) -> JsonRpcResponse:
    """Dispatch a `tools/call` MCP request via O(1) dictionary lookup.

    Args:
        msg_id: JSON-RPC request identifier.
        tool_name: Name of the requested tool.
        args: Arguments dictionary for the tool.
        repository: Optional injected `SnapshotRepository` instance.

    Returns:
        Validated `JsonRpcResponse` instance.
    """
    table = _build_dispatch_table(repository=repository)
    handler = table.get(tool_name or "")
    if handler is None:
        return JsonRpcResponse(
            id=msg_id,
            error=JsonRpcError(
                code=-32601,
                message=f"Unknown tool '{tool_name}'",
            ),
        )

    res = handler(args)
    payload = res.model_dump(exclude_none=True) if isinstance(res, pydantic.BaseModel) else res
    return JsonRpcResponse(
        id=msg_id,
        result=JsonRpcResult(content=[JsonRpcContentItem(type="text", text=json.dumps(payload, indent=2))]),
    )


def handle_mcp_message(
    message: typing.Mapping[str, JsonValue],
    repository: SnapshotRepository | None = None,
) -> JsonRpcResponse:
    """Handle an incoming JSON-RPC / MCP protocol message and generate a response.

    Args:
        message: Parsed JSON-RPC request dictionary.
        repository: Optional injected `SnapshotRepository` instance.

    Returns:
        Validated `JsonRpcResponse` instance.
    """
    raw_id = message.get("id")
    msg_id = raw_id if isinstance(raw_id, (str, int)) else None
    method = str(message.get("method") or "")
    raw_params = message.get("params")
    params = dict(raw_params) if isinstance(raw_params, dict) else {}

    if method == "tools/list":
        tools_list = [dict(t) for t in get_mcp_tools_list()]
        return JsonRpcResponse(
            id=msg_id,
            result=JsonRpcResult(tools=tools_list),
        )

    if method == "tools/call":
        raw_name = params.get("name")
        tool_name = str(raw_name) if raw_name is not None else None
        raw_args = params.get("arguments")
        args = dict(raw_args) if isinstance(raw_args, dict) else {}
        return dispatch_tool_call(msg_id, tool_name, args, repository=repository)

    return JsonRpcResponse(
        id=msg_id,
        error=JsonRpcError(
            code=-32601,
            message=f"Unknown JSON-RPC method '{method}'",
        ),
    )


__all__ = [
    "McpToolSpec",
    "ToolHandlerFn",
    "ToolHandlerResult",
    "dispatch_tool_call",
    "get_mcp_tools_list",
    "handle_mcp_message",
]
