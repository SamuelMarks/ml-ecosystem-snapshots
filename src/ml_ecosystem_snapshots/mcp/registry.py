"""Deterministic MCP tool specification model and dictionary dispatch registry."""

from __future__ import annotations

import json
import typing
import typing_extensions

import pydantic

from ml_ecosystem_snapshots.mcp.tools.domain_ops import build_domain_op_tool_schemas
from ml_ecosystem_snapshots.mcp.protocol import JsonRpcResponseDict, JsonRpcResultDict
from ml_ecosystem_snapshots.models import JsonValue

ToolHandlerResult: typing_extensions.TypeAlias = (
    "pydantic.BaseModel | dict[str, JsonValue] | list[str] | None"
)
ToolHandlerFn: typing_extensions.TypeAlias = (
    "typing.Callable[[dict[str, JsonValue]], ToolHandlerResult]"
)


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


def get_mcp_tools_list() -> list[dict[str, JsonValue]]:
    """Return the MCP schema definition for available tools.

    Returns:
        List of MCP tool definitions with input schemas.
    """
    core_tools: list[dict[str, JsonValue]] = [
        {
            "name": "get_api_signature",
            "description": "Look up the exact ground-truth API signature and parameters for an ML framework function.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "framework": {
                        "type": "string",
                        "description": "Target framework name (e.g. torch, jax, nvidia_sass, stablehlo).",
                    },
                    "api_path": {
                        "type": "string",
                        "description": "Canonical API path or mnemonic.",
                    },
                    "version": {
                        "type": "string",
                        "description": "Optional framework version string (e.g. '2.4.0').",
                    },
                },
                "required": ["framework", "api_path"],
            },
        },
        {
            "name": "search_apis",
            "description": "Search available operations and functions in a framework snapshot by keyword.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "framework": {
                        "type": "string",
                        "description": "Target framework name.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Search keyword or mnemonic prefix.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum matches to return.",
                    },
                    "version": {
                        "type": "string",
                        "description": "Optional framework version string (e.g. '2.4.0').",
                    },
                },
                "required": ["framework", "query"],
            },
        },
        {
            "name": "check_hallucination",
            "description": "Verify whether an API path, positional arguments, or keyword arguments represent LLM hallucinations.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "framework": {
                        "type": "string",
                        "description": "Target framework name.",
                    },
                    "api_path": {
                        "type": "string",
                        "description": "API path to check.",
                    },
                    "kwargs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of keyword arguments to validate.",
                    },
                    "args_count": {
                        "type": "integer",
                        "description": "Optional count of positional arguments passed.",
                    },
                    "strict_kwargs": {
                        "type": "boolean",
                        "description": "Whether to treat unconstrained kwargs as hallucinated (default: true).",
                    },
                    "version": {
                        "type": "string",
                        "description": "Optional framework version string (e.g. '2.4.0').",
                    },
                },
                "required": ["framework", "api_path"],
            },
        },
        {
            "name": "explain_anti_pattern",
            "description": "Provide canonical migration advice and rationale for a flagged anti-pattern or hallucinated argument.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "framework": {
                        "type": "string",
                        "description": "Target framework name (e.g. 'torch', 'jax', 'tensorflow').",
                    },
                    "api_path": {
                        "type": "string",
                        "description": "API path to explain (e.g. 'torch.sum').",
                    },
                    "hallucinated_argument": {
                        "type": "string",
                        "description": "The flagged or hallucinated argument name (e.g. 'axis').",
                    },
                    "passed_value": {
                        "description": "Optional value passed to the argument.",
                    },
                },
                "required": ["framework", "api_path", "hallucinated_argument"],
            },
        },
        {
            "name": "check_sass_instruction",
            "description": "Validate whether an NVIDIA SASS assembly instruction is valid on target SM architecture.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "mnemonic": {
                        "type": "string",
                        "description": "SASS mnemonic (e.g. FADD, WGMMA).",
                    },
                    "operands": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of operand types (e.g. ['R', 'R', 'R']).",
                    },
                    "modifiers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of modifiers (e.g. ['.SAT', '.FTZ']).",
                    },
                    "sm_arch": {
                        "type": "string",
                        "description": "Target SM architecture (e.g. 'sm_80', 'sm_90').",
                    },
                    "control_codes": {
                        "type": "object",
                        "description": "Optional control code dictionary (e.g. {'stall_count': 1, 'latency_ticks': 4}).",
                    },
                },
                "required": ["mnemonic"],
            },
        },
        {
            "name": "check_rdna_instruction",
            "description": "Validate whether an AMD RDNA instruction is valid on target GFX architecture.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "mnemonic": {
                        "type": "string",
                        "description": "RDNA mnemonic (e.g. v_add_f32).",
                    },
                    "operands": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of operand types.",
                    },
                    "encoding": {
                        "type": "string",
                        "description": "Optional instruction encoding format (e.g. VOP2).",
                    },
                    "gfx_arch": {
                        "type": "string",
                        "description": "Target GFX architecture (e.g. 'GFX11/RDNA3').",
                    },
                    "modifiers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of modifiers (e.g. ['-src', 'clamp', 'omod:2']).",
                    },
                    "wave_size": {
                        "type": "integer",
                        "description": "Optional wavefront execution size (32 or 64).",
                    },
                },
                "required": ["mnemonic"],
            },
        },
        {
            "name": "check_ptx_instruction",
            "description": "Validate whether an NVIDIA PTX assembly instruction is valid on target SM architecture.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "mnemonic": {
                        "type": "string",
                        "description": "PTX instruction mnemonic (e.g. add, ld, st, wgmma.mma_async).",
                    },
                    "types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of PTX type qualifiers (e.g. ['.f32'], ['.u64']).",
                    },
                    "operands": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of operand registers or immediates.",
                    },
                    "state_space": {
                        "type": "string",
                        "description": "Optional memory state space qualifier (e.g. '.global', '.shared').",
                    },
                    "scope": {
                        "type": "string",
                        "description": "Optional memory scope qualifier (e.g. '.gpu', '.cta', '.sys').",
                    },
                    "vector_width": {
                        "type": "string",
                        "description": "Optional vector width qualifier (e.g. '.v2', '.v4').",
                    },
                    "sm_arch": {
                        "type": "string",
                        "description": "Target SM architecture (e.g. 'sm_80', 'sm_90').",
                    },
                },
                "required": ["mnemonic"],
            },
        },
        {
            "name": "check_mlir_op",
            "description": "Validate whether an MLIR or StableHLO operation exists with expected signature.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "op_name": {
                        "type": "string",
                        "description": "Qualified dialect op name (e.g. arith.addf, stablehlo.dot_general).",
                    },
                    "operands_count": {
                        "type": "integer",
                        "description": "Optional expected number of SSA operands.",
                    },
                    "attributes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of attribute names to verify.",
                    },
                    "operand_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of SSA operand types to verify against ODS type constraints.",
                    },
                    "result_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of SSA result types to verify.",
                    },
                    "structured_attributes": {
                        "type": "object",
                        "description": "Optional structured attributes (e.g. DotDimensionNumbersAttr, ComparisonDirectionAttr) to validate.",
                    },
                    "regions": {
                        "type": "object",
                        "description": "Optional dictionary of regions mapping name to block_arguments and yield_types.",
                    },
                },
                "required": ["op_name"],
            },
        },
        {
            "name": "check_stablehlo_op",
            "description": "Validate whether a StableHLO operation complies with formal verification rules, dimension numbers, and region constraints.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "op_name": {
                        "type": "string",
                        "description": "StableHLO op name (e.g. stablehlo.dot_general, stablehlo.reduce).",
                    },
                    "operands_count": {
                        "type": "integer",
                        "description": "Optional expected number of SSA operands.",
                    },
                    "attributes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of attribute names to verify.",
                    },
                    "operand_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of SSA operand types.",
                    },
                    "result_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of SSA result types.",
                    },
                    "structured_attributes": {
                        "type": "object",
                        "description": "Optional structured attributes (DotDimensionNumbersAttr, ConvDimensionNumbersAttr).",
                    },
                    "regions": {
                        "type": "object",
                        "description": "Optional dictionary of regions mapping name to block_arguments and yield_types.",
                    },
                },
                "required": ["op_name"],
            },
        },
        {
            "name": "check_code_block",
            "description": "Batch verify a code block for nonexistent APIs, invalid kwargs, and hardware constraints in a single round-trip.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python snippet, SASS assembly, RDNA assembly, or MLIR text.",
                    },
                    "framework": {
                        "type": "string",
                        "description": "Optional framework hint ('torch', 'jax', 'nvidia_sass', 'amd_rdna', 'mlir').",
                    },
                    "version": {
                        "type": "string",
                        "description": "Optional framework version string.",
                    },
                    "sm_arch": {
                        "type": "string",
                        "description": "Optional NVIDIA architecture target (e.g. 'sm_80', 'sm_90').",
                    },
                    "gfx_arch": {
                        "type": "string",
                        "description": "Optional AMD RDNA architecture target (e.g. 'GFX11/RDNA3').",
                    },
                },
                "required": ["code"],
            },
        },
        {
            "name": "translate_concept_arguments",
            "description": "Translate operation arguments across frameworks using concept parameter mapping schemas.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "concept": {
                        "type": "string",
                        "description": "The canonical concept name (e.g. 'matmul', 'reduce_sum', 'softmax', 'convolution').",
                    },
                    "source_framework": {
                        "type": "string",
                        "description": "Source framework identifier ('torch', 'jax', 'numpy', 'tensorflow', 'stablehlo').",
                    },
                    "target_framework": {
                        "type": "string",
                        "description": "Target framework identifier ('torch', 'jax', 'numpy', 'tensorflow', 'stablehlo').",
                    },
                    "source_kwargs": {
                        "type": "object",
                        "description": "Dictionary of source keyword arguments or parameters.",
                    },
                },
                "required": [
                    "concept",
                    "source_framework",
                    "target_framework",
                    "source_kwargs",
                ],
            },
        },
    ]
    return core_tools + build_domain_op_tool_schemas()


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


def _build_dispatch_table() -> dict[str, ToolHandlerFn]:
    """Build the dictionary mapping MCP tool names to handler callables.

    Returns:
        Dictionary mapping tool name string to handler function.
    """
    import ml_ecosystem_snapshots.mcp_server as ms

    return {
        "get_api_signature": lambda a: ms.get_api_signature(
            str(a["framework"]),
            str(a["api_path"]),
            version=_as_str_or_none(a.get("version")),
        ),
        "search_apis": lambda a: ms.search_apis(
            str(a["framework"]),
            str(a["query"]),
            _as_int_or_none(a.get("limit")) or 10,
            version=_as_str_or_none(a.get("version")),
        ),
        "check_hallucination": lambda a: ms.check_hallucination(
            str(a["framework"]),
            str(a["api_path"]),
            _as_str_list(a.get("kwargs")),
            args_count=_as_int_or_none(a.get("args_count")),
            strict_kwargs=bool(a.get("strict_kwargs", True)),
            kwarg_values=_as_dict_or_none(a.get("kwarg_values")),
            version=_as_str_or_none(a.get("version")),
            arg_dtypes=a.get("arg_dtypes"),
            kwarg_dtypes=a.get("kwarg_dtypes"),
            arg_ranks=a.get("arg_ranks"),
            kwarg_ranks=a.get("kwarg_ranks"),
            arg_shapes=a.get("arg_shapes"),
            kwarg_shapes=a.get("kwarg_shapes"),
            strict_c_extensions=bool(a.get("strict_c_extensions", False)),
        ),
        "explain_anti_pattern": lambda a: ms.explain_anti_pattern(
            str(a["framework"]),
            str(a["api_path"]),
            str(a["hallucinated_argument"]),
            passed_value=a.get("passed_value"),
        ),
        "check_sass_instruction": lambda a: ms.check_sass_instruction(
            str(a["mnemonic"]),
            _as_str_list(a.get("operands")),
            _as_str_list(a.get("modifiers")),
            _as_str_or_none(a.get("sm_arch")),
            control_codes=_as_dict_or_none(a.get("control_codes")),
        ),
        "check_rdna_instruction": lambda a: ms.check_rdna_instruction(
            str(a["mnemonic"]),
            _as_str_list(a.get("operands")),
            _as_str_or_none(a.get("encoding")),
            _as_str_or_none(a.get("gfx_arch")),
            modifiers=_as_str_list(a.get("modifiers")),
            wave_size=_as_int_or_none(a.get("wave_size")),
        ),
        "check_ptx_instruction": lambda a: ms.check_ptx_instruction(
            str(a["mnemonic"]),
            types=_as_str_list(a.get("types")),
            operands=_as_str_list(a.get("operands")),
            state_space=_as_str_or_none(a.get("state_space")),
            scope=_as_str_or_none(a.get("scope")),
            vector_width=_as_str_or_none(a.get("vector_width")),
            sm_arch=_as_str_or_none(a.get("sm_arch")),
        ),
        "check_mlir_op": lambda a: ms.check_mlir_op(
            str(a["op_name"]),
            _as_int_or_none(a.get("operands_count")),
            _as_str_list(a.get("attributes")),
            _as_str_list(a.get("operand_types")),
            result_types=_as_str_list(a.get("result_types")),
            structured_attributes=_as_dict_or_none(a.get("structured_attributes")),
            regions=_as_dict_or_none(a.get("regions")),
        ),
        "check_stablehlo_op": lambda a: ms.check_stablehlo_op(
            str(a["op_name"]),
            _as_int_or_none(a.get("operands_count")),
            _as_str_list(a.get("attributes")),
            _as_str_list(a.get("operand_types")),
            result_types=_as_str_list(a.get("result_types")),
            structured_attributes=_as_dict_or_none(a.get("structured_attributes")),
            regions=_as_dict_or_none(a.get("regions")),
        ),
        "check_code_block": lambda a: ms.check_code_block(
            str(a["code"]),
            framework=_as_str_or_none(a.get("framework")),
            version=_as_str_or_none(a.get("version")),
            sm_arch=_as_str_or_none(a.get("sm_arch")),
            gfx_arch=_as_str_or_none(a.get("gfx_arch")),
        ),
        "translate_concept_arguments": lambda a: ms.translate_concept_arguments(
            str(a["concept"]),
            str(a["source_framework"]),
            str(a["target_framework"]),
            _as_dict_or_none(a.get("source_kwargs")) or {},
        ),
        "check_wgsl_op": lambda a: ms.check_wgsl_op(
            str(a["op_name"]),
            inputs_count=_as_int_or_none(a.get("inputs_count")),
            attributes=_as_str_list(a.get("attributes")),
        ),
        "check_onnx_op": lambda a: ms.check_onnx_op(
            str(a["op_name"]),
            domain=str(a.get("domain", "")),
            inputs_count=_as_int_or_none(a.get("inputs_count")),
            attributes=_as_str_list(a.get("attributes")),
        ),
        "check_metal_op": lambda a: ms.check_metal_op(
            str(a["op_name"]),
            address_space=_as_str_or_none(a.get("address_space")),
            inputs_count=_as_int_or_none(a.get("inputs_count")),
            attributes=_as_str_list(a.get("attributes")),
        ),
        "check_wasm_instruction": lambda a: ms.check_wasm_instruction(
            str(a["mnemonic"]),
            operands_count=_as_int_or_none(a.get("operands_count")),
            attributes=_as_str_list(a.get("attributes")),
        ),
        "check_webgl_op": lambda a: ms.check_webgl_op(
            str(a["op_name"]),
            inputs_count=_as_int_or_none(a.get("inputs_count")),
            attributes=_as_str_list(a.get("attributes")),
        ),
        "check_cpp_op": lambda a: ms.check_cpp_op(
            str(a["op_name"]),
            inputs_count=_as_int_or_none(a.get("inputs_count")),
            attributes=_as_str_list(a.get("attributes")),
        ),
        "check_array_api_op": lambda a: ms.check_array_api_op(
            str(a["op_name"]),
            inputs_count=_as_int_or_none(a.get("inputs_count")),
            attributes=_as_str_list(a.get("attributes")),
        ),
        "check_scipy_op": lambda a: ms.check_scipy_op(
            str(a["op_name"]),
            inputs_count=_as_int_or_none(a.get("inputs_count")),
            attributes=_as_str_list(a.get("attributes")),
        ),
        "check_torchvision_op": lambda a: ms.check_torchvision_op(
            str(a["op_name"]),
            inputs_count=_as_int_or_none(a.get("inputs_count")),
            attributes=_as_str_list(a.get("attributes")),
        ),
        "check_torchaudio_op": lambda a: ms.check_torchaudio_op(
            str(a["op_name"]),
            inputs_count=_as_int_or_none(a.get("inputs_count")),
            attributes=_as_str_list(a.get("attributes")),
        ),
        "check_safetensors_op": lambda a: ms.check_safetensors_op(
            str(a["op_name"]),
            inputs_count=_as_int_or_none(a.get("inputs_count")),
            attributes=_as_str_list(a.get("attributes")),
        ),
        "check_aten_op": lambda a: ms.check_aten_op(
            str(a["op_name"]),
            inputs_count=_as_int_or_none(a.get("inputs_count")),
            attributes=_as_str_list(a.get("attributes")),
        ),
        "check_nccl_op": lambda a: ms.check_nccl_op(
            str(a["op_name"]),
            inputs_count=_as_int_or_none(a.get("inputs_count")),
            attributes=_as_str_list(a.get("attributes")),
        ),
        "check_flash_attention_op": lambda a: ms.check_flash_attention_op(
            str(a["op_name"]),
            inputs_count=_as_int_or_none(a.get("inputs_count")),
            attributes=_as_str_list(a.get("attributes")),
        ),
        "check_chex_op": lambda a: ms.check_chex_op(
            str(a["op_name"]),
            inputs_count=_as_int_or_none(a.get("inputs_count")),
            attributes=_as_str_list(a.get("attributes")),
        ),
        "check_grain_op": lambda a: ms.check_grain_op(
            str(a["op_name"]),
            inputs_count=_as_int_or_none(a.get("inputs_count")),
            attributes=_as_str_list(a.get("attributes")),
        ),
        "check_orbax_op": lambda a: ms.check_orbax_op(
            str(a["op_name"]),
            inputs_count=_as_int_or_none(a.get("inputs_count")),
            attributes=_as_str_list(a.get("attributes")),
        ),
        "check_optax_op": lambda a: ms.check_optax_op(
            str(a["op_name"]),
            inputs_count=_as_int_or_none(a.get("inputs_count")),
            attributes=_as_str_list(a.get("attributes")),
        ),
    }


def dispatch_tool_call(
    msg_id: str | int | None,
    tool_name: str | None,
    args: dict[str, JsonValue],
) -> JsonRpcResponseDict:
    """Dispatch a `tools/call` MCP request via O(1) dictionary lookup.

    Args:
        msg_id: JSON-RPC request identifier.
        tool_name: Name of the requested tool.
        args: Arguments dictionary for the tool.

    Returns:
        JSON-RPC 2.0 response dictionary.
    """
    table = _build_dispatch_table()
    handler = table.get(tool_name or "")
    if handler is None:
        return JsonRpcResponseDict(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32601,
                    "message": f"Unknown tool '{tool_name}'",
                },
            }
        )

    res = handler(args)
    payload = (
        res.model_dump(exclude_none=True)
        if isinstance(res, pydantic.BaseModel)
        else res
    )
    return JsonRpcResponseDict(
        {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": JsonRpcResultDict(
                {"content": [{"type": "text", "text": json.dumps(payload, indent=2)}]}
            ),
        }
    )


def handle_mcp_message(
    message: typing.Mapping[str, JsonValue],
) -> JsonRpcResponseDict:
    """Handle an incoming JSON-RPC / MCP protocol message and generate a response.

    Args:
        message: Parsed JSON-RPC request dictionary.

    Returns:
        JSON-RPC response dictionary.
    """
    raw_id = message.get("id")
    msg_id = raw_id if isinstance(raw_id, (str, int)) else None
    method = str(message.get("method") or "")
    raw_params = message.get("params")
    params = dict(raw_params) if isinstance(raw_params, dict) else {}

    if method == "tools/list":
        tools_list: list[JsonValue] = list(get_mcp_tools_list())
        return JsonRpcResponseDict(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": JsonRpcResultDict({"tools": tools_list}),
            }
        )

    if method == "tools/call":
        raw_name = params.get("name")
        tool_name = str(raw_name) if raw_name is not None else None
        raw_args = params.get("arguments")
        args = dict(raw_args) if isinstance(raw_args, dict) else {}
        return dispatch_tool_call(msg_id, tool_name, args)

    return JsonRpcResponseDict(
        {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": -32601,
                "message": f"Unknown JSON-RPC method '{method}'",
            },
        }
    )


__all__ = [
    "McpToolSpec",
    "ToolHandlerFn",
    "ToolHandlerResult",
    "dispatch_tool_call",
    "get_mcp_tools_list",
    "handle_mcp_message",
]
