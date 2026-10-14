"""Deterministic MCP tool specification model and dictionary dispatch registry."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional, Union
from pydantic import BaseModel, ConfigDict
import ml_ecosystem_snapshots.mcp_server as _mcp


class McpToolSpec(BaseModel):
    """Typed specification for an MCP tool endpoint."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Optional[Callable[[Dict[str, Any]], Any]] = None


def _build_dispatch_table() -> Dict[str, Callable[[Dict[str, Any]], Any]]:
    """Construct the deterministic tool-name-to-handler dispatch map.

    Returns:
        Dictionary mapping MCP tool name strings to invocation lambdas.
    """
    return {
        "get_api_signature": lambda args: _mcp.get_api_signature(
            args.get("framework", ""),
            args.get("api_path", ""),
            version=args.get("version"),
        ),
        "search_apis": lambda args: _mcp.search_apis(
            args.get("framework", ""),
            args.get("query", ""),
            args.get("limit", 10),
            version=args.get("version"),
        ),
        "check_hallucination": lambda args: _mcp.check_hallucination(
            args.get("framework", ""),
            args.get("api_path", ""),
            args.get("kwargs", []),
            args_count=args.get("args_count"),
            strict_kwargs=args.get("strict_kwargs", True),
            kwarg_values=args.get("kwarg_values"),
            version=args.get("version"),
            arg_dtypes=args.get("arg_dtypes"),
            kwarg_dtypes=args.get("kwarg_dtypes"),
            arg_ranks=args.get("arg_ranks"),
            kwarg_ranks=args.get("kwarg_ranks"),
            arg_shapes=args.get("arg_shapes"),
            kwarg_shapes=args.get("kwarg_shapes"),
            strict_c_extensions=args.get("strict_c_extensions", False),
        ),
        "explain_anti_pattern": lambda args: _mcp.explain_anti_pattern(
            framework=args.get("framework", ""),
            api_path=args.get("api_path", ""),
            hallucinated_argument=args.get("hallucinated_argument", ""),
            passed_value=args.get("passed_value"),
        ),
        "check_sass_instruction": lambda args: _mcp.check_sass_instruction(
            mnemonic=args.get("mnemonic", ""),
            operands=args.get("operands"),
            modifiers=args.get("modifiers"),
            sm_arch=args.get("sm_arch"),
            control_codes=args.get("control_codes"),
        ),
        "check_rdna_instruction": lambda args: _mcp.check_rdna_instruction(
            mnemonic=args.get("mnemonic", ""),
            operands=args.get("operands"),
            encoding=args.get("encoding"),
            gfx_arch=args.get("gfx_arch"),
            modifiers=args.get("modifiers"),
            wave_size=args.get("wave_size"),
        ),
        "check_ptx_instruction": lambda args: _mcp.check_ptx_instruction(
            mnemonic=args.get("mnemonic", ""),
            types=args.get("types"),
            operands=args.get("operands"),
            state_space=args.get("state_space"),
            scope=args.get("scope"),
            vector_width=args.get("vector_width"),
            sm_arch=args.get("sm_arch"),
        ),
        "check_mlir_op": lambda args: _mcp.check_mlir_op(
            op_name=args.get("op_name", ""),
            operands_count=args.get("operands_count"),
            attributes=args.get("attributes"),
            operand_types=args.get("operand_types"),
            result_types=args.get("result_types"),
            structured_attributes=args.get("structured_attributes"),
            regions=args.get("regions"),
        ),
        "check_stablehlo_op": lambda args: _mcp.check_stablehlo_op(
            op_name=args.get("op_name", ""),
            operands_count=args.get("operands_count"),
            attributes=args.get("attributes"),
            operand_types=args.get("operand_types"),
            result_types=args.get("result_types"),
            structured_attributes=args.get("structured_attributes"),
            regions=args.get("regions"),
        ),
        "check_code_block": lambda args: _mcp.check_code_block(
            code=args.get("code", ""),
            framework=args.get("framework"),
            version=args.get("version"),
            sm_arch=args.get("sm_arch"),
            gfx_arch=args.get("gfx_arch"),
        ),
        "translate_concept_arguments": lambda args: _mcp.translate_concept_arguments(
            concept=args.get("concept", ""),
            source_framework=args.get("source_framework", ""),
            target_framework=args.get("target_framework", ""),
            source_kwargs=args.get("source_kwargs", {}),
        ),
        "check_wgsl_op": lambda args: _mcp.check_wgsl_op(
            op_name=args.get("op_name") or args.get("api_path", ""),
            inputs_count=args.get("inputs_count") or args.get("operands_count"),
            attributes=args.get("attributes"),
        ),
        "check_onnx_op": lambda args: _mcp.check_onnx_op(
            op_name=args.get("op_name") or args.get("api_path", ""),
            domain=args.get("domain", ""),
            inputs_count=args.get("inputs_count"),
            attributes=args.get("attributes"),
        ),
        "check_metal_op": lambda args: _mcp.check_metal_op(
            op_name=args.get("op_name") or args.get("api_path", ""),
            address_space=args.get("address_space"),
            inputs_count=args.get("inputs_count"),
            attributes=args.get("attributes"),
        ),
        "check_wasm_instruction": lambda args: _mcp.check_wasm_instruction(
            mnemonic=args.get("mnemonic") or args.get("op_name", ""),
            operands_count=args.get("operands_count") or args.get("inputs_count"),
            attributes=args.get("attributes"),
        ),
        "check_webgl_op": lambda args: _mcp.check_webgl_op(
            op_name=args.get("op_name") or args.get("api_path", ""),
            inputs_count=args.get("inputs_count"),
            attributes=args.get("attributes"),
        ),
        "check_cpp_op": lambda args: _mcp.check_cpp_op(
            op_name=args.get("op_name") or args.get("api_path", ""),
            inputs_count=args.get("inputs_count"),
            attributes=args.get("attributes"),
        ),
        "check_array_api_op": lambda args: _mcp.check_array_api_op(
            op_name=args.get("op_name") or args.get("api_path", ""),
            inputs_count=args.get("inputs_count") or args.get("args_count"),
            attributes=args.get("attributes") or args.get("kwargs"),
        ),
        "check_scipy_op": lambda args: _mcp.check_scipy_op(
            op_name=args.get("op_name") or args.get("api_path", ""),
            inputs_count=args.get("inputs_count") or args.get("args_count"),
            attributes=args.get("attributes") or args.get("kwargs"),
        ),
        "check_torchvision_op": lambda args: _mcp.check_torchvision_op(
            op_name=args.get("op_name") or args.get("api_path", ""),
            inputs_count=args.get("inputs_count") or args.get("args_count"),
            attributes=args.get("attributes") or args.get("kwargs"),
        ),
        "check_torchaudio_op": lambda args: _mcp.check_torchaudio_op(
            op_name=args.get("op_name") or args.get("api_path", ""),
            inputs_count=args.get("inputs_count") or args.get("args_count"),
            attributes=args.get("attributes") or args.get("kwargs"),
        ),
        "check_safetensors_op": lambda args: _mcp.check_safetensors_op(
            op_name=args.get("op_name") or args.get("api_path", ""),
            inputs_count=args.get("inputs_count") or args.get("args_count"),
            attributes=args.get("attributes") or args.get("kwargs"),
        ),
        "check_aten_op": lambda args: _mcp.check_aten_op(
            op_name=args.get("op_name") or args.get("api_path", ""),
            inputs_count=args.get("inputs_count"),
            attributes=args.get("attributes"),
        ),
        "check_nccl_op": lambda args: _mcp.check_nccl_op(
            op_name=args.get("op_name") or args.get("api_path", ""),
            inputs_count=args.get("inputs_count"),
            attributes=args.get("attributes"),
        ),
        "check_flash_attention_op": lambda args: _mcp.check_flash_attention_op(
            op_name=args.get("op_name") or args.get("api_path", ""),
            inputs_count=args.get("inputs_count") or args.get("args_count"),
            attributes=args.get("attributes") or args.get("kwargs"),
        ),
    }


def dispatch_tool_call(
    msg_id: Optional[Union[int, str]],
    tool_name: Optional[str],
    args: Dict[str, Any],
) -> Dict[str, Any]:
    """Dispatch an MCP `tools/call` invocation using the dictionary dispatch table.

    Args:
        msg_id: JSON-RPC message identifier.
        tool_name: Target MCP tool name.
        args: Tool arguments dictionary.

    Returns:
        JSON-RPC 2.0 response dictionary.
    """
    table = _build_dispatch_table()
    handler = table.get(tool_name or "")
    if handler is None:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": -32601,
                "message": f"Method/tool '{tool_name}' not found",
            },
        }
    res = handler(args)
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]},
    }


__all__ = ["McpToolSpec", "dispatch_tool_call"]
