"""Declarative MCP tool schema definitions and Pydantic input request models."""

from __future__ import annotations

import pydantic

from ml_ecosystem_snapshots.grounding.domain_ops import build_domain_op_tool_schemas
from ml_ecosystem_snapshots.models import JsonValue


class DomainOpToolRequest(pydantic.BaseModel):
    """Pydantic request model for standard domain operation MCP tools."""

    op_name: str = ""
    mnemonic: str = ""
    domain: str = ""
    address_space: str | None = None
    inputs_count: int | None = None
    operands_count: int | None = None
    attributes: list[str] | None = None


CORE_MCP_TOOL_SCHEMAS: list[dict[str, JsonValue]] = [
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


def get_mcp_tools_list() -> list[dict[str, JsonValue]]:
    """Return the MCP schema definition for available tools.

    Returns:
        List of MCP tool definitions with input schemas.
    """
    return list(CORE_MCP_TOOL_SCHEMAS) + build_domain_op_tool_schemas()


__all__ = [
    "CORE_MCP_TOOL_SCHEMAS",
    "DomainOpToolRequest",
    "get_mcp_tools_list",
]
