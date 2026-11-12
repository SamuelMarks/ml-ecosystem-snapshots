"""NVPTX TableGen parser and core arithmetic/comparison PTX specification records."""

from __future__ import annotations

from ml_ecosystem_snapshots.models import IsaInstructionRecordModel, JsonValue
from ml_ecosystem_snapshots.tools.ptx.lexer import tokenize_nvptx_td_defs

PTX_ARITH_OPS: list[dict[str, JsonValue]] = [
    # Arithmetic & Transcendental
    {
        "mnemonic": "add",
        "description": "Integer or floating-point addition.",
        "category": "arithmetic",
        "supported_types": [
            ".u16",
            ".u32",
            ".u64",
            ".s16",
            ".s32",
            ".s64",
            ".f16",
            ".f16x2",
            ".f32",
            ".f64",
        ],
        "vector_widths": [".v2", ".v4"],
        "state_spaces": [],
        "operands": [
            {"name": "d", "role": "dst", "description": "Destination operand"},
            {"name": "a", "role": "src0", "description": "First operand"},
            {"name": "b", "role": "src1", "description": "Second operand"},
        ],
        "min_sm": "sm_50",
    },
    {
        "mnemonic": "sub",
        "description": "Integer or floating-point subtraction.",
        "category": "arithmetic",
        "supported_types": [
            ".u16",
            ".u32",
            ".u64",
            ".s16",
            ".s32",
            ".s64",
            ".f16",
            ".f16x2",
            ".f32",
            ".f64",
        ],
        "vector_widths": [".v2", ".v4"],
        "state_spaces": [],
        "operands": [
            {"name": "d", "role": "dst", "description": "Destination operand"},
            {"name": "a", "role": "src0", "description": "First operand"},
            {"name": "b", "role": "src1", "description": "Second operand"},
        ],
        "min_sm": "sm_50",
    },
    {
        "mnemonic": "mul",
        "description": "Integer or floating-point multiplication.",
        "category": "arithmetic",
        "supported_types": [
            ".u16",
            ".u32",
            ".u64",
            ".s16",
            ".s32",
            ".s64",
            ".f16",
            ".f16x2",
            ".f32",
            ".f64",
        ],
        "vector_widths": [".v2", ".v4"],
        "state_spaces": [],
        "operands": [
            {"name": "d", "role": "dst", "description": "Destination operand"},
            {"name": "a", "role": "src0", "description": "First operand"},
            {"name": "b", "role": "src1", "description": "Second operand"},
        ],
        "min_sm": "sm_50",
    },
    {
        "mnemonic": "fma",
        "description": "Fused multiply-add: d = (a * b) + c with single rounding step.",
        "category": "arithmetic",
        "supported_types": [".f16", ".f16x2", ".bf16", ".bf16x2", ".f32", ".f64"],
        "vector_widths": [".v2"],
        "state_spaces": [],
        "operands": [
            {"name": "d", "role": "dst", "description": "Destination operand"},
            {"name": "a", "role": "src0", "description": "First multiplicand"},
            {"name": "b", "role": "src1", "description": "Second multiplicand"},
            {"name": "c", "role": "src2", "description": "Addend operand"},
        ],
        "min_sm": "sm_50",
    },
    {
        "mnemonic": "mad",
        "description": "Multiply and add: d = (a * b) + c.",
        "category": "arithmetic",
        "supported_types": [
            ".u16",
            ".u32",
            ".u64",
            ".s16",
            ".s32",
            ".s64",
            ".f32",
            ".f64",
        ],
        "vector_widths": [],
        "state_spaces": [],
        "operands": [
            {"name": "d", "role": "dst", "description": "Destination operand"},
            {"name": "a", "role": "src0", "description": "First multiplicand"},
            {"name": "b", "role": "src1", "description": "Second multiplicand"},
            {"name": "c", "role": "src2", "description": "Addend operand"},
        ],
        "min_sm": "sm_50",
    },
    {
        "mnemonic": "div",
        "description": "Divide: d = a / b.",
        "category": "arithmetic",
        "supported_types": [
            ".u16",
            ".u32",
            ".u64",
            ".s16",
            ".s32",
            ".s64",
            ".f16",
            ".f32",
            ".f64",
        ],
        "vector_widths": [],
        "state_spaces": [],
        "operands": [
            {"name": "d", "role": "dst", "description": "Destination operand"},
            {"name": "a", "role": "src0", "description": "Numerator"},
            {"name": "b", "role": "src1", "description": "Denominator"},
        ],
        "min_sm": "sm_50",
    },
    {
        "mnemonic": "min",
        "description": "Minimum of two values.",
        "category": "arithmetic",
        "supported_types": [
            ".u16",
            ".u32",
            ".u64",
            ".s16",
            ".s32",
            ".s64",
            ".f16",
            ".f16x2",
            ".f32",
            ".f64",
        ],
        "vector_widths": [".v2"],
        "state_spaces": [],
        "operands": [
            {"name": "d", "role": "dst", "description": "Destination operand"},
            {"name": "a", "role": "src0", "description": "First operand"},
            {"name": "b", "role": "src1", "description": "Second operand"},
        ],
        "min_sm": "sm_50",
    },
    {
        "mnemonic": "max",
        "description": "Maximum of two values.",
        "category": "arithmetic",
        "supported_types": [
            ".u16",
            ".u32",
            ".u64",
            ".s16",
            ".s32",
            ".s64",
            ".f16",
            ".f16x2",
            ".f32",
            ".f64",
        ],
        "vector_widths": [".v2"],
        "state_spaces": [],
        "operands": [
            {"name": "d", "role": "dst", "description": "Destination operand"},
            {"name": "a", "role": "src0", "description": "First operand"},
            {"name": "b", "role": "src1", "description": "Second operand"},
        ],
        "min_sm": "sm_50",
    },
    {
        "mnemonic": "tanh",
        "description": "Hyperbolic tangent: d = tanh(a).",
        "category": "arithmetic",
        "supported_types": [".f16", ".f16x2", ".f32"],
        "vector_widths": [".v2"],
        "state_spaces": [],
        "operands": [
            {"name": "d", "role": "dst", "description": "Destination operand"},
            {"name": "a", "role": "src0", "description": "Source operand"},
        ],
        "min_sm": "sm_75",
    },
    {
        "mnemonic": "ex2",
        "description": "Base-2 exponential approximation: d = 2^a.",
        "category": "arithmetic",
        "supported_types": [".f16", ".f16x2", ".f32"],
        "vector_widths": [".v2"],
        "state_spaces": [],
        "operands": [
            {"name": "d", "role": "dst", "description": "Destination operand"},
            {"name": "a", "role": "src0", "description": "Source exponent"},
        ],
        "min_sm": "sm_50",
    },
    # Logical & Bitwise
    {
        "mnemonic": "and",
        "description": "Bitwise AND operation.",
        "category": "logical",
        "supported_types": [".b16", ".b32", ".b64", ".pred"],
        "vector_widths": [],
        "state_spaces": [],
        "operands": [
            {"name": "d", "role": "dst", "description": "Destination operand"},
            {"name": "a", "role": "src0", "description": "First operand"},
            {"name": "b", "role": "src1", "description": "Second operand"},
        ],
        "min_sm": "sm_50",
    },
    {
        "mnemonic": "or",
        "description": "Bitwise OR operation.",
        "category": "logical",
        "supported_types": [".b16", ".b32", ".b64", ".pred"],
        "vector_widths": [],
        "state_spaces": [],
        "operands": [
            {"name": "d", "role": "dst", "description": "Destination operand"},
            {"name": "a", "role": "src0", "description": "First operand"},
            {"name": "b", "role": "src1", "description": "Second operand"},
        ],
        "min_sm": "sm_50",
    },
    {
        "mnemonic": "xor",
        "description": "Bitwise XOR operation.",
        "category": "logical",
        "supported_types": [".b16", ".b32", ".b64", ".pred"],
        "vector_widths": [],
        "state_spaces": [],
        "operands": [
            {"name": "d", "role": "dst", "description": "Destination operand"},
            {"name": "a", "role": "src0", "description": "First operand"},
            {"name": "b", "role": "src1", "description": "Second operand"},
        ],
        "min_sm": "sm_50",
    },
    {
        "mnemonic": "not",
        "description": "Bitwise NOT operation.",
        "category": "logical",
        "supported_types": [".b16", ".b32", ".b64", ".pred"],
        "vector_widths": [],
        "state_spaces": [],
        "operands": [
            {"name": "d", "role": "dst", "description": "Destination operand"},
            {"name": "a", "role": "src0", "description": "Source operand"},
        ],
        "min_sm": "sm_50",
    },
    {
        "mnemonic": "shl",
        "description": "Shift bits left.",
        "category": "logical",
        "supported_types": [".b16", ".b32", ".b64"],
        "vector_widths": [],
        "state_spaces": [],
        "operands": [
            {"name": "d", "role": "dst", "description": "Destination operand"},
            {"name": "a", "role": "src0", "description": "Value to shift"},
            {"name": "b", "role": "src1", "description": "Shift amount"},
        ],
        "min_sm": "sm_50",
    },
    {
        "mnemonic": "shr",
        "description": "Shift bits right (logical or arithmetic).",
        "category": "logical",
        "supported_types": [
            ".b16",
            ".b32",
            ".b64",
            ".u16",
            ".u32",
            ".u64",
            ".s16",
            ".s32",
            ".s64",
        ],
        "vector_widths": [],
        "state_spaces": [],
        "operands": [
            {"name": "d", "role": "dst", "description": "Destination operand"},
            {"name": "a", "role": "src0", "description": "Value to shift"},
            {"name": "b", "role": "src1", "description": "Shift amount"},
        ],
        "min_sm": "sm_50",
    },
    # Comparison & Selection
    {
        "mnemonic": "setp",
        "description": "Set predicate based on operand comparison.",
        "category": "comparison",
        "supported_types": [
            ".u16",
            ".u32",
            ".u64",
            ".s16",
            ".s32",
            ".s64",
            ".f16",
            ".f32",
            ".f64",
        ],
        "vector_widths": [],
        "state_spaces": [],
        "operands": [
            {"name": "p", "role": "dst", "description": "Destination predicate"},
            {"name": "a", "role": "src0", "description": "First operand"},
            {"name": "b", "role": "src1", "description": "Second operand"},
        ],
        "min_sm": "sm_50",
    },
    {
        "mnemonic": "selp",
        "description": "Select between two values based on a predicate condition.",
        "category": "comparison",
        "supported_types": [
            ".b16",
            ".b32",
            ".b64",
            ".u16",
            ".u32",
            ".u64",
            ".s16",
            ".s32",
            ".s64",
            ".f16",
            ".f32",
            ".f64",
        ],
        "vector_widths": [],
        "state_spaces": [],
        "operands": [
            {"name": "d", "role": "dst", "description": "Destination operand"},
            {"name": "a", "role": "src0", "description": "Selected if true"},
            {"name": "b", "role": "src1", "description": "Selected if false"},
            {
                "name": "p",
                "role": "src2",
                "description": "Condition predicate operand",
            },
        ],
        "min_sm": "sm_50",
    },
]


def parse_nvptx_td_content(content: str) -> list[IsaInstructionRecordModel]:
    """Extract PTX instruction mnemonics and metadata from NVPTX TableGen content.

    Args:
        content: Raw TableGen text content.

    Returns:
        List of parsed PTX instruction records.
    """
    instructions: list[IsaInstructionRecordModel] = []
    for _def_name, mnemonic in tokenize_nvptx_td_defs(content):
        instructions.append(
            IsaInstructionRecordModel(
                mnemonic=mnemonic,
                category="instruction",
                description=f"PTX {mnemonic} instruction parsed from TableGen.",
                supported_types=[".b32", ".b64"],
                vector_widths=[],
                state_spaces=[],
                operands=[],
                min_sm="sm_50",
            )
        )
    return instructions


__all__ = ["PTX_ARITH_OPS", "parse_nvptx_td_content"]
