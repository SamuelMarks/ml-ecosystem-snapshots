"""WebAssembly 2.0 and Fixed-Width SIMD (v128) Instruction Set Snapshot Extractor.

Formalizes W3C WebAssembly 2.0 instructions, 128-bit vector arithmetic (f32x4, i32x4),
memory access operators (v128.load, v128.store), and lane swizzles.
"""

from typing import Any, Dict, List

from ml_switcheroo_ir.schema.ghost import (
    GhostRef,
    OperandDirection,
    SemanticTier,
)


CANONICAL_WASM_SIMD_OPS: List[Dict[str, Any]] = [
    # 32-bit Floating-Point SIMD (f32x4)
    {
        "mnemonic": "f32x4.add",
        "category": "float_arithmetic",
        "inputs": ["lhs", "rhs"],
        "outputs": ["val"],
        "attributes": [],
        "description": "Lane-wise 32-bit floating point addition across 4 lanes.",
    },
    {
        "mnemonic": "f32x4.sub",
        "category": "float_arithmetic",
        "inputs": ["lhs", "rhs"],
        "outputs": ["val"],
        "attributes": [],
        "description": "Lane-wise 32-bit floating point subtraction across 4 lanes.",
    },
    {
        "mnemonic": "f32x4.mul",
        "category": "float_arithmetic",
        "inputs": ["lhs", "rhs"],
        "outputs": ["val"],
        "attributes": [],
        "description": "Lane-wise 32-bit floating point multiplication across 4 lanes.",
    },
    {
        "mnemonic": "f32x4.div",
        "category": "float_arithmetic",
        "inputs": ["lhs", "rhs"],
        "outputs": ["val"],
        "attributes": [],
        "description": "Lane-wise 32-bit floating point division across 4 lanes.",
    },
    {
        "mnemonic": "f32x4.sqrt",
        "category": "float_arithmetic",
        "inputs": ["val"],
        "outputs": ["res"],
        "attributes": [],
        "description": "Lane-wise 32-bit floating point square root.",
    },
    # 32-bit Integer SIMD (i32x4)
    {
        "mnemonic": "i32x4.add",
        "category": "int_arithmetic",
        "inputs": ["lhs", "rhs"],
        "outputs": ["val"],
        "attributes": [],
        "description": "Lane-wise 32-bit integer addition with wrapping modulo 2^32.",
    },
    {
        "mnemonic": "i32x4.sub",
        "category": "int_arithmetic",
        "inputs": ["lhs", "rhs"],
        "outputs": ["val"],
        "attributes": [],
        "description": "Lane-wise 32-bit integer subtraction with wrapping modulo 2^32.",
    },
    {
        "mnemonic": "i32x4.mul",
        "category": "int_arithmetic",
        "inputs": ["lhs", "rhs"],
        "outputs": ["val"],
        "attributes": [],
        "description": "Lane-wise 32-bit integer multiplication with wrapping modulo 2^32.",
    },
    {
        "mnemonic": "i32x4.dot_i16x8_s",
        "category": "int_arithmetic",
        "inputs": ["lhs", "rhs"],
        "outputs": ["val"],
        "attributes": [],
        "description": "Signed dot product of adjacent 16-bit integer pairs accumulated into 32-bit integer results.",
    },
    # Memory Access (v128)
    {
        "mnemonic": "v128.load",
        "category": "memory",
        "inputs": ["addr"],
        "outputs": ["val"],
        "attributes": [
            {"name": "offset", "type": "int", "default": "0"},
            {"name": "align", "type": "int", "default": "4"},
        ],
        "description": "Load a 128-bit vector from linear memory.",
    },
    {
        "mnemonic": "v128.store",
        "category": "memory",
        "inputs": ["addr", "val"],
        "outputs": [],
        "attributes": [
            {"name": "offset", "type": "int", "default": "0"},
            {"name": "align", "type": "int", "default": "4"},
        ],
        "description": "Store a 128-bit vector into linear memory.",
    },
    # Bitwise & Shuffle
    {
        "mnemonic": "v128.bitselect",
        "category": "bitwise",
        "inputs": ["v1", "v2", "c"],
        "outputs": ["val"],
        "attributes": [],
        "description": "Bitwise selection: (v1 & c) | (v2 & ~c).",
    },
    {
        "mnemonic": "i8x16.shuffle",
        "category": "shuffle",
        "inputs": ["v1", "v2"],
        "outputs": ["val"],
        "attributes": [{"name": "lanes", "type": "list[int]", "default": "[0]*16"}],
        "description": "Permute bytes across two 128-bit vectors according to 16 immediate lane indices.",
    },
]


def _load_wasm_simd_ops() -> List[Dict[str, Any]]:
    """Load canonical WebAssembly SIMD operation specifications.

    Returns:
        List of WebAssembly SIMD instruction dictionaries.
    """
    return CANONICAL_WASM_SIMD_OPS


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Collect WebAssembly SIMD instructions and memory operators.

    Args:
        category: Target SemanticTier enum value.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        List of GhostOperationRef objects representing WASM SIMD instructions.
    """
    if category not in (
        SemanticTier.ARRAY_API,
        SemanticTier.NEURAL,
        SemanticTier.UTIL,
    ):
        return []

    from ml_ecosystem_snapshots.frameworks._base import build_ssa_operation_refs

    ops = _load_wasm_simd_ops()
    return build_ssa_operation_refs(
        ops=ops,
        api_prefix="wasm",
        name_key="mnemonic",
        operand_annotation_fn=lambda op, inp: "v128" if inp != "addr" else "i32",
        operand_description_fn=lambda op, inp: f"WASM operand {inp}",
        operand_direction_fn=lambda op, inp: (
            OperandDirection.WRITE
            if op["mnemonic"] == "v128.store" and inp == "addr"
            else OperandDirection.READ
        ),
        attr_description_prefix="WASM immediate/attribute",
        result_type="v128",
        empty_return_type="None",
        provenance_fn=lambda op: {
            "source_type": "wasm_spec",
            "upstream_version": "2.0",
            "category": op.get("category", "simd"),
        },
        default_doc_fn=lambda mnemonic: f"WebAssembly {mnemonic} instruction.",
        environment_tags=("wasm", "simd", "v128"),
    )
