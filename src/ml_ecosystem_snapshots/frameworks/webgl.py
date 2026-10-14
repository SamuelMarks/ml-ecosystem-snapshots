"""WebGL 2.0 and GLSL ES 3.00 Shading Language Snapshot Extractor.

Formalizes WebGL 2.0 compute shaders, texture fetch intrinsics (texelFetch),
matrix operations (matrixCompMult, transpose), and transcendental functions.
"""

from typing import Any, Dict, List

from ml_switcheroo_ir.schema.ghost import (
    GhostRef,
    OperandDirection,
    SemanticTier,
)


CANONICAL_WEBGL_OPS: List[Dict[str, Any]] = [
    # Texture Fetches & Sampling
    {
        "name": "texelFetch",
        "category": "texture",
        "inputs": ["sampler", "P", "lod"],
        "outputs": ["rgba"],
        "attributes": [
            {"name": "sampler_type", "type": "str", "default": "'sampler2D'"},
        ],
        "description": "Fetch a single texel from a 2D texture using integer coordinate vectors without filtering.",
    },
    {
        "name": "texture",
        "category": "texture",
        "inputs": ["sampler", "P"],
        "outputs": ["rgba"],
        "attributes": [
            {"name": "bias", "type": "float", "default": "None"},
        ],
        "description": "Sample a texture using normalized floating point texture coordinates with filtering.",
    },
    {
        "name": "textureSize",
        "category": "texture",
        "inputs": ["sampler", "lod"],
        "outputs": ["size"],
        "attributes": [],
        "description": "Retrieve the integer dimensions (width, height) of a given mipmap level of a texture.",
    },
    # Matrix Operations
    {
        "name": "matrixCompMult",
        "category": "matrix",
        "inputs": ["x", "y"],
        "outputs": ["result"],
        "attributes": [],
        "description": "Perform component-wise multiplication of two matrices.",
    },
    {
        "name": "transpose",
        "category": "matrix",
        "inputs": ["m"],
        "outputs": ["result"],
        "attributes": [],
        "description": "Transpose a square or rectangular matrix.",
    },
    {
        "name": "inverse",
        "category": "matrix",
        "inputs": ["m"],
        "outputs": ["result"],
        "attributes": [],
        "description": "Compute the matrix inverse of a square floating point matrix.",
    },
    # Float & Vector Math
    {
        "name": "fma",
        "category": "math",
        "inputs": ["a", "b", "c"],
        "outputs": ["result"],
        "attributes": [],
        "description": "Fused multiply-add: (a * b) + c.",
    },
    {
        "name": "mix",
        "category": "math",
        "inputs": ["x", "y", "a"],
        "outputs": ["result"],
        "attributes": [],
        "description": "Linear blend of x and y: x * (1 - a) + y * a.",
    },
    {
        "name": "clamp",
        "category": "math",
        "inputs": ["x", "minVal", "maxVal"],
        "outputs": ["result"],
        "attributes": [],
        "description": "Constrain a value to lie between two limits: min(max(x, minVal), maxVal).",
    },
    {
        "name": "inversesqrt",
        "category": "math",
        "inputs": ["x"],
        "outputs": ["result"],
        "attributes": [],
        "description": "Reciprocal square root: 1.0 / sqrt(x).",
    },
    # Packing & Unpacking
    {
        "name": "packHalf2x16",
        "category": "pack",
        "inputs": ["v"],
        "outputs": ["val"],
        "attributes": [],
        "description": "Pack two 32-bit floating point values into a single 32-bit unsigned integer holding two 16-bit floats.",
    },
    {
        "name": "unpackHalf2x16",
        "category": "pack",
        "inputs": ["v"],
        "outputs": ["vec"],
        "attributes": [],
        "description": "Unpack two 16-bit floating point values from a 32-bit unsigned integer into a 2-component vector.",
    },
]


def _load_webgl_ops() -> List[Dict[str, Any]]:
    """Load canonical WebGL 2.0 / GLSL ES 3.00 operation specifications.

    Returns:
        List of WebGL operation dictionaries.
    """
    return CANONICAL_WEBGL_OPS


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Collect WebGL 2.0 and GLSL ES 3.00 compute shader intrinsics.

    Args:
        category: Target SemanticTier enum value.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        List of GhostOperationRef objects representing WebGL operations.
    """
    if category not in (
        SemanticTier.ARRAY_API,
        SemanticTier.NEURAL,
        SemanticTier.UTIL,
    ):
        return []

    from ml_ecosystem_snapshots.frameworks._base import build_ssa_operation_refs

    ops = _load_webgl_ops()
    return build_ssa_operation_refs(
        ops=ops,
        api_prefix="webgl",
        name_key="name",
        operand_annotation_fn=lambda op, inp: (
            "vec4" if inp in ("x", "y", "a", "b", "c") else "any"
        ),
        operand_description_fn=lambda op, inp: f"WebGL operand {inp}",
        operand_direction_fn=lambda op, inp: OperandDirection.READ,
        attr_description_prefix="WebGL attribute",
        result_type="vec4",
        empty_return_type="None",
        provenance_fn=lambda op: {
            "source_type": "webgl_spec",
            "upstream_version": "2.0",
            "category": op.get("category", "compute"),
        },
        default_doc_fn=lambda op_name: f"WebGL {op_name} operation.",
        environment_tags=("webgl", "glsl", "webgl2"),
    )
