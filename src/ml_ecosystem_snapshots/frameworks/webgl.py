"""WebGL 2.0 and GLSL ES 3.00 Shading Language Snapshot Extractor.

Formalizes WebGL 2.0 compute shaders, texture fetch intrinsics (texelFetch),
matrix operations (matrixCompMult, transpose), and transcendental functions.
"""

from typing import Any, Dict, List

from ml_switcheroo_ir.schema.ghost import (
    GhostOperationRef,
    GhostParam,
    GhostRef,
    GhostResult,
    IRParameterRole,
    OperandDirection,
    ParameterKind,
    SemanticTier,
)
from ..models import ExtendedGhostParam


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

    ops = _load_webgl_ops()
    refs: List[GhostRef] = []

    for op in ops:
        op_name = op["name"]
        api_path = f"webgl.{op_name}"
        params: List[GhostParam] = []

        # SSA operands
        for inp in op.get("inputs", []):
            params.append(
                ExtendedGhostParam(
                    name=inp,
                    kind=ParameterKind.POSITIONAL_ONLY,
                    annotation="vec4" if inp in ("x", "y", "a", "b", "c") else "any",
                    standardized_name=inp,
                    description=f"WebGL operand {inp}",
                    direction=OperandDirection.READ,
                    role=IRParameterRole.OPERAND,
                )
            )

        # Attributes
        attrs_meta: Dict[str, Any] = {}
        for attr in op.get("attributes", []):
            attr_name = attr.get("name", "attr")
            attr_type = attr.get("type", "Any")
            attr_def = attr.get("default")
            params.append(
                ExtendedGhostParam(
                    name=attr_name,
                    kind=ParameterKind.KEYWORD_ONLY,
                    annotation=attr_type,
                    default=str(attr_def) if attr_def is not None else None,
                    standardized_name=attr_name,
                    description=f"WebGL attribute {attr_name}",
                    role=IRParameterRole.ATTRIBUTE,
                )
            )
            attrs_meta[attr_name] = attr

        returns: List[GhostResult] = []
        for out in op.get("outputs", []):
            returns.append(GhostResult(name=out, type="vec4"))

        returns_type = "vec4" if returns else "None"

        provenance = {
            "source_type": "webgl_spec",
            "upstream_version": "2.0",
            "category": op.get("category", "compute"),
        }

        refs.append(
            GhostOperationRef(
                name=op_name,
                api_path=api_path,
                kind="function",
                params=params,
                operands=[
                    p
                    for p in params
                    if getattr(p, "role", None) == IRParameterRole.OPERAND
                ],
                returns=returns,
                returns_type=returns_type,
                docstring=op.get("description", f"WebGL {op_name} operation."),
                environment_tags=["webgl", "glsl", "webgl2"],
                domain_metadata=provenance,
                attributes=attrs_meta if attrs_meta else None,
            )
        )

    return refs
