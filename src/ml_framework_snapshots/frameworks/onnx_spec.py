"""ONNX Operator Specification Extractor.

Extracts official ONNX operator schemas, SSA operands, attribute constraints,
and type signatures from the ONNX operator specification definitions.
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


CANONICAL_ONNX_OPS: List[Dict[str, Any]] = [
    {
        "name": "Add",
        "domain": "",
        "since_version": 14,
        "inputs": ["A", "B"],
        "outputs": ["C"],
        "attributes": [],
        "description": "Element-wise addition of two tensors with broadcasting support.",
    },
    {
        "name": "Sub",
        "domain": "",
        "since_version": 14,
        "inputs": ["A", "B"],
        "outputs": ["C"],
        "attributes": [],
        "description": "Element-wise subtraction of two tensors with broadcasting support.",
    },
    {
        "name": "Mul",
        "domain": "",
        "since_version": 14,
        "inputs": ["A", "B"],
        "outputs": ["C"],
        "attributes": [],
        "description": "Element-wise multiplication of two tensors with broadcasting support.",
    },
    {
        "name": "Div",
        "domain": "",
        "since_version": 14,
        "inputs": ["A", "B"],
        "outputs": ["C"],
        "attributes": [],
        "description": "Element-wise division of two tensors with broadcasting support.",
    },
    {
        "name": "Relu",
        "domain": "",
        "since_version": 14,
        "inputs": ["X"],
        "outputs": ["Y"],
        "attributes": [],
        "description": "Rectified Linear Unit activation: max(0, x).",
    },
    {
        "name": "MatMul",
        "domain": "",
        "since_version": 13,
        "inputs": ["A", "B"],
        "outputs": ["Y"],
        "attributes": [],
        "description": "Matrix product of two tensors with standard batched broadcasting rules.",
    },
    {
        "name": "Gemm",
        "domain": "",
        "since_version": 13,
        "inputs": ["A", "B", "C"],
        "outputs": ["Y"],
        "attributes": [
            {
                "name": "alpha",
                "type": "float",
                "required": False,
                "default": "1.0",
                "description": "Scalar multiplier for product of inputs A and B.",
            },
            {
                "name": "beta",
                "type": "float",
                "required": False,
                "default": "1.0",
                "description": "Scalar multiplier for input C.",
            },
            {
                "name": "transA",
                "type": "int",
                "required": False,
                "default": "0",
                "description": "Whether A should be transposed before multiplying.",
            },
            {
                "name": "transB",
                "type": "int",
                "required": False,
                "default": "0",
                "description": "Whether B should be transposed before multiplying.",
            },
        ],
        "description": "General Matrix Multiplication: alpha * A' * B' + beta * C.",
    },
    {
        "name": "Conv",
        "domain": "",
        "since_version": 11,
        "inputs": ["X", "W", "B"],
        "outputs": ["Y"],
        "attributes": [
            {
                "name": "auto_pad",
                "type": "str",
                "required": False,
                "default": "NOTSET",
                "description": "Auto padding mode string.",
            },
            {
                "name": "dilations",
                "type": "list[int]",
                "required": False,
                "default": "None",
                "description": "Dilation value along each spatial axis.",
            },
            {
                "name": "group",
                "type": "int",
                "required": False,
                "default": "1",
                "description": "Number of groups input channels and output channels are divided into.",
            },
            {
                "name": "kernel_shape",
                "type": "list[int]",
                "required": False,
                "default": "None",
                "description": "The shape of the convolution kernel.",
            },
            {
                "name": "pads",
                "type": "list[int]",
                "required": False,
                "default": "None",
                "description": "Padding for the beginning and ending along each spatial axis.",
            },
            {
                "name": "strides",
                "type": "list[int]",
                "required": False,
                "default": "None",
                "description": "Stride along each spatial axis.",
            },
        ],
        "description": "Standard multi-dimensional spatial convolution.",
    },
    {
        "name": "Reshape",
        "domain": "",
        "since_version": 14,
        "inputs": ["data", "shape"],
        "outputs": ["reshaped"],
        "attributes": [
            {
                "name": "allowzero",
                "type": "int",
                "required": False,
                "default": "0",
                "description": "Whether zero values in shape are kept as zero or copied.",
            }
        ],
        "description": "Reshape the input tensor similar to numpy.reshape.",
    },
    {
        "name": "Softmax",
        "domain": "",
        "since_version": 13,
        "inputs": ["input"],
        "outputs": ["output"],
        "attributes": [
            {
                "name": "axis",
                "type": "int",
                "required": False,
                "default": "-1",
                "description": "The axis along which to compute the softmax.",
            }
        ],
        "description": "The operator computes the softmax normalized exponential values.",
    },
]


def _get_onnx_defs() -> Any:
    """Lazily load the onnx.defs module to prevent unnecessary import overhead.

    Returns:
        The onnx.defs module or None if not installed.
    """
    try:
        import onnx.defs as defs

        return defs
    except (ImportError, Exception):
        return None


def _load_onnx_ops() -> List[Dict[str, Any]]:
    """Load ONNX operator definitions from onnx.defs or canonical fallback.

    Returns:
        List of ONNX operator schema dictionaries.
    """
    defs = _get_onnx_defs()
    if defs is None:
        return CANONICAL_ONNX_OPS

    try:
        schemas = defs.get_all_schemas_with_history()
        # Keep latest version per (domain, name)
        latest_map: Dict[tuple[str, str], Any] = {}
        for s in schemas:
            key = (s.domain, s.name)
            if key not in latest_map or s.since_version > latest_map[key].since_version:
                latest_map[key] = s

        ops: List[Dict[str, Any]] = []
        for (domain, name), s in sorted(
            latest_map.items(), key=lambda x: (x[0][0], x[0][1])
        ):
            inputs = [inp.name for inp in s.inputs]
            outputs = [out.name for out in s.outputs]
            attributes = []
            for attr_name, attr in s.attributes.items():
                attributes.append(
                    {
                        "name": attr_name,
                        "type": str(attr.type).replace("AttrType.", "").lower(),
                        "required": bool(attr.required),
                        "default": str(attr.default_value)
                        if hasattr(attr, "default_value") and attr.default_value
                        else None,
                        "description": getattr(attr, "description", "") or "",
                    }
                )
            ops.append(
                {
                    "name": name,
                    "domain": domain,
                    "since_version": s.since_version,
                    "inputs": inputs,
                    "outputs": outputs,
                    "attributes": attributes,
                    "description": getattr(s, "doc", "") or f"ONNX {name} operator.",
                }
            )
        if ops:
            return ops
    except Exception:
        pass

    return CANONICAL_ONNX_OPS


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Collect ONNX operator definitions for the specified semantic tier.

    Args:
        category: The SemanticTier category.
        include_nonpublic: Whether to include non-public or preview operations.

    Returns:
        List of GhostOperationRef objects representing ONNX operators.
    """
    if category not in (
        SemanticTier.ARRAY_API,
        SemanticTier.NEURAL,
        SemanticTier.UTIL,
    ):
        return []

    ops = _load_onnx_ops()
    refs: List[GhostRef] = []

    for op in ops:
        domain = op.get("domain", "")
        if include_nonpublic or ("preview" not in domain and "internal" not in domain):
            op_name = op["name"]
            prefix = f"onnx.{domain}." if domain else "onnx."
            api_path = f"{prefix}{op_name}"
            params: List[GhostParam] = []

            # SSA inputs
            for inp in op.get("inputs", []):
                params.append(
                    ExtendedGhostParam(
                        name=inp,
                        kind=ParameterKind.POSITIONAL_ONLY,
                        annotation="Tensor",
                        standardized_name=inp,
                        description=f"ONNX operand {inp}",
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
                        description=attr.get(
                            "description", f"ONNX attribute {attr_name}"
                        ),
                        role=IRParameterRole.ATTRIBUTE,
                    )
                )
                attrs_meta[attr_name] = attr

            returns: List[GhostResult] = []
            for out in op.get("outputs", []):
                returns.append(GhostResult(name=out, type="Tensor"))

            returns_type = "Tensor" if returns else "None"

            provenance = {
                "source_type": "onnx_spec",
                "upstream_version": str(op.get("since_version", "1.0")),
                "domain": domain or "ai.onnx",
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
                    docstring=op.get("description", f"ONNX {op_name} operator."),
                    environment_tags=["onnx"],
                    domain_metadata=provenance,
                    attributes=attrs_meta if attrs_meta else None,
                )
            )

    return refs
