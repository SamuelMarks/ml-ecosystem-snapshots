"""ONNX Operator Specification Extractor.

Extracts official ONNX operator schemas, SSA operands, attribute constraints,
and type signatures from the ONNX operator specification definitions.
"""

from typing import Any, Dict, List

from ml_switcheroo_ir.schema.ghost import (
    GhostRef,
    OperandDirection,
    SemanticTier,
)


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
    except Exception:
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
    from ml_ecosystem_snapshots.frameworks._base import build_ssa_operation_refs

    if category not in (
        SemanticTier.ARRAY_API,
        SemanticTier.NEURAL,
        SemanticTier.UTIL,
    ):
        return []

    ops = _load_onnx_ops()
    filtered_ops = [
        op
        for op in ops
        if include_nonpublic
        or (
            "preview" not in str(op.get("domain", ""))
            and "internal" not in str(op.get("domain", ""))
        )
    ]

    return build_ssa_operation_refs(
        ops=filtered_ops,
        api_prefix="onnx",
        name_key="name",
        operand_annotation_fn=lambda _op, _inp: "Tensor",
        operand_description_fn=lambda _op, inp: f"ONNX operand {inp}",
        operand_direction_fn=lambda _op, _inp: OperandDirection.READ,
        attr_description_prefix="ONNX attribute",
        result_type="Tensor",
        empty_return_type="None",
        provenance_fn=lambda op: {
            "source_type": "onnx_spec",
            "upstream_version": str(op.get("since_version", "1.0")),
            "domain": str(op.get("domain") or "ai.onnx"),
        },
        default_doc_fn=lambda op_name: f"ONNX {op_name} operator.",
        environment_tags=("onnx",),
        api_path_fn=lambda op, op_name: (
            f"onnx.{op['domain']}.{op_name}" if op.get("domain") else f"onnx.{op_name}"
        ),
    )
