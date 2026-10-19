"""WebGPU WGSL Dialect Snapshot Extractor.

Extracts built-in functions, compute shader intrinsics, and operations from the WebGPU WGSL specification.
"""

from __future__ import annotations

import json

from ml_switcheroo_ir.schema.ghost import (
    GhostRef,
    OperandDirection,
    SemanticTier,
)
import pydantic

from ml_ecosystem_snapshots.frameworks._base import CanonicalSsaOpSpec


CANONICAL_WGSL_OPS: list[CanonicalSsaOpSpec] = pydantic.TypeAdapter(
    list[CanonicalSsaOpSpec]
).validate_python(
    [
        {
            "name": "storageStore",
            "domain": "wgsl",
            "inputs": ["buffer", "index", "value"],
            "outputs": [],
            "attributes": [
                {
                    "name": "address_space",
                    "type": "str",
                    "required": False,
                    "default": "storage, read_write",
                }
            ],
            "description": "Store a value into a storage buffer at the given index.",
        },
        {
            "name": "storageLoad",
            "domain": "wgsl",
            "inputs": ["buffer", "index"],
            "outputs": ["value"],
            "attributes": [
                {
                    "name": "address_space",
                    "type": "str",
                    "required": False,
                    "default": "storage, read",
                }
            ],
            "description": "Load a value from a storage buffer at the given index.",
        },
        {
            "name": "workgroupBarrier",
            "domain": "wgsl",
            "inputs": [],
            "outputs": [],
            "attributes": [],
            "description": "Synchronize memory operations among invocations in a workgroup.",
        },
        {
            "name": "textureSample",
            "domain": "wgsl",
            "inputs": ["t", "s", "coords"],
            "outputs": ["value"],
            "attributes": [],
            "description": "Samples a texture using a sampler.",
        },
        {
            "name": "textureLoad",
            "domain": "wgsl",
            "inputs": ["t", "coords", "level"],
            "outputs": ["value"],
            "attributes": [],
            "description": "Loads a texel from a texture without filtering.",
        },
        {
            "name": "storageBarrier",
            "domain": "wgsl",
            "inputs": [],
            "outputs": [],
            "attributes": [],
            "description": "Synchronize storage buffer operations across invocations.",
        },
        {
            "name": "atomicAdd",
            "domain": "wgsl",
            "inputs": ["atomic_ptr", "value"],
            "outputs": ["old_value"],
            "attributes": [],
            "description": "Atomically add a value to an atomic variable and return previous value.",
        },
        {
            "name": "atomicSub",
            "domain": "wgsl",
            "inputs": ["atomic_ptr", "value"],
            "outputs": ["old_value"],
            "attributes": [],
            "description": "Atomically subtract a value from an atomic variable and return previous value.",
        },
        {
            "name": "atomicExchange",
            "domain": "wgsl",
            "inputs": ["atomic_ptr", "value"],
            "outputs": ["old_value"],
            "attributes": [],
            "description": "Atomically store a value into an atomic variable and return previous value.",
        },
    ]
)


def _load_wgsl_ops() -> list[CanonicalSsaOpSpec]:
    """Load WGSL operation definitions from ml_switcheroo_ir schema or fallback registry.

    Returns:
        List of WGSL operator specifications.
    """
    try:
        import importlib.resources as pkg_resources
        import ml_switcheroo_ir.schema as ir_schema

        fpath = pkg_resources.files(ir_schema).joinpath("wgsl_ops.json")
        with fpath.open("r", encoding="utf-8") as f:
            data: object = json.load(f)
            if (
                isinstance(data, dict)
                and isinstance(data.get("ops"), list)
                and data["ops"]
            ):
                return pydantic.TypeAdapter(list[CanonicalSsaOpSpec]).validate_python(
                    data["ops"]
                )
    except Exception:
        pass
    return CANONICAL_WGSL_OPS


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> list[GhostRef]:
    """Collect WGSL operations for the specified semantic tier.

    Args:
        category: The SemanticTier category.
        include_nonpublic: Whether to include non-public operations.

    Returns:
        List of GhostOperationRef objects representing WGSL shader operations.
    """
    from ml_ecosystem_snapshots.frameworks._base import build_ssa_operation_refs

    if category not in (
        SemanticTier.ARRAY_API,
        SemanticTier.NEURAL,
        SemanticTier.UTIL,
    ):
        return []

    ops = _load_wgsl_ops()
    return build_ssa_operation_refs(
        ops=ops,
        api_prefix="wgsl",
        name_key="name",
        operand_annotation_fn=lambda _op, _inp: "Value",
        operand_description_fn=lambda _op, inp: f"WGSL operand {inp}",
        operand_direction_fn=lambda op, inp: (
            OperandDirection.WRITE
            if "store" in str(op["name"]).lower() and inp == "value"
            else OperandDirection.READ
        ),
        attr_description_prefix="WGSL attribute",
        result_type="Value",
        empty_return_type="None",
        provenance_fn=lambda op: {
            "source_type": "wgsl_spec",
            "upstream_version": "draft-2024",
            "domain": op.get("domain", "wgsl"),
        },
        default_doc_fn=lambda op_name: f"WebGPU WGSL {op_name} operation.",
        environment_tags=("webgpu",),
    )
