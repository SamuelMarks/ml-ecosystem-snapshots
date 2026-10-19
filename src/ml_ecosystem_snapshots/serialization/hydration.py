"""Schema hydration and v1-to-v2 migration utilities for serialized GhostRef dictionaries."""

from __future__ import annotations

from typing import Any, Dict, List

from ml_switcheroo_ir.schema.ghost import (
    ExtendedGhostParam,
    ExtendedGhostRef,
    GhostInstructionRef,
    GhostIsaRef,
    GhostMlirRef,
    GhostOperationRef,
    GhostParam,
    GhostPythonRef,
    GhostResult,
    IRParameterRole,
    OperandDirection,
    ParameterKind,
)

from ml_ecosystem_snapshots.exceptions import SnapshotSchemaError


def hydrate_ghost_ref(data: Dict[str, Any]) -> ExtendedGhostRef:
    """Hydrate a dictionary payload into a validated `ExtendedGhostRef` or polymorphic subclass.

    Args:
        data: Dictionary representation of a ghost reference.

    Returns:
        Hydrated `ExtendedGhostRef` instance.

    Raises:
        SnapshotSchemaError: If `data` is not a dictionary or violates the `ExtendedGhostRef` schema.
    """
    if not isinstance(data, dict):
        raise SnapshotSchemaError("GhostRef payload must be a dictionary.")
    # Synthesize domain_type and required fields for raw dumps if missing
    if "mnemonic" in data and ("name" not in data or "operands" in data):
        name = str(data.get("name") or data["mnemonic"])
        desc = str(data.get("description") or f"ISA {name} instruction.")
        arch = data.get("architecture")
        valid_archs: List[str] = []
        if isinstance(arch, list):
            valid_archs = [str(a) for a in arch]
        elif isinstance(arch, str):
            valid_archs = [arch]
        modifiers = [str(m) for m in data.get("modifiers", [])]
        operands_list: List[List[str]] = data.get("operands", [])
        max_sig: List[str] = []
        for sig in operands_list:
            if len(sig) > len(max_sig):
                max_sig = sig
        params: List[GhostParam] = []
        for i, raw_op in enumerate(max_sig):
            role = "dst" if i == 0 else f"src{i - 1}"
            direction = OperandDirection.WRITE if i == 0 else OperandDirection.READ
            params.append(
                ExtendedGhostParam(
                    name=f"op{i}",
                    kind=ParameterKind.POSITIONAL_ONLY,
                    annotation=str(raw_op),
                    standardized_name=role,
                    direction=direction,
                    role=IRParameterRole.OPERAND,
                )
            )
        domain_metadata = dict(data.get("domain_metadata") or {})
        domain_metadata.setdefault("architecture", arch)
        domain_metadata.setdefault("valid_architectures", valid_archs)
        domain_metadata.setdefault("modifiers", modifiers)
        domain_metadata.setdefault("operand_signatures", operands_list)
        isa_overloads: List[ExtendedGhostRef] = []
        for sig_idx, sig in enumerate(operands_list):
            sig_params: List[GhostParam] = []
            for idx, op_type in enumerate(sig):
                role = "dst" if idx == 0 else f"src{idx - 1}"
                direction = (
                    OperandDirection.WRITE if idx == 0 else OperandDirection.READ
                )
                sig_params.append(
                    ExtendedGhostParam(
                        name=f"op{idx}",
                        kind=ParameterKind.POSITIONAL_ONLY,
                        annotation=str(op_type),
                        standardized_name=role,
                        direction=direction,
                        role=IRParameterRole.OPERAND,
                    )
                )
            isa_overloads.append(
                ExtendedGhostRef(
                    name=name,
                    api_path=str(data.get("api_path") or f"isa.inst.{name}"),
                    kind="function",
                    params=sig_params,
                    docstring=f"Overload {sig_idx} for {name}: {', '.join(str(s) for s in sig)}",
                    environment_tags=valid_archs,
                    domain_metadata=domain_metadata,
                )
            )
        return GhostIsaRef(
            name=name,
            api_path=str(data.get("api_path") or f"isa.inst.{name}"),
            kind=str(data.get("kind") or "function"),
            params=params,
            docstring=desc,
            environment_tags=valid_archs,
            overloads=isa_overloads if isa_overloads else None,
            domain_metadata=domain_metadata,
        )

    if (
        ("attributes" in data or "results" in data)
        and ("name" not in data or "params" not in data)
        and data.get("domain_type") != "mlir"
    ):
        api_path = str(data.get("api_path") or "mlir.op")
        name = str(data.get("name") or api_path.split(".")[-1])
        mlir_params: List[GhostParam] = []
        for op in data.get("operands", []):
            op_name = op if isinstance(op, str) else str(op.get("name", "arg"))
            mlir_params.append(
                ExtendedGhostParam(
                    name=op_name,
                    kind=ParameterKind.POSITIONAL_ONLY,
                    role=IRParameterRole.OPERAND,
                )
            )
        for attr in data.get("attributes", []):
            attr_name = attr if isinstance(attr, str) else str(attr.get("name", "attr"))
            mlir_params.append(
                ExtendedGhostParam(
                    name=attr_name,
                    kind=ParameterKind.KEYWORD_ONLY,
                    role=IRParameterRole.ATTRIBUTE,
                )
            )
        results = []
        for res in data.get("results", []):
            results.append(
                GhostResult(
                    name=(res if isinstance(res, str) else str(res.get("name", "res"))),
                    type=(
                        str(res.get("type", "Value"))
                        if isinstance(res, dict)
                        else "Value"
                    ),
                )
            )
        raw_traits = data.get("traits")
        raw_regions = data.get("regions")
        raw_successors = data.get("successors")
        raw_attrs = data.get("attributes")
        return GhostMlirRef(
            name=name,
            api_path=api_path,
            kind=str(data.get("kind") or "operation"),
            params=mlir_params,
            returns=results,
            docstring=data.get("docstring") or data.get("description"),
            traits=raw_traits if isinstance(raw_traits, list) else None,
            regions=raw_regions if isinstance(raw_regions, dict) else None,
            successors=raw_successors if isinstance(raw_successors, list) else None,
            domain_metadata=data.get("domain_metadata"),
            attributes=raw_attrs if isinstance(raw_attrs, dict) else None,
        )

    domain_type = data.get("domain_type")
    ref_cls: Any = ExtendedGhostRef
    if domain_type == "python":
        ref_cls = GhostPythonRef
    elif domain_type in ("isa", "instruction"):
        ref_cls = GhostInstructionRef
    elif domain_type in ("mlir", "operation"):
        ref_cls = GhostOperationRef

    if "kind" not in data:
        data = dict(data)
        data["kind"] = "function"

    try:
        if "params" in data and isinstance(data["params"], list):
            data = dict(data)
            hydrated_params = []
            for p in data["params"]:
                if isinstance(p, dict):
                    p_dict = dict(p)
                    if "kind" not in p_dict:
                        p_dict["kind"] = ParameterKind.POSITIONAL_OR_KEYWORD
                    if "default" in p_dict and p_dict["default"] is not None:
                        if not isinstance(p_dict["default"], str):
                            p_dict["default"] = str(p_dict["default"])
                    hydrated_params.append(ExtendedGhostParam.model_validate(p_dict))
                else:
                    hydrated_params.append(p)
            data["params"] = hydrated_params

        res = ref_cls.model_validate(data)
    except (ValueError, TypeError) as exc:
        raise SnapshotSchemaError(f"Invalid GhostRef schema: {exc}") from exc
    assert isinstance(res, ExtendedGhostRef)
    return res


def migrate_ghost_ref_v2(data: Dict[str, Any]) -> ExtendedGhostRef:
    """Seamlessly upgrade v1.x or v2.x dictionaries to the `ExtendedGhostRef` v2.0.0 schema.

    Args:
        data: Dictionary representation of a ghost reference.

    Returns:
        Validated `ExtendedGhostRef` instance.
    """
    data_copy = dict(data)
    if "schema_version" not in data_copy or data_copy["schema_version"] in (
        "1.0",
        "1.0.0",
        "1.2",
    ):
        data_copy["schema_version"] = "2.0.0"

    return hydrate_ghost_ref(data_copy)


__all__ = ["hydrate_ghost_ref", "migrate_ghost_ref_v2"]
