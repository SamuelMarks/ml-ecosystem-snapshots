"""Schema hydration utilities for serialized GhostRef dictionaries."""

from __future__ import annotations

import typing

from ml_switcheroo_ir.schema.ghost import (
    ExtendedGhostParam,
    ExtendedGhostRef,
    GhostInstructionRef,
    GhostIsaRef,
    GhostMlirRef,
    GhostOperationRef,
    GhostParam,
    GhostPythonRef,
    GhostRef,
    GhostResult,
    IRParameterRole,
    OperandDirection,
    ParameterKind,
)

from ml_ecosystem_snapshots.exceptions import SnapshotSchemaError
from ml_ecosystem_snapshots.models import JsonValue, ScalarValue


def hydrate_ghost_ref(
    data: (
        typing.Mapping[str, JsonValue | typing.Sequence[JsonValue | GhostParam]]
        | ScalarValue
    ),
) -> ExtendedGhostRef:
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

    if "mnemonic" in data and ("name" not in data or "operands" in data):
        name = str(data.get("name") or data["mnemonic"])
        desc = str(data.get("description") or f"ISA {name} instruction.")
        arch = data.get("architecture")
        valid_archs: list[str] = []
        if isinstance(arch, list):
            valid_archs = [str(a) for a in arch]
        elif isinstance(arch, str):
            valid_archs = [arch]
        raw_mods = data.get("modifiers")
        modifiers = [str(m) for m in raw_mods] if isinstance(raw_mods, list) else []
        raw_operands = data.get("operands")
        operands_list: list[list[str]] = []
        if isinstance(raw_operands, list) and raw_operands:
            if isinstance(raw_operands[0], list):
                operands_list = [
                    [str(s) for s in sig]
                    for sig in raw_operands
                    if isinstance(sig, list)
                ]
            else:
                operands_list = [
                    [
                        str(op.get("type") or op.get("role") or op.get("name") or "op")
                        if isinstance(op, dict)
                        else str(op)
                        for op in raw_operands
                    ]
                ]
        max_sig: list[str] = []
        for sig in operands_list:
            if len(sig) > len(max_sig):
                max_sig = sig
        params: list[GhostParam] = []
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
        raw_dom_meta = data.get("domain_metadata")
        domain_metadata: dict[str, JsonValue] = (
            dict(raw_dom_meta) if isinstance(raw_dom_meta, dict) else {}
        )
        domain_metadata.setdefault("architecture", arch)
        domain_metadata.setdefault("valid_architectures", list(valid_archs))
        domain_metadata.setdefault("modifiers", list(modifiers))
        domain_metadata.setdefault(
            "operand_signatures", [list(s) for s in operands_list]
        )
        isa_overloads: list[dict[str, JsonValue] | GhostRef] = []
        for sig_idx, sig in enumerate(operands_list):
            sig_params: list[GhostParam] = []
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
            schema_version="2.0.0",
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
        mlir_params: list[GhostParam] = []
        raw_mlir_ops = data.get("operands") or []
        assert isinstance(raw_mlir_ops, list)
        for op in raw_mlir_ops:
            op_name = (
                op
                if isinstance(op, str)
                else str(op.get("name", "arg") if isinstance(op, dict) else "arg")
            )
            mlir_params.append(
                ExtendedGhostParam(
                    name=op_name,
                    kind=ParameterKind.POSITIONAL_ONLY,
                    role=IRParameterRole.OPERAND,
                )
            )
        raw_mlir_attrs = data.get("attributes") or []
        assert isinstance(raw_mlir_attrs, list)
        for attr in raw_mlir_attrs:
            attr_name = (
                attr
                if isinstance(attr, str)
                else str(attr.get("name", "attr") if isinstance(attr, dict) else "attr")
            )
            mlir_params.append(
                ExtendedGhostParam(
                    name=attr_name,
                    kind=ParameterKind.KEYWORD_ONLY,
                    role=IRParameterRole.ATTRIBUTE,
                )
            )
        results: list[GhostResult] = []
        raw_mlir_res = data.get("results") or []
        assert isinstance(raw_mlir_res, list)
        for res in raw_mlir_res:
            results.append(
                GhostResult(
                    name=(
                        res
                        if isinstance(res, str)
                        else str(
                            res.get("name", "res") if isinstance(res, dict) else "res"
                        )
                    ),
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
        raw_doc = data.get("docstring") or data.get("description")
        raw_meta = data.get("domain_metadata")
        return GhostMlirRef(
            name=name,
            api_path=api_path,
            kind=str(data.get("kind") or "operation"),
            schema_version="2.0.0",
            params=mlir_params,
            returns=results,
            docstring=str(raw_doc) if raw_doc is not None else None,
            traits=(
                [str(t) for t in raw_traits] if isinstance(raw_traits, list) else None
            ),
            regions=dict(raw_regions) if isinstance(raw_regions, dict) else None,
            successors=(
                [str(s) for s in raw_successors]
                if isinstance(raw_successors, list)
                else None
            ),
            domain_metadata=dict(raw_meta) if isinstance(raw_meta, dict) else {},
            attributes=dict(raw_attrs) if isinstance(raw_attrs, dict) else None,
        )

    domain_type = data.get("domain_type")
    ref_cls: type[ExtendedGhostRef] = ExtendedGhostRef
    if domain_type == "python":
        ref_cls = GhostPythonRef
    elif domain_type in ("isa", "instruction"):
        ref_cls = GhostInstructionRef
    elif domain_type in ("mlir", "operation"):
        ref_cls = GhostOperationRef

    raw_version = data.get("schema_version", "2.0.0")
    if raw_version != "2.0.0":
        raise SnapshotSchemaError(
            f"Unsupported schema_version '{raw_version}'; expected '2.0.0'."
        )

    data_copy = dict(data)
    data_copy["schema_version"] = "2.0.0"
    if (
        isinstance(data_copy.get("name"), str)
        or isinstance(data_copy.get("api_path"), str)
    ) and "kind" not in data_copy:
        data_copy["kind"] = "function"

    try:
        if "params" in data_copy and isinstance(data_copy["params"], list):
            hydrated_params: list[GhostParam] = []
            for p in data_copy["params"]:
                if isinstance(p, dict):
                    p_dict = dict(p)
                    if (
                        "dtypes" in p_dict
                        and p_dict["dtypes"] is not None
                        and "allowed_dtypes" not in p_dict
                    ) or (
                        "rank" in p_dict
                        and p_dict["rank"] is not None
                        and "rank_constraint" not in p_dict
                    ):
                        raise SnapshotSchemaError(
                            "Legacy v1.x 'dtypes'/'rank' fields are not supported; use 'allowed_dtypes'/'rank_constraint'."
                        )
                    p_dict.pop("dtypes", None)
                    p_dict.pop("rank", None)
                    if "default" in p_dict and p_dict["default"] is not None:
                        if not isinstance(p_dict["default"], str):
                            p_dict["default"] = str(p_dict["default"])
                    hydrated_params.append(ExtendedGhostParam.model_validate(p_dict))
                else:
                    hydrated_params.append(ExtendedGhostParam.model_validate(p))
            data_copy["params"] = [
                hp.model_dump(exclude={"dtypes", "rank"}) for hp in hydrated_params
            ]

        for list_field, key_field in (
            ("operands", "constraint"),
            ("attributes", "type"),
            ("returns", "type"),
        ):
            raw_list = data_copy.get(list_field)
            if isinstance(raw_list, list):
                normalized_items: list[JsonValue] = []
                for item in raw_list:
                    if isinstance(item, dict) and not item.get(key_field):
                        item_copy = dict(item)
                        item_copy[key_field] = "Any"
                        normalized_items.append(item_copy)
                    else:
                        normalized_items.append(item)
                data_copy[list_field] = normalized_items

        validated = ref_cls.model_validate(data_copy)
    except (ValueError, TypeError) as exc:
        raise SnapshotSchemaError(f"Invalid GhostRef schema: {exc}") from exc
    return validated


__all__ = ["hydrate_ghost_ref"]
