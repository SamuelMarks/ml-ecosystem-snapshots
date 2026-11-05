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


def _build_isa_operand_params(sig: list[str]) -> list[GhostParam]:
    """Build positional `ExtendedGhostParam` entries for an ISA operand signature list.

    Args:
        sig: Ordered list of operand type strings (index 0 is destination, subsequent are sources).

    Returns:
        List of `GhostParam` instances representing the instruction operands.
    """
    params: list[GhostParam] = []
    for idx, op_type in enumerate(sig):
        role = "dst" if idx == 0 else f"src{idx - 1}"
        direction = OperandDirection.WRITE if idx == 0 else OperandDirection.READ
        params.append(
            ExtendedGhostParam(
                name=f"op{idx}",
                kind=ParameterKind.POSITIONAL_ONLY,
                annotation=str(op_type),
                standardized_name=role,
                direction=direction,
                role=IRParameterRole.OPERAND,
            )
        )
    return params


def _extract_isa_operand_signatures(raw_operands: JsonValue | typing.Sequence[JsonValue | GhostParam] | None) -> list[list[str]]:
    """Normalize raw ISA `operands` payload into a list of operand signature lists.

    Args:
        raw_operands: Raw `operands` value from an ISA dictionary.

    Returns:
        List of string lists representing valid operand type combinations.
    """
    if not isinstance(raw_operands, list) or not raw_operands:
        return []
    if isinstance(raw_operands[0], list):
        return [[str(s) for s in sig] for sig in raw_operands if isinstance(sig, list)]
    return [[str(op.get("type") or op.get("role") or op.get("name") or "op") if isinstance(op, dict) else str(op) for op in raw_operands]]


def _hydrate_isa_ref(
    data: typing.Mapping[str, JsonValue | typing.Sequence[JsonValue | GhostParam]],
) -> GhostIsaRef:
    """Hydrate a dictionary with ISA mnemonic and operand signatures into a `GhostIsaRef`.

    Args:
        data: Raw dictionary describing an ISA instruction.

    Returns:
        Hydrated `GhostIsaRef` instance.
    """
    name = str(data.get("name") or data["mnemonic"])
    desc = str(data.get("description") or f"ISA {name} instruction.")
    arch = data.get("architecture")
    valid_archs: list[str] = [str(a) for a in arch] if isinstance(arch, list) else ([arch] if isinstance(arch, str) else [])
    raw_mods = data.get("modifiers")
    modifiers = [str(m) for m in raw_mods] if isinstance(raw_mods, list) else []
    operands_list = _extract_isa_operand_signatures(data.get("operands"))

    max_sig: list[str] = []
    for sig in operands_list:
        if len(sig) > len(max_sig):
            max_sig = sig
    params = _build_isa_operand_params(max_sig)

    raw_dom_meta = data.get("domain_metadata")
    domain_metadata: dict[str, JsonValue] = dict(raw_dom_meta) if isinstance(raw_dom_meta, dict) else {}
    domain_metadata.setdefault("architecture", arch if isinstance(arch, (str, list)) or arch is None else str(arch))
    domain_metadata.setdefault("valid_architectures", list(valid_archs))
    domain_metadata.setdefault("modifiers", list(modifiers))
    domain_metadata.setdefault("operand_signatures", [list(s) for s in operands_list])

    api_path = str(data.get("api_path") or f"isa.inst.{name}")
    isa_overloads: list[dict[str, JsonValue] | GhostRef] = [
        ExtendedGhostRef(
            name=name,
            api_path=api_path,
            kind="function",
            params=_build_isa_operand_params(sig),
            docstring=f"Overload {sig_idx} for {name}: {', '.join(str(s) for s in sig)}",
            environment_tags=valid_archs,
            domain_metadata=domain_metadata,
        )
        for sig_idx, sig in enumerate(operands_list)
    ]
    return GhostIsaRef(
        name=name,
        api_path=api_path,
        kind=str(data.get("kind") or "function"),
        schema_version="2.0.0",
        params=params,
        docstring=desc,
        environment_tags=valid_archs,
        overloads=isa_overloads if isa_overloads else None,
        domain_metadata=domain_metadata,
    )


def _hydrate_legacy_mlir_ref(
    data: typing.Mapping[str, JsonValue | typing.Sequence[JsonValue | GhostParam]],
) -> GhostMlirRef:
    """Hydrate a dictionary with MLIR `attributes` or `results` into a `GhostMlirRef`.

    Args:
        data: Raw dictionary describing an MLIR operation without `domain_type == 'mlir'`.

    Returns:
        Hydrated `GhostMlirRef` instance.
    """
    api_path = str(data.get("api_path") or "mlir.op")
    name = str(data.get("name") or api_path.split(".")[-1])
    mlir_params: list[GhostParam] = []

    raw_mlir_ops = data.get("operands") or []
    assert isinstance(raw_mlir_ops, list)
    for op in raw_mlir_ops:
        op_name = op if isinstance(op, str) else str(op.get("name", "arg") if isinstance(op, dict) else "arg")
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
        attr_name = attr if isinstance(attr, str) else str(attr.get("name", "attr") if isinstance(attr, dict) else "attr")
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
                name=(res if isinstance(res, str) else str(res.get("name", "res") if isinstance(res, dict) else "res")),
                type=(str(res.get("type", "Value")) if isinstance(res, dict) else "Value"),
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
        traits=([str(t) for t in raw_traits] if isinstance(raw_traits, list) else None),
        regions=dict(raw_regions) if isinstance(raw_regions, dict) else None,
        successors=([str(s) for s in raw_successors] if isinstance(raw_successors, list) else None),
        domain_metadata=dict(raw_meta) if isinstance(raw_meta, dict) else {},
        attributes=dict(raw_attrs) if isinstance(raw_attrs, dict) else None,
    )


def _normalize_single_v2_param(p: JsonValue | GhostParam) -> ExtendedGhostParam:
    """Validate and normalize a single v2.0.0 parameter dictionary or `GhostParam` object.

    Args:
        p: Raw parameter dictionary or `GhostParam` model instance.

    Returns:
        Validated `ExtendedGhostParam` instance.

    Raises:
        SnapshotSchemaError: If legacy v1.x `dtypes` or `rank` fields are present without v2 equivalents.
    """
    if not isinstance(p, dict):
        return ExtendedGhostParam.model_validate(p)
    p_dict = dict(p)
    has_legacy_dtypes = "dtypes" in p_dict and p_dict["dtypes"] is not None and "allowed_dtypes" not in p_dict
    has_legacy_rank = "rank" in p_dict and p_dict["rank"] is not None and "rank_constraint" not in p_dict
    if has_legacy_dtypes or has_legacy_rank:
        raise SnapshotSchemaError("Legacy v1.x 'dtypes'/'rank' fields are not supported; use 'allowed_dtypes'/'rank_constraint'.")
    p_dict.pop("dtypes", None)
    p_dict.pop("rank", None)
    if "default" in p_dict and p_dict["default"] is not None and not isinstance(p_dict["default"], str):
        p_dict["default"] = str(p_dict["default"])
    return ExtendedGhostParam.model_validate(p_dict)


def _normalize_v2_typed_lists(
    data_copy: dict[str, JsonValue | typing.Sequence[JsonValue | GhostParam]],
) -> None:
    """Populate default `'Any'` constraint/type fields on `operands`, `attributes`, and `returns` lists.

    Args:
        data_copy: Mutable shallow copy of the GhostRef dictionary being hydrated.
    """
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
                    assert not isinstance(item, GhostParam)
                    normalized_items.append(item)
            data_copy[list_field] = normalized_items


_DOMAIN_REF_CLASSES: dict[str, type[ExtendedGhostRef]] = {
    "python": GhostPythonRef,
    "isa": GhostInstructionRef,
    "instruction": GhostInstructionRef,
    "mlir": GhostOperationRef,
    "operation": GhostOperationRef,
}


def hydrate_ghost_ref(
    data: (typing.Mapping[str, JsonValue | typing.Sequence[JsonValue | GhostParam]] | ScalarValue),
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
        return _hydrate_isa_ref(data)

    if ("attributes" in data or "results" in data) and ("name" not in data or "params" not in data) and data.get("domain_type") != "mlir":
        return _hydrate_legacy_mlir_ref(data)

    raw_domain_type = data.get("domain_type")
    ref_cls: type[ExtendedGhostRef] = _DOMAIN_REF_CLASSES.get(str(raw_domain_type), ExtendedGhostRef)

    raw_version = data.get("schema_version", "2.0.0")
    if raw_version != "2.0.0":
        raise SnapshotSchemaError(f"Unsupported schema_version '{raw_version}'; expected '2.0.0'.")

    data_copy: dict[str, JsonValue | typing.Sequence[JsonValue | GhostParam]] = dict(data)
    data_copy["schema_version"] = "2.0.0"
    if (isinstance(data_copy.get("name"), str) or isinstance(data_copy.get("api_path"), str)) and "kind" not in data_copy:
        data_copy["kind"] = "function"

    try:
        raw_params = data_copy.get("params")
        if isinstance(raw_params, list):
            hydrated_params = [_normalize_single_v2_param(p) for p in raw_params]
            data_copy["params"] = [hp.model_dump(exclude={"dtypes", "rank"}) for hp in hydrated_params]

        _normalize_v2_typed_lists(data_copy)
        validated = ref_cls.model_validate(data_copy)
    except (ValueError, TypeError) as exc:
        raise SnapshotSchemaError(f"Invalid GhostRef schema: {exc}") from exc
    return validated


__all__ = ["hydrate_ghost_ref"]
