"""Pure total schema hydration utilities for serialized GhostRef dictionaries."""

from __future__ import annotations

import typing

import typing_extensions
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
from ml_switcheroo_ir.typeclasses.algebra import NonEmpty
from ml_switcheroo_ir.typeclasses.outcome import (
    Err,
    Ok,
    Result,
    Valid,
    Validation,
    invalid_one,
    traverse_result,
    traverse_validation_ne,
)

from ml_ecosystem_snapshots.errors import SnapshotDomainError, SnapshotEnvelopeFault
from ml_ecosystem_snapshots.models import SCHEMA_VERSION, JsonValue, ScalarValue


def _build_isa_operand_params(sig: typing.Sequence[str]) -> list[GhostParam]:
    """Build positional `ExtendedGhostParam` entries for an ISA operand signature sequence.

    Args:
        sig: Ordered sequence of operand type strings (index 0 is destination, subsequent are sources).

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


def _extract_isa_operand_signatures(
    raw_operands: JsonValue | typing.Sequence[JsonValue | GhostParam] | None,
) -> list[list[str]]:
    """Normalize raw ISA `operands` payload into a list of operand signature lists.

    Args:
        raw_operands: Raw `operands` value from an ISA dictionary.

    Returns:
        List of string lists representing valid operand type combinations.
    """
    if not isinstance(raw_operands, (list, tuple)) or not raw_operands:
        return []
    if isinstance(raw_operands[0], (list, tuple)):
        return [[str(s) for s in sig] for sig in raw_operands if isinstance(sig, (list, tuple))]
    return [
        [
            str(op.get("type") or op.get("role") or op.get("name") or "op") if isinstance(op, dict) else str(op)
            for op in raw_operands
        ]
    ]


def _hydrate_isa_ref(
    data: typing.Mapping[str, JsonValue | typing.Sequence[JsonValue | GhostParam]],
) -> Result[SnapshotEnvelopeFault, GhostIsaRef]:
    """Hydrate a dictionary with ISA mnemonic and operand signatures into a `Result` of `GhostIsaRef`.

    Args:
        data: Raw dictionary describing an ISA instruction.

    Returns:
        `Ok(GhostIsaRef)` on success, or `Err(SnapshotEnvelopeFault)` if validation fails.
    """
    name = str(data.get("name") or data["mnemonic"])
    desc = str(data.get("description") or f"ISA {name} instruction.")
    arch = data.get("architecture")
    valid_archs: list[str] = (
        [str(a) for a in arch] if isinstance(arch, (list, tuple)) else ([arch] if isinstance(arch, str) else [])
    )
    raw_mods = data.get("modifiers")
    modifiers = [str(m) for m in raw_mods] if isinstance(raw_mods, (list, tuple)) else []
    operands_list = _extract_isa_operand_signatures(data.get("operands"))

    max_sig: list[str] = []
    for sig in operands_list:
        if len(sig) > len(max_sig):
            max_sig = sig
    params = _build_isa_operand_params(max_sig)

    raw_dom_meta = data.get("domain_metadata")
    domain_metadata: dict[str, JsonValue] = dict(raw_dom_meta) if isinstance(raw_dom_meta, dict) else {}
    domain_metadata.setdefault(
        "architecture",
        list(arch)
        if isinstance(arch, tuple)
        else (arch if isinstance(arch, (str, list)) or arch is None else str(arch)),
    )
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
    return Ok(
        GhostIsaRef.model_validate(
            {
                "name": name,
                "api_path": api_path,
                "kind": "function",
                "schema_version": SCHEMA_VERSION,
                "params": params,
                "docstring": desc,
                "environment_tags": valid_archs,
                "overloads": isa_overloads if isa_overloads else None,
                "domain_metadata": domain_metadata,
            }
        )
    )


def _hydrate_legacy_mlir_ref(
    data: typing.Mapping[str, JsonValue | typing.Sequence[JsonValue | GhostParam]],
) -> Result[SnapshotEnvelopeFault, GhostMlirRef]:
    """Hydrate a dictionary with MLIR `attributes` or `results` into a `Result` of `GhostMlirRef`.

    Args:
        data: Raw dictionary describing an MLIR operation without `domain_type == 'mlir'`.

    Returns:
        `Ok(GhostMlirRef)` on success, or `Err(SnapshotEnvelopeFault)` if validation fails.
    """
    api_path = str(data.get("api_path") or "mlir.op")
    name = str(data.get("name") or api_path.split(".")[-1])
    mlir_params: list[GhostParam] = []

    raw_ops = data.get("operands")
    raw_mlir_ops: list[JsonValue | GhostParam] = list(raw_ops) if isinstance(raw_ops, (list, tuple)) else []
    for op in raw_mlir_ops:
        op_name = op if isinstance(op, str) else str(op.get("name", "arg") if isinstance(op, dict) else "arg")
        mlir_params.append(
            ExtendedGhostParam(
                name=op_name,
                kind=ParameterKind.POSITIONAL_ONLY,
                role=IRParameterRole.OPERAND,
            )
        )

    raw_attrs_in = data.get("attributes")
    raw_mlir_attrs: list[JsonValue | GhostParam] = list(raw_attrs_in) if isinstance(raw_attrs_in, (list, tuple)) else []
    for attr in raw_mlir_attrs:
        attr_name = (
            attr if isinstance(attr, str) else str(attr.get("name", "attr") if isinstance(attr, dict) else "attr")
        )
        mlir_params.append(
            ExtendedGhostParam(
                name=attr_name,
                kind=ParameterKind.KEYWORD_ONLY,
                role=IRParameterRole.ATTRIBUTE,
            )
        )

    results: list[GhostResult] = []
    raw_res = data.get("results")
    raw_mlir_res: list[JsonValue | GhostParam] = list(raw_res) if isinstance(raw_res, (list, tuple)) else []
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
    return Ok(
        GhostMlirRef(
            name=name,
            api_path=api_path,
            kind=str(data.get("kind") or "operation"),
            schema_version=SCHEMA_VERSION,
            params=mlir_params,
            returns=results,
            docstring=str(raw_doc) if raw_doc is not None else None,
            traits=([str(t) for t in raw_traits] if isinstance(raw_traits, (list, tuple)) else None),
            regions=dict(raw_regions) if isinstance(raw_regions, dict) else None,
            successors=([str(s) for s in raw_successors] if isinstance(raw_successors, (list, tuple)) else None),
            domain_metadata=dict(raw_meta) if isinstance(raw_meta, dict) else {},
            attributes=dict(raw_attrs) if isinstance(raw_attrs, dict) else None,
        )
    )


def _normalize_single_v2_param(p: JsonValue | GhostParam) -> Result[SnapshotEnvelopeFault, ExtendedGhostParam]:
    """Validate and normalize a single v2.0.0 parameter dictionary or `GhostParam` totally.

    Args:
        p: Raw parameter dictionary or `GhostParam` model instance.

    Returns:
        `Ok(ExtendedGhostParam)` on success, or `Err(SnapshotEnvelopeFault)` on legacy v1.x or invalid schema.
    """
    try:
        if not isinstance(p, dict):
            return Ok(ExtendedGhostParam.model_validate(p))
        p_dict = dict(p)
        has_legacy_dtypes = "dtypes" in p_dict and p_dict["dtypes"] is not None and "allowed_dtypes" not in p_dict
        has_legacy_rank = "rank" in p_dict and p_dict["rank"] is not None and "rank_constraint" not in p_dict
        if has_legacy_dtypes or has_legacy_rank:
            return Err(
                SnapshotEnvelopeFault(
                    reason="Legacy v1.x 'dtypes'/'rank' fields are not supported; use 'allowed_dtypes'/'rank_constraint'."
                )
            )
        p_dict.pop("dtypes", None)
        p_dict.pop("rank", None)
        if "default" in p_dict and p_dict["default"] is not None and not isinstance(p_dict["default"], str):
            p_dict["default"] = str(p_dict["default"])
        return Ok(ExtendedGhostParam.model_validate(p_dict))
    except (ValueError, TypeError) as exc:
        return Err(SnapshotEnvelopeFault(reason=f"Invalid GhostRef parameter schema: {exc}"))


def _normalize_v2_typed_lists(
    raw_data: typing.Mapping[str, JsonValue | typing.Sequence[JsonValue | GhostParam]],
) -> dict[str, JsonValue | typing.Sequence[JsonValue | GhostParam]]:
    """Purely populate default constraint/type fields on `operands`, `attributes`, and `returns` lists.

    Args:
        raw_data: Input `GhostRef` dictionary mapping (never mutated).

    Returns:
        Fresh dictionary with normalized `operands`, `attributes`, and `returns` lists.
    """
    out: dict[str, JsonValue | typing.Sequence[JsonValue | GhostParam]] = dict(raw_data)
    for list_field, key_field, default_constraint in (
        ("operands", "constraint", "Tensor"),
        ("attributes", "type", "Attribute"),
        ("returns", "type", "Value"),
    ):
        raw_list = out.get(list_field)
        if isinstance(raw_list, (list, tuple)):
            normalized_items: list[JsonValue] = []
            for item in [entry for entry in raw_list if not isinstance(entry, GhostParam)]:
                if isinstance(item, dict) and not item.get(key_field):
                    normalized_items.append({**item, key_field: default_constraint})
                else:
                    normalized_items.append(item)
            out[list_field] = normalized_items
    return out


_DOMAIN_REF_CLASSES: typing.Final[dict[str, type[ExtendedGhostRef]]] = {
    "python": GhostPythonRef,
    "isa": GhostInstructionRef,
    "instruction": GhostInstructionRef,
    "mlir": GhostOperationRef,
    "operation": GhostOperationRef,
}


HydratableInput: typing_extensions.TypeAlias = (
    "typing.Mapping[str, JsonValue | typing.Sequence[JsonValue | GhostParam]] | ScalarValue"
)


def _hydrate_v2_ref_mapping(
    data: typing.Mapping[str, JsonValue | typing.Sequence[JsonValue | GhostParam]],
) -> Result[SnapshotEnvelopeFault, ExtendedGhostRef]:
    """Hydrate a validated v2.0.0 `GhostRef` dictionary mapping into `ExtendedGhostRef`.

    Args:
        data: Mapping representing a standard v2.0.0 `GhostRef`.

    Returns:
        `Ok(ExtendedGhostRef)` on valid schema, else `Err(SnapshotEnvelopeFault)`.
    """
    raw_domain_type = data.get("domain_type")
    ref_cls: type[ExtendedGhostRef] = _DOMAIN_REF_CLASSES.get(str(raw_domain_type), ExtendedGhostRef)

    raw_version = data.get("schema_version", SCHEMA_VERSION)
    if raw_version != SCHEMA_VERSION:
        return Err(
            SnapshotEnvelopeFault(
                reason=f"Unsupported schema_version '{raw_version}'; expected '{SCHEMA_VERSION}'.",
                schema_version=str(raw_version),
            )
        )

    base_copy: dict[str, JsonValue | typing.Sequence[JsonValue | GhostParam]] = dict(data)
    base_copy["schema_version"] = SCHEMA_VERSION
    if (
        isinstance(base_copy.get("name"), str) or isinstance(base_copy.get("api_path"), str)
    ) and "kind" not in base_copy:
        base_copy["kind"] = "function"

    raw_params = base_copy.get("params")
    if isinstance(raw_params, (list, tuple)):
        params_res = traverse_result(raw_params, _normalize_single_v2_param)
        if isinstance(params_res, Err):
            return params_res
        base_copy["params"] = [hp.model_dump(exclude={"dtypes", "rank"}) for hp in params_res.value]

    normalized_data = _normalize_v2_typed_lists(base_copy)
    try:
        return Ok(ref_cls.model_validate(normalized_data))
    except (ValueError, TypeError) as exc:
        return Err(SnapshotEnvelopeFault(reason=f"Invalid GhostRef schema: {exc}", schema_version=SCHEMA_VERSION))


def hydrate_ghost_ref(
    data: HydratableInput,
) -> Result[SnapshotEnvelopeFault, ExtendedGhostRef]:
    """Hydrate a dictionary payload totally into `Result[SnapshotEnvelopeFault, ExtendedGhostRef]`.

    Args:
        data: Dictionary representation of a ghost reference.

    Returns:
        `Ok(ExtendedGhostRef)` on valid schema, else `Err(SnapshotEnvelopeFault)`.
    """
    if not isinstance(data, typing.Mapping):
        return Err(SnapshotEnvelopeFault(reason="GhostRef payload must be a dictionary."))

    if "mnemonic" in data and ("name" not in data or "operands" in data):
        return _hydrate_isa_ref(data)

    if (
        ("attributes" in data or "results" in data)
        and ("name" not in data or "params" not in data)
        and data.get("domain_type") != "mlir"
    ):
        return _hydrate_legacy_mlir_ref(data)

    return _hydrate_v2_ref_mapping(data)


def _hydrate_one_validation(
    item: HydratableInput,
) -> Validation[NonEmpty[SnapshotDomainError], ExtendedGhostRef]:
    """Lift `hydrate_ghost_ref` into a `NonEmpty` error-accumulating `Validation` value.

    Args:
        item: Candidate raw dictionary payload.

    Returns:
        `Valid(ExtendedGhostRef)` or `Invalid(NonEmpty[SnapshotDomainError])`.
    """
    res = hydrate_ghost_ref(item)
    if isinstance(res, Ok):
        return Valid(res.value)
    return invalid_one(res.error)


def hydrate_ghost_refs(
    items: typing.Sequence[HydratableInput],
) -> Validation[NonEmpty[SnapshotDomainError], tuple[ExtendedGhostRef, ...]]:
    """Hydrate a sequence of raw payloads, accumulating all schema errors into `NonEmpty`.

    Args:
        items: Sequence of raw `GhostRef` dictionary payloads.

    Returns:
        `Valid(tuple[ExtendedGhostRef, ...])` when every item succeeds, or
        `Invalid(NonEmpty[SnapshotDomainError])` collecting every schema error.
    """
    return traverse_validation_ne(items, _hydrate_one_validation)


__all__ = ["HydratableInput", "hydrate_ghost_ref", "hydrate_ghost_refs"]
