"""Unified compiler IR grounding verifiers for MLIR and StableHLO dialects."""

from __future__ import annotations

import typing

from ml_switcheroo_ir.schema.ghost import (
    ExtendedGhostParam,
    ExtendedGhostRef,
    GhostMlirRef,
    GhostOperationRef,
    GhostParam,
)

from ml_ecosystem_snapshots.grounding.compiler_attrs import (
    _check_mlir_operand_types,
    _check_mlir_regions,
    _extract_mlir_operands_and_attrs,
    _extract_tensor_rank,
    _find_mlir_op_record,
    _infer_implicit_mlir_traits,
    _validate_mlir_structured_attributes,
)
from ml_ecosystem_snapshots.grounding.domain_ops import (
    validate_metal_op,
    validate_onnx_op,
    validate_wasm_op,
    validate_webgl_op,
    validate_wgsl_op,
)
from ml_ecosystem_snapshots.grounding.engine import GroundingEngine
from ml_ecosystem_snapshots.grounding.models import (
    DiagnosticSeverity,
    GroundingReport,
    IrCheckResult,
)
from ml_ecosystem_snapshots.models import JsonValue

extract_mlir_operands_and_attrs = _extract_mlir_operands_and_attrs
extract_tensor_rank = _extract_tensor_rank
infer_implicit_mlir_traits = _infer_implicit_mlir_traits
validate_mlir_structured_attributes = _validate_mlir_structured_attributes


def _ref_to_op_dict(ref: ExtendedGhostRef) -> dict[str, JsonValue]:
    """Convert an `ExtendedGhostRef` into a normalized dialect operation dictionary.

    Args:
        ref: Hydrated `ExtendedGhostRef` operation specification.

    Returns:
        Normalized operation dictionary with `operands`, `attributes`, and `traits`.
    """
    dumped: dict[str, JsonValue] = ref.model_dump()
    raw_operands = ref.operands if isinstance(ref, (GhostMlirRef, GhostOperationRef)) else None
    source_params: list[GhostParam] = (
        list(raw_operands)
        if raw_operands is not None
        else [
            p
            for p in ref.params
            if "ATTRIBUTE" not in str(p.role if isinstance(p, ExtendedGhostParam) else "") and p.annotation != "Region"
        ]
    )
    op_list: list[dict[str, JsonValue]] = [{"name": p.name, "type": str(p.annotation or "")} for p in source_params]
    if ref.has_varargs and op_list:
        op_list[0]["type"] = f"variadic<{op_list[0].get('type', '')}>"
    dumped["operands"] = list(op_list)
    return dumped


def _check_required_stablehlo_dim_attrs(
    full_op_name: str,
    structured_attributes: typing.Mapping[str, JsonValue] | None,
) -> list[tuple[str, str, str]]:
    """Check mandatory dimension_numbers attributes for StableHLO dot_general and convolution.

    Args:
        full_op_name: Qualified StableHLO operation name.
        structured_attributes: Mapping of attribute names to values.

    Returns:
        List of `(field, message, suggested_fix)` diagnostic tuples.
    """
    attrs = structured_attributes or {}
    diags: list[tuple[str, str, str]] = []
    if full_op_name == "stablehlo.dot_general":
        if "dot_dimension_numbers" not in attrs and "dimension_numbers" not in attrs:
            diags.append(
                (
                    "attributes.dot_dimension_numbers",
                    "Operation 'stablehlo.dot_general' requires 'dot_dimension_numbers' attribute.",
                    "dot_dimension_numbers",
                )
            )
    if full_op_name == "stablehlo.convolution":
        if "dimension_numbers" not in attrs and "conv_dimension_numbers" not in attrs:
            diags.append(
                (
                    "attributes.dimension_numbers",
                    "Operation 'stablehlo.convolution' requires 'dimension_numbers' attribute.",
                    "dimension_numbers",
                )
            )
    return diags


def _check_dialect_operands(
    target_op: str,
    operands: list[str | dict[str, JsonValue]],
    has_variadic: bool,
    operands_count: int | None,
    operand_types: list[str] | None,
    check_suffix_dtypes: bool,
) -> list[tuple[str, str, str | None]]:
    """Validate dialect operand counts and SSA type constraints.

    Args:
        target_op: Qualified operation name.
        operands: Expected operand specifications.
        has_variadic: Whether the operation accepts variadic operands.
        operands_count: Optional passed operand count.
        operand_types: Optional passed SSA operand type strings.
        check_suffix_dtypes: Whether to apply arithmetic suffix dtype checks.

    Returns:
        List of `(field, message, suggested_fix)` diagnostic tuples.
    """
    from ml_ecosystem_snapshots.frameworks.mlir import validate_mlir_type

    diags: list[tuple[str, str, str | None]] = []
    if operands_count is not None and operands and len(operands) != operands_count and not has_variadic:
        diags.append(
            (
                "operand_count",
                f"Operand count mismatch for '{target_op}': expected {len(operands)}, got {operands_count}",
                None,
            )
        )
    if operand_types is None:
        return diags
    if check_suffix_dtypes:
        for op_err in _check_mlir_operand_types(target_op, operands, operand_types):
            fld = (
                f"operands[{op_err.split()[1]}]"
                if op_err.startswith("Operand ") and op_err.split()[1].isdigit()
                else "operands"
            )
            diags.append((fld, op_err, None))
        return diags
    if operands and not has_variadic and len(operand_types) != len(operands):
        diags.append(
            (
                "operands",
                f"Operation '{target_op}' expects {len(operands)} operands, but received {len(operand_types)}.",
                None,
            )
        )
    for i in range(min(len(operands), len(operand_types))):
        op_info = operands[i]
        constraint = str(op_info.get("type", "") if isinstance(op_info, dict) else str(op_info))
        for te in validate_mlir_type(operand_types[i], constraint=constraint):
            diags.append((f"operands[{i}]", f"Operand {i} for '{target_op}': {te}", None))
    return diags


def _validate_dialect_operation(
    target_op: str,
    inst: dict[str, JsonValue],
    *,
    operands_count: int | None = None,
    attributes: list[str] | None = None,
    ssa_types: tuple[list[str] | None, list[str] | None] = (None, None),
    structured_attributes: typing.Mapping[str, JsonValue] | None = None,
    regions: typing.Mapping[str, JsonValue] | None = None,
    flags: tuple[bool, bool] = (False, True),
) -> tuple[
    list[str | dict[str, JsonValue]],
    list[str],
    list[tuple[str, str, str | None]],
]:
    """Shared dialect operation validator powering `validate_stablehlo_op`, `validate_mlir_op`, and `verify_mlir_op`.

    Args:
        target_op: Qualified operation name (e.g. `arith.addf` or `stablehlo.dot_general`).
        inst: Normalized operation dictionary from snapshot or `ExtendedGhostRef`.
        operands_count: Optional explicit operand count to check.
        attributes: Optional list of attribute names to check against `exp_attrs`.
        ssa_types: Tuple of `(operand_types, result_types)`.
        structured_attributes: Optional mapping of structured attribute values.
        regions: Optional mapping of region names to block_arguments and yield_types.
        flags: Tuple of `(enforce_stablehlo_req_attrs, check_suffix_dtypes)`.

    Returns:
        Tuple of `(expected_operands, expected_attributes, list_of_(field, message, suggested_fix))`.
    """
    from ml_ecosystem_snapshots.frameworks.mlir import validate_mlir_traits

    operand_types, result_types = ssa_types
    enforce_stablehlo_req_attrs, check_suffix_dtypes = flags
    diags: list[tuple[str, str, str | None]] = []
    operands, exp_attrs = _extract_mlir_operands_and_attrs(inst)
    has_variadic = any("variadic" in str(o.get("type", "") if isinstance(o, dict) else "").lower() for o in operands)

    if enforce_stablehlo_req_attrs:
        for fld, msg, fix in _check_required_stablehlo_dim_attrs(target_op, structured_attributes):
            diags.append((fld, msg, fix))

    diags.extend(
        _check_dialect_operands(
            target_op,
            operands,
            has_variadic,
            operands_count,
            operand_types,
            check_suffix_dtypes,
        )
    )

    for attr in attributes or []:
        if attr not in exp_attrs:
            diags.append(
                (
                    "attributes",
                    f"Attribute '{attr}' not recognized for '{target_op}'. Expected attributes: {exp_attrs}",
                    None,
                )
            )

    raw_meta = inst.get("domain_metadata")
    meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
    raw_traits = meta.get("traits") or inst.get("traits")
    base_traits = [str(t) for t in raw_traits] if isinstance(raw_traits, list) else []
    traits = _infer_implicit_mlir_traits(target_op, base_traits)
    attr_names = (
        attributes
        if attributes is not None
        else (list(structured_attributes.keys()) if structured_attributes else None)
    )

    for trait_err in validate_mlir_traits(
        op_name=target_op,
        traits=traits,
        operand_types=operand_types if operand_types else None,
        result_types=result_types,
        attributes=attr_names,
        structured_attributes=dict(structured_attributes) if structured_attributes else None,
    ):
        diags.append(("traits", trait_err, None))

    if structured_attributes:
        for attr_err in _validate_mlir_structured_attributes(
            structured_attributes=structured_attributes,
            operand_types=operand_types if operand_types else None,
            result_types=result_types,
        ):
            diags.append(("attributes", attr_err, None))

    for reg_name, reg_spec in (regions or {}).items():
        for reg_err in _check_mlir_regions(target_op, {reg_name: reg_spec}):
            diags.append((f"regions.{reg_name}", reg_err, None))

    return operands, exp_attrs, diags


def verify_mlir_op(
    op_name: str,
    snap: typing.Mapping[str, JsonValue],
    operands_count: int | None = None,
    attributes: list[str] | None = None,
    operand_types: list[str] | None = None,
    result_types: list[str] | None = None,
    structured_attributes: typing.Mapping[str, JsonValue] | None = None,
    regions: typing.Mapping[str, JsonValue] | None = None,
) -> IrCheckResult:
    """Verify an MLIR or StableHLO operation against a resolved dialect snapshot dictionary.

    Args:
        op_name: Qualified operation name (e.g. arith.addf or stablehlo.dot_general).
        snap: Resolved dialect snapshot dictionary.
        operands_count: Optional expected number of SSA operands.
        attributes: Optional list of attribute names to verify.
        operand_types: Optional list of SSA operand type strings.
        result_types: Optional list of SSA result type strings.
        structured_attributes: Optional dictionary of structured attribute values.
        regions: Optional dictionary of regions mapping region name to block_arguments and yield_types.

    Returns:
        Structured `IrCheckResult` report.
    """
    target_op = op_name.strip()
    inst = _find_mlir_op_record(dict(snap), target_op)
    if not inst:
        return IrCheckResult(
            is_valid=False,
            op_exists=False,
            errors=[f"Operation '{target_op}' does not exist in dialect snapshot."],
        )

    operands, exp_attrs, diags = _validate_dialect_operation(
        target_op=target_op,
        inst=inst,
        operands_count=operands_count,
        attributes=attributes,
        ssa_types=(operand_types, result_types),
        structured_attributes=structured_attributes,
        regions=regions,
        flags=(False, True),
    )
    return IrCheckResult(
        is_valid=len(diags) == 0,
        op_exists=True,
        expected_operands=operands,
        expected_attributes=exp_attrs,
        errors=[msg for _fld, msg, _fix in diags],
    )


def validate_stablehlo_op(
    op_name: str,
    operand_types: list[str],
    attributes: typing.Mapping[str, JsonValue],
    engine: GroundingEngine | None = None,
    result_types: list[str] | None = None,
    regions: typing.Mapping[str, JsonValue] | None = None,
) -> GroundingReport:
    """Verify a StableHLO operation against the official StableHLO specification.

    Args:
        op_name: Operation name (e.g., stablehlo.dot_general, dot_general, stablehlo.add).
        operand_types: Types of SSA input operands.
        attributes: Dictionary of operation attributes.
        engine: Optional `GroundingEngine` instance.
        result_types: Optional list of SSA result type strings.
        regions: Optional dictionary of regions mapping region name to block_arguments and yield_types.

    Returns:
        `GroundingReport` containing diagnostic results and grounding status.
    """
    eng = engine or GroundingEngine()
    full_op_name = op_name if op_name.startswith("stablehlo.") else f"stablehlo.{op_name}"
    report = GroundingReport(is_grounded=True, target="stablehlo", symbol=full_op_name)

    ref = eng.get_symbol("stablehlo", full_op_name)
    if not ref:
        suggested = eng.suggest_closest_symbol("stablehlo", full_op_name)
        report.add_diagnostic(
            field="operation",
            message=f"Unrecognized StableHLO operation '{full_op_name}'.",
            severity=DiagnosticSeverity.ERROR,
            suggested_fix=suggested,
        )
        return report

    report.matched_ref = ref
    _ops, _attrs, diags = _validate_dialect_operation(
        target_op=full_op_name,
        inst=_ref_to_op_dict(ref),
        ssa_types=(operand_types, result_types),
        structured_attributes=attributes,
        regions=regions,
        flags=(True, False),
    )
    for fld, msg, fix in diags:
        report.add_diagnostic(field=fld, message=msg, severity=DiagnosticSeverity.ERROR, suggested_fix=fix)
    return report


def validate_mlir_op(
    dialect: str,
    op_name: str,
    operand_count: int,
    attributes: typing.Mapping[str, JsonValue],
    engine: GroundingEngine | None = None,
    operand_types: list[str] | None = None,
    result_types: list[str] | None = None,
) -> GroundingReport:
    """Verify an MLIR operation against core MLIR TableGen definitions.

    Args:
        dialect: The MLIR dialect (e.g., arith, math, tensor, linalg, scf).
        op_name: The operation name or mnemonic (e.g., addf, arith.addf, matmul).
        operand_count: Number of SSA value operands passed to the operation.
        attributes: Dictionary of buildable MLIR attributes.
        engine: Optional `GroundingEngine` instance.
        operand_types: Optional list of SSA operand type strings.
        result_types: Optional list of SSA result type strings.

    Returns:
        `GroundingReport` containing diagnostic results and grounding status.
    """
    eng = engine or GroundingEngine()
    full_path = op_name if op_name.startswith(f"{dialect}.") else f"{dialect}.{op_name}"
    report = GroundingReport(is_grounded=True, target="mlir", symbol=full_path)

    ref = eng.get_symbol("mlir", full_path)
    if not ref:
        suggested = eng.suggest_closest_symbol("mlir", full_path)
        report.add_diagnostic(
            field="operation",
            message=f"Unrecognized MLIR operation '{full_path}'.",
            severity=DiagnosticSeverity.ERROR,
            suggested_fix=suggested,
        )
        return report

    report.matched_ref = ref
    _ops, _attrs, diags = _validate_dialect_operation(
        target_op=full_path,
        inst=_ref_to_op_dict(ref),
        operands_count=operand_count,
        ssa_types=(operand_types, result_types),
        structured_attributes=attributes,
        flags=(False, False),
    )
    for fld, msg, fix in diags:
        report.add_diagnostic(field=fld, message=msg, severity=DiagnosticSeverity.ERROR, suggested_fix=fix)
    return report


__all__ = [
    "_validate_dialect_operation",
    "extract_mlir_operands_and_attrs",
    "extract_tensor_rank",
    "infer_implicit_mlir_traits",
    "validate_metal_op",
    "validate_mlir_op",
    "validate_mlir_structured_attributes",
    "validate_onnx_op",
    "validate_stablehlo_op",
    "validate_wasm_op",
    "validate_webgl_op",
    "validate_wgsl_op",
    "verify_mlir_op",
]
