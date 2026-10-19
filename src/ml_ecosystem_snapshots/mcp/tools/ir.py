"""MCP tool module for compiler IR dialect verification (MLIR and StableHLO)."""

from __future__ import annotations

import re
import typing

from ml_ecosystem_snapshots.mcp.cache import SnapshotRepository, resolve_snapshot
from ml_ecosystem_snapshots.mcp.models import IrCheckResult
from ml_ecosystem_snapshots.models import JsonValue


def _extract_mlir_operands_and_attrs(
    inst: dict[str, JsonValue],
) -> tuple[list[str | dict[str, JsonValue]], list[str]]:
    """Extract expected SSA operands and attribute names from an operation record.

    Args:
        inst: Operation dictionary from a dialect snapshot.

    Returns:
        Tuple of (operands_list, attribute_names_list).
    """
    raw_operands = inst.get("operands")
    operands: list[str | dict[str, JsonValue]] = []
    if isinstance(raw_operands, list):
        operands = [dict(o) if isinstance(o, dict) else str(o) for o in raw_operands]
    else:
        raw_params = inst.get("params") or []
        assert isinstance(raw_params, list)
        operands = [
            dict(p)
            for p in raw_params
            if isinstance(p, dict)
            and str(p.get("kind")) in ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD")
            and str(p.get("role")) != "IRParameterRole.ATTRIBUTE"
            and p.get("annotation") != "Region"
        ]

    raw_attrs = inst.get("attributes")
    exp_attrs: list[str] = []
    if isinstance(raw_attrs, list):
        exp_attrs = [
            str(a.get("name") if isinstance(a, dict) else a) for a in raw_attrs
        ]
    if not exp_attrs:
        raw_params2 = inst.get("params")
        if isinstance(raw_params2, list):
            exp_attrs = [
                str(p.get("name"))
                for p in raw_params2
                if isinstance(p, dict)
                and (
                    str(p.get("kind")) == "KEYWORD_ONLY"
                    or str(p.get("role")) == "IRParameterRole.ATTRIBUTE"
                )
                and p.get("name")
            ]
    return operands, exp_attrs


def _infer_implicit_mlir_traits(target_op: str, base_traits: list[str]) -> list[str]:
    """Augment operation traits with implicit elementwise/type traits for core ops.

    Args:
        target_op: Qualified operation name (e.g. 'arith.addf', 'stablehlo.add').
        base_traits: Explicit trait list from the operation record.

    Returns:
        Augmented list of trait strings.
    """
    traits = list(base_traits)
    is_core_elementwise = (
        target_op.startswith("arith.")
        and any(
            target_op.endswith(s)
            for s in (
                "addf",
                "subf",
                "mulf",
                "divf",
                "addi",
                "subi",
                "muli",
                "remf",
                "remsi",
                "remui",
            )
        )
    ) or (
        target_op.startswith("stablehlo.")
        and any(
            target_op.endswith(s)
            for s in ("add", "subtract", "multiply", "divide", "maximum", "minimum")
        )
    )
    if is_core_elementwise:
        if "SameOperandsAndResultType" not in traits:
            traits.append("SameOperandsAndResultType")
        if "Elementwise" not in traits:
            traits.append("Elementwise")
    return traits


def _extract_tensor_rank(type_str: str) -> int | None:
    """Extract the rank of an MLIR tensor type string such as 'tensor<128x64xf32>'.

    Args:
        type_str: MLIR tensor type string.

    Returns:
        Integer tensor rank, or None if not a ranked tensor string.
    """
    match = re.search(r"tensor<([^>]+)>", type_str)
    return len(match.group(1).split("x")[:-1]) if match else None


def _validate_mlir_structured_attributes(
    structured_attributes: typing.Mapping[str, object],
    operand_types: list[str] | None = None,
    result_types: list[str] | None = None,
) -> list[str]:
    """Validate structured MLIR / StableHLO attributes (dimension numbers, enums, broadcast).

    Args:
        structured_attributes: Dictionary of attribute names to structured values.
        operand_types: Optional list of SSA operand type strings.
        result_types: Optional list of SSA result type strings.

    Returns:
        List of validation error messages.
    """
    from ml_ecosystem_snapshots.frameworks.stablehlo import (
        validate_broadcast_in_dim,
        validate_conv_dimension_numbers,
        validate_dot_dimension_numbers,
        validate_gather_dimension_numbers,
        validate_scatter_dimension_numbers,
    )

    errors: list[str] = []
    for attr_key, attr_val in structured_attributes.items():
        k_low = attr_key.lower().replace("_", "")
        if "comparison" in k_low or "direction" in k_low:
            if str(attr_val).upper() not in ["EQ", "NE", "GE", "GT", "LE", "LT"]:
                errors.append(
                    f"Invalid ComparisonDirectionAttr '{attr_val}': must be one of ['EQ', 'NE', 'GE', 'GT', 'LE', 'LT']"
                )
        elif "precision" in k_low:
            if str(attr_val).upper() not in ["DEFAULT", "HIGH", "HIGHEST"]:
                errors.append(
                    f"Invalid PrecisionAttr '{attr_val}': must be one of ['DEFAULT', 'HIGH', 'HIGHEST']"
                )
        elif "dotdimensionnumbers" in k_low or "dotdimension" in k_low:
            if isinstance(attr_val, dict):
                if attr_val:
                    req_keys = [
                        "lhs_batch_dimensions",
                        "rhs_batch_dimensions",
                        "lhs_contracting_dimensions",
                        "rhs_contracting_dimensions",
                    ]
                    missing = [rk for rk in req_keys if rk not in attr_val]
                    if missing:
                        errors.append(
                            f"DotDimensionNumbersAttr missing required fields: {missing}"
                        )
                    else:
                        lhs_rk = (
                            _extract_tensor_rank(operand_types[0])
                            if operand_types and len(operand_types) >= 2
                            else None
                        )
                        rhs_rk = (
                            _extract_tensor_rank(operand_types[1])
                            if operand_types and len(operand_types) >= 2
                            else None
                        )
                        errors.extend(
                            validate_dot_dimension_numbers(
                                attr_val, lhs_rank=lhs_rk, rhs_rank=rhs_rk
                            )
                        )
            else:
                errors.append("DotDimensionNumbersAttr must be structured dictionary")
        elif "convdimensionnumbers" in k_low or k_low == "dimensionnumbers":
            if isinstance(attr_val, dict):
                if attr_val:
                    req_keys = [
                        "input_batch_dimension",
                        "input_feature_dimension",
                        "input_spatial_dimensions",
                        "kernel_input_feature_dimension",
                        "kernel_output_feature_dimension",
                        "kernel_spatial_dimensions",
                        "output_batch_dimension",
                        "output_feature_dimension",
                        "output_spatial_dimensions",
                    ]
                    missing = [rk for rk in req_keys if rk not in attr_val]
                    if missing:
                        errors.append(
                            f"ConvDimensionNumbersAttr missing required fields: {missing}"
                        )
                    else:
                        in_rk = (
                            _extract_tensor_rank(operand_types[0])
                            if operand_types and len(operand_types) >= 2
                            else None
                        )
                        k_rk = (
                            _extract_tensor_rank(operand_types[1])
                            if operand_types and len(operand_types) >= 2
                            else None
                        )
                        out_rk = (
                            _extract_tensor_rank(result_types[0])
                            if result_types and len(result_types) >= 1
                            else None
                        )
                        errors.extend(
                            validate_conv_dimension_numbers(
                                attr_val,
                                input_rank=in_rk,
                                kernel_rank=k_rk,
                                output_rank=out_rk,
                            )
                        )
            else:
                errors.append("ConvDimensionNumbersAttr must be structured dictionary")
        elif "scatterdimensionnumbers" in k_low:
            if isinstance(attr_val, dict):
                req_keys = [
                    "update_window_dims",
                    "inserted_window_dims",
                    "scatter_dims_to_operand_dims",
                    "index_vector_dim",
                ]
                missing = [rk for rk in req_keys if rk not in attr_val]
                if missing:
                    errors.append(
                        f"ScatterDimensionNumbersAttr missing required fields: {missing}"
                    )
                else:
                    errors.extend(validate_scatter_dimension_numbers(attr_val))
            else:
                errors.append(
                    "ScatterDimensionNumbersAttr must be structured dictionary"
                )
        elif "gatherdimensionnumbers" in k_low:
            if isinstance(attr_val, dict):
                req_keys = [
                    "offset_dims",
                    "collapsed_slice_dims",
                    "start_index_map",
                    "index_vector_dim",
                ]
                missing = [rk for rk in req_keys if rk not in attr_val]
                if missing:
                    errors.append(
                        f"GatherDimensionNumbersAttr missing required fields: {missing}"
                    )
                else:
                    errors.extend(validate_gather_dimension_numbers(attr_val))
            else:
                errors.append(
                    "GatherDimensionNumbersAttr must be structured dictionary"
                )
        elif "broadcastdimensions" in k_low or "broadcast_dimensions" in attr_key:
            if isinstance(attr_val, list):
                op_shape = structured_attributes.get("operand_shape")
                res_shape = structured_attributes.get("result_shape")
                if isinstance(op_shape, list) and isinstance(res_shape, list):
                    op_ints = [int(x) for x in op_shape if isinstance(x, int)]
                    res_ints = [int(x) for x in res_shape if isinstance(x, int)]
                    dim_ints = [int(x) for x in attr_val if isinstance(x, int)]
                    errors.extend(
                        validate_broadcast_in_dim(op_ints, res_ints, dim_ints)
                    )
    return errors


def check_mlir_op(
    op_name: str,
    operands_count: int | None = None,
    attributes: list[str] | None = None,
    operand_types: list[str] | None = None,
    result_types: list[str] | None = None,
    structured_attributes: typing.Mapping[str, object] | None = None,
    regions: typing.Mapping[str, object] | None = None,
    repository: SnapshotRepository | None = None,
) -> IrCheckResult:
    """Validate whether an MLIR or StableHLO operation exists with expected signature.

    Args:
        op_name: Qualified operation name (e.g. 'arith.addf' or 'stablehlo.dot_general').
        operands_count: Optional expected number of SSA operands.
        attributes: Optional list of attribute names to verify.
        operand_types: Optional list of SSA operand type strings to verify against ODS type constraints.
        result_types: Optional list of SSA result type strings.
        structured_attributes: Optional dictionary of structured attribute values to validate.
        regions: Optional dictionary of regions mapping region name to block_arguments and yield_types.
        repository: Optional `SnapshotRepository` instance for snapshot lookup.

    Returns:
        `IrCheckResult` validation report model with is_valid, expected_operands, and errors.
    """
    framework = "stablehlo" if op_name.startswith("stablehlo.") else "mlir"
    snap = resolve_snapshot(framework, repository=repository)
    inst: dict[str, JsonValue] | None = None
    target_op = op_name.strip()
    raw_cats = snap.get("categories") or {}
    assert isinstance(raw_cats, dict)
    for _cat_name, items in raw_cats.items():
        assert isinstance(items, list)
        for item in items:
            assert isinstance(item, dict)
            path = item.get("api_path") or item.get("name")
            if path == target_op:
                inst = dict(item)
                break
        if inst:
            break

    if not inst:
        return IrCheckResult(
            is_valid=False,
            op_exists=False,
            errors=[f"Operation '{target_op}' does not exist in dialect snapshot."],
        )

    errors: list[str] = []
    from ml_ecosystem_snapshots.frameworks.mlir import (
        validate_mlir_traits,
        validate_mlir_type,
    )
    from ml_ecosystem_snapshots.frameworks.stablehlo import validate_stablehlo_region

    operands, exp_attrs = _extract_mlir_operands_and_attrs(inst)

    if operands_count is not None and len(operands) != operands_count:
        has_variadic = any(
            "variadic" in str(o.get("type", "") if isinstance(o, dict) else "").lower()
            for o in operands
        )
        if not has_variadic:
            errors.append(
                f"Operand count mismatch for '{target_op}': expected {len(operands)}, got {operands_count}"
            )

    if attributes:
        for attr in attributes:
            if attr not in exp_attrs:
                errors.append(
                    f"Attribute '{attr}' not recognized for '{target_op}'. Expected attributes: {exp_attrs}"
                )

    raw_meta = inst.get("domain_metadata")
    meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
    raw_traits = meta.get("traits") or inst.get("traits")
    base_traits = [str(t) for t in raw_traits] if isinstance(raw_traits, list) else []
    traits = _infer_implicit_mlir_traits(target_op, base_traits)

    trait_errs = validate_mlir_traits(
        op_name=target_op,
        traits=traits,
        operand_types=operand_types,
        result_types=result_types,
        attributes=attributes,
        structured_attributes=structured_attributes,
    )
    errors.extend(trait_errs)

    if operand_types is not None:
        if len(operands) != len(operand_types):
            has_variadic = any(
                "variadic"
                in str(o.get("type", "") if isinstance(o, dict) else "").lower()
                for o in operands
            )
            if not has_variadic:
                errors.append(
                    f"Operand types count mismatch for '{target_op}': expected {len(operands)}, got {len(operand_types)}"
                )

        for i, (op_info, actual_type) in enumerate(zip(operands, operand_types)):
            constraint = str(
                op_info.get("type", "") if isinstance(op_info, dict) else str(op_info)
            )
            type_errs = validate_mlir_type(actual_type, constraint=constraint)
            for te in type_errs:
                errors.append(f"Operand {i} for '{target_op}': {te}")

        if target_op.endswith("f") or target_op in (
            "math.sin",
            "math.cos",
            "math.tan",
            "math.exp",
            "math.log",
        ):
            for i, act in enumerate(operand_types):
                act_low = act.lower()
                if any(k in act_low for k in ("i8", "i16", "i32", "i64", "int")):
                    errors.append(
                        f"Operand {i} type mismatch for '{target_op}': operation requires floating-point operands, got '{act}'"
                    )
        elif (
            target_op.endswith("i")
            or target_op.endswith("si")
            or target_op.endswith("ui")
        ):
            for i, act in enumerate(operand_types):
                act_low = act.lower()
                if any(k in act_low for k in ("f16", "bf16", "f32", "f64", "float")):
                    errors.append(
                        f"Operand {i} type mismatch for '{target_op}': operation requires integer operands, got '{act}'"
                    )

    if structured_attributes:
        errors.extend(
            _validate_mlir_structured_attributes(
                structured_attributes=structured_attributes,
                operand_types=operand_types,
                result_types=result_types,
            )
        )

    if regions:
        for reg_name, reg_spec in regions.items():
            if isinstance(reg_spec, dict):
                raw_b_args = reg_spec.get("block_arguments") or reg_spec.get(
                    "block_args"
                )
                b_args = (
                    [str(a) for a in raw_b_args] if isinstance(raw_b_args, list) else []
                )
                raw_y_types = reg_spec.get("yield_types")
                y_types = (
                    [str(t) for t in raw_y_types]
                    if isinstance(raw_y_types, list)
                    else []
                )
                errors.extend(
                    validate_stablehlo_region(target_op, reg_name, b_args, y_types)
                )

    return IrCheckResult(
        is_valid=len(errors) == 0,
        op_exists=True,
        expected_operands=operands,
        expected_attributes=exp_attrs,
        errors=errors,
    )


def check_stablehlo_op(
    op_name: str,
    operands_count: int | None = None,
    attributes: list[str] | None = None,
    operand_types: list[str] | None = None,
    result_types: list[str] | None = None,
    structured_attributes: typing.Mapping[str, object] | None = None,
    regions: typing.Mapping[str, object] | None = None,
    repository: SnapshotRepository | None = None,
) -> IrCheckResult:
    """Validate a StableHLO operation against formal verification rules and schemas.

    Args:
        op_name: Qualified operation name (e.g. 'stablehlo.dot_general', 'stablehlo.reduce').
        operands_count: Optional expected number of SSA operands.
        attributes: Optional list of attribute names to verify.
        operand_types: Optional list of SSA operand type strings.
        result_types: Optional list of SSA result type strings.
        structured_attributes: Optional dictionary of structured attribute values.
        regions: Optional dictionary of regions mapping region name to block_arguments and yield_types.
        repository: Optional `SnapshotRepository` instance for snapshot lookup.

    Returns:
        `IrCheckResult` validation report model with is_valid, expected_operands, and errors.
    """
    clean_name = op_name if op_name.startswith("stablehlo.") else f"stablehlo.{op_name}"
    return check_mlir_op(
        op_name=clean_name,
        operands_count=operands_count,
        attributes=attributes,
        operand_types=operand_types,
        result_types=result_types,
        structured_attributes=structured_attributes,
        regions=regions,
        repository=repository,
    )


__all__ = [
    "_extract_mlir_operands_and_attrs",
    "_extract_tensor_rank",
    "_infer_implicit_mlir_traits",
    "_validate_mlir_structured_attributes",
    "check_mlir_op",
    "check_stablehlo_op",
]
