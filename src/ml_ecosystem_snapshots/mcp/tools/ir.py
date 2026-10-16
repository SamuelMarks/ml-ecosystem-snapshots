"""MCP tool module for compiler IR dialect verification (MLIR and StableHLO)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ml_ecosystem_snapshots.mcp.cache import get_framework_snapshot


def check_mlir_op(
    op_name: str,
    operands_count: Optional[int] = None,
    attributes: Optional[List[str]] = None,
    operand_types: Optional[List[str]] = None,
    result_types: Optional[List[str]] = None,
    structured_attributes: Optional[Dict[str, Any]] = None,
    regions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate whether an MLIR or StableHLO operation exists with expected signature.

    Args:
        op_name: Qualified operation name (e.g. 'arith.addf' or 'stablehlo.dot_general').
        operands_count: Optional expected number of SSA operands.
        attributes: Optional list of attribute names to verify.
        operand_types: Optional list of SSA operand type strings to verify against ODS type constraints.
        result_types: Optional list of SSA result type strings.
        structured_attributes: Optional dictionary of structured attribute values to validate.
        regions: Optional dictionary of regions mapping region name to block_arguments and yield_types.

    Returns:
        Validation report dictionary with is_valid, expected_operands, and errors.
    """
    import ml_ecosystem_snapshots.mcp_server as _mcp_server

    active_get_snap = getattr(
        _mcp_server, "get_framework_snapshot", get_framework_snapshot
    )
    framework = "stablehlo" if op_name.startswith("stablehlo.") else "mlir"
    snap = active_get_snap(framework)
    inst = None
    target_op = op_name.strip()
    for _cat_name, items in snap.get("categories", {}).items():
        for item in items:
            path = item.get("api_path") or item.get("name")
            if path == target_op:
                inst = item
                break
        if inst:
            break

    if not inst:
        return {
            "is_valid": False,
            "op_exists": False,
            "errors": [f"Operation '{target_op}' does not exist in dialect snapshot."],
        }

    errors: List[str] = []
    from ml_ecosystem_snapshots.frameworks.mlir import (
        validate_mlir_traits,
        validate_mlir_type,
    )
    from ml_ecosystem_snapshots.frameworks.stablehlo import (
        validate_broadcast_in_dim,
        validate_conv_dimension_numbers,
        validate_dot_dimension_numbers,
        validate_gather_dimension_numbers,
        validate_scatter_dimension_numbers,
        validate_stablehlo_region,
    )

    operands = inst.get("operands")
    if operands is None:
        operands = [
            p
            for p in inst.get("params", [])
            if str(p.get("kind")) in ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD")
            and str(p.get("role")) != "IRParameterRole.ATTRIBUTE"
            and p.get("annotation") != "Region"
        ]
    exp_attrs = [
        a.get("name") if isinstance(a, dict) else str(a)
        for a in (inst.get("attributes") or [])
    ]
    if not exp_attrs:
        exp_attrs = [
            p.get("name")
            for p in inst.get("params", [])
            if str(p.get("kind")) == "KEYWORD_ONLY"
            or str(p.get("role")) == "IRParameterRole.ATTRIBUTE"
        ]

    if operands_count is not None and len(operands) != operands_count:
        has_variadic = any(
            "variadic" in (o.get("type", "") if isinstance(o, dict) else "").lower()
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

    meta = inst.get("domain_metadata") or {}
    traits = list(meta.get("traits") or inst.get("traits") or [])

    if target_op.startswith("arith.") and any(
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
    ):
        if "SameOperandsAndResultType" not in traits:
            traits.append("SameOperandsAndResultType")
        if "Elementwise" not in traits:
            traits.append("Elementwise")
    elif target_op.startswith("stablehlo.") and any(
        target_op.endswith(s)
        for s in ("add", "subtract", "multiply", "divide", "maximum", "minimum")
    ):
        if "SameOperandsAndResultType" not in traits:
            traits.append("SameOperandsAndResultType")
        if "Elementwise" not in traits:
            traits.append("Elementwise")

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
                "variadic" in (o.get("type", "") if isinstance(o, dict) else "").lower()
                for o in operands
            )
            if not has_variadic:
                errors.append(
                    f"Operand types count mismatch for '{target_op}': expected {len(operands)}, got {len(operand_types)}"
                )

        for i, (op_info, actual_type) in enumerate(zip(operands, operand_types)):
            constraint = (
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
                        lhs_rk = None
                        rhs_rk = None
                        if operand_types and len(operand_types) >= 2:
                            m_l = re.search(r"tensor<([^>]+)>", operand_types[0])
                            m_r = re.search(r"tensor<([^>]+)>", operand_types[1])
                            if m_l:
                                lhs_rk = len(m_l.group(1).split("x")[:-1])
                            if m_r:
                                rhs_rk = len(m_r.group(1).split("x")[:-1])
                        errors.extend(
                            validate_dot_dimension_numbers(
                                attr_val, lhs_rank=lhs_rk, rhs_rank=rhs_rk
                            )
                        )
                else:
                    errors.append(
                        "DotDimensionNumbersAttr must be structured dictionary"
                    )
            elif "convdimensionnumbers" in k_low or k_low == "dimensionnumbers":
                if isinstance(attr_val, dict):
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
                        in_rk = None
                        k_rk = None
                        out_rk = None
                        if operand_types and len(operand_types) >= 2:
                            m_in = re.search(r"tensor<([^>]+)>", operand_types[0])
                            m_k = re.search(r"tensor<([^>]+)>", operand_types[1])
                            if m_in:
                                in_rk = len(m_in.group(1).split("x")[:-1])
                            if m_k:
                                k_rk = len(m_k.group(1).split("x")[:-1])
                        if result_types and len(result_types) >= 1:
                            m_out = re.search(r"tensor<([^>]+)>", result_types[0])
                            if m_out:
                                out_rk = len(m_out.group(1).split("x")[:-1])
                        errors.extend(
                            validate_conv_dimension_numbers(
                                attr_val,
                                input_rank=in_rk,
                                kernel_rank=k_rk,
                                output_rank=out_rk,
                            )
                        )
                else:
                    errors.append(
                        "ConvDimensionNumbersAttr must be structured dictionary"
                    )
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
                    if op_shape is not None and res_shape is not None:
                        errors.extend(
                            validate_broadcast_in_dim(op_shape, res_shape, attr_val)
                        )

    if regions:
        for reg_name, reg_spec in regions.items():
            if isinstance(reg_spec, dict):
                b_args = (
                    reg_spec.get("block_arguments") or reg_spec.get("block_args") or []
                )
                y_types = reg_spec.get("yield_types") or []
                errors.extend(
                    validate_stablehlo_region(target_op, reg_name, b_args, y_types)
                )

    return {
        "is_valid": len(errors) == 0,
        "op_exists": True,
        "expected_operands": operands,
        "expected_attributes": exp_attrs,
        "errors": errors,
    }


def check_stablehlo_op(
    op_name: str,
    operands_count: Optional[int] = None,
    attributes: Optional[List[str]] = None,
    operand_types: Optional[List[str]] = None,
    result_types: Optional[List[str]] = None,
    structured_attributes: Optional[Dict[str, Any]] = None,
    regions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate a StableHLO operation against formal verification rules and schemas.

    Args:
        op_name: Qualified operation name (e.g. 'stablehlo.dot_general', 'stablehlo.reduce').
        operands_count: Optional expected number of SSA operands.
        attributes: Optional list of attribute names to verify.
        operand_types: Optional list of SSA operand type strings.
        result_types: Optional list of SSA result type strings.
        structured_attributes: Optional dictionary of structured attribute values.
        regions: Optional dictionary of regions mapping region name to block_arguments and yield_types.

    Returns:
        Validation report dictionary with is_valid, expected_operands, and errors.
    """
    import ml_ecosystem_snapshots.mcp_server as _mcp_server

    active_check_mlir = getattr(_mcp_server, "check_mlir_op", check_mlir_op)
    clean_name = op_name if op_name.startswith("stablehlo.") else f"stablehlo.{op_name}"
    return active_check_mlir(
        op_name=clean_name,
        operands_count=operands_count,
        attributes=attributes,
        operand_types=operand_types,
        result_types=result_types,
        structured_attributes=structured_attributes,
        regions=regions,
    )


__all__ = ["check_mlir_op", "check_stablehlo_op"]
