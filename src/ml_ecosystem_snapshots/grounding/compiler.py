"""Compiler IR grounding verifiers for StableHLO and MLIR core dialects."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from ml_ecosystem_snapshots.grounding.engine import GroundingEngine
from ml_ecosystem_snapshots.grounding.models import (
    DiagnosticSeverity,
    GroundingReport,
)


def validate_stablehlo_op(
    op_name: str,
    operand_types: List[str],
    attributes: Dict[str, Any],
    engine: Optional[GroundingEngine] = None,
    result_types: Optional[List[str]] = None,
    regions: Optional[Dict[str, Any]] = None,
) -> GroundingReport:
    """Verify a StableHLO operation against the official StableHLO specification.

    Delegates trait verification, SSA type constraint validation, structured
    dimension number checks, and region verification to the canonical IR
    validators shared with `check_mlir_op`.

    Args:
        op_name: Operation name (e.g., 'stablehlo.dot_general', 'dot_general', 'stablehlo.add').
        operand_types: Types of SSA input operands (e.g., ['tensor<128x64xf32>', 'tensor<64x256xf32>']).
        attributes: Dictionary of operation attributes (e.g., {'dot_dimension_numbers': {...}}).
        engine: Optional GroundingEngine instance.
        result_types: Optional list of SSA result type strings.
        regions: Optional dictionary of regions mapping region name to block_arguments and yield_types.

    Returns:
        GroundingReport containing diagnostic results and grounding status.
    """
    from ml_ecosystem_snapshots.frameworks.mlir import (
        validate_mlir_traits,
        validate_mlir_type,
    )
    from ml_ecosystem_snapshots.frameworks.stablehlo import validate_stablehlo_region
    from ml_ecosystem_snapshots.mcp.tools.ir import (
        _infer_implicit_mlir_traits,
        _validate_mlir_structured_attributes,
    )

    eng = engine or GroundingEngine()
    full_op_name = (
        op_name if op_name.startswith("stablehlo.") else f"stablehlo.{op_name}"
    )

    report = GroundingReport(
        is_grounded=True,
        target="stablehlo",
        symbol=full_op_name,
    )

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

    # Verify mandatory operation attributes
    if full_op_name == "stablehlo.dot_general":
        if (
            "dot_dimension_numbers" not in attributes
            and "dimension_numbers" not in attributes
        ):
            report.add_diagnostic(
                field="attributes.dot_dimension_numbers",
                message="Operation 'stablehlo.dot_general' requires 'dot_dimension_numbers' attribute.",
                severity=DiagnosticSeverity.ERROR,
                suggested_fix="dot_dimension_numbers",
            )

    if full_op_name == "stablehlo.convolution":
        if (
            "dimension_numbers" not in attributes
            and "conv_dimension_numbers" not in attributes
        ):
            report.add_diagnostic(
                field="attributes.dimension_numbers",
                message="Operation 'stablehlo.convolution' requires 'dimension_numbers' attribute.",
                severity=DiagnosticSeverity.ERROR,
                suggested_fix="dimension_numbers",
            )

    # Verify operand counts
    raw_operands = getattr(ref, "operands", None)
    expected_operands = (
        raw_operands
        if raw_operands is not None
        else [
            p
            for p in ref.params
            if "ATTRIBUTE" not in str(getattr(p, "role", ""))
            and getattr(p, "annotation", None) != "Region"
        ]
    )
    if expected_operands:
        is_variadic = getattr(ref, "has_varargs", False) or any(
            "variadic"
            in str(
                getattr(o, "annotation", "")
                or (o.get("type", "") if isinstance(o, dict) else "")
            ).lower()
            for o in expected_operands
        )
        if not is_variadic and len(operand_types) != len(expected_operands):
            report.add_diagnostic(
                field="operands",
                message=(
                    f"Operation '{full_op_name}' expects {len(expected_operands)} operands, "
                    f"but received {len(operand_types)}."
                ),
                severity=DiagnosticSeverity.ERROR,
            )

        for i, (op_info, actual_type) in enumerate(
            zip(expected_operands, operand_types)
        ):
            constraint = str(
                op_info.get("type", "")
                if isinstance(op_info, dict)
                else getattr(op_info, "annotation", "") or ""
            )
            for te in validate_mlir_type(actual_type, constraint=constraint):
                report.add_diagnostic(
                    field=f"operands[{i}]",
                    message=f"Operand {i} for '{full_op_name}': {te}",
                    severity=DiagnosticSeverity.ERROR,
                )

    meta = ref.domain_metadata or {}
    base_traits = list(meta.get("traits") or getattr(ref, "traits", None) or [])
    traits = _infer_implicit_mlir_traits(full_op_name, base_traits)
    for trait_err in validate_mlir_traits(
        op_name=full_op_name,
        traits=traits,
        operand_types=operand_types if operand_types else None,
        result_types=result_types,
        attributes=list(attributes.keys()) if attributes else None,
        structured_attributes=attributes if attributes else None,
    ):
        report.add_diagnostic(
            field="traits",
            message=trait_err,
            severity=DiagnosticSeverity.ERROR,
        )

    if attributes:
        for attr_err in _validate_mlir_structured_attributes(
            structured_attributes=attributes,
            operand_types=operand_types if operand_types else None,
            result_types=result_types,
        ):
            report.add_diagnostic(
                field="attributes",
                message=attr_err,
                severity=DiagnosticSeverity.ERROR,
            )

    if regions:
        for reg_name, reg_spec in regions.items():
            b_args = reg_spec.get("block_arguments") or reg_spec.get("block_args") or []
            y_types = reg_spec.get("yield_types") or []
            for reg_err in validate_stablehlo_region(
                full_op_name, reg_name, b_args, y_types
            ):
                report.add_diagnostic(
                    field=f"regions.{reg_name}",
                    message=reg_err,
                    severity=DiagnosticSeverity.ERROR,
                )

    return report


def validate_mlir_op(
    dialect: str,
    op_name: str,
    operand_count: int,
    attributes: Dict[str, Any],
    engine: Optional[GroundingEngine] = None,
    operand_types: Optional[List[str]] = None,
    result_types: Optional[List[str]] = None,
) -> GroundingReport:
    """Verify an MLIR operation against core MLIR TableGen definitions.

    Delegates trait verification, SSA type constraint validation, and structured
    attribute checks to the canonical IR validators shared with `check_mlir_op`.

    Args:
        dialect: The MLIR dialect (e.g., 'arith', 'math', 'tensor', 'linalg', 'scf').
        op_name: The operation name or mnemonic (e.g., 'addf', 'arith.addf', 'matmul').
        operand_count: Number of SSA value operands passed to the operation.
        attributes: Dictionary of buildable MLIR attributes.
        engine: Optional GroundingEngine instance.
        operand_types: Optional list of SSA operand type strings.
        result_types: Optional list of SSA result type strings.

    Returns:
        GroundingReport containing diagnostic results and grounding status.
    """
    from ml_ecosystem_snapshots.frameworks.mlir import (
        validate_mlir_traits,
        validate_mlir_type,
    )
    from ml_ecosystem_snapshots.mcp.tools.ir import (
        _infer_implicit_mlir_traits,
        _validate_mlir_structured_attributes,
    )

    eng = engine or GroundingEngine()
    full_path = op_name if op_name.startswith(f"{dialect}.") else f"{dialect}.{op_name}"

    report = GroundingReport(
        is_grounded=True,
        target="mlir",
        symbol=full_path,
    )

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

    # Verify operands count
    raw_operands = getattr(ref, "operands", None)
    expected_operands = (
        raw_operands
        if raw_operands is not None
        else [
            p
            for p in ref.params
            if "ATTRIBUTE" not in str(getattr(p, "role", ""))
            and getattr(p, "annotation", None) != "Region"
        ]
    )
    if expected_operands and operand_count != len(expected_operands):
        is_variadic = getattr(ref, "has_varargs", False) or any(
            "variadic"
            in str(
                getattr(o, "annotation", "")
                or (o.get("type", "") if isinstance(o, dict) else "")
            ).lower()
            for o in expected_operands
        )
        if not is_variadic:
            report.add_diagnostic(
                field="operand_count",
                message=(
                    f"MLIR operation '{full_path}' expects {len(expected_operands)} operands, "
                    f"but received {operand_count}."
                ),
                severity=DiagnosticSeverity.ERROR,
            )

    if operand_types is not None:
        for i, (op_info, actual_type) in enumerate(
            zip(expected_operands, operand_types)
        ):
            constraint = str(
                op_info.get("type", "")
                if isinstance(op_info, dict)
                else getattr(op_info, "annotation", "") or ""
            )
            for te in validate_mlir_type(actual_type, constraint=constraint):
                report.add_diagnostic(
                    field=f"operands[{i}]",
                    message=f"Operand {i} for '{full_path}': {te}",
                    severity=DiagnosticSeverity.ERROR,
                )

    meta = ref.domain_metadata or {}
    base_traits = list(meta.get("traits") or getattr(ref, "traits", None) or [])
    traits = _infer_implicit_mlir_traits(full_path, base_traits)
    for trait_err in validate_mlir_traits(
        op_name=full_path,
        traits=traits,
        operand_types=operand_types,
        result_types=result_types,
        attributes=list(attributes.keys()) if attributes else None,
        structured_attributes=attributes if attributes else None,
    ):
        report.add_diagnostic(
            field="traits",
            message=trait_err,
            severity=DiagnosticSeverity.ERROR,
        )

    if attributes:
        for attr_err in _validate_mlir_structured_attributes(
            structured_attributes=attributes,
            operand_types=operand_types,
            result_types=result_types,
        ):
            report.add_diagnostic(
                field="attributes",
                message=attr_err,
                severity=DiagnosticSeverity.ERROR,
            )

    return report


def _validate_ssa_domain_op(
    target_framework: str,
    symbol_name: str,
    res: Mapping[str, Any],
    unrecognized_label: str,
    missing_field: str = "operation",
    err_field: str = "operation",
    engine: Optional[GroundingEngine] = None,
) -> GroundingReport:
    """Build a GroundingReport from a domain operation verifier dictionary result.

    Args:
        target_framework: Framework identifier (e.g., 'wgsl', 'onnx', 'metal', 'wasm_simd', 'webgl').
        symbol_name: Operation or opcode name checked.
        res: Result dictionary from `check_*_op`.
        unrecognized_label: Human-readable label for unrecognized symbol diagnostics.
        missing_field: Diagnostic field name when symbol is missing.
        err_field: Diagnostic field name for constraint errors.
        engine: Optional `GroundingEngine` instance.

    Returns:
        Populated `GroundingReport` instance.
    """
    eng = engine or GroundingEngine()
    report = GroundingReport(
        is_grounded=True,
        target=target_framework,
        symbol=symbol_name,
    )
    if not res.get("op_exists"):
        suggested = eng.suggest_closest_symbol(target_framework, symbol_name)
        report.add_diagnostic(
            field=missing_field,
            message=f"Unrecognized {unrecognized_label} '{symbol_name}'.",
            severity=DiagnosticSeverity.ERROR,
            suggested_fix=suggested,
        )
        return report

    for err in res.get("errors", []):
        report.add_diagnostic(
            field=err_field,
            message=err,
            severity=DiagnosticSeverity.ERROR,
        )
    return report


def validate_wgsl_op(
    op_name: str,
    operands_count: Optional[int] = None,
    attributes: Optional[List[str]] = None,
    engine: Optional[GroundingEngine] = None,
) -> GroundingReport:
    """Verify a WebGPU WGSL builtin function or memory barrier against ground truth.

    Args:
        op_name: WGSL operation name (e.g., 'workgroupBarrier', 'storageStore').
        operands_count: Optional number of passed arguments.
        attributes: Optional list of operation attributes.
        engine: Optional GroundingEngine instance.

    Returns:
        GroundingReport containing diagnostic results.
    """
    from ml_ecosystem_snapshots.mcp_server import check_wgsl_op

    res = check_wgsl_op(
        op_name=op_name,
        inputs_count=operands_count,
        attributes=attributes,
    )
    return _validate_ssa_domain_op(
        target_framework="wgsl",
        symbol_name=op_name,
        res=res,
        unrecognized_label="WGSL builtin",
        engine=engine,
    )


def validate_onnx_op(
    op_name: str,
    inputs_count: Optional[int] = None,
    attributes: Optional[List[str]] = None,
    engine: Optional[GroundingEngine] = None,
) -> GroundingReport:
    """Verify an ONNX operator specification against the standard schema.

    Args:
        op_name: ONNX operator name (e.g., 'MatMul', 'Conv', 'Relu').
        inputs_count: Optional number of passed inputs.
        attributes: Optional list of attributes.
        engine: Optional GroundingEngine instance.

    Returns:
        GroundingReport containing diagnostic results.
    """
    from ml_ecosystem_snapshots.mcp_server import check_onnx_op

    res = check_onnx_op(
        op_name=op_name,
        inputs_count=inputs_count,
        attributes=attributes,
    )
    return _validate_ssa_domain_op(
        target_framework="onnx",
        symbol_name=op_name,
        res=res,
        unrecognized_label="ONNX operator",
        engine=engine,
    )


def validate_metal_op(
    op_name: str,
    address_space: Optional[str] = None,
    inputs_count: Optional[int] = None,
    attributes: Optional[List[str]] = None,
    engine: Optional[GroundingEngine] = None,
) -> GroundingReport:
    """Verify an Apple Metal Shading Language compute builtin against specifications.

    Args:
        op_name: Metal builtin name (e.g., 'simdgroup_matrix', 'threadgroup_barrier').
        address_space: Optional expected address space (e.g., 'device', 'threadgroup').
        inputs_count: Optional number of passed arguments.
        attributes: Optional list of attributes.
        engine: Optional GroundingEngine instance.

    Returns:
        GroundingReport containing diagnostic results.
    """
    from ml_ecosystem_snapshots.mcp_server import check_metal_op

    res = check_metal_op(
        op_name=op_name,
        address_space=address_space,
        inputs_count=inputs_count,
        attributes=attributes,
    )
    return _validate_ssa_domain_op(
        target_framework="metal",
        symbol_name=op_name,
        res=res,
        unrecognized_label="Metal builtin",
        engine=engine,
    )


def validate_wasm_op(
    mnemonic: str,
    operands_count: Optional[int] = None,
    attributes: Optional[List[str]] = None,
    engine: Optional[GroundingEngine] = None,
) -> GroundingReport:
    """Verify a W3C WebAssembly SIMD opcode mnemonic against standard specifications.

    Args:
        mnemonic: WASM opcode mnemonic (e.g., 'f32x4.add', 'v128.load').
        operands_count: Optional number of passed operands.
        attributes: Optional list of attributes.
        engine: Optional GroundingEngine instance.

    Returns:
        GroundingReport containing diagnostic results.
    """
    from ml_ecosystem_snapshots.mcp_server import check_wasm_instruction

    res = check_wasm_instruction(
        mnemonic=mnemonic,
        operands_count=operands_count,
        attributes=attributes,
    )
    return _validate_ssa_domain_op(
        target_framework="wasm_simd",
        symbol_name=mnemonic,
        res=res,
        unrecognized_label="WebAssembly SIMD opcode",
        missing_field="mnemonic",
        err_field="instruction",
        engine=engine,
    )


def validate_webgl_op(
    op_name: str,
    inputs_count: Optional[int] = None,
    attributes: Optional[List[str]] = None,
    engine: Optional[GroundingEngine] = None,
) -> GroundingReport:
    """Verify a WebGL 2.0 / GLSL ES 3.00 shader builtin against standard specifications.

    Args:
        op_name: GLSL builtin function name (e.g., 'texture', 'dFdx').
        inputs_count: Optional number of passed arguments.
        attributes: Optional list of attributes.
        engine: Optional GroundingEngine instance.

    Returns:
        GroundingReport containing diagnostic results.
    """
    from ml_ecosystem_snapshots.mcp_server import check_webgl_op

    res = check_webgl_op(
        op_name=op_name,
        inputs_count=inputs_count,
        attributes=attributes,
    )
    return _validate_ssa_domain_op(
        target_framework="webgl",
        symbol_name=op_name,
        res=res,
        unrecognized_label="WebGL / GLSL builtin",
        engine=engine,
    )


__all__ = [
    "_validate_ssa_domain_op",
    "validate_metal_op",
    "validate_mlir_op",
    "validate_onnx_op",
    "validate_stablehlo_op",
    "validate_wasm_op",
    "validate_webgl_op",
    "validate_wgsl_op",
]
