"""Compiler IR grounding verifiers for StableHLO and MLIR core dialects."""

from typing import Any, Dict, List, Optional

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
) -> GroundingReport:
    """Verify a StableHLO operation against the official StableHLO specification.

    Args:
        op_name: Operation name (e.g., 'stablehlo.dot_general', 'dot_general', 'stablehlo.add').
        operand_types: Types of SSA input operands (e.g., ['tensor<128x64xf32>', 'tensor<64x256xf32>']).
        attributes: Dictionary of operation attributes (e.g., {'dot_dimension_numbers': {...}}).
        engine: Optional GroundingEngine instance.

    Returns:
        GroundingReport containing diagnostic results and grounding status.
    """
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
        if "dimension_numbers" not in attributes:
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
        else [p for p in ref.params if getattr(p, "role", None) != "ATTRIBUTE"]
    )
    if expected_operands:
        # Check if variadic
        is_variadic = getattr(ref, "has_varargs", False)
        if not is_variadic and len(operand_types) != len(expected_operands):
            report.add_diagnostic(
                field="operands",
                message=(
                    f"Operation '{full_op_name}' expects {len(expected_operands)} operands, "
                    f"but received {len(operand_types)}."
                ),
                severity=DiagnosticSeverity.ERROR,
            )

    return report


def validate_mlir_op(
    dialect: str,
    op_name: str,
    operand_count: int,
    attributes: Dict[str, Any],
    engine: Optional[GroundingEngine] = None,
) -> GroundingReport:
    """Verify an MLIR operation against core MLIR TableGen definitions.

    Args:
        dialect: The MLIR dialect (e.g., 'arith', 'math', 'tensor', 'linalg', 'scf').
        op_name: The operation name or mnemonic (e.g., 'addf', 'arith.addf', 'matmul').
        operand_count: Number of SSA value operands passed to the operation.
        attributes: Dictionary of buildable MLIR attributes.
        engine: Optional GroundingEngine instance.

    Returns:
        GroundingReport containing diagnostic results and grounding status.
    """
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
        else [p for p in ref.params if getattr(p, "role", None) != "ATTRIBUTE"]
    )
    if expected_operands and operand_count != len(expected_operands):
        report.add_diagnostic(
            field="operand_count",
            message=(
                f"MLIR operation '{full_path}' expects {len(expected_operands)} operands, "
                f"but received {operand_count}."
            ),
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

    eng = engine or GroundingEngine()
    report = GroundingReport(
        is_grounded=True,
        target="wgsl",
        symbol=op_name,
    )
    res = check_wgsl_op(
        op_name=op_name,
        inputs_count=operands_count,
        attributes=attributes,
    )
    if not res.get("op_exists"):
        suggested = eng.suggest_closest_symbol("wgsl", op_name)
        report.add_diagnostic(
            field="operation",
            message=f"Unrecognized WGSL builtin '{op_name}'.",
            severity=DiagnosticSeverity.ERROR,
            suggested_fix=suggested,
        )
        return report

    for err in res.get("errors", []):
        report.add_diagnostic(
            field="operation",
            message=err,
            severity=DiagnosticSeverity.ERROR,
        )
    return report


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

    eng = engine or GroundingEngine()
    report = GroundingReport(
        is_grounded=True,
        target="onnx",
        symbol=op_name,
    )
    res = check_onnx_op(
        op_name=op_name,
        inputs_count=inputs_count,
        attributes=attributes,
    )
    if not res.get("op_exists"):
        suggested = eng.suggest_closest_symbol("onnx", op_name)
        report.add_diagnostic(
            field="operation",
            message=f"Unrecognized ONNX operator '{op_name}'.",
            severity=DiagnosticSeverity.ERROR,
            suggested_fix=suggested,
        )
        return report

    for err in res.get("errors", []):
        report.add_diagnostic(
            field="operation",
            message=err,
            severity=DiagnosticSeverity.ERROR,
        )
    return report


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

    eng = engine or GroundingEngine()
    report = GroundingReport(
        is_grounded=True,
        target="metal",
        symbol=op_name,
    )
    res = check_metal_op(
        op_name=op_name,
        address_space=address_space,
        inputs_count=inputs_count,
        attributes=attributes,
    )
    if not res.get("op_exists"):
        suggested = eng.suggest_closest_symbol("metal", op_name)
        report.add_diagnostic(
            field="operation",
            message=f"Unrecognized Metal builtin '{op_name}'.",
            severity=DiagnosticSeverity.ERROR,
            suggested_fix=suggested,
        )
        return report

    for err in res.get("errors", []):
        report.add_diagnostic(
            field="operation",
            message=err,
            severity=DiagnosticSeverity.ERROR,
        )
    return report


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

    eng = engine or GroundingEngine()
    report = GroundingReport(
        is_grounded=True,
        target="wasm_simd",
        symbol=mnemonic,
    )
    res = check_wasm_instruction(
        mnemonic=mnemonic,
        operands_count=operands_count,
        attributes=attributes,
    )
    if not res.get("is_valid") and not res.get("op_exists"):
        suggested = eng.suggest_closest_symbol("wasm_simd", mnemonic)
        report.add_diagnostic(
            field="mnemonic",
            message=f"Unrecognized WebAssembly SIMD opcode '{mnemonic}'.",
            severity=DiagnosticSeverity.ERROR,
            suggested_fix=suggested,
        )
        return report

    for err in res.get("errors", []):
        report.add_diagnostic(
            field="instruction",
            message=err,
            severity=DiagnosticSeverity.ERROR,
        )
    return report


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

    eng = engine or GroundingEngine()
    report = GroundingReport(
        is_grounded=True,
        target="webgl",
        symbol=op_name,
    )
    res = check_webgl_op(
        op_name=op_name,
        inputs_count=inputs_count,
        attributes=attributes,
    )
    if not res.get("op_exists"):
        suggested = eng.suggest_closest_symbol("webgl", op_name)
        report.add_diagnostic(
            field="operation",
            message=f"Unrecognized WebGL / GLSL builtin '{op_name}'.",
            severity=DiagnosticSeverity.ERROR,
            suggested_fix=suggested,
        )
        return report

    for err in res.get("errors", []):
        report.add_diagnostic(
            field="operation",
            message=err,
            severity=DiagnosticSeverity.ERROR,
        )
    return report
