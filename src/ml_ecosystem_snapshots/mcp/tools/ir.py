"""Thin MCP adapter for compiler IR dialect verification (`check_mlir_op`, `check_stablehlo_op`)."""

from __future__ import annotations

import typing

from ml_ecosystem_snapshots.grounding.compiler import verify_mlir_op
from ml_ecosystem_snapshots.mcp.cache import SnapshotRepository, resolve_snapshot
from ml_ecosystem_snapshots.mcp.models import IrCheckResult
from ml_ecosystem_snapshots.models import JsonValue


def check_mlir_op(
    op_name: str,
    operands_count: int | None = None,
    attributes: list[str] | None = None,
    operand_types: list[str] | None = None,
    result_types: list[str] | None = None,
    structured_attributes: typing.Mapping[str, JsonValue] | None = None,
    regions: typing.Mapping[str, JsonValue] | None = None,
    repository: SnapshotRepository | None = None,
) -> IrCheckResult:
    """Validate whether an MLIR or StableHLO operation exists with expected signature.

    Args:
        op_name: Qualified operation name (e.g. arith.addf or stablehlo.dot_general).
        operands_count: Optional expected number of SSA operands.
        attributes: Optional list of attribute names to verify.
        operand_types: Optional list of SSA operand type strings.
        result_types: Optional list of SSA result type strings.
        structured_attributes: Optional dictionary of structured attribute values.
        regions: Optional dictionary of regions mapping region name to block_arguments and yield_types.
        repository: Optional `SnapshotRepository` instance for snapshot lookup.

    Returns:
        `IrCheckResult` validation report model.
    """
    framework = "stablehlo" if op_name.startswith("stablehlo.") else "mlir"
    snap = resolve_snapshot(framework, repository=repository)
    return verify_mlir_op(
        op_name=op_name,
        snap=snap,
        operands_count=operands_count,
        attributes=attributes,
        operand_types=operand_types,
        result_types=result_types,
        structured_attributes=structured_attributes,
        regions=regions,
    )


def check_stablehlo_op(
    op_name: str,
    operands_count: int | None = None,
    attributes: list[str] | None = None,
    operand_types: list[str] | None = None,
    result_types: list[str] | None = None,
    structured_attributes: typing.Mapping[str, JsonValue] | None = None,
    regions: typing.Mapping[str, JsonValue] | None = None,
    repository: SnapshotRepository | None = None,
) -> IrCheckResult:
    """Validate a StableHLO operation against formal verification rules and schemas.

    Args:
        op_name: Qualified operation name (e.g. stablehlo.dot_general, stablehlo.reduce).
        operands_count: Optional expected number of SSA operands.
        attributes: Optional list of attribute names to verify.
        operand_types: Optional list of SSA operand type strings.
        result_types: Optional list of SSA result type strings.
        structured_attributes: Optional dictionary of structured attribute values.
        regions: Optional dictionary of regions mapping region name to block_arguments and yield_types.
        repository: Optional `SnapshotRepository` instance for snapshot lookup.

    Returns:
        `IrCheckResult` validation report model.
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


__all__ = ["check_mlir_op", "check_stablehlo_op"]
