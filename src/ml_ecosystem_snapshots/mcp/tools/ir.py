"""MCP tool module for compiler IR dialect verification (MLIR and StableHLO)."""

from __future__ import annotations

from ml_ecosystem_snapshots.mcp_server import check_mlir_op, check_stablehlo_op

__all__ = ["check_mlir_op", "check_stablehlo_op"]
