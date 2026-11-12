"""Thin MCP adapter for Python API hallucination, anti-pattern, and code-block checks."""

from __future__ import annotations

import typing

from ml_ecosystem_snapshots.grounding.anti_patterns import explain_python_anti_pattern
from ml_ecosystem_snapshots.grounding.code_block import verify_code_block
from ml_ecosystem_snapshots.grounding.dtypes import (
    DTYPE_CANONICAL_ALIASES,
    CallOptionValue,
    matches_rank_constraint,
    normalize_dtype_name,
)
from ml_ecosystem_snapshots.grounding.python_fw import verify_python_call
from ml_ecosystem_snapshots.mcp.cache import SnapshotRepository, resolve_snapshot
from ml_ecosystem_snapshots.mcp.models import (
    AntiPatternExplanationResult,
    CodeBlockCheckResult,
    HallucinationCheckResult,
)
from ml_ecosystem_snapshots.mcp.tools.ir import check_mlir_op
from ml_ecosystem_snapshots.mcp.tools.isa import check_rdna_instruction, check_sass_instruction
from ml_ecosystem_snapshots.mcp.tools.signature import get_api_signature
from ml_ecosystem_snapshots.models import JsonValue


def check_hallucination(
    framework: str,
    api_path: str,
    kwargs: list[str] | None = None,
    args_count: int | None = None,
    version: str | None = None,
    strict_kwargs: bool = True,
    kwarg_values: typing.Mapping[str, JsonValue] | None = None,
    repository: SnapshotRepository | None = None,
    **extra_constraints: CallOptionValue,
) -> HallucinationCheckResult:
    """Verify whether an API call uses valid parameters, arity, dtypes, ranks, and shapes.

    Args:
        framework: Target framework name (e.g. torch, jax).
        api_path: Canonical API path (e.g. torch.sum, jax.numpy.mean).
        kwargs: Optional list of keyword argument names passed by generated code.
        args_count: Optional number of positional arguments passed.
        version: Optional framework version string.
        strict_kwargs: If True, reject unrecognized kwargs even when `**kwargs` is present.
        kwarg_values: Optional mapping of passed keyword argument names to constant values.
        repository: Optional `SnapshotRepository` instance for snapshot lookup.
        **extra_constraints: Optional inline constraint kwargs (`arg_dtypes`, `kwarg_dtypes`, etc.).

    Returns:
        `HallucinationCheckResult` validation report model.
    """
    sig = get_api_signature(framework, api_path, version=version, repository=repository)
    snap = resolve_snapshot(framework, version=version, repository=repository)
    raw_source = snap.get("_snapshot_source")
    fallback_source = "not_found" if not snap.get("categories") else "unknown"
    snapshot_source = str(raw_source) if raw_source is not None else fallback_source
    return verify_python_call(
        framework=framework,
        api_path=api_path,
        sig=sig,
        snapshot_source=snapshot_source,
        kwargs=kwargs,
        args_count=args_count,
        strict_kwargs=strict_kwargs,
        kwarg_values=kwarg_values,
        version=version,
        **extra_constraints,
    )


def explain_anti_pattern(
    framework: str,
    api_path: str,
    hallucinated_argument: str,
    passed_value: JsonValue = None,
    repository: SnapshotRepository | None = None,
) -> AntiPatternExplanationResult:
    """Provide canonical migration advice and rationale for a flagged anti-pattern or hallucinated argument.

    Args:
        framework: Target framework name (e.g. torch, jax, tensorflow).
        api_path: Canonical API path (e.g. torch.sum, jax.numpy.sum).
        hallucinated_argument: Flagged keyword argument name (e.g. axis, keepdims).
        passed_value: Optional literal value passed to the argument.
        repository: Optional `SnapshotRepository` instance for snapshot lookup.

    Returns:
        `AntiPatternExplanationResult` payload.
    """
    return explain_python_anti_pattern(
        framework=framework,
        api_path=api_path,
        hallucinated_argument=hallucinated_argument,
        passed_value=passed_value,
        sig_lookup_fn=lambda fw, path: get_api_signature(fw, path, repository=repository),
    )


def check_code_block(
    code: str,
    framework: str | None = None,
    sm_arch: str | None = None,
    gfx_arch: str | None = None,
    repository: SnapshotRepository | None = None,
) -> CodeBlockCheckResult:
    """Analyze a multi-line code snippet (Python, MLIR, SASS, or RDNA) for hallucinations.

    Args:
        code: Raw source code block string.
        framework: Optional explicit framework override.
        sm_arch: Optional NVIDIA architecture target (e.g. sm_80, sm_90).
        gfx_arch: Optional AMD RDNA architecture target (e.g. GFX11/RDNA3).
        repository: Optional `SnapshotRepository` instance for snapshot lookup.

    Returns:
        `CodeBlockCheckResult` validation report model.
    """
    return verify_code_block(
        code=code,
        call_checker=lambda fw, api, kwargs=None, args_count=None, kwarg_values=None: check_hallucination(
            fw, api, kwargs=kwargs, args_count=args_count, kwarg_values=kwarg_values, repository=repository
        ),
        mlir_checker=lambda op_name: check_mlir_op(op_name, repository=repository),
        sass_checker=lambda mnem, ops, mods, s_arch: check_sass_instruction(
            mnem, operands=ops, modifiers=mods, sm_arch=s_arch, repository=repository
        ),
        rdna_checker=lambda mnem, ops, mods, g_arch: check_rdna_instruction(
            mnem, operands=ops, modifiers=mods, gfx_arch=g_arch, repository=repository
        ),
        framework=framework,
        sm_arch=sm_arch,
        gfx_arch=gfx_arch,
    )


__all__ = [
    "DTYPE_CANONICAL_ALIASES",
    "check_code_block",
    "check_hallucination",
    "explain_anti_pattern",
    "matches_rank_constraint",
    "normalize_dtype_name",
]
