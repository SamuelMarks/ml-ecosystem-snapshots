"""Thin MCP adapter for hardware ISA verification (`check_sass_instruction`, `check_rdna_instruction`, `check_ptx_instruction`)."""

from __future__ import annotations

import typing

from ml_ecosystem_snapshots.grounding.hardware import (
    verify_ptx_instruction,
    verify_rdna_instruction,
    verify_sass_instruction,
)
from ml_ecosystem_snapshots.mcp.cache import SnapshotRepository, resolve_snapshot
from ml_ecosystem_snapshots.mcp.models import IsaCheckResult
from ml_ecosystem_snapshots.models import JsonValue


def check_sass_instruction(
    mnemonic: str,
    operands: list[str] | None = None,
    modifiers: list[str] | None = None,
    sm_arch: str | None = None,
    control_codes: typing.Mapping[str, JsonValue] | None = None,
    repository: SnapshotRepository | None = None,
) -> IsaCheckResult:
    """Validate whether an NVIDIA SASS assembly instruction is valid and supported.

    Args:
        mnemonic: Instruction mnemonic (e.g. FADD, WGMMA).
        operands: Optional list of operand strings.
        modifiers: Optional list of instruction modifiers.
        sm_arch: Optional target SM architecture (e.g. sm_80, sm_90).
        control_codes: Optional control code dictionary.
        repository: Optional `SnapshotRepository` instance for snapshot lookup.

    Returns:
        `IsaCheckResult` validation report model.
    """
    snap = resolve_snapshot("nvidia_sass", repository=repository)
    return verify_sass_instruction(
        mnemonic=mnemonic,
        snap=snap,
        operands=operands,
        modifiers=modifiers,
        sm_arch=sm_arch,
        control_codes=control_codes,
    )


def check_rdna_instruction(
    mnemonic: str,
    operands: list[str] | None = None,
    encoding: str | None = None,
    gfx_arch: str | None = None,
    modifiers: list[str] | None = None,
    wave_size: int | None = None,
    repository: SnapshotRepository | None = None,
) -> IsaCheckResult:
    """Validate whether an AMD RDNA assembly instruction is valid and supported.

    Args:
        mnemonic: Instruction mnemonic (e.g. v_add_f32, s_load_dword).
        operands: Optional list of operand type strings.
        encoding: Optional instruction encoding format (e.g. VOP2, SMEM).
        gfx_arch: Optional target GFX architecture (e.g. GFX11/RDNA3).
        modifiers: Optional list of modifiers.
        wave_size: Optional wavefront execution size (32 or 64).
        repository: Optional `SnapshotRepository` instance for snapshot lookup.

    Returns:
        `IsaCheckResult` validation report model.
    """
    snap = resolve_snapshot("amd_rdna", repository=repository)
    return verify_rdna_instruction(
        mnemonic=mnemonic,
        snap=snap,
        operands=operands,
        encoding=encoding,
        gfx_arch=gfx_arch,
        modifiers=modifiers,
        wave_size=wave_size,
    )


def check_ptx_instruction(
    mnemonic: str,
    types: list[str] | None = None,
    operands: list[str] | None = None,
    state_space: str | None = None,
    scope: str | None = None,
    vector_width: str | None = None,
    sm_arch: str | None = None,
) -> IsaCheckResult:
    """Validate whether an NVIDIA PTX assembly instruction is valid on target SM architecture.

    Args:
        mnemonic: PTX instruction mnemonic (e.g. add, ld, st, wgmma.mma_async).
        types: Optional list of PTX type qualifiers.
        operands: Optional list of register/immediate operand strings.
        state_space: Optional memory state space qualifier.
        scope: Optional memory scope qualifier.
        vector_width: Optional vector width qualifier.
        sm_arch: Optional target SM architecture.

    Returns:
        `IsaCheckResult` validation report model.
    """
    return verify_ptx_instruction(
        mnemonic=mnemonic,
        types=types,
        operands=operands,
        state_space=state_space,
        scope=scope,
        vector_width=vector_width,
        sm_arch=sm_arch,
    )


__all__ = [
    "check_ptx_instruction",
    "check_rdna_instruction",
    "check_sass_instruction",
]
