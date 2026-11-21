"""Monoidal text-block assembly and IR snippet compliance verifiers (MLIR, WGSL, SASS, RDNA)."""

from __future__ import annotations

import re
import typing

from ml_ecosystem_snapshots.compliance.models import SnippetComplianceReport
from ml_ecosystem_snapshots.grounding.compiler import verify_mlir_op
from ml_ecosystem_snapshots.grounding.domain_ops import check_domain_op
from ml_ecosystem_snapshots.grounding.hardware import (
    verify_rdna_instruction,
    verify_sass_instruction,
)
from ml_ecosystem_snapshots.models import JsonValue
from ml_ecosystem_snapshots.serialization.repository import load_or_extract_snapshot
from ml_ecosystem_snapshots.typeclasses import SNIPPET_REPORT_MONOID

_MLIR_OP_RE: typing.Final[re.Pattern[str]] = re.compile(
    r"(?:%\w+\s*=\s*)?([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)\s*(?:\(([^)]*)\))?"
)
_WGSL_CALL_RE: typing.Final[re.Pattern[str]] = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)")
_SASS_LINE_RE: typing.Final[re.Pattern[str]] = re.compile(
    r"^(?:@!?P\d+\s+)?([A-Z0-9_]+)((?:\.[A-Z0-9_]+)*)(?:\s+(.*?))?;?$"
)
_RDNA_LINE_RE: typing.Final[re.Pattern[str]] = re.compile(r"^([a-z0-9_]+)(?:\s+(.*?))?;?$")
_WGSL_IGNORED_KEYWORDS: typing.Final[frozenset[str]] = frozenset(
    {
        "fn",
        "if",
        "for",
        "while",
        "loop",
        "switch",
        "var",
        "let",
        "const",
        "return",
        "vec2",
        "vec3",
        "vec4",
        "mat2x2",
        "mat3x3",
        "mat4x4",
        "array",
        "ptr",
    }
)


def _check_mlir_line(line: str) -> SnippetComplianceReport:
    """Evaluate a single MLIR/StableHLO text line into a `SnippetComplianceReport`.

    Args:
        line: Raw line from an MLIR/StableHLO snippet.

    Returns:
        `SnippetComplianceReport` contribution for `line`.
    """
    cleaned = line.strip()
    if not cleaned or cleaned.startswith("//"):
        return SNIPPET_REPORT_MONOID.empty()

    match = _MLIR_OP_RE.search(cleaned)
    if not match:
        return SNIPPET_REPORT_MONOID.empty()

    op_name = match.group(1)
    if op_name in ("builtin.module", "func.func"):
        return SNIPPET_REPORT_MONOID.empty()

    args_str = match.group(2)
    operands_count = len([a for a in args_str.split(",") if a.strip()]) if args_str is not None else None
    fw = "stablehlo" if op_name.startswith("stablehlo.") else "mlir"
    snap = load_or_extract_snapshot(fw)
    res = verify_mlir_op(op_name, snap=snap, operands_count=operands_count)
    if not res.op_exists:
        return SnippetComplianceReport(
            is_compliant=False,
            total_ops=1,
            verified_ops=0,
            errors=(f"Invalid MLIR operation '{op_name}' on line: '{cleaned}'",),
        )
    if not res.is_valid:
        return SnippetComplianceReport(
            is_compliant=False,
            total_ops=1,
            verified_ops=0,
            errors=res.errors,
        )
    return SnippetComplianceReport(is_compliant=True, total_ops=1, verified_ops=1)


def check_mlir_text_compliance(mlir_text: str) -> SnippetComplianceReport:
    """Verify MLIR / StableHLO text snippet compliance via `SNIPPET_REPORT_MONOID`.

    Args:
        mlir_text: Text snippet of MLIR or StableHLO assembly code.

    Returns:
        Monoid-aggregated `SnippetComplianceReport`.
    """
    return SNIPPET_REPORT_MONOID.mconcat(_check_mlir_line(line) for line in mlir_text.splitlines())


def _check_wgsl_line(line: str) -> SnippetComplianceReport:
    """Evaluate a single WGSL shader line into a `SnippetComplianceReport`.

    Args:
        line: Raw line from a WebGPU WGSL shader.

    Returns:
        `SnippetComplianceReport` contribution for `line`.
    """
    cleaned = line.strip()
    if not cleaned or cleaned.startswith("//"):
        return SNIPPET_REPORT_MONOID.empty()

    line_reports: list[SnippetComplianceReport] = []
    for match in _WGSL_CALL_RE.finditer(cleaned):
        op_name = match.group(1)
        if op_name in _WGSL_IGNORED_KEYWORDS:
            continue
        args_raw = match.group(2).strip()
        inputs_count = len([a for a in args_raw.split(",") if a.strip()]) if args_raw else 0
        res = check_domain_op("wgsl", op_name, inputs_count=inputs_count)
        if res.op_exists:
            if res.is_valid:
                line_reports.append(SnippetComplianceReport(is_compliant=True, total_ops=1, verified_ops=1))
            else:
                line_reports.append(
                    SnippetComplianceReport(
                        is_compliant=False,
                        total_ops=1,
                        verified_ops=0,
                        errors=res.errors,
                    )
                )
    return SNIPPET_REPORT_MONOID.mconcat(line_reports)


def check_wgsl_shader_compliance(shader_text: str) -> SnippetComplianceReport:
    """Verify WebGPU WGSL shader snippet compliance via `SNIPPET_REPORT_MONOID`.

    Args:
        shader_text: Text snippet of WebGPU WGSL shader code.

    Returns:
        Monoid-aggregated `SnippetComplianceReport`.
    """
    return SNIPPET_REPORT_MONOID.mconcat(_check_wgsl_line(line) for line in shader_text.splitlines())


def _check_sass_line(
    line: str,
    snap: typing.Mapping[str, JsonValue],
    sm_arch: str | None,
) -> SnippetComplianceReport:
    """Evaluate a single NVIDIA SASS assembly line into a `SnippetComplianceReport`.

    Args:
        line: Raw SASS assembly line.
        snap: Loaded NVIDIA SASS snapshot dictionary.
        sm_arch: Optional target SM architecture.

    Returns:
        `SnippetComplianceReport` contribution for `line`.
    """
    cleaned = re.sub(r"/\*.*?\*/", "", line).strip()
    if not cleaned or cleaned.startswith("//") or cleaned.startswith("#"):
        return SNIPPET_REPORT_MONOID.empty()

    matched = _SASS_LINE_RE.match(cleaned)
    if not matched:
        return SNIPPET_REPORT_MONOID.empty()

    base_mnemonic = matched.group(1)
    mods_raw = matched.group(2)
    operands_raw = matched.group(3)
    modifiers = [mod for mod in mods_raw.split(".") if mod] if mods_raw else []
    operands = [op.strip() for op in operands_raw.split(",") if op.strip()] if operands_raw else None

    res = verify_sass_instruction(
        base_mnemonic,
        snap=snap,
        operands=operands,
        modifiers=modifiers,
        sm_arch=sm_arch,
    )
    if not res.is_valid:
        return SnippetComplianceReport(
            is_compliant=False,
            total_instructions=1,
            verified_instructions=0,
            errors=res.errors,
        )
    return SnippetComplianceReport(
        is_compliant=True,
        total_instructions=1,
        verified_instructions=1,
    )


def check_sass_assembly_compliance(assembly_text: str, sm_arch: str | None = None) -> SnippetComplianceReport:
    """Verify NVIDIA SASS assembly snippet compliance via `SNIPPET_REPORT_MONOID`.

    Args:
        assembly_text: NVIDIA SASS assembly text.
        sm_arch: Optional target SM architecture (e.g. `'sm_80'`, `'sm_90'`).

    Returns:
        Monoid-aggregated `SnippetComplianceReport`.
    """
    snap = load_or_extract_snapshot("nvidia_sass")
    return SNIPPET_REPORT_MONOID.mconcat(_check_sass_line(line, snap, sm_arch) for line in assembly_text.splitlines())


def _check_rdna_line(
    line: str,
    snap: typing.Mapping[str, JsonValue],
    gfx_arch: str | None,
) -> SnippetComplianceReport:
    """Evaluate a single AMD RDNA assembly line into a `SnippetComplianceReport`.

    Args:
        line: Raw RDNA assembly line.
        snap: Loaded AMD RDNA snapshot dictionary.
        gfx_arch: Optional target GFX architecture.

    Returns:
        `SnippetComplianceReport` contribution for `line`.
    """
    cleaned = re.sub(r"//.*", "", line).strip()
    if not cleaned or cleaned.startswith(";") or cleaned.startswith("#"):
        return SNIPPET_REPORT_MONOID.empty()

    matched = _RDNA_LINE_RE.match(cleaned)
    if not matched:
        return SNIPPET_REPORT_MONOID.empty()

    mnemonic = matched.group(1)
    operands_raw = matched.group(2)
    operands = [op.strip() for op in operands_raw.split(",") if op.strip()] if operands_raw else None

    res = verify_rdna_instruction(mnemonic, snap=snap, operands=operands, gfx_arch=gfx_arch)
    if not res.is_valid:
        return SnippetComplianceReport(
            is_compliant=False,
            total_instructions=1,
            verified_instructions=0,
            errors=res.errors,
        )
    return SnippetComplianceReport(
        is_compliant=True,
        total_instructions=1,
        verified_instructions=1,
    )


def check_rdna_assembly_compliance(assembly_text: str, gfx_arch: str | None = None) -> SnippetComplianceReport:
    """Verify AMD RDNA assembly snippet compliance via `SNIPPET_REPORT_MONOID`.

    Args:
        assembly_text: AMD RDNA assembly text.
        gfx_arch: Optional target GFX architecture (e.g. `'GFX11/RDNA3'`, `'GFX9/CDNA'`).

    Returns:
        Monoid-aggregated `SnippetComplianceReport`.
    """
    snap = load_or_extract_snapshot("amd_rdna")
    return SNIPPET_REPORT_MONOID.mconcat(_check_rdna_line(line, snap, gfx_arch) for line in assembly_text.splitlines())


__all__ = [
    "check_mlir_text_compliance",
    "check_rdna_assembly_compliance",
    "check_sass_assembly_compliance",
    "check_wgsl_shader_compliance",
]
