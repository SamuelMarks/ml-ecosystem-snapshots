"""Text-block assembly and IR snippet compliance verifiers (MLIR, WGSL, SASS, RDNA)."""

from __future__ import annotations

import re

from ml_ecosystem_snapshots.compliance.models import SnippetComplianceReport
from ml_ecosystem_snapshots.grounding.compiler import verify_mlir_op
from ml_ecosystem_snapshots.grounding.domain_ops import check_domain_op
from ml_ecosystem_snapshots.grounding.hardware import (
    verify_rdna_instruction,
    verify_sass_instruction,
)
from ml_ecosystem_snapshots.serialization.repository import load_or_extract_snapshot


def check_mlir_text_compliance(mlir_text: str) -> SnippetComplianceReport:
    """Verify that an MLIR / StableHLO text snippet uses valid operations, attributes, and operands.

    Args:
        mlir_text: Text snippet of MLIR or StableHLO assembly code.

    Returns:
        `SnippetComplianceReport` with `is_compliant`, `total_ops`, `verified_ops`, and `errors`.
    """
    errors: list[str] = []
    total_ops = 0
    verified_ops = 0

    op_pattern = re.compile(r"(?:%\w+\s*=\s*)?([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)\s*(?:\(([^)]*)\))?")

    for line in mlir_text.splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("//"):
            continue

        match = op_pattern.search(cleaned)
        if match:
            op_name = match.group(1)
            if op_name in ("builtin.module", "func.func"):
                continue

            total_ops += 1
            args_str = match.group(2)
            operands_count = len([a for a in args_str.split(",") if a.strip()]) if args_str is not None else None
            fw = "stablehlo" if op_name.startswith("stablehlo.") else "mlir"
            snap = load_or_extract_snapshot(fw)
            res = verify_mlir_op(op_name, snap=snap, operands_count=operands_count)
            if not res.op_exists:
                errors.append(f"Invalid MLIR operation '{op_name}' on line: '{cleaned}'")
            elif not res.is_valid:
                errors.extend(res.errors)
            else:
                verified_ops += 1

    return SnippetComplianceReport(
        is_compliant=len(errors) == 0,
        total_ops=total_ops,
        verified_ops=verified_ops,
        errors=errors,
    )


def check_wgsl_shader_compliance(shader_text: str) -> SnippetComplianceReport:
    """Verify that a WebGPU WGSL shader snippet uses valid builtin operations.

    Args:
        shader_text: Text snippet of WebGPU WGSL shader code.

    Returns:
        `SnippetComplianceReport` with `is_compliant`, `total_ops`, `verified_ops`, and `errors`.
    """
    errors: list[str] = []
    total_ops = 0
    verified_ops = 0

    call_pattern = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)")

    ignored_keywords = {
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

    for line in shader_text.splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("//"):
            continue

        for match in call_pattern.finditer(cleaned):
            op_name = match.group(1)
            if op_name in ignored_keywords:
                continue

            args_raw = match.group(2).strip()
            inputs_count = len([a for a in args_raw.split(",") if a.strip()]) if args_raw else 0
            res = check_domain_op("wgsl", op_name, inputs_count=inputs_count)
            if res.op_exists:
                total_ops += 1
                if res.is_valid:
                    verified_ops += 1
                else:
                    errors.extend(res.errors)

    return SnippetComplianceReport(
        is_compliant=len(errors) == 0,
        total_ops=total_ops,
        verified_ops=verified_ops,
        errors=errors,
    )


def check_sass_assembly_compliance(assembly_text: str, sm_arch: str | None = None) -> SnippetComplianceReport:
    """Verify that an NVIDIA SASS assembly snippet is valid for the target SM architecture.

    Args:
        assembly_text: NVIDIA SASS assembly text.
        sm_arch: Optional target SM architecture (e.g. 'sm_80', 'sm_90').

    Returns:
        `SnippetComplianceReport` with `is_compliant`, `total_instructions`, and `errors`.
    """
    errors: list[str] = []
    total_insts = 0
    verified_insts = 0
    snap = load_or_extract_snapshot("nvidia_sass")

    sass_line_pattern = re.compile(r"^(?:@!?P\d+\s+)?([A-Z0-9_]+)((?:\.[A-Z0-9_]+)*)(?:\s+(.*?))?;?$")

    for line in assembly_text.splitlines():
        cleaned = re.sub(r"/\*.*?\*/", "", line).strip()
        if not cleaned or cleaned.startswith("//") or cleaned.startswith("#"):
            continue

        m = sass_line_pattern.match(cleaned)
        if m:
            base_mnemonic = m.group(1)
            mods_raw = m.group(2)
            operands_raw = m.group(3)

            modifiers = [mod for mod in mods_raw.split(".") if mod] if mods_raw else []
            operands = [op.strip() for op in operands_raw.split(",") if op.strip()] if operands_raw else None

            total_insts += 1
            res = verify_sass_instruction(
                base_mnemonic,
                snap=snap,
                operands=operands,
                modifiers=modifiers,
                sm_arch=sm_arch,
            )
            if not res.is_valid:
                errors.extend(res.errors)
            else:
                verified_insts += 1

    return SnippetComplianceReport(
        is_compliant=len(errors) == 0,
        total_instructions=total_insts,
        verified_instructions=verified_insts,
        errors=errors,
    )


def check_rdna_assembly_compliance(assembly_text: str, gfx_arch: str | None = None) -> SnippetComplianceReport:
    """Verify that an AMD RDNA assembly snippet is valid for the target GFX architecture.

    Args:
        assembly_text: AMD RDNA assembly text.
        gfx_arch: Optional target GFX architecture (e.g. 'GFX11/RDNA3', 'GFX9/CDNA').

    Returns:
        `SnippetComplianceReport` with `is_compliant`, `total_instructions`, and `errors`.
    """
    errors: list[str] = []
    total_insts = 0
    verified_insts = 0
    snap = load_or_extract_snapshot("amd_rdna")

    rdna_line_pattern = re.compile(r"^([a-z0-9_]+)(?:\s+(.*?))?;?$")

    for line in assembly_text.splitlines():
        cleaned = re.sub(r"//.*", "", line).strip()
        if not cleaned or cleaned.startswith(";") or cleaned.startswith("#"):
            continue

        m = rdna_line_pattern.match(cleaned)
        if m:
            mnemonic = m.group(1)
            operands_raw = m.group(2)
            operands = [op.strip() for op in operands_raw.split(",") if op.strip()] if operands_raw else None

            total_insts += 1
            res = verify_rdna_instruction(mnemonic, snap=snap, operands=operands, gfx_arch=gfx_arch)
            if not res.is_valid:
                errors.extend(res.errors)
            else:
                verified_insts += 1

    return SnippetComplianceReport(
        is_compliant=len(errors) == 0,
        total_instructions=total_insts,
        verified_instructions=verified_insts,
        errors=errors,
    )


__all__ = [
    "check_mlir_text_compliance",
    "check_rdna_assembly_compliance",
    "check_sass_assembly_compliance",
    "check_wgsl_shader_compliance",
]
