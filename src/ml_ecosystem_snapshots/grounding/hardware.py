"""Hardware instruction grounding verifiers for NVIDIA SASS and AMD RDNA/CDNA."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ml_ecosystem_snapshots.grounding.engine import GroundingEngine
from ml_ecosystem_snapshots.grounding.models import (
    DiagnosticSeverity,
    GroundingReport,
)


def validate_sass_instruction(
    mnemonic: str,
    architecture: str,
    operands: List[str],
    modifiers: Optional[List[str]] = None,
    engine: Optional[GroundingEngine] = None,
    control_codes: Optional[Dict[str, Any]] = None,
) -> GroundingReport:
    """Verify an NVIDIA SASS instruction against the ground-truth offline ISA catalog.

    Delegates register legality, operand directionality, modifier conflict checks,
    and control code verification to the canonical NVIDIA SASS ISA validators.

    Args:
        mnemonic: The instruction mnemonic (e.g., 'FFMA', 'HMMA', 'WGMMA', 'LDG').
        architecture: Target SM architecture (e.g., 'sm_70', 'sm_80', 'sm_90', 'sm_100').
        operands: List of operand expressions (e.g., ['R1', 'R2', 'R3', 'R4']).
        modifiers: Optional instruction modifier flags (e.g., ['.F16', '.FTZ']).
        engine: Optional GroundingEngine instance. If omitted, a singleton is used.
        control_codes: Optional control code dictionary (e.g., {'stall_count': 1}).

    Returns:
        GroundingReport containing diagnostic results and grounding status.
    """
    from ml_ecosystem_snapshots.frameworks.nvidia_sass import (
        validate_sass_control_code,
        validate_sass_modifiers,
        validate_sass_operand_directionality,
        validate_sass_register,
    )

    eng = engine or GroundingEngine()
    clean_mnemonic = mnemonic.upper().strip()
    clean_arch = architecture.lower().strip()
    mods = modifiers or []

    report = GroundingReport(
        is_grounded=True,
        target="nvidia_sass",
        symbol=clean_mnemonic,
    )

    ref = eng.get_symbol("nvidia_sass", clean_mnemonic)
    if not ref:
        suggested = eng.suggest_closest_symbol("nvidia_sass", clean_mnemonic)
        report.add_diagnostic(
            field="mnemonic",
            message=f"Unrecognized SASS instruction mnemonic '{clean_mnemonic}'.",
            severity=DiagnosticSeverity.ERROR,
            suggested_fix=suggested,
        )
        return report

    report.matched_ref = ref

    # Verify architecture support
    valid_archs = ref.environment_tags or []
    if clean_arch not in valid_archs and valid_archs:
        report.add_diagnostic(
            field="architecture",
            message=(
                f"Instruction '{clean_mnemonic}' is not supported on target architecture '{clean_arch}'. "
                f"Supported architectures: {', '.join(valid_archs)}"
            ),
            severity=DiagnosticSeverity.ERROR,
            suggested_fix=valid_archs[-1] if valid_archs else None,
        )

    # Verify modifier legality and conflicts
    domain_meta = ref.domain_metadata or {}
    allowed_modifiers = domain_meta.get("modifiers") or []
    for mod in mods:
        if allowed_modifiers and mod not in allowed_modifiers:
            report.add_diagnostic(
                field="modifiers",
                message=f"Modifier '{mod}' is not recognized for '{clean_mnemonic}'.",
                severity=DiagnosticSeverity.WARNING,
            )
    if mods:
        for mod_err in validate_sass_modifiers(clean_mnemonic, mods):
            report.add_diagnostic(
                field="modifiers",
                message=mod_err,
                severity=DiagnosticSeverity.ERROR,
            )

    # Verify operand counts
    allowed_sigs = domain_meta.get("operand_signatures") or []
    if allowed_sigs:
        matching_count_sig = [sig for sig in allowed_sigs if len(sig) == len(operands)]
        if not matching_count_sig:
            allowed_counts = sorted(list({len(sig) for sig in allowed_sigs}))
            report.add_diagnostic(
                field="operands",
                message=(
                    f"Instruction '{clean_mnemonic}' expects {allowed_counts} operands, "
                    f"but received {len(operands)}: {operands}."
                ),
                severity=DiagnosticSeverity.ERROR,
            )

    # Delegate register legality and operand directionality checks
    for i, op in enumerate(operands):
        role = "dst" if i == 0 else f"src{i - 1}"
        for reg_err in validate_sass_register(
            op, role=role, sm_arch=clean_arch if clean_arch else None
        ):
            report.add_diagnostic(
                field=f"operands[{i}]",
                message=reg_err,
                severity=DiagnosticSeverity.ERROR,
            )

    for dir_err in validate_sass_operand_directionality(operands, clean_mnemonic):
        report.add_diagnostic(
            field="operands",
            message=dir_err,
            severity=DiagnosticSeverity.ERROR,
        )

    if control_codes is not None:
        for cc_err in validate_sass_control_code(
            clean_mnemonic,
            control_codes,
            sm_arch=clean_arch if clean_arch else None,
        ):
            report.add_diagnostic(
                field="control_codes",
                message=cc_err,
                severity=DiagnosticSeverity.ERROR,
            )

    return report


def validate_rdna_instruction(
    mnemonic: str,
    gfx_arch: str,
    operands: List[str],
    modifiers: Optional[List[str]] = None,
    engine: Optional[GroundingEngine] = None,
    encoding: Optional[str] = None,
    wave_size: Optional[int] = None,
) -> GroundingReport:
    """Verify an AMD RDNA / CDNA instruction against the ground-truth offline ISA catalog.

    Delegates architecture resolution, microarchitecture/wavefront rules, modifier
    validation, register legality, and constant-bus limits to the canonical AMD RDNA
    validators.

    Args:
        mnemonic: The instruction mnemonic (e.g., 'V_ADD_F32', 'V_DUAL_FMAC_F32').
        gfx_arch: Target GFX generation (e.g., 'GFX9', 'GFX10', 'GFX11', 'GFX12').
        operands: List of operand expressions (e.g., ['v0', 'v1', 's0']).
        modifiers: Optional modifier flags (e.g., ['clamp', 'omod']).
        engine: Optional GroundingEngine instance.
        encoding: Optional instruction encoding format (e.g., 'VOP2', 'VOP3').
        wave_size: Optional wavefront execution size (32 or 64).

    Returns:
        GroundingReport containing diagnostic results and grounding status.
    """
    from ml_ecosystem_snapshots.frameworks.amd_rdna import (
        resolve_gfx_architectures,
        validate_rdna_constant_bus,
        validate_rdna_microarchitecture,
        validate_rdna_modifiers,
        validate_rdna_register,
    )

    eng = engine or GroundingEngine()
    clean_mnemonic = mnemonic.upper().strip()

    report = GroundingReport(
        is_grounded=True,
        target="amd_rdna",
        symbol=clean_mnemonic,
    )

    ref = eng.get_symbol("amd_rdna", clean_mnemonic)
    if not ref:
        suggested = eng.suggest_closest_symbol("amd_rdna", clean_mnemonic)
        report.add_diagnostic(
            field="mnemonic",
            message=f"Unrecognized AMD RDNA instruction mnemonic '{clean_mnemonic}'.",
            severity=DiagnosticSeverity.ERROR,
            suggested_fix=suggested,
        )
        return report

    report.matched_ref = ref
    domain_meta = ref.domain_metadata or {}
    actual_encoding = getattr(ref, "encoding", None) or domain_meta.get("encoding")
    arch_str = getattr(ref, "architecture", None) or domain_meta.get("architecture")
    supported_archs = (
        domain_meta.get("valid_architectures")
        or ref.environment_tags
        or resolve_gfx_architectures(arch_str)
    )

    if gfx_arch and supported_archs:
        norm_gfx = gfx_arch.strip()
        if not any(
            norm_gfx.upper() == a.upper() or a.upper().startswith(norm_gfx.upper())
            for a in supported_archs
        ):
            report.add_diagnostic(
                field="gfx_arch",
                message=(
                    f"Instruction '{clean_mnemonic}' is not supported on '{gfx_arch}'. "
                    f"Supported: {', '.join(supported_archs)}"
                ),
                severity=DiagnosticSeverity.ERROR,
            )

    for uarch_err in validate_rdna_microarchitecture(
        clean_mnemonic.lower(), gfx_arch=gfx_arch, wave_size=wave_size
    ):
        report.add_diagnostic(
            field="gfx_arch",
            message=uarch_err,
            severity=DiagnosticSeverity.ERROR,
        )

    if (
        encoding
        and actual_encoding
        and encoding.upper() != str(actual_encoding).upper()
    ):
        report.add_diagnostic(
            field="encoding",
            message=(
                f"Encoding mismatch for '{clean_mnemonic}': specified '{encoding}', "
                f"actual '{actual_encoding}'"
            ),
            severity=DiagnosticSeverity.ERROR,
        )

    if modifiers:
        for mod_err in validate_rdna_modifiers(modifiers, encoding=actual_encoding):
            report.add_diagnostic(
                field="modifiers",
                message=mod_err,
                severity=DiagnosticSeverity.ERROR,
            )

    op_sigs = domain_meta.get("operand_signatures") or []
    if op_sigs and not any(":" in op for op in operands):
        matching_len = [sig for sig in op_sigs if len(sig) == len(operands)]
        if not matching_len:
            report.add_diagnostic(
                field="operands",
                message=(
                    f"Operand count mismatch for '{clean_mnemonic}': got {len(operands)}, "
                    f"expected {[len(s) for s in op_sigs]}"
                ),
                severity=DiagnosticSeverity.ERROR,
            )

    # Check 64-bit even-index register alignment: v[n:n+1] requires n to be even
    even_reg_pattern = re.compile(r"^[vs]\[(\d+):(\d+)\]$", re.IGNORECASE)
    for idx, op in enumerate(operands):
        match = even_reg_pattern.match(op.strip())
        if match:
            start_reg = int(match.group(1))
            end_reg = int(match.group(2))
            if end_reg - start_reg == 1 and (start_reg % 2 != 0):
                report.add_diagnostic(
                    field=f"operands[{idx}]",
                    message=(
                        f"64-bit register pair '{op}' must be even-aligned. "
                        f"Register index {start_reg} is odd."
                    ),
                    severity=DiagnosticSeverity.ERROR,
                    suggested_fix=f"v[{start_reg - 1}:{start_reg}]",
                )
                continue
        for reg_err in validate_rdna_register(op, gfx_arch=gfx_arch):
            report.add_diagnostic(
                field=f"operands[{idx}]",
                message=reg_err,
                severity=DiagnosticSeverity.ERROR,
            )

    for cbus_err in validate_rdna_constant_bus(
        operands, encoding=actual_encoding, gfx_arch=gfx_arch
    ):
        report.add_diagnostic(
            field="operands",
            message=cbus_err,
            severity=DiagnosticSeverity.ERROR,
        )

    return report


def validate_ptx_instruction(
    mnemonic: str,
    sm_arch: Optional[str] = None,
    types: Optional[List[str]] = None,
    operands: Optional[List[str]] = None,
    state_space: Optional[str] = None,
    scope: Optional[str] = None,
    vector_width: Optional[str] = None,
    engine: Optional[GroundingEngine] = None,
) -> GroundingReport:
    """Verify an NVIDIA PTX instruction against the ground-truth offline ISA catalog.

    Args:
        mnemonic: The PTX instruction mnemonic (e.g., 'add', 'wgmma.mma_async').
        sm_arch: Optional target SM architecture (e.g., 'sm_70', 'sm_80', 'sm_90').
        types: Optional list of PTX type qualifiers (e.g. ['.f32', '.f16']).
        operands: Optional list of register/memory operands.
        state_space: Optional memory state space (e.g. '.global', '.shared').
        scope: Optional memory visibility scope (e.g. '.cta', '.gpu').
        vector_width: Optional vector width (e.g. '.v2', '.v4').
        engine: Optional GroundingEngine instance.

    Returns:
        GroundingReport containing diagnostic results and grounding status.
    """
    from ml_ecosystem_snapshots.frameworks.nvidia_ptx import (
        _load_exhaustive_ptx,
        validate_ptx_instruction as raw_validate_ptx,
    )

    eng = engine or GroundingEngine()
    clean_mnemonic = mnemonic.lower().strip()

    report = GroundingReport(
        is_grounded=True,
        target="nvidia_ptx",
        symbol=clean_mnemonic,
    )

    ptx_db = {inst["mnemonic"]: inst for inst in _load_exhaustive_ptx()}
    if clean_mnemonic not in ptx_db:
        suggested = eng.suggest_closest_symbol("nvidia_ptx", clean_mnemonic)
        report.add_diagnostic(
            field="mnemonic",
            message=f"Unrecognized NVIDIA PTX instruction mnemonic '{clean_mnemonic}'.",
            severity=DiagnosticSeverity.ERROR,
            suggested_fix=suggested,
        )
        return report

    ref = eng.get_symbol("nvidia_ptx", clean_mnemonic)
    if ref:
        report.matched_ref = ref

    errors = raw_validate_ptx(
        clean_mnemonic,
        types=types,
        operands=operands,
        state_space=state_space,
        scope=scope,
        vector_width=vector_width,
        sm_arch=sm_arch,
    )
    for err in errors:
        report.add_diagnostic(
            field="instruction",
            message=err,
            severity=DiagnosticSeverity.ERROR,
        )

    return report


__all__ = [
    "validate_ptx_instruction",
    "validate_rdna_instruction",
    "validate_sass_instruction",
]
