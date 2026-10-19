"""MCP tool module for hardware ISA instruction verification (NVIDIA SASS, AMD RDNA, NVIDIA PTX)."""

from __future__ import annotations

import re
import typing

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
        mnemonic: Instruction mnemonic (e.g. 'FADD', 'WGMMA').
        operands: Optional list of operand strings (e.g. ['R', 'R', 'R']).
        modifiers: Optional list of instruction modifiers (e.g. ['.SAT', '.FTZ']).
        sm_arch: Optional target SM architecture (e.g. 'sm_80', 'sm_90').
        control_codes: Optional control code dictionary (e.g. {'stall_count': 1, 'latency_ticks': 4}).
        repository: Optional `SnapshotRepository` instance for snapshot lookup.

    Returns:
        `IsaCheckResult` validation report model with is_valid, supported_architectures, and errors.
    """
    snap = resolve_snapshot("nvidia_sass", repository=repository)
    inst: dict[str, JsonValue] | None = None
    raw_mnemonic = mnemonic.upper().strip()
    extracted_modifiers: list[str] = []
    if "." in raw_mnemonic:
        tokens = raw_mnemonic.split(".")
        target_name = tokens[0].strip()
        extracted_modifiers = [
            f".{t.strip()}" if not t.startswith(".") else t.strip()
            for t in tokens[1:]
            if t.strip()
        ]
    else:
        target_name = raw_mnemonic

    combined_modifiers = list(modifiers or [])
    for em in extracted_modifiers:
        if em not in combined_modifiers:
            combined_modifiers.append(em)
    modifiers = combined_modifiers if combined_modifiers else None

    raw_cats = snap.get("categories") or {}
    assert isinstance(raw_cats, dict)
    for _cat, items in raw_cats.items():
        assert isinstance(items, list)
        for item in items:
            assert isinstance(item, dict)
            name = str(item.get("mnemonic") or item.get("name") or "")
            if name and name.upper() == target_name:
                inst = dict(item)
                break
        if inst:
            break

    if not inst:
        return IsaCheckResult(
            is_valid=False,
            mnemonic_exists=False,
            errors=[
                f"Instruction mnemonic '{target_name}' does not exist in SASS ISA."
            ],
        )

    errors: list[str] = []
    raw_meta = inst.get("domain_metadata")
    meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
    from ml_ecosystem_snapshots.frameworks.nvidia_sass import (
        parse_sass_modifiers,
        resolve_sm_architectures,
        validate_sass_control_code,
        validate_sass_modifiers,
        validate_sass_operand_directionality,
        validate_sass_register,
    )

    raw_arch = inst.get("architecture") or meta.get("architecture")
    raw_valid_archs = meta.get("valid_architectures")
    supported_archs: list[str] = (
        [str(a) for a in raw_valid_archs]
        if isinstance(raw_valid_archs, list) and raw_valid_archs
        else resolve_sm_architectures(raw_arch)
    )
    raw_meta_mods = meta.get("modifiers")
    raw_inst_mods = inst.get("modifiers")
    inst_mods_list = (
        [str(m) for m in raw_inst_mods] if isinstance(raw_inst_mods, list) else []
    )
    valid_modifiers: list[str] = (
        [str(m) for m in raw_meta_mods]
        if isinstance(raw_meta_mods, list) and raw_meta_mods
        else parse_sass_modifiers(inst_mods_list)
    )

    if target_name.startswith("WGMMA"):
        supported_archs = [a for a in supported_archs if a in ("sm_90", "sm_100")]
        if sm_arch and sm_arch not in ("sm_90", "sm_100"):
            errors.append(
                f"WGMMA instructions are strictly supported on sm_90+ architectures, but target architecture is '{sm_arch}'."
            )

    if any(p in target_name for p in ("FP4", "FP6", "MXFP", "E2M1", "E3M2")):
        supported_archs = [a for a in supported_archs if a in ("sm_100",)]
        if sm_arch and sm_arch not in ("sm_100",):
            errors.append(
                f"Microscopic scaling and FP4/FP6 instructions ('{target_name}') are strictly supported on sm_100+ architectures, but target architecture is '{sm_arch}'."
            )

    if sm_arch and sm_arch not in supported_archs:
        errors.append(
            f"Instruction '{target_name}' is not supported on target architecture '{sm_arch}'. "
            f"Supported architectures: {', '.join(supported_archs)}"
        )

    if sm_arch and operands:
        if any(
            op == "UR" or str(op).startswith("UR") for op in operands
        ) and sm_arch in ["sm_70"]:
            errors.append(
                f"Uniform registers (UR) are only supported on sm_75+ architectures, but target architecture is '{sm_arch}'."
            )

    if modifiers:
        for mod in modifiers:
            norm_mod = mod if mod.startswith(".") else f".{mod}"
            if norm_mod not in valid_modifiers:
                errors.append(
                    f"Modifier '{norm_mod}' is not recognized for '{target_name}'. "
                    f"Valid modifiers: {', '.join(valid_modifiers)}"
                )
        mod_conflicts = validate_sass_modifiers(target_name, modifiers)
        errors.extend(mod_conflicts)

    if operands is not None:
        raw_sigs = meta.get("operand_signatures") or inst.get("operands")
        op_sigs = (
            [sig for sig in raw_sigs if isinstance(sig, list)]
            if isinstance(raw_sigs, list)
            else []
        )
        if op_sigs:
            matching_len = [sig for sig in op_sigs if len(sig) == len(operands)]
            if not matching_len:
                errors.append(
                    f"Operand count mismatch for '{target_name}': got {len(operands)}, "
                    f"expected {[len(s) for s in op_sigs]}"
                )

        for i, op in enumerate(operands):
            role = "dst" if i == 0 else f"src{i - 1}"
            reg_errs = validate_sass_register(op, role=role, sm_arch=sm_arch)
            errors.extend(reg_errs)

        dir_errs = validate_sass_operand_directionality(operands, target_name)
        errors.extend(dir_errs)

    if control_codes is not None:
        cc_errs = validate_sass_control_code(
            target_name, control_codes, sm_arch=sm_arch
        )
        errors.extend(cc_errs)

    return IsaCheckResult(
        is_valid=len(errors) == 0,
        mnemonic_exists=True,
        supported_architectures=supported_archs,
        valid_modifiers=valid_modifiers,
        errors=errors,
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
        mnemonic: Instruction mnemonic (e.g. 'v_add_f32', 's_load_dword').
        operands: Optional list of operand type strings.
        encoding: Optional instruction encoding format (e.g. 'VOP2', 'SMEM').
        gfx_arch: Optional target GFX architecture (e.g. 'GFX11/RDNA3').
        modifiers: Optional list of modifiers (e.g. ['-src', 'clamp', 'omod:2']).
        wave_size: Optional wavefront execution size (32 or 64).
        repository: Optional `SnapshotRepository` instance for snapshot lookup.

    Returns:
        `IsaCheckResult` validation report model with is_valid, supported_architectures, and errors.
    """
    snap = resolve_snapshot("amd_rdna", repository=repository)
    inst: dict[str, JsonValue] | None = None
    target_name = mnemonic.lower().strip()
    extracted_suffix = None
    suffix_match = re.match(
        r"^(v_[a-z0-9_]+)_(e32|e64|dpp\d*|sdwa|b32|b64)$", target_name
    )
    if suffix_match:
        base_target = suffix_match.group(1)
        extracted_suffix = f"_{suffix_match.group(2)}"
    else:
        base_target = target_name

    raw_cats = snap.get("categories") or {}
    assert isinstance(raw_cats, dict)
    for _cat, items in raw_cats.items():
        assert isinstance(items, list)
        for item in items:
            assert isinstance(item, dict)
            name = str(item.get("mnemonic") or item.get("name") or "").lower()
            if name in (target_name, base_target):
                inst = dict(item)
                break
        if inst:
            break

    if not inst:
        from ml_ecosystem_snapshots.frameworks.amd_rdna import (
            _get_canonical_fallback_rdna,
        )

        for fb_item in _get_canonical_fallback_rdna():
            name = str(fb_item.get("mnemonic") or fb_item.get("name") or "").lower()
            if name in (target_name, base_target):
                inst = dict(fb_item)
                break

    if not inst:
        return IsaCheckResult(
            is_valid=False,
            mnemonic_exists=False,
            errors=[
                f"Instruction mnemonic '{target_name}' does not exist in RDNA ISA."
            ],
        )

    if extracted_suffix:
        if modifiers is None:
            modifiers = [extracted_suffix]
        elif extracted_suffix not in modifiers:
            modifiers = list(modifiers) + [extracted_suffix]

    errors: list[str] = []
    raw_meta = inst.get("domain_metadata")
    meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
    from ml_ecosystem_snapshots.frameworks.amd_rdna import (
        resolve_gfx_architectures,
        validate_rdna_constant_bus,
        validate_rdna_microarchitecture,
        validate_rdna_modifiers,
        validate_rdna_register,
    )

    raw_arch = inst.get("architecture") or meta.get("architecture")
    raw_valid_archs = meta.get("valid_architectures")
    supported_archs: list[str] = (
        [str(a) for a in raw_valid_archs]
        if isinstance(raw_valid_archs, list) and raw_valid_archs
        else resolve_gfx_architectures(raw_arch)
    )
    raw_enc = inst.get("encoding") or meta.get("encoding")
    actual_encoding = str(raw_enc) if raw_enc is not None else None

    uarch_errors = validate_rdna_microarchitecture(
        target_name, gfx_arch=gfx_arch, wave_size=wave_size
    )
    errors.extend(uarch_errors)

    if gfx_arch and gfx_arch not in supported_archs:
        errors.append(
            f"Instruction '{target_name}' is not supported on '{gfx_arch}'. "
            f"Supported: {', '.join(supported_archs)}"
        )

    if encoding and actual_encoding and encoding.upper() != actual_encoding.upper():
        errors.append(
            f"Encoding mismatch for '{target_name}': specified '{encoding}', actual '{actual_encoding}'"
        )

    if modifiers:
        mod_errors = validate_rdna_modifiers(modifiers, encoding=actual_encoding)
        errors.extend(mod_errors)

    if operands is not None:
        raw_sigs = meta.get("operand_signatures") or inst.get("operands")
        op_sigs = (
            [sig for sig in raw_sigs if isinstance(sig, list)]
            if isinstance(raw_sigs, list)
            else []
        )
        if op_sigs:
            matching_len = [sig for sig in op_sigs if len(sig) == len(operands)]
            if not matching_len:
                errors.append(
                    f"Operand count mismatch for '{target_name}': got {len(operands)}, "
                    f"expected {[len(s) for s in op_sigs]}"
                )

        for op in operands:
            reg_errs = validate_rdna_register(op, gfx_arch=gfx_arch)
            errors.extend(reg_errs)

        cbus_errs = validate_rdna_constant_bus(
            operands, encoding=actual_encoding, gfx_arch=gfx_arch
        )
        errors.extend(cbus_errs)

    return IsaCheckResult(
        is_valid=len(errors) == 0,
        mnemonic_exists=True,
        encoding=actual_encoding,
        supported_architectures=supported_archs,
        errors=errors,
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
        mnemonic: PTX instruction mnemonic (e.g. 'add', 'ld', 'st', 'wgmma.mma_async').
        types: Optional list of PTX type qualifiers (e.g. ['.f32'], ['.u64']).
        operands: Optional list of register/immediate operand strings.
        state_space: Optional memory state space qualifier (e.g. '.global').
        scope: Optional memory scope qualifier (e.g. '.gpu', '.cta', '.sys').
        vector_width: Optional vector width qualifier (e.g. '.v2', '.v4').
        sm_arch: Optional target SM architecture (e.g. 'sm_80', 'sm_90').

    Returns:
        `IsaCheckResult` validation report model with is_valid, mnemonic_exists, and errors.
    """
    from ml_ecosystem_snapshots.frameworks.nvidia_ptx import (
        _load_exhaustive_ptx,
        validate_ptx_instruction,
    )

    clean_mnem = mnemonic.strip().lower()
    ptx_db = {str(inst["mnemonic"]): inst for inst in _load_exhaustive_ptx()}
    if clean_mnem not in ptx_db:
        return IsaCheckResult(
            is_valid=False,
            mnemonic_exists=False,
            errors=[f"Instruction mnemonic '{mnemonic}' does not exist in PTX ISA."],
        )

    errors = validate_ptx_instruction(
        clean_mnem,
        types=types,
        operands=operands,
        state_space=state_space,
        scope=scope,
        vector_width=vector_width,
        sm_arch=sm_arch,
    )

    inst_meta = ptx_db[clean_mnem]
    raw_supp_types = inst_meta.get("supported_types")
    supp_types = (
        [str(t) for t in raw_supp_types] if isinstance(raw_supp_types, list) else []
    )
    return IsaCheckResult(
        is_valid=len(errors) == 0,
        mnemonic_exists=True,
        category=str(inst_meta.get("category") or "instruction"),
        min_sm=str(inst_meta.get("min_sm") or "sm_50"),
        supported_types=supp_types,
        errors=errors,
    )


__all__ = [
    "check_ptx_instruction",
    "check_rdna_instruction",
    "check_sass_instruction",
]
