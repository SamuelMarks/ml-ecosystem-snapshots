"""Core SASS, RDNA, and PTX mnemonic, modifier, register, and architecture verifiers."""

from __future__ import annotations

import re
import typing

from ml_ecosystem_snapshots.grounding.models import IsaCheckResult
from ml_ecosystem_snapshots.models import JsonValue, ensure_json_dict, ensure_json_list


def _parse_sass_mnemonic_and_modifiers(
    mnemonic: str,
    modifiers: list[str] | None,
) -> tuple[str, list[str] | None]:
    """Split a dotted SASS mnemonic into its base instruction name and normalized modifier list.

    Args:
        mnemonic: Raw SASS instruction mnemonic (e.g., 'FADD.FTZ' or 'WGMMA').
        modifiers: Optional explicit list of modifier strings.

    Returns:
        Tuple of `(target_mnemonic_name, combined_modifiers_or_none)`.
    """
    raw_mnemonic = mnemonic.upper().strip()
    extracted_modifiers: list[str] = []
    if "." in raw_mnemonic:
        tokens = raw_mnemonic.split(".")
        target_name = tokens[0].strip()
        extracted_modifiers = [f".{t.strip()}" if not t.startswith(".") else t.strip() for t in tokens[1:] if t.strip()]
    else:
        target_name = raw_mnemonic

    combined_modifiers = list(modifiers or [])
    for em in extracted_modifiers:
        if em not in combined_modifiers:
            combined_modifiers.append(em)
    return target_name, (combined_modifiers if combined_modifiers else None)


def _find_sass_instruction_record(
    snap: dict[str, JsonValue],
    target_name: str,
) -> dict[str, JsonValue] | None:
    """Locate an NVIDIA SASS instruction record in the snapshot categories.

    Args:
        snap: Categorized NVIDIA SASS snapshot dictionary.
        target_name: Uppercase base instruction mnemonic.

    Returns:
        Matching instruction dictionary, or None if not found.
    """
    raw_cats = ensure_json_dict(snap.get("categories") or {})
    for _cat, raw_items in raw_cats.items():
        items = ensure_json_list(raw_items)
        for raw_item in items:
            item = ensure_json_dict(raw_item)
            name = str(item.get("mnemonic") or item.get("name") or "")
            if name and name.upper() == target_name:
                return dict(item)
    return None


def _check_sass_architecture_constraints(
    target_name: str,
    sm_arch: str | None,
    operands: list[str] | None,
    supported_archs: list[str],
) -> tuple[list[str], list[str]]:
    """Validate SASS instruction support against the target SM architecture.

    Args:
        target_name: Uppercase base SASS mnemonic.
        sm_arch: Optional target SM architecture string (e.g., 'sm_80', 'sm_90').
        operands: Optional list of operand strings.
        supported_archs: Initial list of supported SM architectures for the instruction.

    Returns:
        Tuple of `(filtered_supported_archs, architecture_errors)`.
    """
    errors: list[str] = []
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
            f"Instruction '{target_name}' is not supported on target architecture '{sm_arch}'. Supported architectures: {', '.join(supported_archs)}"
        )

    if sm_arch and operands:
        if any(op == "UR" or str(op).startswith("UR") for op in operands) and sm_arch in ["sm_70"]:
            errors.append(
                f"Uniform registers (UR) are only supported on sm_75+ architectures, but target architecture is '{sm_arch}'."
            )

    return supported_archs, errors


def _check_sass_operands_list(
    target_name: str,
    operands: list[str],
    sm_arch: str | None,
    inst: dict[str, JsonValue],
    meta: dict[str, JsonValue],
) -> list[str]:
    """Validate SASS operand counts, register syntax, and operand directionality.

    Args:
        target_name: Uppercase base SASS mnemonic.
        operands: List of operand strings passed to the instruction.
        sm_arch: Optional target SM architecture string.
        inst: Raw SASS instruction dictionary from the snapshot.
        meta: Domain metadata dictionary from the instruction record.

    Returns:
        List of operand validation error strings.
    """
    from ml_ecosystem_snapshots.frameworks.nvidia_sass import (
        validate_sass_operand_directionality,
        validate_sass_register,
    )

    errors: list[str] = []
    raw_sigs = meta.get("operand_signatures") or inst.get("operands")
    op_sigs = [sig for sig in raw_sigs if isinstance(sig, list)] if isinstance(raw_sigs, list) else []
    if op_sigs:
        matching_len = [sig for sig in op_sigs if len(sig) == len(operands)]
        if not matching_len:
            errors.append(
                f"Operand count mismatch for '{target_name}': got {len(operands)}, expected {[len(s) for s in op_sigs]}"
            )

    for i, op in enumerate(operands):
        role = "dst" if i == 0 else f"src{i - 1}"
        errors.extend(validate_sass_register(op, role=role, sm_arch=sm_arch))

    errors.extend(validate_sass_operand_directionality(operands, target_name))
    return errors


def verify_sass_instruction(
    mnemonic: str,
    snap: typing.Mapping[str, JsonValue],
    operands: list[str] | None = None,
    modifiers: list[str] | None = None,
    sm_arch: str | None = None,
    control_codes: typing.Mapping[str, JsonValue] | None = None,
) -> IsaCheckResult:
    """Validate whether an NVIDIA SASS assembly instruction is valid and supported.

    Args:
        mnemonic: Instruction mnemonic (e.g. 'FADD', 'WGMMA').
        operands: Optional list of operand strings (e.g. ['R', 'R', 'R']).
        modifiers: Optional list of instruction modifiers (e.g. ['.SAT', '.FTZ']).
        sm_arch: Optional target SM architecture (e.g. 'sm_80', 'sm_90').
        control_codes: Optional control code dictionary (e.g. {'stall_count': 1, 'latency_ticks': 4}).
        snap: Categorized NVIDIA SASS snapshot dictionary.

    Returns:
        `IsaCheckResult` validation report model with is_valid, supported_architectures, and errors.
    """
    target_name, modifiers = _parse_sass_mnemonic_and_modifiers(mnemonic, modifiers)
    inst = _find_sass_instruction_record(dict(snap), target_name)

    if not inst:
        return IsaCheckResult(
            is_valid=False,
            mnemonic_exists=False,
            errors=[f"Instruction mnemonic '{target_name}' does not exist in SASS ISA."],
        )

    raw_meta = inst.get("domain_metadata")
    meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
    from ml_ecosystem_snapshots.frameworks.nvidia_sass import (
        parse_sass_modifiers,
        resolve_sm_architectures,
        validate_sass_control_code,
        validate_sass_modifiers,
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
    inst_mods_list = [str(m) for m in raw_inst_mods] if isinstance(raw_inst_mods, list) else []
    valid_modifiers: list[str] = (
        [str(m) for m in raw_meta_mods]
        if isinstance(raw_meta_mods, list) and raw_meta_mods
        else parse_sass_modifiers(inst_mods_list)
    )

    supported_archs, errors = _check_sass_architecture_constraints(
        target_name=target_name,
        sm_arch=sm_arch,
        operands=operands,
        supported_archs=supported_archs,
    )

    if modifiers:
        for mod in modifiers:
            norm_mod = mod if mod.startswith(".") else f".{mod}"
            if norm_mod not in valid_modifiers:
                errors.append(
                    f"Modifier '{norm_mod}' is not recognized for '{target_name}'. Valid modifiers: {', '.join(valid_modifiers)}"
                )
        errors.extend(validate_sass_modifiers(target_name, modifiers))

    if operands is not None:
        errors.extend(_check_sass_operands_list(target_name, operands, sm_arch, inst, meta))

    if control_codes is not None:
        errors.extend(validate_sass_control_code(target_name, control_codes, sm_arch=sm_arch))

    return IsaCheckResult(
        is_valid=len(errors) == 0,
        mnemonic_exists=True,
        supported_architectures=supported_archs,
        valid_modifiers=valid_modifiers,
        errors=errors,
    )


def _find_rdna_instruction_record(
    snap: dict[str, JsonValue],
    target_name: str,
    base_target: str,
) -> dict[str, JsonValue] | None:
    """Locate an AMD RDNA instruction record in the snapshot or canonical fallback catalog.

    Args:
        snap: Categorized AMD RDNA snapshot dictionary.
        target_name: Full lowercase instruction mnemonic.
        base_target: Base lowercase mnemonic stripped of encoding suffix.

    Returns:
        Matched instruction dictionary, or None if not found.
    """
    raw_cats = ensure_json_dict(snap.get("categories") or {})
    for _cat, raw_items in raw_cats.items():
        items = ensure_json_list(raw_items)
        for raw_item in items:
            item = ensure_json_dict(raw_item)
            name = str(item.get("mnemonic") or item.get("name") or "").lower()
            if name in (target_name, base_target):
                return dict(item)

    from ml_ecosystem_snapshots.frameworks.amd_rdna import (
        _get_canonical_fallback_rdna,
    )

    for fb_item in _get_canonical_fallback_rdna():
        name = str(fb_item.get("mnemonic") or fb_item.get("name") or "").lower()
        if name in (target_name, base_target):
            return dict(fb_item)
    return None


def _check_rdna_operands_list(
    target_name: str,
    operands: list[str],
    gfx_arch: str | None,
    actual_encoding: str | None,
    inst: dict[str, JsonValue],
    meta: dict[str, JsonValue],
) -> list[str]:
    """Validate AMD RDNA operand count, register legality, and constant bus constraints.

    Args:
        target_name: Lowercase RDNA instruction mnemonic.
        operands: List of operand strings passed to the instruction.
        gfx_arch: Optional target GFX architecture string.
        actual_encoding: Resolved instruction encoding format string.
        inst: Raw RDNA instruction dictionary from the snapshot.
        meta: Domain metadata dictionary from the instruction record.

    Returns:
        List of operand validation error strings.
    """
    from ml_ecosystem_snapshots.frameworks.amd_rdna import (
        validate_rdna_constant_bus,
        validate_rdna_register,
    )

    errors: list[str] = []
    raw_sigs = meta.get("operand_signatures") or inst.get("operands")
    op_sigs = [sig for sig in raw_sigs if isinstance(sig, list)] if isinstance(raw_sigs, list) else []
    if op_sigs:
        matching_len = [sig for sig in op_sigs if len(sig) == len(operands)]
        if not matching_len:
            errors.append(
                f"Operand count mismatch for '{target_name}': got {len(operands)}, expected {[len(s) for s in op_sigs]}"
            )

    for op in operands:
        errors.extend(validate_rdna_register(op, gfx_arch=gfx_arch))

    errors.extend(validate_rdna_constant_bus(operands, encoding=actual_encoding, gfx_arch=gfx_arch))
    return errors


def verify_rdna_instruction(
    mnemonic: str,
    snap: typing.Mapping[str, JsonValue],
    operands: list[str] | None = None,
    encoding: str | None = None,
    gfx_arch: str | None = None,
    modifiers: list[str] | None = None,
    wave_size: int | None = None,
) -> IsaCheckResult:
    """Validate whether an AMD RDNA assembly instruction is valid and supported.

    Args:
        mnemonic: Instruction mnemonic (e.g. 'v_add_f32', 's_load_dword').
        operands: Optional list of operand type strings.
        encoding: Optional instruction encoding format (e.g. 'VOP2', 'SMEM').
        gfx_arch: Optional target GFX architecture (e.g. 'GFX11/RDNA3').
        modifiers: Optional list of modifiers (e.g. ['-src', 'clamp', 'omod:2']).
        wave_size: Optional wavefront execution size (32 or 64).
        snap: Categorized AMD RDNA snapshot dictionary.

    Returns:
        `IsaCheckResult` validation report model with is_valid, supported_architectures, and errors.
    """
    target_name = mnemonic.lower().strip()
    extracted_suffix = None
    suffix_match = re.match(r"^(v_[a-z0-9_]+)_(e32|e64|dpp\d*|sdwa|b32|b64)$", target_name)
    if suffix_match:
        base_target = suffix_match.group(1)
        extracted_suffix = f"_{suffix_match.group(2)}"
    else:
        base_target = target_name

    inst = _find_rdna_instruction_record(dict(snap), target_name, base_target)
    if not inst:
        return IsaCheckResult(
            is_valid=False,
            mnemonic_exists=False,
            errors=[f"Instruction mnemonic '{target_name}' does not exist in RDNA ISA."],
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
        validate_rdna_microarchitecture,
        validate_rdna_modifiers,
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

    errors.extend(validate_rdna_microarchitecture(target_name, gfx_arch=gfx_arch, wave_size=wave_size))

    if gfx_arch and gfx_arch not in supported_archs:
        errors.append(
            f"Instruction '{target_name}' is not supported on '{gfx_arch}'. Supported: {', '.join(supported_archs)}"
        )

    if encoding and actual_encoding and encoding.upper() != actual_encoding.upper():
        errors.append(f"Encoding mismatch for '{target_name}': specified '{encoding}', actual '{actual_encoding}'")

    if modifiers:
        errors.extend(validate_rdna_modifiers(modifiers, encoding=actual_encoding))

    if operands is not None:
        errors.extend(_check_rdna_operands_list(target_name, operands, gfx_arch, actual_encoding, inst, meta))

    return IsaCheckResult(
        is_valid=len(errors) == 0,
        mnemonic_exists=True,
        encoding=actual_encoding,
        supported_architectures=supported_archs,
        errors=errors,
    )


def verify_ptx_instruction(
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
    supp_types = [str(t) for t in raw_supp_types] if isinstance(raw_supp_types, list) else []
    return IsaCheckResult(
        is_valid=len(errors) == 0,
        mnemonic_exists=True,
        category=str(inst_meta.get("category") or "instruction"),
        min_sm=str(inst_meta.get("min_sm") or "sm_50"),
        supported_types=supp_types,
        errors=errors,
    )


__all__ = [
    "verify_ptx_instruction",
    "verify_rdna_instruction",
    "verify_sass_instruction",
]
