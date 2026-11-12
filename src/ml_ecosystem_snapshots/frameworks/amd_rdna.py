"""AMD RDNA (Assembly) Extractor.

Provides a static snapshot of AMD RDNA instructions with TableGen-derived encodings,
register classes, operand directionality, and GFX architecture targeting.
"""

from __future__ import annotations

import json
import os
import re
import typing
from enum import Enum

from ml_switcheroo_ir.schema.ghost import (
    GhostParam,
    GhostRef,
    ParameterKind,
    SemanticTier,
)

from ml_ecosystem_snapshots.frameworks._base import CollectorDomain, DomainCollectorAdapter

from ..models import (
    ExtendedGhostParam,
    ExtendedGhostRef,
    GhostIsaRef,
    IRParameterRole,
    JsonValue,
    OperandDirection,
)

ALL_GFX_ARCHITECTURES: list[str] = [
    "GFX9/CDNA",
    "GFX10/RDNA1",
    "GFX10.3/RDNA2",
    "GFX11/RDNA3",
    "GFX11.5",
    "GFX12/RDNA4",
]


class RDNAEncodingProfile(str, Enum):
    """Instruction encoding profiles in AMD RDNA/CDNA architectures."""

    VOP1 = "VOP1"
    VOP2 = "VOP2"
    VOP3 = "VOP3"
    VOP3P = "VOP3P"
    VOPC = "VOPC"
    VOPD = "VOPD"
    SOP1 = "SOP1"
    SOP2 = "SOP2"
    SOPK = "SOPK"
    SOPP = "SOPP"
    SMEM = "SMEM"
    FLAT = "FLAT"
    GLOBAL = "GLOBAL"
    SCRATCH = "SCRATCH"
    DS = "DS"


class RDNARegisterClass(str, Enum):
    """Register classes in AMD RDNA and CDNA ISAs."""

    VGPR_32 = "VGPR_32"
    VReg_64 = "VReg_64"
    VReg_96 = "VReg_96"
    VReg_128 = "VReg_128"
    VReg_256 = "VReg_256"
    VReg_512 = "VReg_512"
    SGPR_32 = "SGPR_32"
    SReg_64 = "SReg_64"
    AReg_32 = "AReg_32"


RDNA_SOURCE_MODIFIERS: set[str] = {
    "-src",
    "|src|",
    "clamp",
    "op_sel",
    "neg_lo",
    "neg_hi",
}

RDNA_OUTPUT_MODIFIERS: set[str] = {
    "omod",
    "omod:2",
    "omod:4",
    "omod:div2",
}


_RDNA_SLICE_ALIGNMENTS: dict[int, int] = {
    1: 2,
    3: 4,
    7: 8,
    15: 16,
}


def _validate_rdna_slice_register(operand: str, slice_match: re.Match[str]) -> list[str]:
    """Validate register bounds, span, and alignment for a sliced RDNA register range.

    Args:
        operand: Original register operand string for error reporting.
        slice_match: Regex match containing register prefix, start index, and end index.

    Returns:
        List of register range validation errors.
    """
    reg_type = slice_match.group(1).lower()
    start_idx = int(slice_match.group(2))
    end_idx = int(slice_match.group(3))
    span = end_idx - start_idx

    max_idx = 105 if reg_type == "s" else 255
    if start_idx > max_idx or end_idx > max_idx:
        return [f"Register index out of range in '{operand}': maximum index for {reg_type.upper()} is {max_idx}."]

    align = _RDNA_SLICE_ALIGNMENTS.get(span)
    if align is None:
        return [
            f"Invalid register span in '{operand}': expected pair (span 1), quad (span 3), oct (span 7), or hex (span 15), got span {span}."
        ]

    if start_idx % align != 0:
        if span == 1:
            return [
                f"Register alignment error: 64-bit register pair '{operand}' must start on an even register index (got {start_idx})."
            ]
        bits = (span + 1) * 32
        return [
            f"Register alignment error: {bits}-bit register '{operand}' must start on a modulo-{align} register index (got {start_idx})."
        ]
    return []


_RDNA_SINGLE_REG_LIMITS: tuple[tuple[str, int, str], ...] = (
    ("v", 255, "Vector register index v{idx} out of range (v0-v255)."),
    ("s", 105, "Scalar register index s{idx} out of range (s0-s105)."),
    ("a", 255, "Accumulator register index a{idx} out of range (a0-a255)."),
)


def _validate_rdna_single_register(cleaned: str) -> list[str]:
    """Validate a single 32-bit vector, scalar, or accumulator register index.

    Args:
        cleaned: Stripped register token string.

    Returns:
        List of single-register index range error messages.
    """
    for prefix, limit, msg_template in _RDNA_SINGLE_REG_LIMITS:
        reg_match = re.match(rf"^{prefix}(\d+)$", cleaned, re.IGNORECASE)
        if reg_match:
            idx = int(reg_match.group(1))
            return [msg_template.format(idx=idx)] if idx > limit else []
    return []


def validate_rdna_register(operand: str, gfx_arch: str | None = None) -> list[str]:
    """Validate a single AMD RDNA/CDNA register specification against alignment and bounds.

    Validates:
        - 64-bit register pair (v[n:n+1], s[n:n+1], a[n:n+1]) requiring even starting index
        - 128-bit register quad (v[n:n+3], a[n:n+3], s[n:n+3]) requiring modulo-4 starting index
        - CDNA matrix accumulator (a[...], AReg_32) restricted to GFX9/CDNA architectures
        - Individual 32-bit registers (v0-v255, s0-s105, a0-a255)

    Args:
        operand: Raw operand string (e.g. 'v[0:1]', 's[1:2]', 'a[0:3]', 'v0').
        gfx_arch: Target GFX architecture string (e.g. 'GFX11/RDNA3', 'GFX9/CDNA').

    Returns:
        List of register validation errors.
    """
    errors: list[str] = []
    cleaned = operand.strip()

    # Memory wrapper handling: e.g. [v0, s[0:1]]
    if cleaned.startswith("[") and not cleaned.lower().startswith(("v[", "s[", "a[")) and cleaned.endswith("]"):
        inner = cleaned[1:-1].strip()
        for part in (p.strip() for p in inner.split(",") if p.strip()):
            errors.extend(validate_rdna_register(part, gfx_arch=gfx_arch))
        return errors

    # Check CDNA matrix accumulators
    is_cdna_acc = bool(re.match(r"^a(\d+|\[.*?\])$", cleaned, re.IGNORECASE) or cleaned in ("AReg_32", "a[n:n+3]"))
    if is_cdna_acc and gfx_arch and "CDNA" not in gfx_arch and "GFX9" not in gfx_arch:
        errors.append(f"Matrix accumulator operand '{operand}' is only supported on GFX9/CDNA, not '{gfx_arch}'.")

    # Check range/slice registers: v[0:1], s[0:3], a[0:3]
    slice_match = re.match(r"^([vVsSaA])\[(\d+):(\d+)\]$", cleaned)
    if slice_match:
        errors.extend(_validate_rdna_slice_register(operand, slice_match))
        return errors

    errors.extend(_validate_rdna_single_register(cleaned))
    return errors


def validate_rdna_constant_bus(
    operands: list[str],
    encoding: str | None = None,
    gfx_arch: str | None = None,
) -> list[str]:
    """Validate AMD hardware constant bus limitation on pre-RDNA architectures (GFX9/CDNA).

    Pre-RDNA architectures cannot read more than one SGPR or literal constant
    across the vector source operands of an instruction.

    Args:
        operands: Full list of operand strings.
        encoding: Instruction encoding profile (e.g. 'VOP1', 'VOP2', 'VOP3').
        gfx_arch: Target GFX architecture (e.g. 'GFX9/CDNA').

    Returns:
        List of constant bus limitation violation errors.
    """
    errors: list[str] = []
    if not operands or len(operands) <= 1:
        return errors

    # Constant bus limit applies primarily to pre-RDNA (GFX9/CDNA) vector instructions
    is_pre_rdna = gfx_arch and ("CDNA" in gfx_arch or "GFX9" in gfx_arch)
    enc = (encoding or "").upper()
    is_vector_op = enc.startswith("VOP")

    if is_pre_rdna and is_vector_op:
        # Source operands (operand 0 is dst, or operand 1 for VOPC dst=VCC)
        src_start = 2 if enc == "VOPC" else 1
        src_ops = operands[src_start:]

        constant_bus_reads = 0
        for op in src_ops:
            cleaned = op.strip().lower()
            # SGPR scalar register
            is_sgpr = bool(
                re.match(r"^s\d+$", cleaned)
                or re.match(r"^s\[.*?\]$", cleaned)
                or cleaned in ("sgpr", "sgpr_32", "sreg_64")
            )
            # Literal constant / immediate
            is_imm = bool(re.match(r"^-?(0x[0-9a-fA-F]+|\d+(\.\d+)?)$", cleaned) or cleaned in ("imm32", "simm16"))
            if is_sgpr or is_imm:
                constant_bus_reads += 1

        if constant_bus_reads > 1:
            errors.append(
                f"Constant bus limitation violation on '{gfx_arch}': vector instructions cannot read more than one SGPR or literal constant across vector source operands (found {constant_bus_reads})."
            )

    return errors


VOPD_VALID_OPCODES: set[str] = {
    "v_dual_fmac_f32",
    "v_dual_fma_f32",
    "v_dual_add_f32",
    "v_dual_sub_f32",
    "v_dual_mul_f32",
    "v_dual_cndmask_b32",
    "v_dual_mov_b32",
    "v_dual_dot2acc_f32_f16",
    "v_dual_dot2acc_f32_bf16",
}


def validate_vopd_pairing(
    opx: typing.Mapping[str, JsonValue],
    opy: typing.Mapping[str, JsonValue],
    gfx_arch: str | None = None,
) -> list[str]:
    """Validate dual-issue VOPD instruction pairing rules for RDNA3/GFX11.

    Rules:
        - VOPD is only supported on GFX11/RDNA3 and GFX11.5
        - Both OpX and OpY mnemonics must belong to legal VOPD opcodes
        - OpX and OpY cannot write to the same destination VGPR

    Args:
        opx: Dictionary describing OpX instruction (keys: mnemonic, dst, src0, src1).
        opy: Dictionary describing OpY instruction (keys: mnemonic, dst, src0, src1).
        gfx_arch: Target architecture string.

    Returns:
        List of VOPD pairing error messages.
    """
    errors: list[str] = []

    if gfx_arch and "GFX11" not in gfx_arch and "RDNA3" not in gfx_arch:
        errors.append(f"VOPD dual-issue encoding is only supported on GFX11/RDNA3, got '{gfx_arch}'.")

    mnem_x = str(opx.get("mnemonic", "")).lower()
    mnem_y = str(opy.get("mnemonic", "")).lower()

    if mnem_x not in VOPD_VALID_OPCODES:
        errors.append(f"Instruction '{mnem_x}' is not a valid VOPD OpX opcode.")
    if mnem_y not in VOPD_VALID_OPCODES:
        errors.append(f"Instruction '{mnem_y}' is not a valid VOPD OpY opcode.")

    dst_x = str(opx.get("dst", "")).strip().lower()
    dst_y = str(opy.get("dst", "")).strip().lower()

    if dst_x and dst_y and dst_x == dst_y:
        errors.append(f"VOPD destination conflict: OpX and OpY cannot write to the same destination VGPR '{dst_x}'.")

    return errors


def validate_rdna_modifiers(modifiers: list[str], encoding: str | None = None) -> list[str]:
    """Validate source and output modifier legality against the instruction encoding.

    Args:
        modifiers: List of modifier strings (e.g. ['-src', 'clamp', 'omod:2']).
        encoding: Instruction encoding profile (e.g. 'VOP3', 'VOP1', 'SOP2').

    Returns:
        List of modifier validation error messages.
    """
    errors: list[str] = []
    enc = (encoding or "").upper()

    for mod in modifiers:
        clean_mod = mod.strip().lower()

        # Source modifiers: -src, |src|, clamp, op_sel, neg_lo, neg_hi
        if clean_mod in RDNA_SOURCE_MODIFIERS:
            if enc not in ("VOP3", "VOP3P", "VOPD"):
                errors.append(
                    f"Source modifier '{mod}' is only valid on VOP3/VOP3P encodings, but instruction encoding is '{encoding}'."
                )

        # Output modifiers: omod:2, omod:4, omod:div2
        if clean_mod in RDNA_OUTPUT_MODIFIERS:
            if enc != "VOP3":
                errors.append(
                    f"Output modifier '{mod}' is only valid on VOP3 encodings, but instruction encoding is '{encoding}'."
                )

    return errors


def validate_rdna_microarchitecture(
    mnemonic: str,
    gfx_arch: str | None = None,
    wave_size: int | None = None,
) -> list[str]:
    """Validate microarchitecture and wave-size execution constraints for RDNA/CDNA instructions.

    Args:
        mnemonic: The instruction mnemonic (e.g. 'v_dual_fma_f32', 'v_mfma_f32_16x16x16f16').
        gfx_arch: Target GFX microarchitecture.
        wave_size: Wavefront execution size (32 or 64).

    Returns:
        List of microarchitecture error messages.
    """
    errors: list[str] = []
    m = mnemonic.lower().strip()

    # Guard dual-issue instructions strictly to GFX11/RDNA3 and GFX12/RDNA4
    if m.startswith("v_dual_"):
        if gfx_arch and ("GFX9" in gfx_arch or "GFX10" in gfx_arch):
            errors.append(
                f"Dual-issue instruction '{mnemonic}' is strictly supported on GFX11+ (RDNA3/RDNA4), not '{gfx_arch}'."
            )
        if wave_size == 64:
            errors.append(f"Dual-issue instruction '{mnemonic}' strictly requires Wave32 execution mode.")

    # Guard matrix accumulator instructions strictly to GFX9/CDNA
    if m.startswith("v_mfma_") or m.startswith("v_smfmac_"):
        if gfx_arch and "CDNA" not in gfx_arch and "GFX9" not in gfx_arch:
            errors.append(
                f"Matrix accumulator instruction '{mnemonic}' is strictly supported on GFX9/CDNA architectures, not '{gfx_arch}'."
            )

    # Guard wave-size mode per architecture
    if wave_size is not None and gfx_arch:
        if wave_size == 32 and ("CDNA" in gfx_arch or "GFX9" in gfx_arch):
            errors.append(f"Architecture '{gfx_arch}' executes in Wave64 mode and does not support Wave32 execution.")

    return errors


_DEFAULT_RDNA_ARCHITECTURES: list[str] = [
    "GFX10/RDNA1",
    "GFX10.3/RDNA2",
    "GFX11/RDNA3",
    "GFX11.5",
    "GFX12/RDNA4",
]

_EXACT_GFX_ARCH_MAP: dict[str, list[str]] = {arch: [arch] for arch in ALL_GFX_ARCHITECTURES}


def resolve_gfx_architectures(arch_spec: JsonValue) -> list[str]:
    """Resolve an architecture specification into discrete targeted GFX microarchitectures.

    Args:
        arch_spec: Architecture specification string (e.g. 'GFX10+', 'GFX11/RDNA3').

    Returns:
        List of targeted GFX architectures.
    """
    if not arch_spec:
        return list(_DEFAULT_RDNA_ARCHITECTURES)

    spec = str(arch_spec).strip()
    if spec in _EXACT_GFX_ARCH_MAP:
        return list(_EXACT_GFX_ARCH_MAP[spec])
    if "GFX11" in spec:
        return ["GFX11/RDNA3", "GFX11.5", "GFX12/RDNA4"]
    if "GFX12" in spec:
        return ["GFX12/RDNA4"]

    return list(_DEFAULT_RDNA_ARCHITECTURES)


def tokenize_rdna_line(
    line: str,
) -> tuple[str, list[str], str | None, list[str]]:
    """Tokenize an AMD RDNA assembly line into mnemonic, operands, encoding suffix, and modifiers.

    Args:
        line: A single RDNA assembly instruction line (e.g. 'v_add_f32_e32 v0, v1, v2 clamp').

    Returns:
        A tuple of (base_mnemonic, operands, encoding_suffix, modifiers).
    """
    clean = line.strip().rstrip(";")
    if not clean:
        return "", [], None, []

    tokens = clean.split(None, 1)
    inst_token = tokens[0].lower()
    rest = tokens[1] if len(tokens) > 1 else ""

    encoding_suffix = None
    suffix_match = re.match(r"^(v_[a-z0-9_]+)_(e32|e64|dpp\d*|sdwa|b32|b64)$", inst_token)
    if suffix_match:
        base_mnemonic = suffix_match.group(1)
        encoding_suffix = f"_{suffix_match.group(2)}"
    else:
        base_mnemonic = inst_token

    # Parse operands and trailing modifiers
    raw_parts = [p.strip() for p in rest.split(",") if p.strip()]
    operands: list[str] = []
    modifiers: list[str] = []

    known_mod_prefixes = (
        "clamp",
        "omod:",
        "op_sel:",
        "neg_lo:",
        "neg_hi:",
        "quad_perm:",
        "row_mask:",
        "bank_mask:",
    )

    for i, part in enumerate(raw_parts):
        subparts = part.split()
        if i == len(raw_parts) - 1 and len(subparts) > 1:
            operands.append(subparts[0])
            for sub in subparts[1:]:
                if any(sub.startswith(p) for p in known_mod_prefixes):
                    modifiers.append(sub)
                else:
                    operands.append(sub)
        else:
            if any(part.startswith(p) for p in known_mod_prefixes):
                modifiers.append(part)
            else:
                operands.append(part)

    if encoding_suffix and encoding_suffix not in modifiers:
        modifiers.append(encoding_suffix)

    return base_mnemonic, operands, encoding_suffix, modifiers


def _get_canonical_fallback_rdna() -> list[dict[str, JsonValue]]:
    """Generate canonical baseline AMD RDNA instructions for offline operation when unbuilt.

    Returns:
        List of AMD RDNA instruction dictionaries.
    """
    return [
        {
            "mnemonic": "v_add_f32",
            "architecture": "GFX10+",
            "encoding": "VOP2",
            "modifiers": ["_e32", "_e64", "clamp"],
            "operands": [
                ["VGPR", "VGPR", "VGPR"],
                ["VGPR", "SGPR", "VGPR"],
                ["VGPR", "imm32", "VGPR"],
            ],
            "description": "Vector floating point 32-bit addition.",
        },
        {
            "mnemonic": "v_sub_f32",
            "architecture": "GFX10+",
            "encoding": "VOP2",
            "modifiers": ["_e32", "_e64", "clamp"],
            "operands": [
                ["VGPR", "VGPR", "VGPR"],
                ["VGPR", "SGPR", "VGPR"],
                ["VGPR", "imm32", "VGPR"],
            ],
            "description": "Vector floating point 32-bit subtraction.",
        },
        {
            "mnemonic": "v_mul_f32",
            "architecture": "GFX10+",
            "encoding": "VOP2",
            "modifiers": ["_e32", "_e64", "clamp"],
            "operands": [
                ["VGPR", "VGPR", "VGPR"],
                ["VGPR", "SGPR", "VGPR"],
                ["VGPR", "imm32", "VGPR"],
            ],
            "description": "Vector floating point 32-bit multiplication.",
        },
        {
            "mnemonic": "v_fma_f32",
            "architecture": "GFX10+",
            "encoding": "VOP3",
            "modifiers": ["_e64", "clamp", "neg_lo", "neg_hi"],
            "operands": [
                ["VGPR", "VGPR", "VGPR", "VGPR"],
                ["VGPR", "SGPR", "VGPR", "VGPR"],
            ],
            "description": "Vector floating point 32-bit fused multiply-add.",
        },
        {
            "mnemonic": "v_fmac_f32",
            "architecture": "GFX10+",
            "encoding": "VOP2",
            "modifiers": ["_e32", "_e64"],
            "operands": [
                ["VGPR", "VGPR", "VGPR"],
            ],
            "description": "Vector floating point 32-bit multiply-accumulate.",
        },
        {
            "mnemonic": "v_dual_fmac_f32",
            "architecture": "GFX11/RDNA3",
            "encoding": "VOPD",
            "modifiers": ["dual"],
            "operands": [
                ["VGPR", "VGPR", "VGPR"],
            ],
            "description": "RDNA3 dual-issue fused multiply-accumulate.",
        },
        {
            "mnemonic": "v_dual_add_f32",
            "architecture": "GFX11/RDNA3",
            "encoding": "VOPD",
            "modifiers": ["dual"],
            "operands": [
                ["VGPR", "VGPR", "VGPR"],
            ],
            "description": "RDNA3 dual-issue addition.",
        },
        {
            "mnemonic": "v_dot4c_i32_i8",
            "architecture": "GFX10+",
            "encoding": "VOP3",
            "modifiers": ["clamp"],
            "operands": [
                ["VGPR", "VGPR", "VGPR", "VGPR"],
            ],
            "description": "Vector dot product 4 8-bit integers with 32-bit integer accumulation.",
        },
        {
            "mnemonic": "v_mov_b32",
            "architecture": "GFX10+",
            "encoding": "VOP1",
            "modifiers": ["_e32", "_e64"],
            "operands": [
                ["VGPR", "VGPR"],
                ["VGPR", "SGPR"],
                ["VGPR", "imm32"],
            ],
            "description": "Vector 32-bit move.",
        },
        {
            "mnemonic": "s_add_u32",
            "architecture": "GFX10+",
            "encoding": "SOP2",
            "modifiers": [],
            "operands": [
                ["SGPR", "SGPR", "SGPR"],
                ["SGPR", "SGPR", "imm32"],
            ],
            "description": "Scalar 32-bit unsigned integer addition.",
        },
        {
            "mnemonic": "s_mov_b32",
            "architecture": "GFX10+",
            "encoding": "SOP1",
            "modifiers": [],
            "operands": [
                ["SGPR", "SGPR"],
                ["SGPR", "imm32"],
            ],
            "description": "Scalar 32-bit move.",
        },
        {
            "mnemonic": "s_waitcnt",
            "architecture": "GFX10+",
            "encoding": "SOPP",
            "modifiers": [],
            "operands": [
                ["imm16"],
            ],
            "description": "Wait for memory counts.",
        },
        {
            "mnemonic": "global_load_dword",
            "architecture": "GFX10+",
            "encoding": "GLOBAL",
            "modifiers": ["glc", "slc"],
            "operands": [
                ["VGPR", "ADDR"],
            ],
            "description": "Global memory load 32-bit dword.",
        },
        {
            "mnemonic": "global_store_dword",
            "architecture": "GFX10+",
            "encoding": "GLOBAL",
            "modifiers": ["glc", "slc"],
            "operands": [
                ["ADDR", "VGPR"],
            ],
            "description": "Global memory store 32-bit dword.",
        },
    ]


def _load_exhaustive_rdna() -> list[dict[str, JsonValue]]:
    """Load the exhaustive AMD RDNA JSON instructions file or fallback to canonical records.

    Returns:
        The loaded JSON list of AMD RDNA instruction dictionaries.
    """
    json_path = os.path.join(os.path.dirname(__file__), "amd_rdna_exhaustive.json")
    if not os.path.exists(json_path):
        return _get_canonical_fallback_rdna()
    with open(json_path, encoding="utf-8") as f:
        data: JsonValue = json.load(f)
        if isinstance(data, dict):
            if "categories" in data and isinstance(data["categories"], dict):
                all_ops: list[dict[str, JsonValue]] = []
                for cat_ops in data["categories"].values():
                    if isinstance(cat_ops, list):
                        all_ops.extend(item for item in cat_ops if isinstance(item, dict))
                return all_ops
            raw_cand = data.get("instructions") or data.get("operations") or data.get("items") or []
            return [item for item in (raw_cand if isinstance(raw_cand, list) else []) if isinstance(item, dict)]
        return [item for item in (data if isinstance(data, list) else []) if isinstance(item, dict)]


def _build_rdna_operand_params(sig: list[str]) -> list[GhostParam]:
    """Build positional `ExtendedGhostParam` entries from an RDNA operand type signature.

    Args:
        sig: Ordered list of RDNA operand type tokens (e.g. `['VGPR', 'SGPR', 'VGPR']`).

    Returns:
        List of `GhostParam` definitions with directionality and standardized role names.
    """
    params: list[GhostParam] = []
    for i, op_type in enumerate(sig):
        if op_type in ("VCC", "SCC"):
            role = "condition"
            direction = OperandDirection.PREDICATE
        elif op_type == "EXEC":
            role = "exec_mask"
            direction = OperandDirection.READ
        else:
            role = "dst" if i == 0 else f"src{i - 1}"
            direction = OperandDirection.WRITE if i == 0 else OperandDirection.READ

        params.append(
            ExtendedGhostParam(
                name=role if role not in ("condition", "exec_mask") or i == 0 else f"{role}_{i}",
                kind=ParameterKind.POSITIONAL_ONLY,
                annotation=op_type,
                standardized_name=role,
                description=f"Operand {i} ({role}) of type {op_type}",
                direction=direction,
                role=IRParameterRole.OPERAND,
            )
        )
    return params


def collect_api(category: SemanticTier, include_nonpublic: bool = False) -> list[GhostRef]:
    """Entrypoint to collect the AMD RDNA API signature.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered.
    """
    if category != SemanticTier.UTIL:
        return []

    refs: list[GhostRef] = []

    for inst_data in _load_exhaustive_rdna():
        mnemonic = str(inst_data["mnemonic"])
        encoding = str(inst_data.get("encoding", "Unknown"))
        base_desc = str(inst_data.get("description", f"AMD RDNA {mnemonic} instruction."))
        raw_mods = inst_data.get("modifiers", [])
        modifiers: list[str] = [str(m) for m in (raw_mods if isinstance(raw_mods, list) else [])]
        arch = inst_data.get("architecture")
        valid_archs = resolve_gfx_architectures(arch)
        env_tags = ["rocm"] + valid_archs

        docstring = f"AMD RDNA {mnemonic} instruction.\nEncoding: {encoding}\nDescription: {base_desc}\nModifiers: {', '.join(modifiers) if modifiers else 'None'}\nTarget Architectures: {', '.join(valid_archs)}"

        raw_operands = inst_data.get("operands", [])
        operands_list: list[list[str]] = [
            [str(op) for op in sig]
            for sig in (raw_operands if isinstance(raw_operands, list) else [])
            if isinstance(sig, list)
        ]

        max_sig: list[str] = max(operands_list, key=len, default=[])
        params = _build_rdna_operand_params(max_sig)

        domain_metadata: dict[str, JsonValue] = {
            "encoding": encoding,
            "architecture": arch,
            "valid_architectures": list(valid_archs),
            "modifiers": list(modifiers),
            "operand_signatures": [list(s) for s in operands_list],
        }

        overloads: list[dict[str, JsonValue] | GhostRef] = [
            ExtendedGhostRef(
                name=mnemonic,
                api_path=f"amd_rdna.inst.{mnemonic}",
                kind="instruction",
                params=_build_rdna_operand_params(sig),
                docstring=f"Overload {sig_idx} for {mnemonic}: {', '.join(sig)}",
                environment_tags=env_tags,
                domain_metadata=domain_metadata,
            )
            for sig_idx, sig in enumerate(operands_list)
        ]

        refs.append(
            GhostIsaRef.model_validate(
                {
                    "name": mnemonic,
                    "api_path": f"amd_rdna.inst.{mnemonic}",
                    "kind": "instruction",
                    "params": params,
                    "docstring": docstring,
                    "environment_tags": env_tags,
                    "overloads": overloads,
                    "domain_metadata": domain_metadata,
                    "instruction_modifiers": modifiers,
                    "supported_architectures": valid_archs,
                }
            )
        )

    return refs


def collect_domain(include_nonpublic: bool = False) -> list[GhostRef]:
    """Collect AMD RDNA hardware instruction schemas without `SemanticTier` shoehorning.

    Args:
        include_nonpublic: Whether to include non-public instructions.

    Returns:
        List of `GhostIsaRef` items.
    """
    return collect_api(SemanticTier.UTIL, include_nonpublic)


DOMAIN_COLLECTOR = DomainCollectorAdapter(CollectorDomain.HARDWARE_ISA, "instructions", collect_domain)
get_canonical_fallback_rdna = _get_canonical_fallback_rdna
load_exhaustive_rdna = _load_exhaustive_rdna
