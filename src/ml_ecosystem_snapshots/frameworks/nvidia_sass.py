"""NVIDIA SASS (Assembly) Extractor.

Provides a static snapshot of NVIDIA SASS instructions with accurate operand modeling,
predicate guards, instruction modifiers, and multi-architecture targeting.
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
    IsaControlCodesSpec,
    IsaStructuredOperandSpec,
    ParameterKind,
    SemanticTier,
)

from ml_ecosystem_snapshots.frameworks._base import CollectorDomain, DomainCollectorAdapter

from ..models import (
    ExtendedGhostParam,
    ExtendedGhostRef,
    GhostIsaRef,
    IRParameterRole,
    IsaInstructionRecordModel,
    JsonValue,
    OperandDirection,
)

ALL_SM_ARCHITECTURES: list[str] = [
    "sm_70",
    "sm_75",
    "sm_80",
    "sm_86",
    "sm_89",
    "sm_90",
    "sm_100",
]

DEFAULT_SASS_CONTROL_CODE_SCHEMA: dict[str, JsonValue] = {
    "latency_ticks": {
        "type": "integer",
        "description": "Cycles before destination register is ready for subsequent dependent instructions.",
        "default": 4,
    },
    "yield_flag": {
        "type": "boolean",
        "description": "Yield warp execution to warp scheduler on current cycle (Y flag).",
        "default": False,
    },
    "stall_count": {
        "type": "integer",
        "description": "Number of scheduler cycles to stall instruction issuing after this instruction (0-15).",
        "default": 1,
    },
    "read_barrier_mask": {
        "type": "integer",
        "description": "Read barrier mask (0-5 dependency barriers to wait on for register read completion).",
        "default": 0,
    },
    "write_barrier_mask": {
        "type": "integer",
        "description": "Write barrier mask (0-5 dependency barriers to set for register write completion).",
        "default": 0,
    },
    "register_reuse_flags": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Register reuse cache flags (e.g. ['R0.reuse', 'R1.reuse']).",
        "default": [],
    },
}


class SASSInstructionFamily(str, Enum):
    """Instruction families in NVIDIA SASS architecture."""

    ARITHMETIC = "arithmetic"
    MEMORY = "memory"
    TENSOR = "tensor"
    CONTROL = "control"
    BARRIER = "barrier"
    SPECIAL = "special"


class SASSModifier(str, Enum):
    """Structured enum of standard NVIDIA SASS instruction modifiers."""

    SAT = ".SAT"
    RN = ".RN"
    RZ = ".RZ"
    RM = ".RM"
    RP = ".RP"
    FTZ = ".FTZ"
    STRONG = ".STRONG"
    CG = ".CG"
    CS = ".CS"
    LU = ".LU"
    CV = ".CV"


ROUNDING_MODIFIERS: set[str] = {".RN", ".RZ", ".RM", ".RP"}
CACHE_MODIFIERS: set[str] = {".CG", ".CS", ".LU", ".CV"}


def get_sass_instruction_family(mnemonic: str) -> str:
    """Determine the SASS instruction family based on mnemonic pattern.

    Args:
        mnemonic: The instruction mnemonic (e.g. 'FADD', 'LDG', 'HMMA16816').

    Returns:
        The instruction family string.
    """
    m = mnemonic.upper().strip()
    if any(m.startswith(p) for p in ("BAR", "DEPBAR", "SYNC")):
        return SASSInstructionFamily.BARRIER.value
    if any(m.startswith(p) for p in ("BRA", "RET", "EXIT", "WARPSYNC", "JMP", "CALL", "BR")):
        return SASSInstructionFamily.CONTROL.value
    if any(m.startswith(p) for p in ("HMMA", "IMMA", "BMMA", "DMMA", "WGMMA")):
        return SASSInstructionFamily.TENSOR.value
    if any(
        m.startswith(p)
        for p in (
            "LD",
            "LDG",
            "LDS",
            "LDL",
            "LDC",
            "ST",
            "STG",
            "STS",
            "STL",
            "ATOM",
            "ATOMG",
            "ATOMS",
            "RED",
            "TMA",
            "LDGSTS",
            "LDSM",
        )
    ):
        return SASSInstructionFamily.MEMORY.value
    if any(m.startswith(p) for p in ("MUFU", "COS", "SIN", "EX2", "LG2", "RCP", "RSQ")):
        return SASSInstructionFamily.SPECIAL.value
    return SASSInstructionFamily.ARITHMETIC.value


def get_sass_latency_range(family: str, sm_arch: str | None = None) -> tuple[int, int]:
    """Retrieve the legal cycle latency bounds for an instruction family on target architecture.

    Args:
        family: Instruction family name ('arithmetic', 'memory', 'tensor', 'special', 'control', 'barrier').
        sm_arch: Target SM architecture string (e.g. 'sm_80', 'sm_90').

    Returns:
        Tuple of (min_latency_cycles, max_latency_cycles).
    """
    arch = sm_arch.lower().strip() if sm_arch else "sm_80"
    is_modern = arch in ("sm_90", "sm_100")

    if family == SASSInstructionFamily.ARITHMETIC.value:
        return (3, 5) if is_modern else (4, 6)
    if family == SASSInstructionFamily.MEMORY.value:
        return (16, 256) if is_modern else (19, 256)
    if family == SASSInstructionFamily.TENSOR.value:
        return (16, 32) if is_modern else (8, 16)
    if family == SASSInstructionFamily.SPECIAL.value:
        return (14, 16)
    if family in (
        SASSInstructionFamily.CONTROL.value,
        SASSInstructionFamily.BARRIER.value,
    ):
        return (1, 4)
    return (1, 16)


def _validate_sass_register_pair_or_quad(operand: str, start_idx: int, end_idx: int) -> list[str]:
    """Validate 64-bit register pair or 128-bit register quad alignment and index bounds.

    Args:
        operand: Original register operand string for diagnostic messages.
        start_idx: Starting register index in the range.
        end_idx: Ending register index in the range.

    Returns:
        List of alignment or range validation error messages.
    """
    if start_idx > 255 or end_idx > 255:
        return [f"Register index out of range in '{operand}': max register index is R255."]
    span = end_idx - start_idx
    if span == 1:
        if start_idx % 2 != 0:
            return [f"64-bit register pair '{operand}' must start on an even register index (got R{start_idx})."]
        return []
    if span == 3:
        if start_idx % 4 != 0:
            return [f"128-bit register quad '{operand}' must start on a modulo-4 register index (got R{start_idx})."]
        return []
    return [f"Invalid register span in '{operand}': expected pair (span 1) or quad (span 3), got span {span}."]


def _validate_sass_uniform_register_pair(
    operand: str,
    start_idx: int,
    end_idx: int,
    sm_arch: str | None,
) -> list[str]:
    """Validate uniform register pair architecture support, index bounds, and alignment.

    Args:
        operand: Original uniform register pair operand string.
        start_idx: Starting uniform register index.
        end_idx: Ending uniform register index.
        sm_arch: Target SM architecture identifier.

    Returns:
        List of uniform register pair validation error messages.
    """
    errors: list[str] = []
    if sm_arch == "sm_70":
        errors.append(
            f"Uniform registers (UR) are only supported on sm_75+ architectures, but target architecture is '{sm_arch}'."
        )
    if start_idx > 63 or end_idx > 63:
        errors.append(f"Uniform register index out of range in '{operand}': max is UR63.")
    elif end_idx - start_idx != 1:
        errors.append(f"Invalid uniform register pair span in '{operand}': expected span 1, got {end_idx - start_idx}.")
    elif start_idx % 2 != 0:
        errors.append(f"Uniform register pair '{operand}' must start on an even register index (got UR{start_idx}).")
    return errors


def _validate_sass_scalar_register(cleaned: str, sm_arch: str | None) -> list[str]:
    """Validate a single non-pair SASS scalar, uniform, predicate, or barrier register token.

    Args:
        cleaned: Cleaned register token stripped of reuse flags and predicate prefixes.
        sm_arch: Target SM architecture identifier.

    Returns:
        List of register bounds or architecture compatibility error messages.
    """
    errors: list[str] = []
    r_match = re.match(r"^R(\d+)$", cleaned, re.IGNORECASE)
    if r_match and int(r_match.group(1)) > 255:
        errors.append(f"32-bit vector register index R{int(r_match.group(1))} out of range (R0-R255).")

    ur_match = re.match(r"^UR(\d+)$", cleaned, re.IGNORECASE)
    if cleaned.upper() == "URZ" or ur_match:
        if sm_arch == "sm_70":
            errors.append(
                f"Uniform registers (UR) are only supported on sm_75+ architectures, but target architecture is '{sm_arch}'."
            )
        if ur_match and int(ur_match.group(1)) > 63:
            errors.append(f"Uniform register index UR{int(ur_match.group(1))} out of range (UR0-UR63).")

    p_match = re.match(r"^!?P(\d+)$", cleaned, re.IGNORECASE)
    if p_match and int(p_match.group(1)) > 7:
        errors.append(f"Predicate register index P{int(p_match.group(1))} out of range (P0-P7).")

    up_match = re.match(r"^!?UP(\d+)$", cleaned, re.IGNORECASE)
    if cleaned.upper() in ("UPT", "!UPT") or up_match:
        if sm_arch == "sm_70":
            errors.append(
                f"Uniform predicate registers (UP) are only supported on sm_75+ architectures, but target architecture is '{sm_arch}'."
            )
        if up_match and int(up_match.group(1)) > 7:
            errors.append(f"Uniform predicate register index UP{int(up_match.group(1))} out of range (UP0-UP7).")

    b_match = re.match(r"^B(\d+)$", cleaned, re.IGNORECASE)
    if b_match and int(b_match.group(1)) > 15:
        errors.append(f"Barrier register index B{int(b_match.group(1))} out of range (B0-B15).")

    return errors


def validate_sass_register(operand: str, role: str = "src", sm_arch: str | None = None) -> list[str]:
    """Validate a single SASS register specification against hardware alignment and indexing rules.

    Validates:
        - 32-bit registers (R0-R255, RZ)
        - 64-bit aligned register pairs (R0:R1, R2:R3 requiring even starting index and 0 <= n <= 254)
        - 128-bit aligned register quads (R0:R3, R4:R7 requiring modulo-4 starting index and 0 <= n <= 252)
        - Uniform registers (UR0-UR63, URZ) and pairs (UR0:UR1, requiring even starting index) on sm_75+
        - Predicate registers (P0-P7, PT) and Uniform Predicate registers (UP0-UP7, UPT) on sm_75+
        - Barrier register IDs (B0-B15)
        - Legality of register reuse flags (R*.reuse) exclusively on source operand slots

    Args:
        operand: Raw operand string (e.g. 'R0', 'R0:R1', 'R1:R2', 'UR0', 'P0', 'R0.reuse').
        role: Operand role ('dst', 'src0', 'src1', etc.).
        sm_arch: Target SM architecture (e.g. 'sm_70', 'sm_80', 'sm_90').

    Returns:
        List of error strings identified during register validation.
    """
    errors: list[str] = []
    cleaned = operand.strip()

    # Check for register reuse flag
    if ".reuse" in cleaned.lower():
        if role == "dst":
            errors.append(f"Register reuse flag '.reuse' is not permitted on destination operand '{operand}'.")
        cleaned = re.sub(r"\.reuse", "", cleaned, flags=re.IGNORECASE).strip()

    # Strip predicate guard prefix '@'
    if cleaned.startswith("@"):
        cleaned = cleaned[1:].strip()

    # Check for memory operand wrapper like [R1] or [R2+0x4]
    if cleaned.startswith("[") and cleaned.endswith("]"):
        inner = cleaned[1:-1].strip()
        reg_part = re.split(r"[+\-\*]", inner)[0].strip()
        if reg_part:
            return validate_sass_register(reg_part, role=role, sm_arch=sm_arch)
        return []

    # Check 64-bit / 128-bit vector register range: R<start>:R<end> or R[<start>:<end>]
    quad_pair_match = re.match(r"^R(\d+):R(\d+)$", cleaned, re.IGNORECASE) or re.match(
        r"^R\[(\d+):(\d+)\]$", cleaned, re.IGNORECASE
    )
    if quad_pair_match:
        errors.extend(
            _validate_sass_register_pair_or_quad(operand, int(quad_pair_match.group(1)), int(quad_pair_match.group(2)))
        )
        return errors

    # Check uniform register pair: UR<start>:UR<end> or UR[<start>:<end>]
    ur_pair_match = re.match(r"^UR(\d+):UR(\d+)$", cleaned, re.IGNORECASE) or re.match(
        r"^UR\[(\d+):(\d+)\]$", cleaned, re.IGNORECASE
    )
    if ur_pair_match:
        errors.extend(
            _validate_sass_uniform_register_pair(
                operand,
                int(ur_pair_match.group(1)),
                int(ur_pair_match.group(2)),
                sm_arch,
            )
        )
        return errors

    errors.extend(_validate_sass_scalar_register(cleaned, sm_arch))
    return errors


def validate_sass_operand_directionality(operands: list[str], mnemonic: str) -> list[str]:
    """Validate operand directionality to ensure read-only elements are not assigned to write slots.

    Args:
        operands: List of operand strings passed to instruction.
        mnemonic: The instruction mnemonic.

    Returns:
        List of directionality error messages.
    """
    errors: list[str] = []
    if not operands:
        return errors

    family = get_sass_instruction_family(mnemonic)
    # For branch, barrier, and store instructions, operand 0 is not a standard scalar write register
    is_store = mnemonic.upper().startswith("ST")
    is_control_or_barrier = family in (
        SASSInstructionFamily.CONTROL.value,
        SASSInstructionFamily.BARRIER.value,
    )

    if not is_store and not is_control_or_barrier:
        dst = operands[0].strip()
        # Immediate constants cannot be destination
        if re.match(r"^-?(0x[0-9a-fA-F]+|\d+(\.\d+)?)$", dst) or dst in ("I", "FI"):
            errors.append(f"Operand 0 is a destination (write) slot and cannot be an immediate: '{dst}'.")
        elif "c[" in dst.lower() or "cx[" in dst.lower() or dst in ("c[bank][offset]", "cx[bank][offset]"):
            errors.append(f"Operand 0 is a destination (write) slot and cannot be constant bank memory: '{dst}'.")

    c_bank_count = sum(
        1
        for op in operands
        if "c[" in op.lower() or "cx[" in op.lower() or op in ("c[bank][offset]", "cx[bank][offset]")
    )
    if c_bank_count > 1:
        errors.append(
            f"Hardware resource conflict in '{mnemonic}': SASS instructions permit at most 1 constant bank reference per cycle due to shared read ports (found {c_bank_count})."
        )

    return errors


def validate_sass_modifiers(mnemonic: str, modifiers: list[str]) -> list[str]:
    """Validate instruction modifier combinations for illegal conflicts.

    Checks:
        - Conflicting IEEE-754 rounding mode modifiers (.RN, .RZ, .RM, .RP)
        - Conflicting cache policy modifiers (.CG, .CS, .LU, .CV)
        - Saturation modifier (.SAT) legality per instruction family

    Args:
        mnemonic: The instruction mnemonic.
        modifiers: List of modifiers to validate.

    Returns:
        List of modifier conflict errors.
    """
    errors: list[str] = []
    norm_mods = [m if m.startswith(".") else f".{m}" for m in modifiers]

    # Check rounding mode conflicts
    rounding_present = [m for m in norm_mods if m in ROUNDING_MODIFIERS]
    if len(rounding_present) > 1:
        errors.append(f"Conflicting rounding mode modifiers in '{mnemonic}': {sorted(rounding_present)}.")

    # Check cache policy conflicts
    cache_present = [m for m in norm_mods if m in CACHE_MODIFIERS]
    if len(cache_present) > 1:
        errors.append(f"Conflicting cache policy modifiers in '{mnemonic}': {sorted(cache_present)}.")

    # Check .SAT validity
    if ".SAT" in norm_mods:
        family = get_sass_instruction_family(mnemonic)
        if family in (
            SASSInstructionFamily.MEMORY.value,
            SASSInstructionFamily.BARRIER.value,
            SASSInstructionFamily.CONTROL.value,
        ):
            errors.append(
                f"Saturation modifier '.SAT' is not supported on instruction family '{family}' for '{mnemonic}'."
            )

    return errors


def _validate_sass_control_timing(
    family: str,
    control_codes: typing.Mapping[str, JsonValue],
    sm_arch: str | None,
) -> list[str]:
    """Validate stall counts, dependency barrier scoreboard masks, and cycle latency bounds.

    Args:
        family: Resolved SASS instruction family name.
        control_codes: Mapping of control code field names to values.
        sm_arch: Optional target SM architecture string.

    Returns:
        List of timing or scoreboard validation error messages.
    """
    errors: list[str] = []

    if "stall_count" in control_codes:
        stall = control_codes["stall_count"]
        if not isinstance(stall, int) or stall < 0 or stall > 15:
            errors.append(f"Stall count cycle {stall} is out of valid range (0-15).")

    max_barrier = 7 if sm_arch in ("sm_90", "sm_100") else 5
    for b_field in ("read_barrier_mask", "write_barrier_mask"):
        if b_field in control_codes:
            mask = control_codes[b_field]
            if not isinstance(mask, int) or mask < 0 or mask > max_barrier:
                name = "Read" if "read" in b_field else "Write"
                errors.append(f"{name} dependency barrier scoreboard {mask} is out of valid range (0-{max_barrier}).")

    if "latency_ticks" in control_codes and sm_arch:
        latency = control_codes["latency_ticks"]
        min_ticks, max_ticks = get_sass_latency_range(family, sm_arch)
        if not isinstance(latency, int) or latency < min_ticks or latency > max_ticks:
            errors.append(
                f"Latency ticks {latency} out of bounds for {family} on {sm_arch}: expected between {min_ticks} and {max_ticks} cycles."
            )

    return errors


def validate_sass_control_code(
    mnemonic: str,
    control_codes: typing.Mapping[str, JsonValue],
    sm_arch: str | None = None,
) -> list[str]:
    """Validate control code attributes according to microarchitecture hardware constraints.

    Args:
        mnemonic: Instruction mnemonic.
        control_codes: Dictionary containing control code specifications.
        sm_arch: Target SM architecture string.

    Returns:
        List of control code error messages.
    """
    family = get_sass_instruction_family(mnemonic)
    errors = _validate_sass_control_timing(family, control_codes, sm_arch)

    # Register reuse cache flags (.reuse) validation
    if "register_reuse_flags" in control_codes:
        reuse_flags = control_codes["register_reuse_flags"]
        if isinstance(reuse_flags, list):
            for rf in reuse_flags:
                rf_str = str(rf)
                if not rf_str.endswith(".reuse"):
                    errors.append(f"Malformed register reuse flag '{rf_str}': expected 'R*.reuse'.")
                reg_clean = rf_str.replace(".reuse", "")
                reg_errs = validate_sass_register(reg_clean, sm_arch=sm_arch)
                errors.extend(reg_errs)

    # Dual-issue restrictions
    if control_codes.get("dual_issue") and family in (
        SASSInstructionFamily.TENSOR.value,
        SASSInstructionFamily.BARRIER.value,
    ):
        errors.append(f"Instruction '{mnemonic}' in family '{family}' does not support dual-issue execution.")

    # Hopper / Blackwell WGMMA and TMA asynchronous instruction validation
    if (mnemonic.startswith("WGMMA") or "TMA" in mnemonic) and sm_arch and sm_arch not in ("sm_90", "sm_100"):
        errors.append(f"Instruction '{mnemonic}' requires Hopper or Blackwell architecture (sm_90+), got '{sm_arch}'.")

    return errors


def parse_sass_modifiers(modifiers: list[str]) -> list[str]:
    """Parse and normalize instruction modifiers into validated structured representations.

    Args:
        modifiers: List of raw modifier strings (e.g. ['SAT', 'RN'] or ['.SAT', '.RN']).

    Returns:
        List of normalized modifier strings.
    """
    normalized: list[str] = []
    valid_values = {m.value for m in SASSModifier}
    for mod in modifiers:
        prefix = mod if str(mod).startswith(".") else f".{mod}"
        if prefix in valid_values and prefix not in normalized:
            normalized.append(prefix)
        elif prefix not in normalized:
            normalized.append(prefix)
    return sorted(normalized)


def tokenize_sass_line(
    line: str,
) -> tuple[str | None, str, list[str], list[str]]:
    """Tokenize a SASS assembly line into predicate, mnemonic, modifiers, and operands.

    Args:
        line: A single SASS assembly instruction line (e.g. '@P0 FADD.FTZ.RN R0, R1, R2;').

    Returns:
        A tuple of (predicate, base_mnemonic, modifiers, operands).
    """
    clean = line.strip().rstrip(";")
    if not clean:
        return None, "", [], []

    predicate = None
    pred_match = re.match(r"^(@!?U?P\d+|@!?PT|@!?UPT)\s+(.*)$", clean)
    if pred_match:
        predicate = pred_match.group(1)
        clean = pred_match.group(2).strip()

    tokens = clean.split(None, 1)
    inst_token = tokens[0]
    operands_part = tokens[1] if len(tokens) > 1 else ""

    subtokens = inst_token.split(".")
    base_mnemonic = subtokens[0].upper()
    modifiers = [f".{m}" for m in subtokens[1:] if m]

    operands = [op.strip() for op in operands_part.split(",") if op.strip()]
    return predicate, base_mnemonic, modifiers, operands


def resolve_sm_architectures(
    arch_spec: str | typing.Sequence[str] | JsonValue,
) -> list[str]:
    """Resolve an SM architecture specification into discrete targeted SM architectures.

    Args:
        arch_spec: Architecture specification string (e.g. 'sm_80+', 'sm_70+') or list of discrete SM archs.

    Returns:
        List of discrete SM architectures targeted (sm_70 through sm_100).
    """
    if not arch_spec:
        return list(ALL_SM_ARCHITECTURES)
    if isinstance(arch_spec, list):
        return [str(a) for a in arch_spec if str(a) in ALL_SM_ARCHITECTURES]
    clean = str(arch_spec).lower().strip()
    if clean.endswith("+"):
        base = clean[:-1]
        if base in ALL_SM_ARCHITECTURES:
            idx = ALL_SM_ARCHITECTURES.index(base)
            return ALL_SM_ARCHITECTURES[idx:]
    if clean in ALL_SM_ARCHITECTURES:
        return [clean]
    return list(ALL_SM_ARCHITECTURES)


_SASS_OPERAND_TYPE_MAP: dict[str, str] = {
    "R": "R",
    "RegOperand": "R",
    "UR": "UR",
    "UniformRegOperand": "UR",
    "P": "P",
    "PredicateOperand": "P",
    "I": "I",
    "IntIMMOperand": "I",
    "FI": "FI",
    "FloatIMMOperand": "FI",
    "ADDR": "ADDR",
    "AddressOperand": "ADDR",
}


def normalize_sass_operand_type(raw_op: str) -> str:
    """Normalize raw operand names into canonical SASS operand categories.

    Maps:
        'R' -> 32-bit vector register
        'UR' -> Uniform register
        'P' -> Predicate register
        'I' -> Integer immediate
        'FI' -> Float immediate
        'ADDR' -> Memory address
        'c[x][y]' -> Constant bank memory c[bank][offset]

    Args:
        raw_op: Raw operand string from ISA metadata.

    Returns:
        Canonical operand category string.
    """
    cleaned = raw_op.strip()
    if cleaned in _SASS_OPERAND_TYPE_MAP:
        return _SASS_OPERAND_TYPE_MAP[cleaned]
    if "cx[" in cleaned:
        return "cx[bank][offset]"
    if "c[" in cleaned:
        return "c[bank][offset]"
    return cleaned


def _classify_sass_operand_reg_classes(op: str) -> list[str]:
    """Classify the register and constant-bank classes referenced by a SASS operand token.

    Args:
        op: Raw SASS operand token string.

    Returns:
        List of canonical register class identifiers ('UR', 'R', 'P', 'UP', 'B', 'CBANK').
    """
    reg_classes: list[str] = []
    op_u = op.upper()
    if "UR" in op_u:
        reg_classes.append("UR")
    if "R" in op_u and "UR" not in op_u and "BAR" not in op_u:
        reg_classes.append("R")
    if "P" in op_u and "UP" not in op_u and "DEPBAR" not in op_u:
        reg_classes.append("P")
    if "UP" in op_u:
        reg_classes.append("UP")
    if "B" in op_u:
        reg_classes.append("B")
    if "C[" in op_u or "CX[" in op_u:
        reg_classes.append("CBANK")
    return reg_classes


def build_structured_sass_operands(operands: list[str], mnemonic: str) -> list[dict[str, JsonValue]]:
    """Build structured operand records detailing roles, register classes, and immediate constraints.

    Args:
        operands: Raw list of operand tokens (e.g. ['R0', 'R1', 'c[0x0][0x10]']).
        mnemonic: Instruction mnemonic (e.g. 'FADD', 'STG', 'BRA').

    Returns:
        List of structured operand dictionaries detailing index, role, register_classes, and imm limits.
    """
    structured: list[dict[str, JsonValue]] = []
    is_store = mnemonic.startswith("ST")
    is_branch = mnemonic in ("BRA", "BRX", "JMP", "JMX", "CALL", "RET")
    is_barrier = mnemonic in ("DEPBAR", "BAR", "SYNC", "WARPSYNC")

    for idx, op in enumerate(operands):
        role = "dest" if idx == 0 and not (is_store or is_branch or is_barrier) else f"src{idx - 1 if idx > 0 else 0}"
        if is_store and idx == 0:
            role = "address"
        elif is_branch:
            role = "target"
        elif is_barrier:
            role = "barrier"

        reg_classes = _classify_sass_operand_reg_classes(op)

        imm_type: str | None = None
        if re.match(r"^-?\d+\.\d+", op):
            imm_type = "float32"
        elif re.match(r"^-?(0x[0-9a-fA-F]+|\d+)$", op):
            imm_type = "int32"

        rec: dict[str, JsonValue] = {
            "index": idx,
            "raw_token": op,
            "role": role,
            "register_classes": list(reg_classes),
        }
        if imm_type:
            rec["immediate_type"] = imm_type
        structured.append(rec)
    return structured


def _get_canonical_fallback_sass() -> list[IsaInstructionRecordModel]:
    """Generate canonical baseline SASS instructions for offline operation when unbuilt.

    Returns:
        List of SASS instruction dictionaries.
    """
    from ..tools.scrape_nvidia_sass import build_expanded_sass_catalog

    base: list[dict[str, JsonValue]] = [
        {
            "mnemonic": "MOV",
            "modifiers": [],
            "operands": [
                ["R", "R"],
                ["R", "CBANK"],
                ["R", "IMM"],
                ["R", "c[bank][offset]"],
            ],
            "architecture": [
                "sm_70",
                "sm_75",
                "sm_80",
                "sm_86",
                "sm_89",
                "sm_90",
                "sm_100",
            ],
            "description": "Move register, constant, or immediate.",
        },
        {
            "mnemonic": "FADD",
            "modifiers": [".FTZ", ".SAT", ".RN", ".RZ", ".RM", ".RP"],
            "operands": [["R", "R", "R"]],
            "architecture": [
                "sm_70",
                "sm_75",
                "sm_80",
                "sm_86",
                "sm_89",
                "sm_90",
                "sm_100",
            ],
            "description": "Floating point addition.",
        },
        {
            "mnemonic": "FSUB",
            "modifiers": [".FTZ", ".SAT", ".RN", ".RZ", ".RM", ".RP"],
            "operands": [["R", "R", "R"]],
            "architecture": [
                "sm_70",
                "sm_75",
                "sm_80",
                "sm_86",
                "sm_89",
                "sm_90",
                "sm_100",
            ],
            "description": "Floating point subtraction.",
        },
        {
            "mnemonic": "FMUL",
            "modifiers": [".FTZ", ".SAT", ".RN", ".RZ", ".RM", ".RP"],
            "operands": [["R", "R", "R"]],
            "architecture": [
                "sm_70",
                "sm_75",
                "sm_80",
                "sm_86",
                "sm_89",
                "sm_90",
                "sm_100",
            ],
            "description": "Floating point multiplication.",
        },
        {
            "mnemonic": "FFMA",
            "modifiers": [".FTZ", ".SAT", ".RN", ".RZ", ".RM", ".RP"],
            "operands": [["R", "R", "R", "R"]],
            "architecture": [
                "sm_70",
                "sm_75",
                "sm_80",
                "sm_86",
                "sm_89",
                "sm_90",
                "sm_100",
            ],
            "description": "Floating point fused multiply-add.",
        },
        {
            "mnemonic": "IADD",
            "modifiers": [".X", ".PO"],
            "operands": [["R", "R", "R"]],
            "architecture": [
                "sm_70",
                "sm_75",
                "sm_80",
                "sm_86",
                "sm_89",
                "sm_90",
                "sm_100",
            ],
            "description": "Integer addition.",
        },
        {
            "mnemonic": "IADD3",
            "modifiers": [".X"],
            "operands": [["R", "R", "R", "R"]],
            "architecture": [
                "sm_70",
                "sm_75",
                "sm_80",
                "sm_86",
                "sm_89",
                "sm_90",
                "sm_100",
            ],
            "description": "3-input integer addition.",
        },
        {
            "mnemonic": "IMAD",
            "modifiers": [".WIDE", ".U32", ".S32"],
            "operands": [["R", "R", "R", "R"]],
            "architecture": [
                "sm_70",
                "sm_75",
                "sm_80",
                "sm_86",
                "sm_89",
                "sm_90",
                "sm_100",
            ],
            "description": "Integer multiply-add.",
        },
        {
            "mnemonic": "LDG",
            "modifiers": [".E", ".STRONG", ".LU", ".CV", ".CONSTANT"],
            "operands": [["R", "ADDR"]],
            "architecture": [
                "sm_70",
                "sm_75",
                "sm_80",
                "sm_86",
                "sm_89",
                "sm_90",
                "sm_100",
            ],
            "description": "Load from global memory.",
        },
        {
            "mnemonic": "LDS",
            "modifiers": [".U8", ".S8", ".16", ".32", ".64", ".128"],
            "operands": [["R", "ADDR"]],
            "architecture": [
                "sm_70",
                "sm_75",
                "sm_80",
                "sm_86",
                "sm_89",
                "sm_90",
                "sm_100",
            ],
            "description": "Load from shared memory.",
        },
        {
            "mnemonic": "STG",
            "modifiers": [".E", ".STRONG", ".LU", ".CV"],
            "operands": [["ADDR", "R"]],
            "architecture": [
                "sm_70",
                "sm_75",
                "sm_80",
                "sm_86",
                "sm_89",
                "sm_90",
                "sm_100",
            ],
            "description": "Store to global memory.",
        },
        {
            "mnemonic": "STS",
            "modifiers": [".U8", ".S8", ".16", ".32", ".64", ".128"],
            "operands": [["ADDR", "R"]],
            "architecture": [
                "sm_70",
                "sm_75",
                "sm_80",
                "sm_86",
                "sm_89",
                "sm_90",
                "sm_100",
            ],
            "description": "Store to shared memory.",
        },
        {
            "mnemonic": "BRA",
            "modifiers": [".U", ".DIV"],
            "operands": [["ADDR"]],
            "architecture": [
                "sm_70",
                "sm_75",
                "sm_80",
                "sm_86",
                "sm_89",
                "sm_90",
                "sm_100",
            ],
            "description": "Branch relative.",
        },
        {
            "mnemonic": "EXIT",
            "modifiers": [],
            "operands": [],
            "architecture": [
                "sm_70",
                "sm_75",
                "sm_80",
                "sm_86",
                "sm_89",
                "sm_90",
                "sm_100",
            ],
            "description": "Thread execution exit.",
        },
        {
            "mnemonic": "S2R",
            "modifiers": [],
            "operands": [["R", "SR"]],
            "architecture": [
                "sm_70",
                "sm_75",
                "sm_80",
                "sm_86",
                "sm_89",
                "sm_90",
                "sm_100",
            ],
            "description": "Special register to general register.",
        },
        {
            "mnemonic": "WGMMA",
            "modifiers": [".F16", ".BF16", ".TF32"],
            "operands": [["R", "R", "R", "R"]],
            "architecture": ["sm_90", "sm_100"],
            "description": "Hopper Warpgroup MMA.",
        },
        {
            "mnemonic": "BMMA",
            "modifiers": [".F16", ".BF16"],
            "operands": [["R", "R", "R", "R"]],
            "architecture": ["sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_100"],
            "description": "Bit Matrix Multiply and Accumulate.",
        },
        {
            "mnemonic": "HMMA",
            "modifiers": [".F16", ".F32"],
            "operands": [["R", "R", "R", "R"]],
            "architecture": [
                "sm_70",
                "sm_75",
                "sm_80",
                "sm_86",
                "sm_89",
                "sm_90",
                "sm_100",
            ],
            "description": "Half precision Matrix Multiply and Accumulate.",
        },
        {
            "mnemonic": "IMMA",
            "modifiers": [".S8", ".U8"],
            "operands": [["R", "R", "R", "R"]],
            "architecture": [
                "sm_72",
                "sm_75",
                "sm_80",
                "sm_86",
                "sm_89",
                "sm_90",
                "sm_100",
            ],
            "description": "Integer Matrix Multiply and Accumulate.",
        },
    ]
    return build_expanded_sass_catalog(base)


def _load_exhaustive_sass() -> list[IsaInstructionRecordModel]:
    """Loads the exhaustive NVIDIA SASS json dump or falls back to canonical offline catalog.

    Returns:
        The loaded JSON list of SASS instruction dictionaries.
    """
    json_path = os.path.join(os.path.dirname(__file__), "nvidia_sass_exhaustive.json")
    if not os.path.exists(json_path):
        return _get_canonical_fallback_sass()
    with open(json_path, encoding="utf-8") as f:
        data: JsonValue = json.load(f)
        if isinstance(data, dict):
            if "categories" in data and isinstance(data["categories"], dict):
                all_ops: list[IsaInstructionRecordModel] = []
                for cat_ops in data["categories"].values():
                    if isinstance(cat_ops, list):
                        all_ops.extend(
                            IsaInstructionRecordModel.model_validate(item) for item in cat_ops if isinstance(item, dict)
                        )
                return all_ops
            raw_cand = data.get("instructions") or data.get("operations") or data.get("items") or []
            return [
                IsaInstructionRecordModel.model_validate(item)
                for item in (raw_cand if isinstance(raw_cand, list) else [])
                if isinstance(item, dict)
            ]
        return [
            IsaInstructionRecordModel.model_validate(item)
            for item in (data if isinstance(data, list) else [])
            if isinstance(item, dict)
        ]


def collect_api(category: SemanticTier, include_nonpublic: bool = False) -> list[GhostRef]:
    """Entrypoint to collect the NVIDIA SASS API signature.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered.
    """
    if category != SemanticTier.UTIL:
        return []

    refs: list[GhostRef] = []

    sass_data = _load_exhaustive_sass()

    for inst_data in sass_data:
        mnemonic = inst_data.mnemonic
        base_desc = inst_data.description or f"NVIDIA SASS {mnemonic} instruction."
        raw_modifiers = [str(m) for m in inst_data.modifiers]
        parsed_modifiers = parse_sass_modifiers(raw_modifiers)

        arch: JsonValue = (
            list(inst_data.architecture) if isinstance(inst_data.architecture, list) else inst_data.architecture
        )
        valid_archs = resolve_sm_architectures(arch)
        env_tags = ["cuda"] + valid_archs

        docstring = f"NVIDIA SASS {mnemonic} instruction.\nDescription: {base_desc}\nModifiers: {', '.join(parsed_modifiers) if parsed_modifiers else 'None'}\nTarget Architectures: {', '.join(valid_archs)}\nPredicate Guard: Supported (@P0, @!P1)"

        operands_list: list[list[str]] = [
            [str(op) for op in sig] for sig in inst_data.operands if isinstance(sig, list)
        ]

        # Find the maximum length signature for the canonical representation
        max_sig: list[str] = []
        for sig in operands_list:
            if len(sig) > len(max_sig):
                max_sig = sig

        params: list[GhostParam] = []
        for i, raw_op in enumerate(max_sig):
            op_type = normalize_sass_operand_type(raw_op)
            role = "dst" if i == 0 else f"src{i - 1}"
            direction = OperandDirection.WRITE if i == 0 else OperandDirection.READ
            params.append(
                ExtendedGhostParam(
                    name=role,
                    kind=ParameterKind.POSITIONAL_ONLY,
                    annotation=op_type,
                    standardized_name=role,
                    description=f"Operand {i} ({role}) of type {op_type}",
                    direction=direction,
                    role=IRParameterRole.OPERAND,
                )
            )

        arch_json: JsonValue = (
            [str(a) for a in arch] if isinstance(arch, list) else (str(arch) if arch is not None else None)
        )
        domain_metadata: dict[str, JsonValue] = {
            "architecture": arch_json,
            "valid_architectures": list(valid_archs),
            "modifiers": list(parsed_modifiers),
            "predicate_supported": True,
            "operand_signatures": [list(s) for s in operands_list],
            "control_code_schema": DEFAULT_SASS_CONTROL_CODE_SCHEMA,
        }

        # Build overloads for all operand variations
        overloads: list[dict[str, JsonValue] | GhostRef] = []
        for sig_idx, sig in enumerate(operands_list):
            overload_params: list[GhostParam] = []
            for i, raw_op in enumerate(sig):
                op_type = normalize_sass_operand_type(raw_op)
                role = "dst" if i == 0 else f"src{i - 1}"
                direction = OperandDirection.WRITE if i == 0 else OperandDirection.READ
                overload_params.append(
                    ExtendedGhostParam(
                        name=role,
                        kind=ParameterKind.POSITIONAL_ONLY,
                        annotation=op_type,
                        standardized_name=role,
                        description=f"Operand {i} ({role}) of type {op_type}",
                        direction=direction,
                        role=IRParameterRole.OPERAND,
                    )
                )
            overloads.append(
                ExtendedGhostRef(
                    name=mnemonic,
                    api_path=f"nvidia_sass.inst.{mnemonic}",
                    kind="instruction",
                    params=overload_params,
                    docstring=f"Overload {sig_idx} for {mnemonic}: {', '.join(sig)}",
                    environment_tags=env_tags,
                    domain_metadata=domain_metadata,
                )
            )

        raw_struct_ops = build_structured_sass_operands(max_sig, mnemonic)

        refs.append(
            GhostIsaRef.model_validate(
                {
                    "name": mnemonic,
                    "api_path": f"nvidia_sass.inst.{mnemonic}",
                    "kind": "instruction",
                    "params": params,
                    "docstring": docstring,
                    "environment_tags": env_tags,
                    "overloads": overloads,
                    "domain_metadata": domain_metadata,
                    "predicate_guards": ["@P0", "@!P0", "@P1", "@!P1", "@PT"],
                    "instruction_modifiers": parsed_modifiers,
                    "supported_architectures": valid_archs,
                    "control_codes": IsaControlCodesSpec(
                        latency=4,
                        stall_count=1,
                        yield_flag=False,
                        read_barrier=0,
                        write_barrier=0,
                        wait_barrier_mask=0,
                    ),
                    "structured_operands": [IsaStructuredOperandSpec.model_validate(op) for op in raw_struct_ops],
                }
            )
        )

    return refs


def collect_domain(include_nonpublic: bool = False) -> list[GhostRef]:
    """Collect NVIDIA SASS hardware instruction schemas without `SemanticTier` shoehorning.

    Args:
        include_nonpublic: Whether to include non-public instructions.

    Returns:
        List of `GhostIsaRef` items.
    """
    return collect_api(SemanticTier.UTIL, include_nonpublic)


DOMAIN_COLLECTOR = DomainCollectorAdapter(CollectorDomain.HARDWARE_ISA, "instructions", collect_domain)
get_canonical_fallback_sass = _get_canonical_fallback_sass
load_exhaustive_sass = _load_exhaustive_sass
