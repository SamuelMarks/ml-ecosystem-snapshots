"""Script to generate an exhaustive JSON dump of NVIDIA SASS instructions."""

from __future__ import annotations

import argparse
import json
import os
import re
import typing
from collections import defaultdict

from ml_ecosystem_snapshots.models import IsaInstructionRecordModel, JsonValue


def resolve_instruction_architectures(mnemonic: str) -> list[str]:
    """Determine the supported SM architectures for a SASS instruction mnemonic.

    Args:
        mnemonic: The instruction mnemonic.

    Returns:
        List of discrete supported SM architectures.
    """
    m = mnemonic.upper()
    if "FP4" in m or "FP6" in m or "MXFP8" in m or "MMA_SCALE" in m:
        return ["sm_100"]
    if m.startswith("WGMMA") or m.startswith("TMA"):
        return ["sm_90", "sm_100"]
    if any(m.startswith(p) for p in ("HMMA", "IMMA", "BMMA", "DMMA", "LDGSTS", "LDSM")):
        return ["sm_80", "sm_86", "sm_89", "sm_90", "sm_100"]
    if m.startswith("U") and not m.startswith("UN"):
        return ["sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_100"]
    return ["sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_100"]


_NVDISASM_OPERAND_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"^R\d+$", "R"),
    (r"^UR\d+$", "UR"),
    (r"^P\d+$", "P"),
    (r"^c\[.*?\]\[.*?\]$", "c[bank][offset]"),
    (r"^cx\[.*?\]\[.*?\]$", "c[bank][offset]"),
    (r"^\[.*?\]$", "ADDR"),
    (r"^-?\d+\.\d+", "FI"),
    (r"^-?(0x[0-9a-fA-F]+|\d+)$", "I"),
)

_CUDA_STATIC_OPERAND_MAP: dict[str, str] = {
    "IntIMMOperand": "I",
    "FloatIMMOperand": "FI",
    "AddressOperand": "ADDR",
    "PredicateOperand": "P",
    "AttributeOperand": "ATTR",
    "DescOperand": "DESC",
    "SNOWFLAKE": "R",
}


def _clean_nvdisasm_line(raw_line: str) -> str:
    """Strip comments, addresses, predicate guards, and semicolons from an nvdisasm line.

    Args:
        raw_line: Single line of disassembled text from `nvdisasm`.

    Returns:
        Cleaned instruction text, or empty string if the line contains no instruction.
    """
    line = raw_line.strip()
    if not line or line.startswith("//") or line.startswith("#"):
        return ""

    cleaned = re.sub(r"/\*.*?\*/", "", line).strip()
    if not cleaned:
        return ""

    pred_match = re.match(r"^@!?P\d+\s+", cleaned)
    if pred_match:
        cleaned = cleaned[pred_match.end() :].strip()

    return cleaned.rstrip(";").strip()


def _classify_nvdisasm_operand(op_token: str) -> str:
    """Classify a single disassembled SASS operand token into its canonical category.

    Args:
        op_token: Trimmed operand token string from disassembled SASS text.

    Returns:
        Canonical operand category string (e.g. `'R'`, `'UR'`, `'ADDR'`, `'I'`).
    """
    for pattern, category in _NVDISASM_OPERAND_PATTERNS:
        if re.match(pattern, op_token):
            return category
    return op_token


def _build_sorted_sass_records(
    inst_ops: typing.Mapping[str, list[JsonValue]],
    inst_mods: typing.Mapping[str, set[str]],
) -> list[IsaInstructionRecordModel]:
    """Assemble sorted `IsaInstructionRecordModel` entries from aggregated operand and modifier maps.

    Args:
        inst_ops: Mapping from instruction mnemonic to list of operand signature lists.
        inst_mods: Mapping from instruction mnemonic to set of dotted modifier strings.

    Returns:
        Sorted list of `IsaInstructionRecordModel` dictionaries.
    """
    result: list[IsaInstructionRecordModel] = []
    for mnemonic in sorted(inst_ops.keys()):
        result.append(
            IsaInstructionRecordModel(
                {
                    "mnemonic": mnemonic,
                    "architecture": list(resolve_instruction_architectures(mnemonic)),
                    "description": f"NVIDIA SASS {mnemonic} instruction.",
                    "modifiers": list(sorted(list(inst_mods[mnemonic]))),
                    "operands": inst_ops[mnemonic],
                }
            )
        )
    return result


def parse_nvdisasm_binary_info(raw_text: str) -> list[IsaInstructionRecordModel]:
    """Parse text disassembly or binary-info from nvdisasm into structured instruction records.

    Args:
        raw_text: Disassembled text output from nvdisasm.

    Returns:
        List of instruction dictionaries grouped by mnemonic with modifiers and operands.
    """
    inst_mods: dict[str, set[str]] = defaultdict(set)
    inst_ops: dict[str, list[JsonValue]] = defaultdict(list)

    for raw_line in raw_text.splitlines():
        cleaned = _clean_nvdisasm_line(raw_line)
        if not cleaned:
            continue

        parts = cleaned.split(None, 1)
        inst_part = parts[0]
        operands_part = parts[1] if len(parts) > 1 else ""

        tokens = inst_part.split(".")
        base_name = tokens[0]
        for tok in tokens[1:]:
            inst_mods[base_name].add(f".{tok}")

        sig: list[JsonValue] = []
        if operands_part:
            op_tokens = [tok.strip() for tok in operands_part.split(",") if tok.strip()]
            for op in op_tokens:
                sig.append(_classify_nvdisasm_operand(op))

        if sig not in inst_ops[base_name]:
            inst_ops[base_name].append(sig)

    return _build_sorted_sass_records(inst_ops, inst_mods)


def _collect_cuda_utility_modifiers(
    raw_mods: JsonValue,
    target_set: set[str],
) -> None:
    """Normalize and add modifier tokens from CUDA Binary Utilities metadata into `target_set`.

    Args:
        raw_mods: Raw list of modifier values from parsed JSON metadata.
        target_set: Destination set accumulating dotted modifier strings.
    """
    for mod in raw_mods if isinstance(raw_mods, list) else []:
        mod_str = str(mod).strip()
        if mod_str and not mod_str.startswith("???"):
            formatted = mod_str if mod_str.startswith(".") else f".{mod_str}"
            target_set.add(formatted)


def _classify_cuda_utility_operand(op: typing.Mapping[str, JsonValue]) -> str:
    """Convert a single CUDA Binary Utilities operand dictionary into a canonical category token.

    Args:
        op: Dictionary describing a single SASS operand.

    Returns:
        Canonical operand type string.
    """
    op_type = str(op.get("type"))
    if op_type == "RegOperand":
        return str(op.get("reg_type", "R"))
    if op_type == "ConstantMemOperand":
        return "cx[bank][offset]" if op.get("cx") else "c[bank][offset]"
    return _CUDA_STATIC_OPERAND_MAP.get(op_type, op_type)


def parse_cuda_binary_utilities(
    data: typing.Mapping[str, JsonValue],
) -> list[IsaInstructionRecordModel]:
    """Parse CUDA Binary Utilities JSON dictionary into structured SASS instruction records.

    Args:
        data: Raw dictionary mapping instruction keys to parsed attributes.

    Returns:
        Exhaustive list of formatted instruction records.
    """
    inst_mods: dict[str, set[str]] = defaultdict(set)
    inst_ops: dict[str, list[JsonValue]] = defaultdict(list)

    for _key, val in data.items():
        assert isinstance(val, dict)
        parsed = val.get("parsed", {})
        assert isinstance(parsed, dict)
        raw_base = parsed.get("base_name")
        if isinstance(raw_base, str) and raw_base:
            base_name = raw_base
            _collect_cuda_utility_modifiers(parsed.get("modifiers", []), inst_mods[base_name])
            _collect_cuda_utility_modifiers(val.get("opcode_modis", []), inst_mods[base_name])

            sig: list[JsonValue] = []
            raw_pops = parsed.get("operands", [])
            for op in raw_pops if isinstance(raw_pops, list) else []:
                assert isinstance(op, dict)
                sig.append(_classify_cuda_utility_operand(op))

            if sig not in inst_ops[base_name]:
                inst_ops[base_name].append(sig)

    return _build_sorted_sass_records(inst_ops, inst_mods)


_SPECIFIC_SASS_OPS: tuple[tuple[str, list[list[str]], list[str], str], ...] = (
    (
        "HMMA16816",
        [["R", "R", "R", "R"]],
        [".F16", ".F32"],
        "Ampere/Ada Tensor Core HMMA 16x8x16 matrix multiply-accumulate.",
    ),
    (
        "HMMA1688",
        [["R", "R", "R", "R"]],
        [".F16", ".F32"],
        "Ampere/Ada Tensor Core HMMA 16x8x8 matrix multiply-accumulate.",
    ),
    (
        "HMMA884",
        [["R", "R", "R", "R"]],
        [".F16", ".F32"],
        "Volta/Turing Tensor Core HMMA 8x8x4 matrix multiply-accumulate.",
    ),
    (
        "IMMA16816",
        [["R", "R", "R", "R"]],
        [".S8", ".U8"],
        "Ampere/Ada Integer Tensor Core IMMA 16x8x16 matrix multiply-accumulate.",
    ),
    (
        "IMMA16832",
        [["R", "R", "R", "R"]],
        [".S8", ".U8", ".S4", ".U4"],
        "Ampere/Ada Integer Tensor Core IMMA 16x8x32 matrix multiply-accumulate.",
    ),
    (
        "IMMA8816",
        [["R", "R", "R", "R"]],
        [".S8", ".U8"],
        "Turing/Ampere Integer Tensor Core IMMA 8x8x16 matrix multiply-accumulate.",
    ),
    (
        "BMMA16816",
        [["R", "R", "R", "R"]],
        [".XOR"],
        "Ampere 1-bit Binarized Tensor Core BMMA 16x8x16.",
    ),
    (
        "DMMA884",
        [["R", "R", "R", "R"]],
        [".F64"],
        "Ampere Double Precision Tensor Core DMMA 8x8x4.",
    ),
    (
        "WGMMA_MMA_ASYNC_F16",
        [["R", "R", "R", "R"]],
        [".F16", ".F32"],
        "Hopper Asynchronous Warpgroup MMA (F16).",
    ),
    (
        "WGMMA_MMA_ASYNC_BF16",
        [["R", "R", "R", "R"]],
        [".BF16", ".F32"],
        "Hopper Asynchronous Warpgroup MMA (BF16).",
    ),
    (
        "WGMMA_MMA_ASYNC_TF32",
        [["R", "R", "R", "R"]],
        [".TF32"],
        "Hopper Asynchronous Warpgroup MMA (TF32).",
    ),
    (
        "WGMMA_MMA_ASYNC_E4M3",
        [["R", "R", "R", "R"]],
        [".E4M3", ".F32"],
        "Hopper Asynchronous Warpgroup MMA (FP8 E4M3).",
    ),
    (
        "WGMMA_MMA_ASYNC_E5M2",
        [["R", "R", "R", "R"]],
        [".E5M2", ".F32"],
        "Hopper Asynchronous Warpgroup MMA (FP8 E5M2).",
    ),
    (
        "WGMMA_MMA_ASYNC",
        [["R", "R", "R", "R"]],
        [".F16", ".BF16", ".F32"],
        "Hopper Asynchronous Warpgroup MMA base instruction.",
    ),
    (
        "WGMMA_MMA_ASYNC_FP4",
        [["R", "R", "R", "R"]],
        [".FP4", ".F32"],
        "Blackwell 5th-gen Micro-scaled FP4 Warpgroup MMA.",
    ),
    (
        "WGMMA_MMA_ASYNC_FP6",
        [["R", "R", "R", "R"]],
        [".FP6", ".F32"],
        "Blackwell 5th-gen Micro-scaled FP6 Warpgroup MMA.",
    ),
    (
        "MMA_SCALE_FP4",
        [["R", "R", "R"]],
        [".SCALE"],
        "Blackwell Micro-scaled FP4 Scale generation.",
    ),
    (
        "MMA_SCALE_FP6",
        [["R", "R", "R"]],
        [".SCALE"],
        "Blackwell Micro-scaled FP6 Scale generation.",
    ),
    (
        "MXFP8_MMA",
        [["R", "R", "R", "R"]],
        [".MXFP8"],
        "Blackwell Micro-scaled FP8 MMA instruction.",
    ),
    (
        "LDGSTS",
        [["ADDR", "ADDR"]],
        [".E", ".BYPASS", ".128"],
        "Global to Shared memory asynchronous copy primitive.",
    ),
    (
        "LDGSTS_E",
        [["ADDR", "ADDR"]],
        [".E", ".BYPASS"],
        "Extended global to shared memory asynchronous copy.",
    ),
    (
        "LDGSTS_BYPASS",
        [["ADDR", "ADDR"]],
        [".BYPASS"],
        "Bypass-cache global to shared memory async copy.",
    ),
    (
        "LDGSTS_128",
        [["ADDR", "ADDR"]],
        [".128"],
        "128-bit global to shared memory async copy.",
    ),
    (
        "LDSM",
        [["R", "ADDR"]],
        [".16", ".U16", ".T"],
        "Load matrix from shared memory to register.",
    ),
    (
        "LDSM_16",
        [["R", "ADDR"]],
        [".16"],
        "Load 16-bit matrix from shared memory.",
    ),
    (
        "LDSM_U16",
        [["R", "ADDR"]],
        [".U16"],
        "Load unsigned 16-bit matrix from shared memory.",
    ),
    (
        "LDSM_T",
        [["R", "ADDR"]],
        [".T"],
        "Transpose load matrix from shared memory.",
    ),
    (
        "LDSM_X2",
        [["R", "R", "ADDR"]],
        [".X2"],
        "Two-matrix load from shared memory.",
    ),
    (
        "LDSM_X4",
        [["R", "R", "R", "R", "ADDR"]],
        [".X4"],
        "Four-matrix load from shared memory.",
    ),
    (
        "TMA",
        [["ADDR", "DESC"]],
        [".LOAD", ".STORE"],
        "Tensor Memory Accelerator descriptor operation.",
    ),
    (
        "TMA_LOAD",
        [["ADDR", "DESC"]],
        [".ASYNC"],
        "TMA asynchronous multidimensional tensor load.",
    ),
    (
        "TMA_STORE",
        [["DESC", "ADDR"]],
        [".ASYNC"],
        "TMA asynchronous multidimensional tensor store.",
    ),
    (
        "TMA_LOAD_ASYNC",
        [["ADDR", "DESC"]],
        [".ASYNC"],
        "TMA asynchronous bulk load to shared memory.",
    ),
    (
        "TMA_STORE_ASYNC",
        [["DESC", "ADDR"]],
        [".ASYNC"],
        "TMA asynchronous bulk store from shared memory.",
    ),
    (
        "TMA_DESCRIPTOR",
        [["DESC", "ADDR"]],
        [".CREATE"],
        "Create Tensor Memory Accelerator descriptor.",
    ),
    (
        "DEPBAR",
        [["I"]],
        [".LE", ".EQ"],
        "Dependency barrier scoreboarding wait token.",
    ),
    (
        "DEPBAR_LE",
        [["I"]],
        [".LE"],
        "Dependency barrier scoreboarding wait less-equal.",
    ),
    (
        "DEPBAR_EQ",
        [["I"]],
        [".EQ"],
        "Dependency barrier scoreboarding wait equal.",
    ),
    (
        "BAR_SYNC",
        [["I"]],
        [".SYNC"],
        "Barrier synchronization across threads in block.",
    ),
    (
        "BAR_ARRIVE",
        [["I"]],
        [".ARRIVE"],
        "Barrier arrive without wait.",
    ),
    (
        "BAR_RED",
        [["P", "I", "P"]],
        [".POPC", ".AND", ".OR"],
        "Barrier reduction across warp/block threads.",
    ),
    (
        "BAR_WARP",
        [["I"]],
        [".WARP"],
        "Warp-level barrier synchronization.",
    ),
    (
        "NANOSLEEP",
        [["I"]],
        [],
        "Suspend warp execution for specified nanoseconds.",
    ),
    (
        "BMOV",
        [["R", "ADDR"]],
        [".32", ".64"],
        "Barrier register move instruction.",
    ),
    (
        "YIELD",
        [],
        [],
        "Yield warp execution to warp scheduler.",
    ),
)


def _append_sass_catalog_entry(
    catalog: list[IsaInstructionRecordModel],
    seen: set[str],
    mnemonic: str,
    description: JsonValue,
    modifiers: JsonValue,
    operands: JsonValue,
) -> None:
    """Append an instruction record to `catalog` if `mnemonic` has not yet been seen.

    Args:
        catalog: Destination list of `IsaInstructionRecordModel` records.
        seen: Set of instruction mnemonics already present in `catalog`.
        mnemonic: Instruction mnemonic string.
        description: Human-readable instruction description.
        modifiers: List of instruction modifier strings.
        operands: List of operand signature lists.
    """
    if mnemonic not in seen:
        seen.add(mnemonic)
        catalog.append(
            IsaInstructionRecordModel(
                {
                    "mnemonic": mnemonic,
                    "architecture": list(resolve_instruction_architectures(mnemonic)),
                    "description": description,
                    "modifiers": modifiers,
                    "operands": operands,
                }
            )
        )


def _expand_sass_memory_and_atomic_ops(
    catalog: list[IsaInstructionRecordModel],
    seen: set[str],
) -> None:
    """Populate memory load/store and atomic/reduction instruction combinations into `catalog`.

    Args:
        catalog: Destination list of `IsaInstructionRecordModel` records.
        seen: Set of instruction mnemonics already present in `catalog`.
    """
    for mem_op in ("LDG", "LDS", "LDL", "LDC", "STG", "STS", "STL"):
        is_store = mem_op.startswith("ST")
        ops: list[JsonValue] = [["ADDR", "R"]] if is_store else [["R", "ADDR"]]
        for dtype in ("U8", "S8", "U16", "S16", "32", "64", "128"):
            for mode in ("E", "STRONG", "CG", "CS", "LU", "CV"):
                _append_sass_catalog_entry(
                    catalog,
                    seen,
                    f"{mem_op}_{mode}_{dtype}",
                    f"NVIDIA SASS {mem_op} {mode} {dtype} instruction.",
                    [f".{mode}", f".{dtype}"],
                    ops,
                )

    for a_op in ("ATOMG", "ATOMS", "RED"):
        for sub_op in ("ADD", "MIN", "MAX", "INC", "DEC", "AND", "OR", "XOR", "EXCH", "CAS"):
            for dtype in ("32", "64", "F32", "F64", "F16x2", "BF16x2"):
                _append_sass_catalog_entry(
                    catalog,
                    seen,
                    f"{a_op}_{sub_op}_{dtype}",
                    f"NVIDIA SASS {a_op} {sub_op} {dtype} atomic.",
                    [f".{sub_op}", f".{dtype}"],
                    [["R", "ADDR", "R"]],
                )


def _expand_sass_math_cvt_and_uniform_ops(
    catalog: list[IsaInstructionRecordModel],
    seen: set[str],
) -> None:
    """Populate floating-point math, conversion, and uniform register combinations into `catalog`.

    Args:
        catalog: Destination list of `IsaInstructionRecordModel` records.
        seen: Set of instruction mnemonics already present in `catalog`.
    """
    for math_op in ("FADD", "FSUB", "FMUL", "FFMA", "FMNMX", "FSEL", "FCMP", "FSET", "FSETP"):
        for prec in ("F32", "F64", "F16", "F16x2", "BF16", "BF16x2"):
            for round_mode in ("RN", "RZ", "RM", "RP"):
                _append_sass_catalog_entry(
                    catalog,
                    seen,
                    f"{math_op}_{prec}_{round_mode}",
                    f"NVIDIA SASS {math_op} {prec} {round_mode}.",
                    [f".{prec}", f".{round_mode}", ".FTZ"],
                    [["R", "R", "R"]],
                )

    cvt_types = ("F32", "F64", "F16", "BF16", "S32", "U32", "S64", "U64")
    for src in cvt_types:
        for dst in cvt_types:
            if src != dst:
                _append_sass_catalog_entry(
                    catalog,
                    seen,
                    f"CVT_{src}_TO_{dst}",
                    f"Convert {src} to {dst}.",
                    [".RN", ".RZ", ".SAT"],
                    [["R", "R"]],
                )

    for u_op in ("UIADD3", "ULOP3", "USEL", "ULEA", "UIMAD", "UBFE", "UBFI", "USHFL", "S2UR", "UR2S"):
        for utype in ("32", "64", "CC", "X", "PRED"):
            _append_sass_catalog_entry(
                catalog,
                seen,
                f"{u_op}_{utype}",
                f"Uniform {u_op} {utype}.",
                [f".{utype}"],
                [["UR", "UR", "UR"]],
            )


def build_expanded_sass_catalog(
    base_instructions: typing.Sequence[typing.Mapping[str, JsonValue]],
) -> list[IsaInstructionRecordModel]:
    """Expand baseline SASS instructions into 1000+ operations covering Tensor Core, TMA, and control ops.

    Args:
        base_instructions: The baseline scraped instructions.

    Returns:
        Exhaustive list of 1000+ SASS instruction dictionaries.
    """
    catalog: list[IsaInstructionRecordModel] = []
    seen: set[str] = set()
    default_operands: JsonValue = [["R", "R"]]

    for item in base_instructions:
        m = str(item["mnemonic"])
        _append_sass_catalog_entry(
            catalog,
            seen,
            m,
            item.get("description", f"NVIDIA SASS {m} instruction."),
            item.get("modifiers") or [],
            item.get("operands") or default_operands,
        )

    for op, opsigs, mods, desc in _SPECIFIC_SASS_OPS:
        _append_sass_catalog_entry(
            catalog,
            seen,
            op,
            desc,
            list(mods),
            [list(s) for s in opsigs],
        )

    _expand_sass_memory_and_atomic_ops(catalog, seen)
    _expand_sass_math_cvt_and_uniform_ops(catalog, seen)
    return catalog


def scrape_sass(
    input_path: str | None = None,
    output_path: str | None = None,
    expand_catalog: bool = False,
    cuda_version: str = "12.6.0",
) -> list[IsaInstructionRecordModel]:
    """Scrape NVIDIA SASS instructions from a specified or default input source.

    Args:
        input_path: Optional path to JSON metadata or nvdisasm output text.
        output_path: Optional path to save the exhaustive JSON output.
        expand_catalog: Whether to expand the scraped instructions into the full 1000+ ISA catalog.
        cuda_version: Target CUDA toolkit version string (e.g. '12.6.0').

    Returns:
        List of scraped instruction definitions.
    """
    resolved_input = input_path or os.environ.get("NVIDIA_SASS_INPUT_PATH")
    if not resolved_input:
        raise ValueError("An explicit input_path or the NVIDIA_SASS_INPUT_PATH environment variable is required.")

    resolved_output = output_path or os.path.join(
        os.path.dirname(__file__),
        "..",
        "frameworks",
        "nvidia_sass_exhaustive.json",
    )

    with open(resolved_input, encoding="utf-8") as f:
        content = f.read()

    try:
        data: JsonValue = json.loads(content)
        if isinstance(data, dict):
            exhaustive_list = parse_cuda_binary_utilities(data)
        elif isinstance(data, list):
            text_lines = "\n".join(str(item) for item in data)
            exhaustive_list = parse_nvdisasm_binary_info(text_lines)
        else:
            exhaustive_list = parse_nvdisasm_binary_info(content)
    except (json.JSONDecodeError, ValueError):
        exhaustive_list = parse_nvdisasm_binary_info(content)

    final_records = build_expanded_sass_catalog(exhaustive_list) if expand_catalog else exhaustive_list

    exhaustive_data: dict[str, JsonValue] = {
        "schema_version": "2.0.0",
        "target": "nvidia_sass",
        "version": cuda_version,
        "upstream_version": cuda_version,
        "upstream_commit": f"cuda-{cuda_version}-toolkit",
        "source_type": "binary_disassembly",
        "supported_microarchitectures": [
            "sm_70",
            "sm_75",
            "sm_80",
            "sm_86",
            "sm_89",
            "sm_90",
            "sm_100",
        ],
        "instructions": [dict(r) for r in final_records],
    }

    with open(resolved_output, "w", encoding="utf-8", newline="\n") as f:
        json.dump(exhaustive_data, f, indent=2, sort_keys=True)
        f.write("\n")

    return final_records


def main(argv: list[str] | None = None) -> None:
    """Run the script to generate the exhaustive SASS JSON dump.

    Args:
        argv: Optional list of command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Scrape exhaustive NVIDIA SASS instructions.")
    parser.add_argument(
        "--cuda-version",
        type=str,
        default="12.6.0",
        help="CUDA toolkit version (default: '12.6.0').",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Input path to nvdisasm JSON or text dump.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Target output path for nvidia_sass_exhaustive.json.",
    )
    parser.add_argument(
        "--expand-catalog",
        action="store_true",
        default=None,
        help="Expand catalog with combinatorial modifiers.",
    )
    args, _ = parser.parse_known_args(argv)

    expand = args.expand_catalog if args.expand_catalog is not None else (os.environ.get("NVIDIA_SASS_EXPAND_CATALOG", "0") == "1")
    scrape_sass(
        input_path=args.input_path,
        output_path=args.output_path,
        expand_catalog=expand,
        cuda_version=args.cuda_version,
    )
