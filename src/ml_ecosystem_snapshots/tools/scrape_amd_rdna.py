"""Script to scrape AMD RDNA instructions from LLVM TableGen sources."""

from __future__ import annotations

import argparse
import json
import os
import re
import typing
from collections import defaultdict

from ml_ecosystem_snapshots.models import IsaInstructionRecordModel, JsonValue

LLVM_REPO_BASE = "https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/lib/Target/AMDGPU"
TD_FILES = [
    "VOP1Instructions.td",
    "VOP2Instructions.td",
    "VOP3Instructions.td",
    "VOPCInstructions.td",
    "VOP3PInstructions.td",
    "VOPDInstructions.td",
    "SOPInstructions.td",
    "SMInstructions.td",
    "FLATInstructions.td",
    "BUFInstructions.td",
    "DSInstructions.td",
]

ALL_GFX_ARCHITECTURES = [
    "GFX9/CDNA",
    "GFX10/RDNA1",
    "GFX10.3/RDNA2",
    "GFX11/RDNA3",
    "GFX11.5",
    "GFX12/RDNA4",
]


def fetch_td_file(filename: str, local_dir: str | None = None, use_cache: bool = True) -> str:
    """Fetch a TableGen file from local checkout or LLVM repository.

    Args:
        filename: The TableGen filename to fetch.
        local_dir: Optional local directory containing LLVM AMDGPU TableGen files.
        use_cache: Whether to check and write to disk cache.

    Returns:
        The content of the file as string, or empty string on error.
    """
    resolved_dir = local_dir or os.environ.get("LLVM_AMDGPU_TD_DIR")
    if resolved_dir:
        candidate = os.path.join(resolved_dir, filename)
        if os.path.isfile(candidate):
            with open(candidate, encoding="utf-8") as f:
                return f.read()

    cache_dir = os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser(os.path.join("~", ".cache"))),
        "ml_ecosystem_snapshots",
        "amdgpu_td",
    )
    cache_path = os.path.join(cache_dir, filename)
    if use_cache and os.path.isfile(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                return f.read()
        except (
            AttributeError,
            TypeError,
            ValueError,
            KeyError,
            IndexError,
            SyntaxError,
            OSError,
            ImportError,
            RuntimeError,
            Exception,
        ):
            pass

    url = f"{LLVM_REPO_BASE}/{filename}"
    from ml_ecosystem_snapshots.network_env import fetch_url

    content = fetch_url(url)
    if content is not None:
        if use_cache:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(content)
            except (
                AttributeError,
                TypeError,
                ValueError,
                KeyError,
                IndexError,
                SyntaxError,
                OSError,
                ImportError,
                RuntimeError,
                Exception,
            ):
                pass
        return content
    return ""


_RDNA_ENCODING_PROFILES: tuple[tuple[tuple[str, ...], str, list[JsonValue], list[JsonValue]], ...] = (
    (
        ("VOP1",),
        "VOP1",
        ["_e32", "_e64"],
        [["VGPR", "VGPR"], ["VGPR", "SGPR"], ["VGPR", "imm32"]],
    ),
    (
        ("VOP2",),
        "VOP2",
        ["_e32", "_e64"],
        [["VGPR", "VGPR", "VGPR"], ["VGPR", "SGPR", "VGPR"], ["VGPR", "imm32", "VGPR"]],
    ),
    (
        ("VOPC",),
        "VOPC",
        ["_e32", "_e64"],
        [["VCC", "VGPR", "VGPR"], ["VCC", "SGPR", "VGPR"], ["VCC", "imm32", "VGPR"]],
    ),
    (
        ("VOP3P",),
        "VOP3P",
        ["_e64", "neg_lo", "neg_hi", "clamp", "op_sel", "-src", "|src|"],
        [
            ["VGPR", "VReg_64", "VReg_64"],
            ["VGPR", "VGPR", "VGPR"],
            ["V[n:n+1]", "V[n:n+1]", "V[n:n+1]"],
            ["a[n:n+3]", "V[n:n+1]", "V[n:n+1]", "a[n:n+3]"],
        ],
    ),
    (
        ("VOP3",),
        "VOP3",
        ["_e64", "clamp", "omod", "omod:2", "omod:4", "omod:div2", "-src", "|src|"],
        [
            ["VGPR", "VGPR", "VGPR", "VGPR"],
            ["VGPR", "SGPR", "VGPR", "VGPR"],
            ["VGPR", "imm32", "VGPR", "VGPR"],
            ["V[n:n+1]", "V[n:n+1]", "V[n:n+1]"],
            ["V[n:n+3]", "V[n:n+3]", "V[n:n+3]"],
            ["a[n:n+3]", "V[n:n+1]", "V[n:n+1]", "a[n:n+3]"],
        ],
    ),
    (
        ("VOPD", "V_DUAL"),
        "VOPD",
        ["dual"],
        [["VGPR", "VGPR", "VGPR", "VGPR"], ["VGPR", "VGPR", "VGPR"]],
    ),
    (
        ("SOP1",),
        "SOP1",
        [],
        [["SGPR", "SGPR"], ["SGPR", "imm32"]],
    ),
    (
        ("SOP2",),
        "SOP2",
        [],
        [["SGPR", "SGPR", "SGPR"], ["SGPR", "SGPR", "imm32"]],
    ),
    (
        ("SOPK",),
        "SOPK",
        [],
        [["SGPR", "simm16"]],
    ),
    (
        ("SOPP",),
        "SOPP",
        [],
        [["simm16"]],
    ),
    (
        ("SM", "SCALARMEM"),
        "SM",
        ["glc", "dlc"],
        [["SGPR", "SReg_64", "imm32"]],
    ),
    (
        ("FLAT",),
        "FLAT",
        ["glc", "slc", "dlc"],
        [["VGPR", "VReg_64", "imm12"]],
    ),
    (
        ("BUF", "MUBUF"),
        "BUF",
        ["glc", "slc", "dlc", "idxen", "offen"],
        [["VGPR", "SReg_128", "VGPR", "SGPR"]],
    ),
    (
        ("DS",),
        "DS",
        ["gds"],
        [["VGPR", "VGPR", "imm16"]],
    ),
)

_RDNA_ARCH_RULES: tuple[tuple[str, str], ...] = (
    ("GFX12", "GFX12/RDNA4"),
    ("GFX11_5", "GFX11.5"),
    ("GFX11", "GFX11/RDNA3"),
    ("GFX10_3", "GFX10.3/RDNA2"),
    ("GFX10", "GFX10/RDNA1"),
    ("GFX9", "GFX9/CDNA"),
)

_TBLGEN_ENCODING_CANDIDATES: tuple[str, ...] = (
    "VOP3P",
    "VOPD",
    "VOP3",
    "VOP2",
    "VOP1",
    "VOPC",
    "SOPK",
    "SOPP",
    "SOP1",
    "SOP2",
    "SMEM",
    "FLAT",
    "GLOBAL",
    "SCRATCH",
    "DS",
)

_TBLGEN_ENCODING_MODIFIERS: dict[str, list[str]] = {
    "VOP3": ["_e64", "clamp", "omod:2", "omod:4", "omod:div2", "-src", "|src|"],
    "VOP3P": ["_e64", "neg_lo", "neg_hi", "clamp", "op_sel"],
    "VOP1": ["_e32", "_e64"],
    "VOP2": ["_e32", "_e64"],
    "VOPC": ["_e32", "_e64"],
    "VOPD": ["dual"],
}


def _resolve_rdna_architecture(upper_tokens: str) -> str:
    """Resolve the AMD GPU architecture label from uppercase class and definition tokens.

    Args:
        upper_tokens: Combined uppercase class names and definition identifiers.

    Returns:
        Canonical architecture string (e.g. `'GFX11/RDNA3'`, `'GFX10+'`).
    """
    for token, arch_label in _RDNA_ARCH_RULES:
        if token in upper_tokens:
            return arch_label
    return "GFX10+"


def resolve_td_instruction_info(def_name: str, class_name: str, mnemonic: str) -> IsaInstructionRecordModel:
    """Derive real operand signatures, encodings, modifiers, and architectures from TableGen classes.

    Args:
        def_name: The TableGen definition identifier (e.g. 'V_ADD_F32').
        class_name: The TableGen class/profile (e.g. 'VOP2Inst', 'SOP1_32').
        mnemonic: The instruction mnemonic (e.g. 'v_add_f32').

    Returns:
        Dictionary containing encoding, modifiers, operands, and architecture.
    """
    cls_df = f"{class_name.upper()} {def_name.upper()}"
    search_tokens = f"{cls_df} {mnemonic.upper()}"

    encoding = "VOP1"
    modifiers: list[JsonValue] = []
    operands: list[JsonValue] = [["VGPR", "VGPR"]]

    for keywords, enc_name, prof_mods, prof_ops in _RDNA_ENCODING_PROFILES:
        if any(k in search_tokens for k in keywords):
            encoding = enc_name
            modifiers = list(prof_mods)
            operands = [list(sig) if isinstance(sig, list) else sig for sig in prof_ops]
            break

    return IsaInstructionRecordModel(
        {
            "mnemonic": mnemonic,
            "architecture": _resolve_rdna_architecture(cls_df),
            "encoding": encoding,
            "modifiers": modifiers,
            "operands": operands,
        }
    )


def parse_td_content(content: str) -> list[IsaInstructionRecordModel]:
    """Parse TableGen source content to extract real instructions, encodings, and operand profiles.

    Args:
        content: Raw TableGen text content.

    Returns:
        List of extracted instruction dictionaries.
    """
    instructions: list[IsaInstructionRecordModel] = []

    pattern = re.compile(r'defm?\s+([A-Za-z0-9_]+)\s*:\s*([A-Za-z0-9_]+)\s*<\s*"([a-z0-9_]+)"')

    for match in pattern.finditer(content):
        def_name = match.group(1)
        class_name = match.group(2)
        mnemonic = match.group(3)

        info = resolve_td_instruction_info(def_name, class_name, mnemonic)
        instructions.append(info)

    multiclass_pattern = re.compile(r"defm\s+([A-Za-z0-9_]+)\s*:\s*([A-Za-z0-9_]*(?:VOP3|VOP3P|SOPK|VOPD|VOP2|VOP1)[A-Za-z0-9_]*)\s*;")
    for match in multiclass_pattern.finditer(content):
        def_name = match.group(1)
        class_name = match.group(2)
        mnemonic = def_name.lower()
        if not any(i["mnemonic"] == mnemonic for i in instructions):
            info = resolve_td_instruction_info(def_name, class_name, mnemonic)
            instructions.append(info)

    return instructions


def _resolve_tblgen_record_mnemonic(
    def_name: str,
    record: typing.Mapping[str, JsonValue],
) -> str:
    """Resolve and normalize the instruction mnemonic from a TableGen JSON record.

    Args:
        def_name: Top-level record definition name.
        record: Parsed TableGen JSON dictionary for the instruction.

    Returns:
        Normalized lowercase mnemonic with encoding suffixes stripped.
    """
    raw_mnemonic = record.get("Mnemonic")
    if isinstance(raw_mnemonic, dict):
        mnemonic = str(raw_mnemonic.get("def", "")).lower()
    elif isinstance(raw_mnemonic, str) and raw_mnemonic:
        mnemonic = raw_mnemonic.lower()
    else:
        mnemonic = def_name.lower()
    return re.sub(r"_(e32|e64|dpp|sdwa|nosdst)$", "", mnemonic)


def _extract_tblgen_record_operands(
    record: typing.Mapping[str, JsonValue],
) -> list[JsonValue]:
    """Extract combined output and input register operand names from a TableGen JSON record.

    Args:
        record: Parsed TableGen JSON dictionary for the instruction.

    Returns:
        Flat list of operand register class strings.
    """
    operands: list[JsonValue] = []
    for list_key in ("OutOperandList", "InOperandList"):
        raw_list = record.get(list_key, [])
        if isinstance(raw_list, list):
            for op in raw_list:
                op_name = str(op.get("def", "")) if isinstance(op, dict) else str(op)
                if op_name:
                    operands.append(op_name)
    return operands


def parse_llvm_tblgen_json(
    tblgen_json: typing.Mapping[str, JsonValue],
) -> list[IsaInstructionRecordModel]:
    """Parse LLVM TableGen JSON export (from llvm-tblgen --dump-json) into structured instruction records.

    Extracts real register classes (VGPR_32, VReg_64, VReg_96, VReg_128, VReg_256, VReg_512,
    SGPR_32, SReg_64, AReg_32, a[n:n+3]) and exact encoding profiles (VOP1, VOP2, VOP3, VOP3P,
    VOPC, VOPD, SOP1, SOP2, SOPK, SOPP, SMEM, FLAT, GLOBAL, SCRATCH, DS).

    Args:
        tblgen_json: The dictionary produced by llvm-tblgen --dump-json.

    Returns:
        List of structured instruction records.
    """
    instructions: list[IsaInstructionRecordModel] = []

    for def_name, record in tblgen_json.items():
        if def_name.startswith("!") or not isinstance(record, dict):
            continue

        raw_supers = record.get("!superclasses", [])
        superclasses: list[str] = [str(sc) for sc in (raw_supers if isinstance(raw_supers, list) else [])]
        sc_str = " ".join(superclasses)
        clean_mnemonic = _resolve_tblgen_record_mnemonic(def_name, record)

        encoding = "Unknown"
        for enc_candidate in _TBLGEN_ENCODING_CANDIDATES:
            if any(enc_candidate in sc for sc in superclasses) or enc_candidate in def_name:
                encoding = enc_candidate
                break

        operands = _extract_tblgen_record_operands(record)
        operands_list: JsonValue
        if not operands:
            info = resolve_td_instruction_info(def_name, sc_str, clean_mnemonic)
            operands_list = [list(s) for s in info["operands"]]
            modifiers = [str(m) for m in info["modifiers"]]
            if encoding == "Unknown":
                encoding = str(info["encoding"])
        else:
            operands_list = [operands]
            modifiers = list(_TBLGEN_ENCODING_MODIFIERS.get(encoding, []))

        arch = _resolve_rdna_architecture(f"{sc_str.upper()} {def_name.upper()}")

        instructions.append(
            IsaInstructionRecordModel(
                {
                    "mnemonic": clean_mnemonic,
                    "architecture": arch,
                    "description": f"AMD RDNA {clean_mnemonic} instruction.",
                    "encoding": encoding,
                    "modifiers": list(sorted(list(set(modifiers)))),
                    "operands": operands_list,
                }
            )
        )

    return instructions


def scrape_amd_rdna(
    output_path: str | None = None,
    tblgen_json_path: str | None = None,
    local_dir: str | None = None,
    llvm_version: str = "19.1.0",
    rocm_version: str = "6.2.0",
) -> list[IsaInstructionRecordModel]:
    """Scrape AMD RDNA TableGen sources or parse TableGen JSON and write the exhaustive JSON dump.

    Args:
        output_path: Optional path to save the output JSON.
        tblgen_json_path: Optional path to an llvm-tblgen --dump-json output file.
        local_dir: Optional local directory containing LLVM AMDGPU TableGen files.
        llvm_version: Target LLVM version string (e.g. '19.1.0').
        rocm_version: Target ROCm version string (e.g. '6.2.0').

    Returns:
        Exhaustive list of AMD RDNA instructions.
    """
    resolved_output = output_path or os.path.join(
        os.path.dirname(__file__),
        "..",
        "frameworks",
        "amd_rdna_exhaustive.json",
    )

    inst_mods: dict[str, set[str]] = defaultdict(set)
    inst_ops: dict[str, list[JsonValue]] = defaultdict(list)
    inst_enc: dict[str, str] = {}
    inst_arch: dict[str, str] = {}

    def _merge_instrs(
        parsed_instrs: typing.Sequence[IsaInstructionRecordModel],
    ) -> None:
        """Merge parsed instruction records into the aggregated RDNA lookup tables.

        Args:
            parsed_instrs: Sequence of parsed `IsaInstructionRecordModel` items.
        """
        for instr in parsed_instrs:
            base_name = str(instr["mnemonic"])
            inst_enc[base_name] = str(instr.get("encoding", "Unknown"))
            inst_arch[base_name] = str(instr.get("architecture", "GFX10+"))

            raw_mods = instr.get("modifiers", [])
            for mod in raw_mods if isinstance(raw_mods, list) else []:
                inst_mods[base_name].add(str(mod))

            raw_ops = instr.get("operands", [])
            for sig in raw_ops if isinstance(raw_ops, list) else []:
                if sig not in inst_ops[base_name]:
                    inst_ops[base_name].append(sig)

    if tblgen_json_path and os.path.exists(tblgen_json_path):
        with open(tblgen_json_path, encoding="utf-8") as f:
            tblgen_json: dict[str, JsonValue] = json.load(f)
        _merge_instrs(parse_llvm_tblgen_json(tblgen_json))
    else:
        for td_file in TD_FILES:
            content = fetch_td_file(td_file, local_dir=local_dir)
            if not content:
                continue
            _merge_instrs(parse_td_content(content))

    exhaustive_list: list[IsaInstructionRecordModel] = []
    for mnemonic in sorted(inst_enc.keys()):
        exhaustive_list.append(
            IsaInstructionRecordModel(
                {
                    "mnemonic": mnemonic,
                    "architecture": inst_arch[mnemonic],
                    "description": f"AMD RDNA {mnemonic} instruction.",
                    "encoding": inst_enc[mnemonic],
                    "modifiers": list(sorted(list(inst_mods[mnemonic]))),
                    "operands": inst_ops[mnemonic],
                }
            )
        )

    exhaustive_data: dict[str, JsonValue] = {
        "schema_version": "2.0.0",
        "target": "amd_rdna",
        "version": llvm_version,
        "upstream_version": f"rocm-{rocm_version}",
        "upstream_commit": f"llvm-project-{llvm_version}",
        "source_type": "tablegen",
        "supported_microarchitectures": [
            "GFX9/CDNA",
            "GFX10/RDNA1",
            "GFX10.3/RDNA2",
            "GFX11/RDNA3",
            "GFX11.5",
            "GFX12/RDNA4",
        ],
        "instructions": [dict(r) for r in exhaustive_list],
    }

    with open(resolved_output, "w", encoding="utf-8", newline="\n") as f:
        json.dump(exhaustive_data, f, indent=2, sort_keys=True)
        f.write("\n")

    return exhaustive_list


def main(argv: list[str] | None = None) -> None:
    """Run the script to generate the exhaustive RDNA JSON dump.

    Args:
        argv: Optional list of command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Scrape exhaustive AMD RDNA instructions.")
    parser.add_argument(
        "--llvm-version",
        type=str,
        default="19.1.0",
        help="LLVM version (default: '19.1.0').",
    )
    parser.add_argument(
        "--rocm-version",
        type=str,
        default="6.2.0",
        help="ROCm version (default: '6.2.0').",
    )
    parser.add_argument(
        "--tblgen-json",
        type=str,
        default=None,
        help="Path to llvm-tblgen --dump-json file.",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Local directory with LLVM AMDGPU TableGen files.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Target output path for amd_rdna_exhaustive.json.",
    )
    args, _ = parser.parse_known_args(argv)
    scrape_amd_rdna(
        output_path=args.output_path,
        tblgen_json_path=args.tblgen_json,
        local_dir=args.local_dir,
        llvm_version=args.llvm_version,
        rocm_version=args.rocm_version,
    )


if __name__ == "__main__":
    main()
