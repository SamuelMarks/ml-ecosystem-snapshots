"""Golden equivalence tests asserting exact operation presence and scraper cleanliness."""

from __future__ import annotations

import json
import os


def _load_dataset_or_canonical(fw_dir: str, filename: str) -> list[dict[str, object]]:
    """Load an exhaustive JSON file if present on disk, or fall back to canonical framework records.

    Args:
        fw_dir: Directory containing framework modules and optional JSON dumps.
        filename: Target JSON filename (e.g., 'stablehlo_exhaustive.json').

    Returns:
        List of operation or instruction dictionary records.
    """
    fpath = os.path.join(fw_dir, filename)
    if os.path.isfile(fpath) and os.path.getsize(fpath) >= 10000:
        with open(fpath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            return list(
                raw.get("categories", {}).get("UTIL")
                or raw.get("instructions")
                or raw.get("operations")
                or raw.get("items")
                or []
            )
        return list(raw)

    from ml_switcheroo_ir.schema.ghost import SemanticTier

    if filename == "stablehlo_exhaustive.json":
        from ml_ecosystem_snapshots.frameworks.stablehlo import (
            collect_api as shlo_collect,
        )

        return [r.model_dump() for r in shlo_collect(SemanticTier.UTIL)]
    if filename == "mlir_exhaustive.json":
        from ml_ecosystem_snapshots.frameworks.mlir import collect_api as mlir_collect

        return [r.model_dump() for r in mlir_collect(SemanticTier.UTIL)]
    if filename == "nvidia_sass_exhaustive.json":
        from ml_ecosystem_snapshots.frameworks.nvidia_sass import (
            collect_api as sass_collect,
        )

        return [r.model_dump() for r in sass_collect(SemanticTier.UTIL)]
    if filename == "nvidia_ptx_exhaustive.json":
        from ml_ecosystem_snapshots.frameworks.nvidia_ptx import (
            collect_api as ptx_collect,
        )

        return [r.model_dump() for r in ptx_collect(SemanticTier.UTIL)]
    if filename == "amd_rdna_exhaustive.json":
        from ml_ecosystem_snapshots.frameworks.amd_rdna import (
            collect_api as rdna_collect,
        )

        return [r.model_dump() for r in rdna_collect(SemanticTier.UTIL)]
    return []


def test_golden_symbol_counts_and_mandatory_ops() -> None:
    """Verify that golden datasets contain all mandatory operations and minimum symbol thresholds."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    fw_dir = os.path.join(repo_root, "src", "ml_ecosystem_snapshots", "frameworks")

    # 1. StableHLO
    shlo_data = _load_dataset_or_canonical(fw_dir, "stablehlo_exhaustive.json")
    assert len(shlo_data) >= 10, f"StableHLO golden count too low: {len(shlo_data)}"
    shlo_names = {op.get("api_path") or op.get("name") for op in shlo_data}
    for req in [
        "stablehlo.dot_general",
        "stablehlo.convolution",
        "stablehlo.reduce",
        "stablehlo.while",
        "stablehlo.gather",
        "stablehlo.scatter",
        "stablehlo.add",
        "stablehlo.multiply",
    ]:
        assert req in shlo_names, f"Mandatory StableHLO operation missing: {req}"

    # 2. Core MLIR
    mlir_data = _load_dataset_or_canonical(fw_dir, "mlir_exhaustive.json")
    assert len(mlir_data) >= 10, f"MLIR golden count too low: {len(mlir_data)}"
    mlir_names = {op.get("api_path") or op.get("name") for op in mlir_data}
    for req in [
        "arith.addf",
        "arith.mulf",
        "math.exp",
        "tensor.extract",
        "linalg.matmul",
        "scf.for",
    ]:
        assert req in mlir_names, f"Mandatory MLIR operation missing: {req}"

    # 3. NVIDIA SASS
    sass_data = _load_dataset_or_canonical(fw_dir, "nvidia_sass_exhaustive.json")
    assert len(sass_data) >= 10, f"SASS golden count too low: {len(sass_data)}"
    sass_mnemonics = {
        str(op.get("mnemonic") or op.get("name") or "").upper() for op in sass_data
    }
    for req in ["FFMA", "HMMA", "WGMMA", "LDG", "STG", "BRA", "EXIT"]:
        assert req in sass_mnemonics, f"Mandatory SASS instruction missing: {req}"

    # 4. NVIDIA PTX
    ptx_data = _load_dataset_or_canonical(fw_dir, "nvidia_ptx_exhaustive.json")
    assert len(ptx_data) >= 10, f"PTX golden count too low: {len(ptx_data)}"
    ptx_names = {
        str(op.get("mnemonic") or op.get("name") or "").lower().split(".")[0]
        for op in ptx_data
    }
    for req in ["add", "sub", "mul", "atom", "and", "ret"]:
        assert req in ptx_names, f"Mandatory PTX instruction missing: {req}"

    # 5. AMD RDNA
    rdna_data = _load_dataset_or_canonical(fw_dir, "amd_rdna_exhaustive.json")
    assert len(rdna_data) >= 10, f"RDNA golden count too low: {len(rdna_data)}"
    rdna_names = {
        str(op.get("mnemonic") or op.get("name") or "").upper() for op in rdna_data
    }
    for req in ["V_ADD_F32", "V_FMAC_F32", "V_MUL_F32"]:
        assert req in rdna_names, f"Mandatory RDNA instruction missing: {req}"


def test_golden_datasets_zero_leak_tokens() -> None:
    """Verify zero unparsed regex or scraper leak tokens in any exhaustive or canonical dataset."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    fw_dir = os.path.join(repo_root, "src", "ml_ecosystem_snapshots", "frameworks")

    leak_tokens = [
        "???0",
        "???1",
        "SNOWFLAKE",
        "<placeholder>",
        "UNKNOWN_OPERAND",
        "AttributeOperand",
        "DescOperand",
    ]

    for fname in [
        "stablehlo_exhaustive.json",
        "mlir_exhaustive.json",
        "nvidia_sass_exhaustive.json",
        "nvidia_ptx_exhaustive.json",
        "amd_rdna_exhaustive.json",
    ]:
        records = _load_dataset_or_canonical(fw_dir, fname)
        assert len(records) > 0, f"No records loaded for {fname}"
        content = json.dumps(records)
        for token in leak_tokens:
            assert token not in content, f"Leak token '{token}' discovered in {fname}"
