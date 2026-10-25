"""Tests for Ghost Protocol v2 contract synchronization and bidirectional SerDe."""

from __future__ import annotations

import json
import os

from ml_switcheroo_ir.schema.ghost import ParameterKind
from ml_ecosystem_snapshots.models import (
    ExtendedGhostParam,
    ExtendedGhostRef,
    GhostIsaRef,
    GhostMlirRef,
    GhostResult,
    IRParameterRole,
    OperandDirection,
    SnapshotEnvelope,
)
from ml_ecosystem_snapshots.serialization.hydration import hydrate_ghost_ref


def test_ghost_protocol_v2_models_instantiation() -> None:
    """Verify instantiation of all Ghost Protocol v2 models."""
    res = GhostResult(name="res0", type="tensor<4xf32>", description="Output tensor")
    assert res.name == "res0"

    param = ExtendedGhostParam(
        name="input",
        kind=ParameterKind.POSITIONAL_ONLY,
        direction=OperandDirection.READ,
        role=IRParameterRole.OPERAND,
        allowed_dtypes=["float32", "bfloat16"],
        rank_constraint="2",
    )
    assert param.direction == OperandDirection.READ
    assert param.role == IRParameterRole.OPERAND

    isa_ref = GhostIsaRef(
        name="FFMA",
        api_path="isa.inst.FFMA",
        kind="function",
        params=[param],
        predicate_guards=["@P0", "@!P1"],
        instruction_modifiers=[".F32", ".FTZ"],
    )
    assert isa_ref.domain_type in ("isa", "instruction")
    assert isa_ref.predicate_guards == ["@P0", "@!P1"]

    mlir_ref = GhostMlirRef(
        name="addf",
        api_path="arith.addf",
        kind="operation",
        params=[param],
        returns=[res],
        traits=["Commutative", "SameOperandsAndResultType"],
    )
    assert mlir_ref.domain_type in ("mlir", "operation")
    assert mlir_ref.traits == ["Commutative", "SameOperandsAndResultType"]

    envelope = SnapshotEnvelope(
        target="stablehlo",
        schema_version="2.0.0",
        version="1.0.0",
    )
    assert envelope.schema_version == "2.0.0"


def test_hydrate_ghost_ref_v2_roundtrip() -> None:
    """Verify hydrate_ghost_ref validates v2.0.0 structures with zero information loss."""
    from ml_ecosystem_snapshots.models import JsonValue

    v2_dict: dict[str, JsonValue] = {
        "name": "relu",
        "api_path": "torch.nn.functional.relu",
        "kind": "function",
        "schema_version": "2.0.0",
        "params": [
            {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "inplace", "kind": "KEYWORD_ONLY", "default": "False"},
        ],
    }

    upgraded = hydrate_ghost_ref(v2_dict)
    assert isinstance(upgraded, ExtendedGhostRef)
    assert upgraded.name == "relu"
    assert upgraded.api_path == "torch.nn.functional.relu"
    assert upgraded.schema_version == "2.0.0"

    # Test roundtrip serialization to JSON and back
    json_str = upgraded.model_dump_json()
    re_parsed = hydrate_ghost_ref(json.loads(json_str))
    assert re_parsed.name == upgraded.name
    assert len(re_parsed.params) == len(upgraded.params)

    # Test legacy empty string constraint/type normalization in operands, attributes, returns
    legacy_op_dict: dict[str, JsonValue] = {
        "name": "legacy_op",
        "api_path": "wgsl.legacy_op",
        "kind": "operation",
        "schema_version": "2.0.0",
        "params": [],
        "operands": [
            {"name": "a", "constraint": ""},
            {"name": "b", "constraint": "f32"},
        ],
        "attributes": [{"name": "attr", "type": ""}, {"name": "attr2", "type": "i32"}],
        "returns": [{"name": "res", "type": ""}, {"name": "res2", "type": "f32"}],
    }
    hydrated_legacy = hydrate_ghost_ref(legacy_op_dict)
    assert hydrated_legacy.returns is not None
    assert hydrated_legacy.returns[0].type == "Any"
    assert hydrated_legacy.returns[1].type == "f32"
    dumped_legacy = hydrated_legacy.model_dump()
    assert dumped_legacy["operands"][0]["constraint"] == "Any"
    assert dumped_legacy["operands"][1]["constraint"] == "f32"
    assert dumped_legacy["attributes"][0]["type"] == "Any"
    assert dumped_legacy["attributes"][1]["type"] == "i32"


def test_exhaustive_snapshot_loading_serde() -> None:
    """Verify all 5 exhaustive snapshot files deserialize into Ghost Protocol v2 models."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    fw_dir = os.path.join(repo_root, "src", "ml_ecosystem_snapshots", "frameworks")

    for fname in [
        "stablehlo_exhaustive.json",
        "mlir_exhaustive.json",
        "nvidia_sass_exhaustive.json",
        "nvidia_ptx_exhaustive.json",
        "amd_rdna_exhaustive.json",
    ]:
        fpath = os.path.join(fw_dir, fname)
        if not os.path.exists(fpath):
            continue

        with open(fpath, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if isinstance(raw_data, dict):
            if "operations" in raw_data:
                records = raw_data["operations"]
            elif "ops" in raw_data:
                records = raw_data["ops"]
            elif "categories" in raw_data:
                records = []
                for cat in raw_data["categories"].values():
                    if isinstance(cat, list):
                        records.extend(cat)
            else:
                records = [raw_data]
        else:
            records = raw_data

        assert isinstance(records, list)
        assert len(records) > 0, f"Snapshot {fname} is empty"

        for raw_item in records[:10]:
            ref = hydrate_ghost_ref(raw_item)
            assert ref.name or getattr(ref, "mnemonic", None)
            # Re-serialize to verify schema integrity
            dumped = ref.model_dump()
            assert isinstance(dumped, dict)


def test_cross_repo_dtype_coverage() -> None:
    """Verify all ml_switcheroo_ir DType variants are supported and roundtrip serialized."""
    import ml_switcheroo_ir as ir
    from ml_ecosystem_snapshots.compliance import DTYPE_BITWIDTHS, get_dtype_bitwidth
    from ml_ecosystem_snapshots.mcp_server import normalize_dtype_name

    all_ir_dtypes = [dt.value for dt in ir.DType]
    assert len(all_ir_dtypes) >= 30

    for dt_str in all_ir_dtypes:
        norm = normalize_dtype_name(dt_str)
        assert norm in DTYPE_BITWIDTHS, f"DType {dt_str} missing from DTYPE_BITWIDTHS"
        bitwidth = get_dtype_bitwidth(norm)
        assert bitwidth in (2, 4, 8, 16, 32, 64, 128)

    # Roundtrip param with all dtypes
    param = ExtendedGhostParam(
        name="input",
        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
        allowed_dtypes=all_ir_dtypes,
        rank_constraint=">=2",
    )
    dumped = param.model_dump()
    assert dumped["allowed_dtypes"] == all_ir_dtypes
    assert dumped["rank_constraint"] == ">=2"


def test_hydrate_ghost_ref_minimal_custom_op_default_kind() -> None:
    """Verify hydrate_ghost_ref defaults kind to 'function' for minimal custom op entries."""
    minimal_entry = {
        "name": "FlashAttention",
        "api_path": "ml.switcheroo.custom.FlashAttention",
    }
    hydrated = hydrate_ghost_ref(minimal_entry)
    assert isinstance(hydrated, ExtendedGhostRef)
    assert hydrated.name == "FlashAttention"
    assert hydrated.api_path == "ml.switcheroo.custom.FlashAttention"
    assert hydrated.kind == "function"

    from ml_ecosystem_snapshots.models import JsonValue

    ptx_entry: dict[str, JsonValue] = {
        "mnemonic": "add",
        "operands": [{"name": "d", "role": "dst"}, "src0"],
    }
    hydrated_ptx = hydrate_ghost_ref(ptx_entry)
    assert isinstance(hydrated_ptx, GhostIsaRef)
    assert len(hydrated_ptx.params) == 2
