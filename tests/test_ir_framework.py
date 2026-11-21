"""Unit tests for ML-Switcheroo IR API snapshot extractor and framework integration."""

from __future__ import annotations

import inspect
import typing

import ml_switcheroo_ir as ir
import pytest
from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier, SnapshotEnvelope
from ml_switcheroo_ir.typeclasses.outcome import raise_on_err

from ml_ecosystem_snapshots.api import (
    extract_snapshot,
    get_pkg_version,
    validate_snapshot_envelope,
)
from ml_ecosystem_snapshots.frameworks.ir import (
    IR_EXPORTED_SYMBOLS,
    collect_api,
    extract_ir_summary_doc,
    inspect_ir_symbol,
)
from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    get_member_dict,
    get_runtime_signature,
)


def test_collect_api_categories() -> None:
    """Test collect_api returns all expected IR v0.0.8 symbols in strict 1-to-1 parity."""
    ir_members = get_member_dict(ir)
    raw_all = ir_members.get("__all__")
    assert isinstance(raw_all, list)
    expected_public = {str(sym) for sym in raw_all if not str(sym).startswith("_")}
    assert set(IR_EXPORTED_SYMBOLS) == expected_public
    assert len(IR_EXPORTED_SYMBOLS) == 153

    for tier in (SemanticTier.ARRAY_API, SemanticTier.NEURAL, SemanticTier.UTIL):
        refs: list[GhostRef] = collect_api(tier)
        assert len(refs) == 153
        names = {ref.name for ref in refs}
        assert names == expected_public

        for ref in refs:
            assert ref.api_path == f"ml_switcheroo_ir.{ref.name}"
            assert ref.docstring is not None
            assert len(ref.docstring) > 0


def test_collect_api_unsupported_category() -> None:
    """Test collect_api returns an empty list for categories without IR mappings."""
    refs = collect_api(SemanticTier.OPTIMIZER)
    assert refs == []


def test_ir_graph_and_node_signatures() -> None:
    """Test LogicalGraph, IRGraph, LogicalNode, and IRNode parameter parity."""
    refs = collect_api(SemanticTier.ARRAY_API)
    ref_by_name: dict[str, GhostRef] = {ref.name: ref for ref in refs}

    lg_ref = ref_by_name["LogicalGraph"]
    ir_graph_ref = ref_by_name["IRGraph"]
    assert lg_ref.kind == "class"
    assert ir_graph_ref.kind == "class"
    assert [p.name for p in lg_ref.params] == [p.name for p in ir_graph_ref.params]
    lg_params = {p.name: p for p in lg_ref.params}
    for expected_field in ("name", "nodes", "outputs", "mesh", "annotations", "metadata", "sorted_nodes"):
        assert expected_field in lg_params

    ln_ref = ref_by_name["LogicalNode"]
    ir_node_ref = ref_by_name["IRNode"]
    assert ln_ref.kind == "class"
    assert ir_node_ref.kind == "class"
    assert [p.name for p in ln_ref.params] == [p.name for p in ir_node_ref.params]
    ln_params = {p.name: p for p in ln_ref.params}
    for expected_field in ("id", "op_type", "attributes", "sharding", "subgraphs", "device", "stream"):
        assert expected_field in ln_params


def test_ir_symbol_signatures_and_annotations() -> None:
    """Test parameter signatures and annotations for extracted IR symbols."""
    refs = collect_api(SemanticTier.ARRAY_API)
    ref_by_name: dict[str, GhostRef] = {ref.name: ref for ref in refs}

    shape_ref = ref_by_name["TensorShape"]
    assert shape_ref.kind == "class"
    assert "dims" in {p.name for p in shape_ref.params}

    ts_ref = ref_by_name["TensorSpec"]
    assert ts_ref.kind == "class"
    ts_params = {p.name: p for p in ts_ref.params}
    assert "shape" in ts_params
    assert "dtype" in ts_params
    assert ts_params["dtype"].annotation is not None
    assert "DTypeLikeProtocol" in ts_params["dtype"].annotation
    assert "sparsity" in ts_params
    assert ts_params["sparsity"].default == "None"

    ps_ref = ref_by_name["PartitionSpec"]
    assert ps_ref.kind == "class"
    assert "axes" in {p.name for p in ps_ref.params}

    zt_ref = ref_by_name["ZeroTangent"]
    nt_ref = ref_by_name["NoTangent"]
    assert zt_ref.kind == "class"
    assert nt_ref.kind == "class"
    assert "id" in {p.name for p in zt_ref.params}
    assert "id" in {p.name for p in nt_ref.params}

    edge_ref = ref_by_name["LogicalEdge"]
    assert edge_ref.kind == "class"
    edge_params = {p.name: p for p in edge_ref.params}
    assert edge_params["source_idx"].default == "0"
    assert edge_params["source_idx"].annotation == "int | None"
    assert edge_params["target_idx"].default == "0"
    assert edge_params["value_name"].default == "None"

    topo_ref = ref_by_name["topological_sort"]
    assert topo_ref.kind == "function"
    assert topo_ref.returns_type == "list[LogicalNode]"
    topo_params = {p.name: p for p in topo_ref.params}
    assert "graph" in topo_params
    assert topo_params["strict"].default == "False"

    # Verify v0.0.8 Haskell-style typeclasses, identifiers, and proof witnesses
    for cls_sym in ("Ok", "Err", "Valid", "Invalid", "Eq", "Ord", "Semigroup", "Monoid", "Codec", "ProvenGraph"):
        assert ref_by_name[cls_sym].kind == "class"
        assert len(ref_by_name[cls_sym].params) >= 1

    for alias_or_const in ("Result", "Validation", "NodeId", "NonBoolInt", "ALL_DIALECT_CATALOG_WITNESSES"):
        assert ref_by_name[alias_or_const].kind == "class"
        assert ref_by_name[alias_or_const].params == []

    for fn_or_pass in ("prove_acyclic_sorted", "prove_schema_valid", "graph_from_json_result", "CSE_PASS"):
        assert ref_by_name[fn_or_pass].kind == "function"
        assert len(ref_by_name[fn_or_pass].params) >= 1


def test_extract_summary_doc_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _extract_summary_doc fallback when inspect.getdoc returns None.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setattr(inspect, "getdoc", lambda _target: None)
    assert extract_ir_summary_doc("Dummy", ir.LogicalGraph, "IR class") == "IR class Dummy."
    unannotated_ref = inspect_ir_symbol("unannotated_fn", lambda alpha: alpha)
    assert unannotated_ref.kind == "function"
    assert unannotated_ref.returns_type is None
    assert [p.name for p in unannotated_ref.params] == ["alpha"]


def test_ir_v008_aliases_and_structural_protocols() -> None:
    """Verify IRGraph, IRNode, and all structural protocols resolve and validate in the v0.0.8 IR snapshot."""
    protocol_names = (
        "DTypeLikeProtocol",
        "DeviceLikeProtocol",
        "SymbolicNodeProtocol",
        "SymbolicDimProtocol",
        "SymbolicIntWrapperProtocol",
        "CustomRuleHandleProtocol",
        "LoopOptionsProtocol",
        "MeshLikeProtocol",
        "ArrayBufferProtocol",
        "SubgraphLikeProtocol",
    )
    ir_members = get_member_dict(ir)
    assert ir_members["IRGraph"] is ir.LogicalGraph
    assert ir_members["IRNode"] is ir.LogicalNode

    refs_by_name = {ref.name: ref for ref in collect_api(SemanticTier.NEURAL)}
    for proto_name in protocol_names:
        assert proto_name in ir_members
        ref = refs_by_name[proto_name]
        assert ref.kind == "class"
        assert ref.api_path == f"ml_switcheroo_ir.{proto_name}"
        assert ref.docstring is not None and len(ref.docstring) > 0

    snap = raise_on_err(extract_snapshot("ir")).to_wire_dict()
    assert snap.get("version") == "0.0.8"
    env = raise_on_err(validate_snapshot_envelope(snap))
    assert isinstance(env, SnapshotEnvelope)
    assert env.version == "0.0.8"
    neural_names = {entry["name"] if isinstance(entry, dict) else entry.name for entry in env.categories["neural"]}
    for sym in ("IRGraph", "IRNode", "ProvenGraph", "Result", "Validation", *protocol_names):
        assert sym in neural_names


def test_ir_live_signature_reflection_parity() -> None:
    """Verify every callable GhostRef in frameworks/ir.py matches live ml_switcheroo_ir signatures."""
    refs = collect_api(SemanticTier.NEURAL)
    ir_members = get_member_dict(ir)
    non_sig = {"DType", "DiagnosticSeverity", "Ordering", "CyclicGraphError"}
    for ref in refs:
        assert ref.name in ir_members, f"Missing symbol {ref.name} in ml_switcheroo_ir"
        obj = ir_members[ref.name]
        if (
            ref.name in non_sig
            or typing.get_origin(obj) is not None
            or "__supertype__" in get_member_dict(obj)
            or not callable(obj)
        ):
            assert ref.params == []
            continue
        sig = get_runtime_signature(obj)
        live_params = list(sig.parameters.values())
        assert [p.name for p in ref.params] == [p.name for p in live_params], f"Parameter name mismatch for {ref.name}"
        for idx, ghost_p in enumerate(ref.params):
            live_p = live_params[idx]
            assert ghost_p.kind.value == live_p.kind.name, f"Parameter kind mismatch on {ref.name}.{ghost_p.name}"
            has_live_default = live_p.default is not inspect.Parameter.empty
            has_ghost_default = ghost_p.default is not None
            assert has_ghost_default == has_live_default, f"Default presence mismatch on {ref.name}.{ghost_p.name}"


def test_extract_snapshot_ir_envelope() -> None:
    """Test end-to-end extraction and validation of the IR snapshot envelope."""
    snap = raise_on_err(extract_snapshot("ir")).to_wire_dict()
    assert snap.get("version") == "0.0.8"
    assert "categories" in snap
    categories = snap["categories"]
    assert isinstance(categories, dict)
    assert "neural" in categories
    assert "array" in categories
    assert "util" in categories

    env = raise_on_err(validate_snapshot_envelope(snap))
    assert isinstance(env, SnapshotEnvelope)


def test_get_pkg_version_ir() -> None:
    """Test get_pkg_version resolving 'ir' and 'ml_switcheroo_ir' versions."""
    ver_ir = get_pkg_version("ir")
    assert ver_ir == "0.0.8"

    ver_sw = get_pkg_version("ml_switcheroo_ir")
    assert ver_sw == ver_ir


def test_ir_version_resolution_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_pkg_version('ir') fallback when module import or metadata fails.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import importlib.metadata
    import sys

    monkeypatch.setitem(sys.modules, "ml_switcheroo_ir", None)
    ver = get_pkg_version("ir")
    assert ver != "unknown"

    def raise_meta_err(_dist: str) -> str:
        """Raise RuntimeError for metadata lookup.

        Args:
            _dist: Distribution name.

        Returns:
            Never returns.

        Raises:
            RuntimeError: Simulated metadata failure.
        """
        raise RuntimeError("metadata missing")

    monkeypatch.setattr(importlib.metadata, "version", raise_meta_err)
    assert get_pkg_version("ir") == "0.0.8"
