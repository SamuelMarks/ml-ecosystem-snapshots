"""Unit tests for ML-Switcheroo IR API snapshot extractor and framework integration."""

from typing import Any, Dict, List

from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier, SnapshotEnvelope
from ml_framework_snapshots.api import (
    extract_snapshot,
    get_pkg_version,
    validate_snapshot_envelope,
)
from ml_framework_snapshots.frameworks.ir import collect_api


def test_collect_api_categories() -> None:
    """Test collect_api returns all expected IR symbols for supported semantic tiers."""
    for tier in (SemanticTier.ARRAY_API, SemanticTier.NEURAL, SemanticTier.UTIL):
        refs: List[GhostRef] = collect_api(tier)
        assert len(refs) == 15

        names = {ref.name for ref in refs}
        assert "LogicalGraph" in names
        assert "LogicalNode" in names
        assert "TensorSpec" in names
        assert "TensorShape" in names
        assert "LogicalEdge" in names
        assert "LogicalMesh" in names
        assert "LogicalAxis" in names
        assert "PartitionSpec" in names
        assert "NodeDict" in names
        assert "DType" in names
        assert "eliminate_dead_nodes" in names
        assert "eliminate_common_subexpressions" in names
        assert "propagate_shapes_and_constants" in names
        assert "estimate_graph_communication_volume" in names
        assert "topological_sort" in names


def test_collect_api_unsupported_category() -> None:
    """Test collect_api returns an empty list for categories without IR mappings."""
    refs = collect_api(SemanticTier.OPTIMIZER)
    assert refs == []


def test_ir_symbol_signatures_and_annotations() -> None:
    """Test parameter signatures and annotations for extracted IR symbols."""
    refs = collect_api(SemanticTier.ARRAY_API)
    ref_by_name: Dict[str, GhostRef] = {ref.name: ref for ref in refs}

    # Verify TensorSpec
    ts_ref = ref_by_name["TensorSpec"]
    assert ts_ref.kind == "class"
    ts_params = {p.name: p for p in ts_ref.params}
    assert "shape" in ts_params
    assert "dtype" in ts_params
    assert "sparsity" in ts_params
    assert ts_params["sparsity"].default == "None"

    # Verify LogicalEdge
    edge_ref = ref_by_name["LogicalEdge"]
    assert edge_ref.kind == "class"
    edge_params = {p.name: p for p in edge_ref.params}
    assert edge_params["source_idx"].default == "0"
    assert edge_params["target_idx"].default == "0"
    assert edge_params["value_name"].default == "None"

    # Verify topological_sort
    topo_ref = ref_by_name["topological_sort"]
    assert topo_ref.kind == "function"
    assert topo_ref.returns_type == "list[LogicalNode]"
    topo_params = {p.name: p for p in topo_ref.params}
    assert topo_params["strict"].default == "False"


def test_extract_snapshot_ir_envelope() -> None:
    """Test end-to-end extraction and validation of the IR snapshot envelope."""
    snap: Dict[str, Any] = extract_snapshot("ir")
    assert snap.get("version") is not None
    assert snap.get("version") != "unknown"
    assert "categories" in snap
    assert "neural" in snap["categories"]
    assert "array" in snap["categories"]
    assert "util" in snap["categories"]

    # Validate snapshot schema compliance
    env = validate_snapshot_envelope(snap)
    assert isinstance(env, SnapshotEnvelope)


def test_get_pkg_version_ir() -> None:
    """Test get_pkg_version resolving 'ir' and 'ml_switcheroo_ir' versions."""
    ver_ir = get_pkg_version("ir")
    assert ver_ir != "unknown"
    assert ver_ir.count(".") >= 1

    ver_sw = get_pkg_version("ml_switcheroo_ir")
    assert ver_sw == ver_ir


def test_get_pkg_version_ir_fallback(monkeypatch: Any) -> None:
    """Test get_pkg_version fallback when importing ml_switcheroo_ir raises Exception."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        """Simulate missing ml_switcheroo_ir."""
        if name == "ml_switcheroo_ir":
            raise ImportError("simulated error")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    ver = get_pkg_version("ir")
    assert ver != "unknown"
