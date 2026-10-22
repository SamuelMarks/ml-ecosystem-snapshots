"""Unit tests for ML-Switcheroo IR API snapshot extractor and framework integration."""

from __future__ import annotations

import inspect
import pytest

import ml_switcheroo_ir as ir
from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier, SnapshotEnvelope
from ml_ecosystem_snapshots.api import (
    extract_snapshot,
    get_pkg_version,
    validate_snapshot_envelope,
)
from ml_ecosystem_snapshots.frameworks.ir import collect_api


def test_collect_api_categories() -> None:
    """Test collect_api returns all expected IR symbols for supported semantic tiers."""
    for tier in (SemanticTier.ARRAY_API, SemanticTier.NEURAL, SemanticTier.UTIL):
        refs: list[GhostRef] = collect_api(tier)
        assert len(refs) == 47

        names = {ref.name for ref in refs}
        expected_symbols = {
            "LogicalGraph",
            "LogicalNode",
            "TensorSpec",
            "TensorShape",
            "LogicalEdge",
            "LogicalMesh",
            "LogicalAxis",
            "PartitionSpec",
            "NodeDict",
            "DType",
            "EdgeList",
            "ZeroTangent",
            "NoTangent",
            "SymInt",
            "SymVar",
            "SymConst",
            "SymBinaryOp",
            "SymUnaryOp",
            "SymPiecewise",
            "WebRTCSignalingTopology",
            "WebRTCPeerConfig",
            "PipelineTopologyConfig",
            "MicrobatchSplittingConfig",
            "MeshMappingConfig",
            "StageCommunicationConfig",
            "DependencyConfig",
            "PipelineScheduleConfig",
            "SchedulePhaseConfig",
            "SymbolicSolver",
            "SymbolicConstraintTracker",
            "ShapeTracker",
            "ParameterTranslationEngine",
            "eliminate_dead_nodes",
            "eliminate_common_subexpressions",
            "propagate_shapes_and_constants",
            "estimate_communication_volume",
            "estimate_graph_communication_volume",
            "topological_sort",
            "export_to_python",
            "export_schemas",
            "get_json_schema",
            "generate_typescript_definitions",
            "get_abstract_op",
            "broadcast_shapes",
            "broadcast_dimension",
            "matmul_shape",
            "normalize_axis",
        }
        assert names == expected_symbols


def test_collect_api_unsupported_category() -> None:
    """Test collect_api returns an empty list for categories without IR mappings."""
    refs = collect_api(SemanticTier.OPTIMIZER)
    assert refs == []


def test_ir_symbol_signatures_and_annotations() -> None:
    """Test parameter signatures and annotations for extracted IR symbols."""
    refs = collect_api(SemanticTier.ARRAY_API)
    ref_by_name: dict[str, GhostRef] = {ref.name: ref for ref in refs}

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
    assert edge_params["source_idx"].annotation == "int | None"
    assert edge_params["target_idx"].default == "0"
    assert edge_params["value_name"].default == "None"

    # Verify topological_sort
    topo_ref = ref_by_name["topological_sort"]
    assert topo_ref.kind == "function"
    assert topo_ref.returns_type == "list[LogicalNode]"
    topo_params = {p.name: p for p in topo_ref.params}
    assert topo_params["strict"].default == "False"


def test_ir_live_signature_reflection_parity() -> None:
    """Verify every extracted GhostRef in frameworks/ir.py matches live ml_switcheroo_ir signatures."""
    refs = collect_api(SemanticTier.NEURAL)
    for ref in refs:
        assert hasattr(ir, ref.name), f"Missing symbol {ref.name} in ml_switcheroo_ir"
        obj = getattr(ir, ref.name)
        if ref.name == "DType":
            continue
        sig = inspect.signature(obj)
        live_params = list(sig.parameters.values())
        assert [p.name for p in ref.params] == [
            p.name for p in live_params
        ], f"Parameter name mismatch for {ref.name}"
        for ghost_p, live_p in zip(ref.params, live_params):
            assert (
                ghost_p.kind.value == live_p.kind.name
            ), f"Parameter kind mismatch on {ref.name}.{ghost_p.name}"
            has_live_default = live_p.default is not inspect.Parameter.empty
            has_ghost_default = ghost_p.default is not None
            assert (
                has_ghost_default == has_live_default
            ), f"Default presence mismatch on {ref.name}.{ghost_p.name}"


def test_extract_snapshot_ir_envelope() -> None:
    """Test end-to-end extraction and validation of the IR snapshot envelope."""
    snap = extract_snapshot("ir")
    assert snap.get("version") is not None
    assert snap.get("version") != "unknown"
    assert "categories" in snap
    categories = snap["categories"]
    assert isinstance(categories, dict)
    assert "neural" in categories
    assert "array" in categories
    assert "util" in categories

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


def test_get_pkg_version_ir_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_pkg_version fallback when importing ml_switcheroo_ir raises Exception.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import builtins

    real_import = getattr(builtins, "__import__")

    def mock_import(name: str, *args: object, **kwargs: object) -> object:
        """Mock import function.

        Args:
            name: The module name.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The imported module or raises.
        """
        if name == "ml_switcheroo_ir":
            raise ImportError("simulated error")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    ver = get_pkg_version("ir")
    assert ver != "unknown"

    import importlib.metadata

    def raise_meta_err(_dist: str) -> str:
        """Raise PackageNotFoundError for metadata lookup.

        Args:
            _dist: Distribution name.

        Returns:
            Never returns.

        Raises:
            RuntimeError: Simulated metadata failure.
        """
        raise RuntimeError("metadata missing")

    monkeypatch.setattr(importlib.metadata, "version", raise_meta_err)
    assert get_pkg_version("ir") != "unknown"
