"""Unit tests for WebGPU WGSL dialect extractor, compliance checker, and CLI."""

import argparse
import os
import tempfile
from typing import Any, Dict, List
import pytest

from ml_switcheroo_ir.schema.ghost import (
    GhostRef,
    IRParameterRole,
    OperandDirection,
    SemanticTier,
    SnapshotEnvelope,
)
from ml_framework_snapshots.api import extract_snapshot, validate_snapshot_envelope
from ml_framework_snapshots.cli import cmd_check_wgsl
from ml_framework_snapshots.compliance import check_wgsl_shader_compliance
from ml_framework_snapshots.frameworks.wgsl import collect_api
from ml_framework_snapshots.mcp_server import check_wgsl_op


def test_collect_api_wgsl_categories() -> None:
    """Test collect_api returns WGSL operations for supported tiers and empty for others."""
    for tier in (SemanticTier.ARRAY_API, SemanticTier.NEURAL, SemanticTier.UTIL):
        refs: List[GhostRef] = collect_api(tier)
        assert len(refs) == 7
        names = {r.name for r in refs}
        assert "storageStore" in names
        assert "storageLoad" in names
        assert "workgroupBarrier" in names
        assert "storageBarrier" in names
        assert "atomicAdd" in names
        assert "atomicSub" in names
        assert "atomicExchange" in names

    assert collect_api(SemanticTier.OPTIMIZER) == []


def test_wgsl_op_properties_and_roles() -> None:
    """Test operand directions, parameter roles, and attributes of WGSL operations."""
    from ml_framework_snapshots.models import ExtendedGhostParam

    refs = collect_api(SemanticTier.ARRAY_API)
    by_name: Dict[str, GhostRef] = {r.name: r for r in refs}

    # storageStore: value operand has WRITE direction
    store_ref = by_name["storageStore"]
    assert store_ref.kind == "function"
    store_params: Dict[str, ExtendedGhostParam] = {
        p.name: p for p in store_ref.params if isinstance(p, ExtendedGhostParam)
    }
    assert store_params["buffer"].direction == OperandDirection.READ
    assert store_params["index"].direction == OperandDirection.READ
    assert store_params["value"].direction == OperandDirection.WRITE
    assert store_params["address_space"].role == IRParameterRole.ATTRIBUTE

    # atomicAdd: atomic_ptr and value
    add_ref = by_name["atomicAdd"]
    add_params: Dict[str, ExtendedGhostParam] = {
        p.name: p for p in add_ref.params if isinstance(p, ExtendedGhostParam)
    }
    assert add_params["atomic_ptr"].direction == OperandDirection.READ
    assert add_params["value"].direction == OperandDirection.READ


def test_extract_snapshot_wgsl_envelope() -> None:
    """Test end-to-end WGSL snapshot extraction and envelope validation."""
    snap: Dict[str, Any] = extract_snapshot("wgsl")
    assert snap.get("version") == "draft-2024"
    assert "categories" in snap
    assert "neural" in snap["categories"]

    env = validate_snapshot_envelope(snap)
    assert isinstance(env, SnapshotEnvelope)


def test_check_wgsl_op_validation() -> None:
    """Test check_wgsl_op for valid and invalid inputs, counts, and attributes."""
    # Valid operation
    res_valid = check_wgsl_op(
        "storageStore", inputs_count=3, attributes=["address_space"]
    )
    assert res_valid["is_valid"] is True
    assert res_valid["op_exists"] is True

    # Bad inputs count
    res_bad_count = check_wgsl_op("storageStore", inputs_count=1)
    assert res_bad_count["is_valid"] is False
    assert any("expects 3 inputs" in err for err in res_bad_count["errors"])

    # Invalid attribute
    res_bad_attr = check_wgsl_op("storageStore", attributes=["invalid_attr"])
    assert res_bad_attr["is_valid"] is False
    assert any(
        "Attribute 'invalid_attr' is not valid" in err for err in res_bad_attr["errors"]
    )

    # Unknown operation
    res_unknown = check_wgsl_op("nonexistent_op")
    assert res_unknown["is_valid"] is False
    assert res_unknown["op_exists"] is False


def test_check_wgsl_shader_compliance() -> None:
    """Test check_wgsl_shader_compliance parsing shader text and identifying WGSL operations."""
    shader_code = """
    // Compute shader snippet
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let v = vec3(1, 2, 3);
        let val = storageLoad(buf, id.x);
        workgroupBarrier();
        storageStore(out_buf, id.x, val);
    }
    """
    res = check_wgsl_shader_compliance(shader_code)
    assert res["is_compliant"] is True
    assert res["verified_ops"] >= 3
    assert res["errors"] == []

    from unittest.mock import patch

    with patch(
        "ml_framework_snapshots.mcp_server.check_wgsl_op",
        return_value={"op_exists": True, "is_valid": False, "errors": ["Mock error"]},
    ):
        res_err = check_wgsl_shader_compliance("storageStore(1);")
        assert res_err["is_compliant"] is False
        assert "Mock error" in res_err["errors"]


def test_cmd_check_wgsl_cli(capsys: pytest.CaptureFixture[str]) -> None:
    """Test cmd_check_wgsl CLI routing for valid, invalid, and file-based arguments."""
    # 1. Valid operation CLI
    args_valid = argparse.Namespace(
        op_name="atomicAdd",
        inputs_count=2,
        attributes=None,
        file=None,
    )
    cmd_check_wgsl(args_valid)
    out_valid = capsys.readouterr().out
    assert "WGSL Operation 'atomicAdd' is valid." in out_valid

    # 2. Invalid operation CLI exits with 1
    args_invalid = argparse.Namespace(
        op_name="nonexistent_op",
        inputs_count=None,
        attributes=None,
        file=None,
    )
    with pytest.raises(SystemExit):
        cmd_check_wgsl(args_invalid)

    # 3. Missing op_name and file exits with 1
    args_empty = argparse.Namespace(
        op_name=None,
        inputs_count=None,
        attributes=None,
        file=None,
    )
    with pytest.raises(SystemExit):
        cmd_check_wgsl(args_empty)

    # 4. File-based verification
    with tempfile.NamedTemporaryFile("w", suffix=".wgsl", delete=False) as f:
        f.write("fn foo() { storageBarrier(); }")
        temp_path = f.name

    try:
        args_file = argparse.Namespace(
            op_name=None,
            inputs_count=None,
            attributes=None,
            file=temp_path,
        )
        cmd_check_wgsl(args_file)
        out_file = capsys.readouterr().out
        assert "WGSL shader verified compliant" in out_file
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # 5. Nonexistent file exits with 1
    args_bad_file = argparse.Namespace(
        op_name=None,
        inputs_count=None,
        attributes=None,
        file="/tmp/nonexistent_wgsl_file.wgsl",
    )
    with pytest.raises(SystemExit):
        cmd_check_wgsl(args_bad_file)

    # 6. Non-compliant file exits with 1
    with tempfile.NamedTemporaryFile("w", suffix=".wgsl", delete=False) as f_bad:
        f_bad.write("fn test() { storageStore(); }")
        bad_path = f_bad.name

    try:
        from unittest.mock import patch

        with patch(
            "ml_framework_snapshots.compliance.check_wgsl_shader_compliance",
            return_value={"is_compliant": False, "errors": ["Mock error"]},
        ):
            args_bad_content = argparse.Namespace(
                op_name=None,
                inputs_count=None,
                attributes=None,
                file=bad_path,
            )
            with pytest.raises(SystemExit):
                cmd_check_wgsl(args_bad_content)
    finally:
        if os.path.exists(bad_path):
            os.remove(bad_path)


def test_load_wgsl_ops_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _load_wgsl_ops returns CANONICAL_WGSL_OPS when schema resources are unavailable or empty."""
    from unittest.mock import MagicMock
    from ml_framework_snapshots.frameworks.wgsl import (
        _load_wgsl_ops,
        CANONICAL_WGSL_OPS,
    )

    # Case 1: Exception on open
    mock_file = MagicMock()
    mock_file.open.side_effect = IOError("Mock open error")
    mock_schema_res = MagicMock()
    mock_schema_res.joinpath.return_value = mock_file
    monkeypatch.setattr("importlib.resources.files", lambda _: mock_schema_res)
    ops = _load_wgsl_ops()
    assert ops == CANONICAL_WGSL_OPS

    # Case 2: Empty ops in json
    import io

    mock_file_empty = MagicMock()
    mock_file_empty.open.return_value = io.StringIO('{"ops": []}')
    mock_schema_res_empty = MagicMock()
    mock_schema_res_empty.joinpath.return_value = mock_file_empty
    monkeypatch.setattr("importlib.resources.files", lambda _: mock_schema_res_empty)
    ops_empty = _load_wgsl_ops()
    assert ops_empty == CANONICAL_WGSL_OPS

    # Case 3: Outer import failure
    def mock_files_raise(_: Any) -> Any:
        """Mock failure in files resource resolution."""
        raise RuntimeError("Outer failure")

    monkeypatch.setattr("importlib.resources.files", mock_files_raise)
    ops_err = _load_wgsl_ops()
    assert ops_err == CANONICAL_WGSL_OPS


def test_wgsl_roundtrip_serialization() -> None:
    """Verify 100% roundtrip serialization through GhostOperationRef and Pydantic models."""
    from ml_framework_snapshots.frameworks.wgsl import collect_api
    from ml_switcheroo_ir.schema.ghost import GhostOperationRef, SemanticTier

    refs = collect_api(SemanticTier.ARRAY_API)
    assert len(refs) > 0
    for ref in refs:
        assert isinstance(ref, GhostOperationRef)
        serialized = ref.model_dump()
        reconstructed = GhostOperationRef.model_validate(serialized)
        assert reconstructed.name == ref.name
        assert reconstructed.api_path == ref.api_path
        assert reconstructed.operands is not None
        assert ref.operands is not None
        assert len(reconstructed.operands) == len(ref.operands)
        assert reconstructed.params is not None
        assert ref.params is not None
        assert len(reconstructed.params) == len(ref.params)
