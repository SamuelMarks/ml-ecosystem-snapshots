"""Tests for WebGL 2.0 / GLSL ES 3.00 compute shader snapshot extractor, MCP server, and CLI."""

import argparse
import pytest

from ml_ecosystem_snapshots.frameworks.webgl import (
    CANONICAL_WEBGL_OPS,
    _load_webgl_ops,
    collect_api,
)
from ml_ecosystem_snapshots.mcp_server import check_webgl_op
from ml_ecosystem_snapshots.cli import cmd_check_webgl
from ml_switcheroo_ir.schema.ghost import (
    GhostOperationRef,
    SemanticTier,
)


def test_collect_api_webgl_categories() -> None:
    """Verify collect_api populates GhostRefs for valid tiers and rejects invalid ones."""
    refs_array = collect_api(SemanticTier.ARRAY_API)
    assert len(refs_array) > 0

    refs_neural = collect_api(SemanticTier.NEURAL)
    assert len(refs_neural) > 0

    refs_util = collect_api(SemanticTier.UTIL)
    assert len(refs_util) > 0

    refs_invalid = collect_api(SemanticTier.ACTIVATION)
    assert refs_invalid == []


def test_webgl_op_properties_and_roles() -> None:
    """Verify that extracted WebGL operations have valid operands, attributes, and results."""
    refs = collect_api(SemanticTier.ARRAY_API)
    op_map = {ref.name: ref for ref in refs if isinstance(ref, GhostOperationRef)}

    assert "texelFetch" in op_map
    fetch_ref = op_map["texelFetch"]
    assert fetch_ref.api_path == "webgl.texelFetch"
    assert fetch_ref.operands is not None and len(fetch_ref.operands) == 3
    assert fetch_ref.returns is not None and len(fetch_ref.returns) == 1
    assert fetch_ref.returns_type == "vec4"

    assert "fma" in op_map
    fma_ref = op_map["fma"]
    assert fma_ref.operands is not None and len(fma_ref.operands) == 3


def test_check_webgl_op_validation() -> None:
    """Test check_webgl_op validating existence, operands count, and attributes."""
    # 1. Valid operation
    res_valid = check_webgl_op("texelFetch", inputs_count=3)
    assert res_valid["is_valid"] is True
    assert res_valid["op_exists"] is True
    assert len(res_valid["errors"]) == 0

    # 2. Prefix stripping (webgl.texelFetch)
    res_prefix = check_webgl_op("webgl.texelFetch", inputs_count=3)
    assert res_prefix["is_valid"] is True

    # 3. Operands count mismatch
    res_count_err = check_webgl_op("texelFetch", inputs_count=1)
    assert res_count_err["is_valid"] is False
    assert any("expects 3 inputs, but got 1" in e for e in res_count_err["errors"])

    # 4. Valid attributes
    res_attr = check_webgl_op("texelFetch", attributes=["sampler_type"])
    assert res_attr["is_valid"] is True

    # 5. Invalid attribute
    res_bad_attr = check_webgl_op("texelFetch", attributes=["bad_attr"])
    assert res_bad_attr["is_valid"] is False
    assert any("Attribute 'bad_attr' is not valid" in e for e in res_bad_attr["errors"])

    # 6. Non-existent operator
    res_nonexistent = check_webgl_op("nonexistent_glsl_op")
    assert res_nonexistent["is_valid"] is False
    assert res_nonexistent["op_exists"] is False
    assert any("Unknown WebGL operation" in e for e in res_nonexistent["errors"])


def test_cmd_check_webgl_cli(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI command check-webgl for valid and invalid queries."""
    # 1. Valid invocation
    args_valid = argparse.Namespace(
        op_name="matrixCompMult",
        inputs_count=2,
        attributes=None,
    )
    cmd_check_webgl(args_valid)
    out_valid = capsys.readouterr().out
    assert "WebGL Operation 'matrixCompMult' is valid." in out_valid

    # 2. Attributes filtering
    args_attrs = argparse.Namespace(
        op_name="texture",
        inputs_count=None,
        attributes="bias",
    )
    cmd_check_webgl(args_attrs)
    out_attrs = capsys.readouterr().out
    assert "WebGL Operation 'texture' is valid." in out_attrs

    # 3. Invalid operation exits 1
    args_invalid = argparse.Namespace(
        op_name="nonexistent_glsl_op",
        inputs_count=None,
        attributes=None,
    )
    with pytest.raises(SystemExit) as exc_info:
        cmd_check_webgl(args_invalid)
    assert exc_info.value.code == 1
    out_invalid = capsys.readouterr().out
    assert "WebGL Operation 'nonexistent_glsl_op' Invalid:" in out_invalid


def test_webgl_roundtrip_serialization() -> None:
    """Verify roundtrip serialization through GhostOperationRef and Pydantic models."""
    refs = collect_api(SemanticTier.ARRAY_API)
    assert len(refs) > 0
    for ref in refs:
        assert isinstance(ref, GhostOperationRef)
        serialized = ref.model_dump()
        reconstructed = GhostOperationRef.model_validate(serialized)
        assert reconstructed.name == ref.name
        assert reconstructed.api_path == ref.api_path
        assert len(reconstructed.operands or []) == len(ref.operands or [])
        assert len(reconstructed.params or []) == len(ref.params or [])


def test_load_webgl_ops_catalog() -> None:
    """Verify _load_webgl_ops catalog returns standard canonical definitions."""
    ops = _load_webgl_ops()
    assert ops == CANONICAL_WEBGL_OPS
