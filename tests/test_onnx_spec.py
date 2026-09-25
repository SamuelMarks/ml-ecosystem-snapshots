"""Tests for ONNX operator specification extractor, MCP server verification, and CLI."""

import argparse
import sys
import types
from typing import Any
import pytest

from ml_ecosystem_snapshots.frameworks.onnx_spec import (
    CANONICAL_ONNX_OPS,
    _get_onnx_defs,
    _load_onnx_ops,
    collect_api,
)
from ml_ecosystem_snapshots.mcp_server import check_onnx_op
from ml_ecosystem_snapshots.cli import cmd_check_onnx
from ml_switcheroo_ir.schema.ghost import (
    GhostOperationRef,
    IRParameterRole,
    SemanticTier,
)


def test_collect_api_onnx_categories() -> None:
    """Verify collect_api populates GhostRefs across valid tiers and filters preview domains."""
    refs_array = collect_api(SemanticTier.ARRAY_API, include_nonpublic=False)
    assert len(refs_array) > 0

    refs_neural = collect_api(SemanticTier.NEURAL, include_nonpublic=False)
    assert len(refs_neural) > 0

    refs_util = collect_api(SemanticTier.UTIL, include_nonpublic=False)
    assert len(refs_util) > 0

    refs_invalid = collect_api(SemanticTier.ACTIVATION, include_nonpublic=False)
    assert refs_invalid == []

    refs_with_preview = collect_api(SemanticTier.ARRAY_API, include_nonpublic=True)
    assert len(refs_with_preview) >= len(refs_array)


def test_onnx_op_properties_and_roles() -> None:
    """Verify that extracted ONNX operations have valid operands, attributes, and results."""
    refs = collect_api(SemanticTier.ARRAY_API)
    op_map = {ref.name: ref for ref in refs if isinstance(ref, GhostOperationRef)}

    assert "Add" in op_map
    add_ref = op_map["Add"]
    assert add_ref.api_path == "onnx.Add"
    assert add_ref.operands is not None and len(add_ref.operands) == 2
    assert getattr(add_ref.operands[0], "role", None) == IRParameterRole.OPERAND
    assert getattr(add_ref.operands[1], "role", None) == IRParameterRole.OPERAND
    assert add_ref.returns is not None and len(add_ref.returns) == 1
    assert add_ref.returns[0].name == "C"

    assert "Conv" in op_map
    conv_ref = op_map["Conv"]
    attr_params = [
        p
        for p in conv_ref.params
        if getattr(p, "role", None) == IRParameterRole.ATTRIBUTE
    ]
    assert len(attr_params) > 0


def test_check_onnx_op_validation() -> None:
    """Test check_onnx_op validating existence, operands count, and attributes."""
    # 1. Valid Add
    res_valid = check_onnx_op("Add", inputs_count=2)
    assert res_valid["is_valid"] is True
    assert res_valid["op_exists"] is True
    assert len(res_valid["errors"]) == 0

    # 2. Input count mismatch
    res_mismatch = check_onnx_op("Add", inputs_count=5)
    assert res_mismatch["is_valid"] is False
    assert any("expects 2 inputs, but got 5" in e for e in res_mismatch["errors"])

    # 3. Valid attributes
    res_attrs = check_onnx_op("Conv", attributes=["kernel_shape", "strides"])
    assert res_attrs["is_valid"] is True

    # 4. Invalid attribute
    res_bad_attr = check_onnx_op("Conv", attributes=["nonexistent_attr"])
    assert res_bad_attr["is_valid"] is False
    assert any(
        "Attribute 'nonexistent_attr' is not valid" in e for e in res_bad_attr["errors"]
    )

    # 5. Non-existent operator
    res_nonexistent = check_onnx_op("NonExistentOp")
    assert res_nonexistent["is_valid"] is False
    assert res_nonexistent["op_exists"] is False
    assert any("Unknown ONNX operation" in e for e in res_nonexistent["errors"])

    # 6. Domain fallback when domain="" matches op from another domain
    with pytest.MonkeyPatch.context() as mp:
        dummy_domain_op = [
            {
                "name": "CustomClassifier",
                "domain": "ai.onnx.ml",
                "since_version": 1,
                "inputs": ["X"],
                "outputs": ["Y"],
                "attributes": [],
                "description": "Domain specific op",
            }
        ]
        mp.setattr(
            "ml_ecosystem_snapshots.frameworks.onnx_spec._load_onnx_ops",
            lambda: dummy_domain_op,
        )
        res_domain_fallback = check_onnx_op("CustomClassifier", domain="")
        assert res_domain_fallback["is_valid"] is True
        assert res_domain_fallback["op_exists"] is True


def test_cmd_check_onnx_cli(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI command check-onnx for valid and invalid queries."""
    # 1. Valid invocation
    args_valid = argparse.Namespace(
        op_name="Add",
        domain="",
        inputs_count=2,
        attributes=None,
    )
    cmd_check_onnx(args_valid)
    out_valid = capsys.readouterr().out
    assert "ONNX Operation 'Add' is valid." in out_valid

    # 2. Attributes filtering
    args_attrs = argparse.Namespace(
        op_name="Conv",
        domain="",
        inputs_count=None,
        attributes="kernel_shape, strides",
    )
    cmd_check_onnx(args_attrs)
    out_attrs = capsys.readouterr().out
    assert "ONNX Operation 'Conv' is valid." in out_attrs

    # 3. Invalid operation triggers sys.exit(1)
    args_invalid = argparse.Namespace(
        op_name="NonExistentOp",
        domain="",
        inputs_count=None,
        attributes=None,
    )
    with pytest.raises(SystemExit) as exc_info:
        cmd_check_onnx(args_invalid)
    assert exc_info.value.code == 1
    out_invalid = capsys.readouterr().out
    assert "ONNX Operation 'NonExistentOp' Invalid:" in out_invalid


def test_load_onnx_ops_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test fallback paths in _load_onnx_ops when onnx.defs is absent or fails."""
    # 1. Fallback when onnx.defs is None
    monkeypatch.setattr(
        "ml_ecosystem_snapshots.frameworks.onnx_spec._get_onnx_defs",
        lambda: None,
    )
    ops_fallback = _load_onnx_ops()
    assert ops_fallback == CANONICAL_ONNX_OPS

    # 2. Fallback when onnx.defs raises Exception
    class BrokenDefs:
        """Mock defs raising exception."""

        def get_all_schemas_with_history(self) -> Any:
            """Raise simulation error."""
            raise RuntimeError("Corrupt schema defs")

    monkeypatch.setattr(
        "ml_ecosystem_snapshots.frameworks.onnx_spec._get_onnx_defs",
        lambda: BrokenDefs(),
    )
    ops_err = _load_onnx_ops()
    assert ops_err == CANONICAL_ONNX_OPS


def test_get_onnx_defs_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _get_onnx_defs gracefully catches ImportError."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        """Mock import raising ImportError on onnx.

        Args:
            name: Module name.
            *args: Positional args.
            **kwargs: Keyword args.

        Returns:
            Imported module.

        Raises:
            ImportError: When importing onnx.
        """
        if "onnx" in name:
            raise ImportError("Simulated onnx import error")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    assert _get_onnx_defs() is None


def test_get_onnx_defs_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _get_onnx_defs returns module when import succeeds.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    fake_onnx = types.ModuleType("onnx")
    fake_defs = types.ModuleType("onnx.defs")
    setattr(fake_onnx, "defs", fake_defs)
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx)
    monkeypatch.setitem(sys.modules, "onnx.defs", fake_defs)
    assert _get_onnx_defs() is fake_defs


def test_onnx_schema_attributes_with_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _load_onnx_ops extracts schema attributes with explicit default values.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """

    class FakeAttribute:
        """Mock Schema attribute with type and default value."""

        def __init__(self, typ: str, required: bool, default_val: Any) -> None:
            """Initialize fake attribute.

            Args:
                typ: Attribute type name.
                required: Whether attribute is required.
                default_val: Default value representation.
            """
            self.type = typ
            self.required = required
            self.default_value = default_val

    class FakeSchemaWithAttr:
        """Mock Schema containing inputs, outputs, and attributes."""

        def __init__(self) -> None:
            """Initialize fake schema with attributes."""
            self.name = "AttrOp"
            self.domain = ""
            self.since_version = 1
            self.inputs: list[Any] = []
            self.outputs: list[Any] = []
            self.attributes = {
                "alpha": FakeAttribute("FLOAT", False, 1.5),
                "beta": FakeAttribute("INT", False, None),
            }
            self.doc = "Operation with attribute default."

    class MockDefsWithAttrs:
        """Mock Defs provider returning schemas with attributes."""

        def get_all_schemas_with_history(self) -> Any:
            """Return schemas with attributes.

            Returns:
                List of schema objects.
            """
            return [FakeSchemaWithAttr()]

    monkeypatch.setattr(
        "ml_ecosystem_snapshots.frameworks.onnx_spec._get_onnx_defs",
        lambda: MockDefsWithAttrs(),
    )
    ops = _load_onnx_ops()
    assert len(ops) == 1
    assert len(ops[0]["attributes"]) == 2
    attr_map = {a["name"]: a for a in ops[0]["attributes"]}
    assert attr_map["alpha"]["default"] == "1.5"
    assert attr_map["beta"]["default"] is None


def test_onnx_collect_no_outputs_no_attrs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify collect_api handles ops without outputs or attributes."""
    dummy_ops = [
        {
            "name": "SinkOp",
            "domain": "",
            "since_version": 1,
            "inputs": ["in_data"],
            "outputs": [],
            "attributes": [],
            "description": "Sink operation with no outputs",
        }
    ]
    monkeypatch.setattr(
        "ml_ecosystem_snapshots.frameworks.onnx_spec._load_onnx_ops",
        lambda: dummy_ops,
    )
    refs = collect_api(SemanticTier.ARRAY_API)
    assert len(refs) == 1
    assert isinstance(refs[0], GhostOperationRef)
    assert refs[0].returns_type == "None"
    assert refs[0].attributes is None


def test_onnx_filter_preview_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify collect_api skips preview domain operations when include_nonpublic is False."""
    preview_ops = [
        {
            "name": "PreviewOp",
            "domain": "ai.onnx.preview.training",
            "since_version": 1,
            "inputs": ["x"],
            "outputs": ["y"],
            "attributes": [],
            "description": "Preview operation",
        }
    ]
    monkeypatch.setattr(
        "ml_ecosystem_snapshots.frameworks.onnx_spec._load_onnx_ops",
        lambda: preview_ops,
    )
    refs_public = collect_api(SemanticTier.ARRAY_API, include_nonpublic=False)
    assert len(refs_public) == 0
    refs_nonpublic = collect_api(SemanticTier.ARRAY_API, include_nonpublic=True)
    assert len(refs_nonpublic) == 1


def test_onnx_schema_older_version_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _load_onnx_ops keeps latest version when multiple schema versions exist."""

    class FakeSchema:
        """Mock Schema object."""

        def __init__(self, name: str, domain: str, since_version: int) -> None:
            """Initialize fake schema."""
            self.name = name
            self.domain = domain
            self.since_version = since_version
            self.inputs: list[Any] = []
            self.outputs: list[Any] = []
            self.attributes: dict[str, Any] = {}
            self.doc = "Fake schema"

    class MockDefs:
        """Mock Defs object."""

        def get_all_schemas_with_history(self) -> Any:
            """Return schema versions."""
            return [
                FakeSchema("TestOp", "", 10),
                FakeSchema("TestOp", "", 5),
            ]

    monkeypatch.setattr(
        "ml_ecosystem_snapshots.frameworks.onnx_spec._get_onnx_defs",
        lambda: MockDefs(),
    )
    ops = _load_onnx_ops()
    assert len(ops) == 1
    assert ops[0]["since_version"] == 10


def test_onnx_load_empty_schemas(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify fallback when schemas list is empty."""

    class EmptyDefs:
        """Mock defs returning empty list."""

        def get_all_schemas_with_history(self) -> Any:
            """Return empty schema list."""
            return []

    monkeypatch.setattr(
        "ml_ecosystem_snapshots.frameworks.onnx_spec._get_onnx_defs",
        lambda: EmptyDefs(),
    )
    ops = _load_onnx_ops()
    assert ops == CANONICAL_ONNX_OPS


def test_onnx_roundtrip_serialization() -> None:
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
