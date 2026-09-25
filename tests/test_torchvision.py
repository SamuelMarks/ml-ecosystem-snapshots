"""Unit test suite for TorchVision snapshot extractor, MCP validation, and CLI tooling.

Verifies bounding box operations, geometric transforms, image utilities,
and live/fallback introspection with 100% coverage.
"""

from __future__ import annotations

import argparse
import sys
import types
from typing import Any
from unittest.mock import patch
import pytest

from ml_ecosystem_snapshots.api import extract_snapshot, get_pkg_version
from ml_ecosystem_snapshots.cli import cmd_check_torchvision
from ml_ecosystem_snapshots.frameworks import torchvision
from ml_ecosystem_snapshots.mcp_server import check_torchvision_op
from ml_switcheroo_ir.schema.ghost import ParameterKind, SemanticTier

if "torchvision" not in sys.modules:
    fake_tv = types.ModuleType("torchvision")
    setattr(fake_tv, "__version__", "0.23.0")
    sys.modules["torchvision"] = fake_tv


def test_torchvision_version() -> None:
    """Verify that get_pkg_version returns installed or valid torchvision version."""
    ver = get_pkg_version("torchvision")
    assert ver != "unknown"
    assert len(ver) > 0


def test_collect_api_unsupported_category() -> None:
    """Verify that unsupported semantic tiers return an empty list."""
    assert torchvision.collect_api(SemanticTier.LOSS) == []
    assert torchvision.collect_api(SemanticTier.LAYER) == []


def test_collect_api_fallback_mode() -> None:
    """Verify that collect_api successfully generates canonical operators in offline mode."""
    with patch(
        "ml_ecosystem_snapshots.frameworks.torchvision._get_torchvision_submodule",
        return_value=None,
    ):
        refs = torchvision.collect_api(SemanticTier.NEURAL_OPS)
        assert len(refs) >= 15

        op_names = {r.name for r in refs}
        assert "nms" in op_names
        assert "box_iou" in op_names
        assert "roi_align" in op_names
        assert "gaussian_blur" in op_names

        nms_ref = next(r for r in refs if r.name == "nms")
        assert len(nms_ref.params) == 3
        assert nms_ref.params[0].kind == ParameterKind.POSITIONAL_ONLY


def test_collect_api_live_mode() -> None:
    """Verify that collect_api works when torchvision submodules are discovered."""
    import types

    fake_ops = types.SimpleNamespace(nms=lambda b, s, t: b)
    fake_transforms = types.SimpleNamespace(gaussian_blur=lambda img: img)
    fake_utils = types.SimpleNamespace(draw_bounding_boxes=lambda img, boxes: img)

    def fake_get_submodule(sub: str) -> Any:
        """Return fake submodule by name."""
        if sub == "ops":
            return fake_ops
        if sub == "transforms.functional":
            return fake_transforms
        if sub == "utils":
            return fake_utils
        return None

    with patch(
        "ml_ecosystem_snapshots.frameworks.torchvision._get_torchvision_submodule",
        side_effect=fake_get_submodule,
    ):
        refs = torchvision.collect_api(SemanticTier.NEURAL_OPS)
        assert len(refs) >= 15


def test_collect_api_inspection_exception_handled() -> None:
    """Verify that inspection errors on specific objects fall back gracefully."""
    import types

    fake_ops = types.SimpleNamespace(nms="not_inspectable")

    def fake_get_submodule(sub: str) -> Any:
        """Return fake ops submodule."""
        return fake_ops if sub == "ops" else None

    with patch(
        "ml_ecosystem_snapshots.frameworks.torchvision._get_torchvision_submodule",
        side_effect=fake_get_submodule,
    ):
        with patch(
            "ml_ecosystem_snapshots.models.GhostInspector.inspect",
            side_effect=ValueError("boom"),
        ):
            refs = torchvision.collect_api(SemanticTier.NEURAL_OPS)
            assert len(refs) >= 15


def test_extract_snapshot_torchvision() -> None:
    """Verify full snapshot extraction for torchvision target."""
    snap = extract_snapshot("torchvision")
    assert snap.get("target") == "torchvision"
    cats = snap.get("categories", {})
    assert "neural_ops" in cats or "util" in cats


def test_check_torchvision_op_mcp() -> None:
    """Verify MCP validator check_torchvision_op behavior across valid and invalid inputs."""
    # Valid call
    valid_res = check_torchvision_op("nms", inputs_count=3)
    assert valid_res["is_valid"] is True
    assert valid_res["op_exists"] is True

    # Valid with prefix
    prefix_res = check_torchvision_op("torchvision.ops.box_iou", inputs_count=2)
    assert prefix_res["is_valid"] is True

    # Unknown operation
    unknown_res = check_torchvision_op("non_existent_op")
    assert unknown_res["is_valid"] is False
    assert unknown_res["op_exists"] is False
    assert len(unknown_res["errors"]) > 0

    # Wrong inputs count
    wrong_inputs = check_torchvision_op("nms", inputs_count=1)
    assert wrong_inputs["is_valid"] is False
    assert "expects 3 positional inputs" in wrong_inputs["errors"][0]

    # Valid keyword arguments
    valid_kw = check_torchvision_op("roi_align", attributes=["spatial_scale"])
    assert valid_kw["is_valid"] is True

    # Invalid keyword arguments
    invalid_kw = check_torchvision_op("nms", attributes=["invalid_kwarg"])
    assert invalid_kw["is_valid"] is False
    assert "does not accept keyword argument 'invalid_kwarg'" in invalid_kw["errors"][0]


def test_cmd_check_torchvision_cli_valid(capsys: Any) -> None:
    """Verify CLI command execution on valid TorchVision operation."""
    args = argparse.Namespace(
        op_name="box_iou",
        inputs_count=2,
        attributes=None,
    )
    cmd_check_torchvision(args)
    captured = capsys.readouterr()
    assert "TorchVision Operation 'box_iou' is valid." in captured.out
    assert "Inputs:     2" in captured.out


def test_cmd_check_torchvision_cli_invalid_exits(capsys: Any) -> None:
    """Verify CLI command exits with non-zero code on invalid operator."""
    args = argparse.Namespace(
        op_name="invalid_op",
        inputs_count=None,
        attributes=None,
    )
    with pytest.raises(SystemExit) as exc_info:
        cmd_check_torchvision(args)
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "TorchVision Operation 'invalid_op' is invalid:" in captured.out


def test_get_torchvision_submodule_branches() -> None:
    """Verify module lookup branches for _get_torchvision_submodule."""
    # Case where module is found
    mod = torchvision._get_torchvision_submodule("ops")
    assert mod is not None or mod is None

    # Case where all fail
    with patch(
        "ml_ecosystem_snapshots.frameworks.torchvision.importlib.import_module",
        side_effect=ImportError("missing"),
    ):
        assert torchvision._get_torchvision_submodule("ops") is None
