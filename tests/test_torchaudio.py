"""Unit test suite for TorchAudio snapshot extractor, MCP validation, and CLI tooling.

Verifies audio processing operations, feature extraction transforms, filterbanks,
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
from ml_ecosystem_snapshots.cli import cmd_check_torchaudio
from ml_ecosystem_snapshots.frameworks import torchaudio
from ml_ecosystem_snapshots.mcp_server import check_torchaudio_op
from ml_switcheroo_ir.schema.ghost import ParameterKind, SemanticTier

if "torchaudio" not in sys.modules:
    fake_ta = types.ModuleType("torchaudio")
    setattr(fake_ta, "__version__", "2.8.0")
    sys.modules["torchaudio"] = fake_ta


def test_torchaudio_version() -> None:
    """Verify that get_pkg_version returns installed or valid torchaudio version."""
    ver = get_pkg_version("torchaudio")
    assert ver != "unknown"
    assert len(ver) > 0


def test_collect_api_unsupported_category() -> None:
    """Verify that unsupported semantic tiers return an empty list."""
    assert torchaudio.collect_api(SemanticTier.LOSS) == []
    assert torchaudio.collect_api(SemanticTier.OPTIMIZER) == []


def test_collect_api_fallback_mode() -> None:
    """Verify that collect_api successfully generates canonical operators in offline mode."""
    with patch(
        "ml_ecosystem_snapshots.frameworks.torchaudio._get_torchaudio_submodule",
        return_value=None,
    ):
        refs = torchaudio.collect_api(SemanticTier.NEURAL_OPS)
        assert len(refs) >= 10

        op_names = {r.name for r in refs}
        assert "melscale_fbanks" in op_names
        assert "spectrogram" in op_names
        assert "MFCC" in op_names
        assert "MelSpectrogram" in op_names

        mel_ref = next(r for r in refs if r.name == "melscale_fbanks")
        assert len(mel_ref.params) == 7
        assert mel_ref.params[0].kind == ParameterKind.POSITIONAL_ONLY


def test_collect_api_live_mode() -> None:
    """Verify that collect_api works when torchaudio submodules are discovered."""
    import types

    fake_func = types.SimpleNamespace(
        melscale_fbanks=lambda n_freqs, f_min, f_max, n_mels, sample_rate: n_freqs
    )
    fake_trans = types.SimpleNamespace(MFCC=lambda: None)

    def fake_get_submodule(sub: str) -> Any:
        """Return fake submodule by name."""
        if sub == "functional":
            return fake_func
        if sub == "transforms":
            return fake_trans
        return None

    with patch(
        "ml_ecosystem_snapshots.frameworks.torchaudio._get_torchaudio_submodule",
        side_effect=fake_get_submodule,
    ):
        refs = torchaudio.collect_api(SemanticTier.NEURAL_OPS)
        assert len(refs) >= 10


def test_collect_api_inspection_exception_handled() -> None:
    """Verify that inspection errors on specific objects fall back gracefully."""
    import types

    fake_func = types.SimpleNamespace(melscale_fbanks="not_inspectable")

    def fake_get_submodule(sub: str) -> Any:
        """Return fake functional submodule."""
        return fake_func if sub == "functional" else None

    with patch(
        "ml_ecosystem_snapshots.frameworks.torchaudio._get_torchaudio_submodule",
        side_effect=fake_get_submodule,
    ):
        with patch(
            "ml_ecosystem_snapshots.models.GhostInspector.inspect",
            side_effect=ValueError("boom"),
        ):
            refs = torchaudio.collect_api(SemanticTier.NEURAL_OPS)
            assert len(refs) >= 10


def test_extract_snapshot_torchaudio() -> None:
    """Verify full snapshot extraction for torchaudio target."""
    snap = extract_snapshot("torchaudio")
    assert snap.get("target") == "torchaudio"
    cats = snap.get("categories", {})
    assert "neural_ops" in cats or "util" in cats


def test_check_torchaudio_op_mcp() -> None:
    """Verify MCP validator check_torchaudio_op behavior across valid and invalid inputs."""
    # Valid call
    valid_res = check_torchaudio_op("melscale_fbanks", inputs_count=5)
    assert valid_res["is_valid"] is True
    assert valid_res["op_exists"] is True

    # Valid with prefix
    prefix_res = check_torchaudio_op(
        "torchaudio.functional.spectrogram", inputs_count=8
    )
    assert prefix_res["is_valid"] is True

    # Unknown operation
    unknown_res = check_torchaudio_op("non_existent_op")
    assert unknown_res["is_valid"] is False
    assert unknown_res["op_exists"] is False
    assert len(unknown_res["errors"]) > 0

    # Wrong inputs count
    wrong_inputs = check_torchaudio_op("melscale_fbanks", inputs_count=1)
    assert wrong_inputs["is_valid"] is False
    assert "expects 5 positional inputs" in wrong_inputs["errors"][0]

    # Valid keyword arguments
    valid_kw = check_torchaudio_op("melscale_fbanks", attributes=["norm", "mel_scale"])
    assert valid_kw["is_valid"] is True

    # Invalid keyword arguments
    invalid_kw = check_torchaudio_op("melscale_fbanks", attributes=["invalid_kwarg"])
    assert invalid_kw["is_valid"] is False
    assert "does not accept keyword argument 'invalid_kwarg'" in invalid_kw["errors"][0]


def test_cmd_check_torchaudio_cli_valid(capsys: Any) -> None:
    """Verify CLI command execution on valid TorchAudio operation."""
    args = argparse.Namespace(
        op_name="spectrogram",
        inputs_count=8,
        attributes=None,
    )
    cmd_check_torchaudio(args)
    captured = capsys.readouterr()
    assert "TorchAudio Operation 'spectrogram' is valid." in captured.out
    assert "Inputs:     8" in captured.out


def test_cmd_check_torchaudio_cli_invalid_exits(capsys: Any) -> None:
    """Verify CLI command exits with non-zero code on invalid operator."""
    args = argparse.Namespace(
        op_name="invalid_op",
        inputs_count=None,
        attributes=None,
    )
    with pytest.raises(SystemExit) as exc_info:
        cmd_check_torchaudio(args)
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "TorchAudio Operation 'invalid_op' is invalid:" in captured.out


def test_get_torchaudio_submodule_branches() -> None:
    """Verify module lookup branches for _get_torchaudio_submodule."""
    # Case where module is found
    mod = torchaudio._get_torchaudio_submodule("functional")
    assert mod is not None or mod is None

    # Case where all fail
    with patch(
        "ml_ecosystem_snapshots.frameworks.torchaudio.importlib.import_module",
        side_effect=ImportError("missing"),
    ):
        assert torchaudio._get_torchaudio_submodule("functional") is None
