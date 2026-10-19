"""Unit tests for the structured domain exception hierarchy in `exceptions.py`."""

from __future__ import annotations

import pytest

from ml_ecosystem_snapshots.cli.resolver import resolve_snapshot_path
from ml_ecosystem_snapshots.exceptions import (
    CExtensionSignatureError,
    ComplianceVerificationError,
    GroundingValidationError,
    IntrospectionError,
    OfflineModeNetworkError,
    SnapshotError,
    SnapshotNotFoundError,
    SnapshotSchemaError,
)
from ml_ecosystem_snapshots.network_env import fetch_url


def test_exception_inheritance_and_instantiation() -> None:
    """Verify every domain exception inherits from SnapshotError and appropriate stdlib base classes."""
    assert issubclass(SnapshotNotFoundError, SnapshotError)
    assert issubclass(SnapshotNotFoundError, FileNotFoundError)

    assert issubclass(SnapshotSchemaError, SnapshotError)
    assert issubclass(SnapshotSchemaError, ValueError)

    assert issubclass(IntrospectionError, SnapshotError)
    assert issubclass(CExtensionSignatureError, IntrospectionError)
    assert issubclass(CExtensionSignatureError, SnapshotError)

    assert issubclass(ComplianceVerificationError, SnapshotError)
    assert issubclass(GroundingValidationError, SnapshotError)

    assert issubclass(OfflineModeNetworkError, SnapshotError)
    assert issubclass(OfflineModeNetworkError, RuntimeError)

    for exc_cls in (
        SnapshotError,
        SnapshotNotFoundError,
        SnapshotSchemaError,
        IntrospectionError,
        CExtensionSignatureError,
        ComplianceVerificationError,
        GroundingValidationError,
        OfflineModeNetworkError,
    ):
        inst = exc_cls("domain error message")
        assert isinstance(inst, SnapshotError)
        assert "domain error message" in str(inst)


def test_fetch_url_offline_mode_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `fetch_url` raises `OfflineModeNetworkError` when `strict_offline=True` in offline mode.

    Args:
        monkeypatch: Pytest monkeypatch fixture for setting environment variables.
    """
    monkeypatch.setenv("ML_SNAPSHOTS_OFFLINE", "1")
    assert fetch_url("https://example.com/snap.json", strict_offline=False) is None
    with pytest.raises(OfflineModeNetworkError, match="blocked in offline mode"):
        fetch_url("https://example.com/snap.json", strict_offline=True)


def test_resolve_snapshot_path_not_found_error() -> None:
    """Verify `resolve_snapshot_path` raises `SnapshotNotFoundError` when `must_exist=True`."""
    nonexistent = "definitely_missing_snapshot_xyz_999.json"
    assert resolve_snapshot_path(nonexistent, must_exist=False) == nonexistent
    with pytest.raises(SnapshotNotFoundError, match="Snapshot file not found"):
        resolve_snapshot_path(nonexistent, must_exist=True)


def test_schema_and_introspection_domain_errors() -> None:
    """Verify SnapshotSchemaError, CExtensionSignatureError, and IntrospectionError are raised on invalid inputs."""
    from typing import Any
    from ml_ecosystem_snapshots.api import validate_snapshot_envelope
    from ml_ecosystem_snapshots.c_extension_sig import _parse_c_sig_ast_arguments
    from ml_ecosystem_snapshots.introspection.ast_kwargs import _parse_callable_ast
    from ml_ecosystem_snapshots.serialization.hydration import hydrate_ghost_ref

    invalid_payload: Any = "not_a_dict"
    with pytest.raises(SnapshotSchemaError, match="must be a dictionary"):
        hydrate_ghost_ref(invalid_payload)

    with pytest.raises(SnapshotSchemaError, match="Invalid GhostRef schema"):
        hydrate_ghost_ref({"name": 123, "api_path": None, "params": "invalid"})

    with pytest.raises(SnapshotSchemaError, match="Invalid snapshot envelope schema"):
        validate_snapshot_envelope({"target": "torch", "categories": "not_a_dict"})

    with pytest.raises(CExtensionSignatureError, match="Failed to parse"):
        _parse_c_sig_ast_arguments("??? invalid syntax $$$")

    with pytest.raises(IntrospectionError, match="Unable to parse AST"):
        _parse_callable_ast(len)
