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
    """Verify `fetch_url` returns `Err(OfflineModeNetworkError)` in offline mode.

    Args:
        monkeypatch: Pytest monkeypatch fixture for setting environment variables.
    """
    from ml_switcheroo_ir.typeclasses.outcome import Err

    monkeypatch.setenv("ML_SNAPSHOTS_OFFLINE", "1")
    res_soft = fetch_url("https://example.com/snap.json", strict_offline=False)
    assert isinstance(res_soft, Err)
    assert isinstance(res_soft.error, OfflineModeNetworkError)
    res_strict = fetch_url("https://example.com/snap.json", strict_offline=True)
    assert isinstance(res_strict, Err)
    assert isinstance(res_strict.error, OfflineModeNetworkError)
    assert "blocked in offline mode" in str(res_strict.error)


def test_resolve_snapshot_path_not_found_error() -> None:
    """Verify `resolve_snapshot_path` returns `Err(SnapshotNotFoundError)` when missing."""
    from ml_switcheroo_ir.typeclasses.outcome import Err, value_or

    nonexistent = "definitely_missing_snapshot_xyz_999.json"
    res = resolve_snapshot_path(nonexistent)
    assert isinstance(res, Err)
    assert isinstance(res.error, SnapshotNotFoundError)
    assert "Snapshot file not found" in str(res.error)
    assert value_or(res, nonexistent) == nonexistent


def test_schema_and_introspection_domain_errors() -> None:
    """Verify SnapshotSchemaError, CExtensionSignatureError, and IntrospectionError outcomes on invalid inputs."""
    from ml_switcheroo_ir.typeclasses.outcome import Err

    from ml_ecosystem_snapshots.api import validate_snapshot_envelope
    from ml_ecosystem_snapshots.c_extension_sig import parse_c_sig_ast_arguments
    from ml_ecosystem_snapshots.introspection.ast_kwargs import parse_callable_ast
    from ml_ecosystem_snapshots.serialization.hydration import hydrate_ghost_ref

    invalid_payload: str = "not_a_dict"
    hyd1 = hydrate_ghost_ref(invalid_payload)
    assert isinstance(hyd1, Err)
    assert isinstance(hyd1.error, SnapshotSchemaError)
    assert "must be a dictionary" in str(hyd1.error)

    hyd2 = hydrate_ghost_ref({"name": 123, "api_path": None, "params": "invalid"})
    assert isinstance(hyd2, Err)
    assert isinstance(hyd2.error, SnapshotSchemaError)
    assert "Invalid GhostRef schema" in str(hyd2.error)

    env_res = validate_snapshot_envelope({"target": "torch", "categories": "not_a_dict"})
    assert isinstance(env_res, Err)
    assert isinstance(env_res.error, SnapshotSchemaError)
    assert "Invalid snapshot envelope schema" in str(env_res.error)

    c_sig_res = parse_c_sig_ast_arguments("??? invalid syntax $$$")
    assert isinstance(c_sig_res, Err)
    assert isinstance(c_sig_res.error, CExtensionSignatureError)
    assert "Failed to parse" in str(c_sig_res.error)

    with pytest.raises(IntrospectionError, match="Unable to parse AST"):
        parse_callable_ast(len)
