"""Unit tests for `exceptions.py` boundary carriers and fault propagation."""

from __future__ import annotations

import pytest
from ml_switcheroo_ir.typeclasses.outcome import Err, value_or

from ml_ecosystem_snapshots.api import validate_snapshot_envelope
from ml_ecosystem_snapshots.c_extension_sig import parse_c_sig_ast_arguments
from ml_ecosystem_snapshots.cli.resolver import resolve_snapshot_path
from ml_ecosystem_snapshots.errors import (
    CExtensionParseFault,
    IntrospectionFault,
    OfflineNetworkFault,
    SnapshotEnvelopeFault,
    SnapshotNotFoundFault,
    format_snapshot_error,
)
from ml_ecosystem_snapshots.exceptions import (
    SnapshotError,
    raise_snapshot_boundary,
)
from ml_ecosystem_snapshots.introspection.ast_kwargs import parse_callable_ast
from ml_ecosystem_snapshots.network_env import fetch_url
from ml_ecosystem_snapshots.serialization.hydration import hydrate_ghost_ref


def test_snapshot_error_wraps_domain_error() -> None:
    """Verify `SnapshotError` wraps closed `SnapshotDomainError` variants and renders diagnostics."""
    fault = SnapshotNotFoundFault(target="torch", version="2.4.0", searched_paths=("/opt/snaps",))
    err = SnapshotError(fault)
    assert isinstance(err, Exception)
    assert err.domain_error == fault
    assert str(err) == format_snapshot_error(fault)


def test_fetch_url_offline_network_fault(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `fetch_url` returns `Err(OfflineNetworkFault)` in offline mode.

    Args:
        monkeypatch: Pytest monkeypatch fixture for setting environment variables.
    """
    monkeypatch.setenv("ML_SNAPSHOTS_OFFLINE", "1")
    res_soft = fetch_url("https://example.com/snap.json", strict_offline=False)
    assert isinstance(res_soft, Err)
    assert isinstance(res_soft.error, OfflineNetworkFault)
    res_strict = fetch_url("https://example.com/snap.json", strict_offline=True)
    assert isinstance(res_strict, Err)
    assert isinstance(res_strict.error, OfflineNetworkFault)
    assert res_strict.error.operation == "offline_blocked"


def test_resolve_snapshot_path_not_found_fault() -> None:
    """Verify `resolve_snapshot_path` returns `Err(SnapshotNotFoundFault)` when missing."""
    nonexistent = "definitely_missing_snapshot_xyz_999.json"
    res = resolve_snapshot_path(nonexistent)
    assert isinstance(res, Err)
    assert isinstance(res.error, SnapshotNotFoundFault)
    assert res.error.target == nonexistent
    assert value_or(res, nonexistent) == nonexistent

    with pytest.raises(SnapshotError) as exc_info:
        raise_snapshot_boundary(res)
    assert exc_info.value.domain_error == res.error


def test_schema_and_introspection_domain_faults() -> None:
    """Verify `SnapshotEnvelopeFault`, `CExtensionParseFault`, and `IntrospectionFault` outcomes."""
    invalid_payload: str = "not_a_dict"
    hyd1 = hydrate_ghost_ref(invalid_payload)
    assert isinstance(hyd1, Err)
    assert isinstance(hyd1.error, SnapshotEnvelopeFault)
    assert "must be a dictionary" in hyd1.error.reason

    hyd2 = hydrate_ghost_ref({"name": 123, "api_path": None, "params": "invalid"})
    assert isinstance(hyd2, Err)
    assert isinstance(hyd2.error, SnapshotEnvelopeFault)
    assert "Invalid GhostRef schema" in hyd2.error.reason

    env_res = validate_snapshot_envelope({"target": "torch", "categories": "not_a_dict"})
    assert isinstance(env_res, Err)
    assert isinstance(env_res.error, SnapshotEnvelopeFault)
    assert "Invalid snapshot envelope schema" in env_res.error.reason

    c_sig_res = parse_c_sig_ast_arguments("??? invalid syntax $$$")
    assert isinstance(c_sig_res, Err)
    assert isinstance(c_sig_res.error, CExtensionParseFault)
    assert "Failed to parse" in c_sig_res.error.reason

    ast_res = parse_callable_ast(len)
    assert isinstance(ast_res, Err)
    assert isinstance(ast_res.error, IntrospectionFault)
    assert "Unable to parse AST" in ast_res.error.reason
