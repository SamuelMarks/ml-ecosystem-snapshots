"""Edge case tests for coverage gaps."""

import pytest
import os
import sys
from unittest.mock import patch
from typing import Any


def test_ghostref_from_obj_griffe_import_error() -> None:
    """Test GhostInspector handles missing griffe import gracefully."""
    from ml_ecosystem_snapshots.models import GhostInspector

    with patch.dict(sys.modules, {"griffe": None}):

        def dummy_fn() -> None:
            """Dummy function."""
            pass

        ref = GhostInspector.inspect(dummy_fn, "dummy_fn")
        assert ref.name == "dummy_fn"


def test_extended_ghost_ref_dict_with_non_string_default() -> None:
    """Test hydrate method coerces non-string default values to strings."""
    from ml_ecosystem_snapshots.models import GhostInspector

    data = {
        "name": "foo",
        "api_path": "foo",
        "kind": "function",
        "params": [{"name": "a", "default": 123, "kind": "POSITIONAL_OR_KEYWORD"}],
    }
    ref = GhostInspector.hydrate(data)
    assert ref.params[0].default == "123"


def test_build_stablehlo_snapshot_fetch_fail() -> None:
    """Test extract_ops throws RuntimeError when both fetch endpoints fail."""
    from ml_ecosystem_snapshots.tools.build_stablehlo_snapshot import extract_ops

    with patch("ml_ecosystem_snapshots.utils.fetch_url", return_value=None):
        with pytest.raises(RuntimeError):
            extract_ops(source_content=None)


def test_fetch_amdgpu_td_file_cache_write_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test fetch_td_file gracefully ignores caching when os.makedirs fails.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
    """
    from ml_ecosystem_snapshots.tools.scrape_amd_rdna import fetch_td_file

    monkeypatch.setattr("ml_ecosystem_snapshots.utils.fetch_url", lambda url: "content")

    def mock_makedirs(*args: Any, **kwargs: Any) -> None:
        """Mock makedirs throwing OSError.

        Args:
            args: Variadic positional arguments or CLI namespace.
            kwargs: Variadic keyword arguments.
        """
        raise OSError("Permission denied")

    monkeypatch.setattr(os, "makedirs", mock_makedirs)
    res = fetch_td_file("foo.td", use_cache=True)
    assert res == "content"


def test_fetch_amdgpu_td_file_fetch_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test fetch_td_file returns empty string if fetch_url fails.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
    """
    from ml_ecosystem_snapshots.tools.scrape_amd_rdna import fetch_td_file

    monkeypatch.setattr("ml_ecosystem_snapshots.utils.fetch_url", lambda url: None)
    res = fetch_td_file("foo.td", use_cache=True)
    assert res == ""


def test_fetch_amdgpu_td_file_no_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test fetch_td_file successful fetch with no cache.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
    """
    from ml_ecosystem_snapshots.tools.scrape_amd_rdna import fetch_td_file

    monkeypatch.setattr("ml_ecosystem_snapshots.utils.fetch_url", lambda url: "content")
    res = fetch_td_file("foo.td", use_cache=False)
    assert res == "content"


def test_scrape_sass_no_input() -> None:
    """Test scrape_sass raises ValueError when no inputs are provided."""
    from ml_ecosystem_snapshots.tools.scrape_nvidia_sass import scrape_sass

    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="input_path"):
            scrape_sass(input_path=None)
