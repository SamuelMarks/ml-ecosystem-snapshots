"""Tests for pytest configuration in tests/conftest.py."""

from __future__ import annotations

import resource
from unittest import mock

from conftest import pytest_configure

from ml_ecosystem_snapshots.introspection.static_parsers import _GRIFFE_CACHE


def test_pytest_configure_success() -> None:
    """Test pytest_configure adjusts RLIMIT_NOFILE to min(4096, hard) and populates _GRIFFE_CACHE."""
    mock_config = mock.MagicMock()
    with mock.patch("resource.getrlimit", return_value=(256, 8192)):
        with mock.patch("resource.setrlimit") as mock_setrlimit:
            pytest_configure(mock_config)
            mock_setrlimit.assert_called_once_with(resource.RLIMIT_NOFILE, (4096, 8192))
            assert "__loader__" in _GRIFFE_CACHE
            assert "torch" in _GRIFFE_CACHE


def test_pytest_configure_already_high() -> None:
    """Test pytest_configure when limit is already higher than target."""
    mock_config = mock.MagicMock()
    with mock.patch("resource.getrlimit", return_value=(8192, 8192)):
        with mock.patch("resource.setrlimit") as mock_setrlimit:
            pytest_configure(mock_config)
            mock_setrlimit.assert_not_called()
            assert "jax" in _GRIFFE_CACHE


def test_pytest_configure_exception() -> None:
    """Test pytest_configure handles OSError gracefully without raising and still populates _GRIFFE_CACHE."""
    mock_config = mock.MagicMock()
    _GRIFFE_CACHE.pop("__loader__", None)
    with mock.patch("resource.getrlimit", side_effect=OSError("Limit error")) as mock_getrlimit:
        pytest_configure(mock_config)
        mock_getrlimit.assert_called_once_with(resource.RLIMIT_NOFILE)
        assert callable(_GRIFFE_CACHE.get("__loader__"))
