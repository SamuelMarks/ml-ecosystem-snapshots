"""Tests for pytest configuration in tests/conftest.py."""

from unittest import mock
from conftest import pytest_configure


def test_pytest_configure_success() -> None:
    """Test pytest_configure successfully adjusts RLIMIT_NOFILE."""
    mock_config = mock.MagicMock()
    with mock.patch("resource.getrlimit", return_value=(256, 8192)):
        with mock.patch("resource.setrlimit") as mock_setrlimit:
            pytest_configure(mock_config)
            mock_setrlimit.assert_called_once()


def test_pytest_configure_already_high() -> None:
    """Test pytest_configure when limit is already higher than target."""
    mock_config = mock.MagicMock()
    with mock.patch("resource.getrlimit", return_value=(8192, 8192)):
        with mock.patch("resource.setrlimit") as mock_setrlimit:
            pytest_configure(mock_config)
            mock_setrlimit.assert_not_called()


def test_pytest_configure_exception() -> None:
    """Test pytest_configure handles OSError gracefully without raising."""
    mock_config = mock.MagicMock()
    with mock.patch(
        "resource.getrlimit", side_effect=OSError("Limit error")
    ) as mock_getrlimit:
        pytest_configure(mock_config)
        mock_getrlimit.assert_called_once()
