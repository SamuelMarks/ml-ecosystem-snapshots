"""Default value and type annotation sanitization utilities for Ghost introspection."""

from __future__ import annotations

from typing import Any, Optional, Tuple
from ml_ecosystem_snapshots.models import (
    sanitize_param_default as _sanitize_param_default,
    sanitize_type_str as _sanitize_type_str,
)


def sanitize_param_default(val: Any) -> Tuple[Optional[str], Optional[str], bool]:
    """Sanitize a parameter default value, scrubbing memory pointers and factory callables.

    Args:
        val: Raw default value from `inspect.Parameter` or AST.

    Returns:
        Tuple of `(default_value_str, default_factory_str, is_mandatory)`.
    """
    return _sanitize_param_default(val)


def sanitize_type_str(typ_str: Optional[str]) -> Optional[str]:
    """Sanitize type hint strings into canonical PEP-585 / PEP-604 syntax.

    Args:
        typ_str: Raw type annotation string or None.

    Returns:
        Sanitized type string or None.
    """
    return _sanitize_type_str(typ_str)


__all__ = ["sanitize_param_default", "sanitize_type_str"]
