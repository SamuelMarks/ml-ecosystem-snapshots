"""Introspection subpackage for live and static ML framework symbol extraction.

Provides modular components for AST keyword-argument discovery, decorator
unwrapping, static CDD/Griffe docstring and AST adapters, C-extension signature
parsing, default value and type string sanitization, and the core `GhostInspector` engine.
"""

from __future__ import annotations

from ml_ecosystem_snapshots.introspection.ast_kwargs import (
    extract_accepted_kwargs_from_ast,
)
from ml_ecosystem_snapshots.c_extension_sig import (
    CExtensionSignature,
    extract_c_extension_signature,
)
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.sanitize import (
    sanitize_param_default,
    sanitize_type_str,
)
from ml_ecosystem_snapshots.introspection.static_parsers import preload_griffe_cache
from ml_ecosystem_snapshots.introspection.unwrap import unwrap_decorated_target

__all__ = [
    "CExtensionSignature",
    "GhostInspector",
    "extract_accepted_kwargs_from_ast",
    "extract_c_extension_signature",
    "preload_griffe_cache",
    "sanitize_param_default",
    "sanitize_type_str",
    "unwrap_decorated_target",
]
