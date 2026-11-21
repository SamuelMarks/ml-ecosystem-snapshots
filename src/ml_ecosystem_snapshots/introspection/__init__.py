"""Introspection subpackage for live and static ML framework symbol extraction.

Provides modular components for AST keyword-argument discovery, decorator
unwrapping, static CDD/Griffe docstring and AST adapters, C-extension signature
parsing, default value and type string sanitization, and the core `GhostInspector` engine.
"""

from __future__ import annotations

from ml_ecosystem_snapshots.c_extension_sig import (
    CExtensionSignature,
    ParsedCSignatureLine,
    build_c_extension_signature,
    extract_c_extension_signature,
)
from ml_ecosystem_snapshots.introspection.ast_kwargs import (
    extract_accepted_kwargs_from_ast,
)
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    STATIC_GRIFFE_TYPES,
    DefaultValueInput,
    DescriptorProtocol,
    InspectableCallableProtocol,
    LivePythonTarget,
    StaticGriffeNode,
    WrappedTargetProtocol,
    get_member_dict,
    get_runtime_signature,
    get_runtime_source,
)
from ml_ecosystem_snapshots.introspection.sanitize import (
    TYPE_STR_SANITIZE_ENDO,
    FactoryDefault,
    LiteralDefault,
    MandatoryDefault,
    SanitizedDefault,
    UnrepresentableDefault,
    default_value_or_none,
    sanitize_param_default,
    sanitize_type_str,
)
from ml_ecosystem_snapshots.introspection.static_parsers import (
    clear_griffe_cache,
    load_griffe_module_safely,
    preload_griffe_cache,
)
from ml_ecosystem_snapshots.introspection.unwrap import unwrap_decorated_target

__all__ = [
    "CExtensionSignature",
    "DefaultValueInput",
    "DescriptorProtocol",
    "FactoryDefault",
    "GhostInspector",
    "InspectableCallableProtocol",
    "LiteralDefault",
    "LivePythonTarget",
    "MandatoryDefault",
    "ParsedCSignatureLine",
    "STATIC_GRIFFE_TYPES",
    "SanitizedDefault",
    "StaticGriffeNode",
    "TYPE_STR_SANITIZE_ENDO",
    "UnrepresentableDefault",
    "WrappedTargetProtocol",
    "build_c_extension_signature",
    "clear_griffe_cache",
    "default_value_or_none",
    "extract_accepted_kwargs_from_ast",
    "extract_c_extension_signature",
    "get_member_dict",
    "get_runtime_signature",
    "get_runtime_source",
    "load_griffe_module_safely",
    "preload_griffe_cache",
    "sanitize_param_default",
    "sanitize_type_str",
    "unwrap_decorated_target",
]
