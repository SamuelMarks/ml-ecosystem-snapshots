"""Models for Ghost API representations.

Ghost Core: Introspection Abstraction Layer.

This module provides the data structures and re-exports the inspection and
hydration logic required to decouple framework analysis from the live
environment. It enables the system to operate in "Ghost Mode" (WASM/CI) by
working against cached snapshots instead of requiring heavy libraries
(Torch/TensorFlow) to be installed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from ml_switcheroo_ir.schema.ghost import (
    ExtendedGhostParam as ExtendedGhostParam,
    ExtendedGhostRef as ExtendedGhostRef,
    GhostInstructionRef as GhostInstructionRef,
    GhostIsaRef as GhostIsaRef,
    GhostMlirRef as GhostMlirRef,
    GhostOperationRef as GhostOperationRef,
    GhostParam as GhostParam,
    GhostPythonRef as GhostPythonRef,
    GhostRef as GhostRef,
    GhostResult as GhostResult,
    IRParameterRole as IRParameterRole,
    OperandDirection as OperandDirection,
    ParameterKind as ParameterKind,
    SnapshotEnvelope as SnapshotEnvelope,
)

# --- Backward-compatibility re-exports from introspection / serialization / utils ---
from ml_ecosystem_snapshots.introspection.ast_kwargs import (
    extract_accepted_kwargs_from_ast,
)
from ml_ecosystem_snapshots.introspection.inspector import (
    FRAMEWORK_CAPABILITIES,
    GhostInspector,
    STANDARD_ARG_MAP,
    STANDARD_ENUM_MAP,
    to_parameter_kind,
)
from ml_ecosystem_snapshots.introspection.sanitize import (
    sanitize_param_default,
    sanitize_type_str,
)
from ml_ecosystem_snapshots.introspection.static_parsers import (
    _GRIFFE_CACHE,
    preload_griffe_cache,
)
from ml_ecosystem_snapshots.serialization.hydration import (
    hydrate_ghost_ref,
    migrate_ghost_ref_v2,
)
from ml_ecosystem_snapshots.utils import (
    extract_c_extension_signature,
    extract_griffe_docstring_metadata,
    get_framework_docstring_parser,
    resolve_griffe_parser,
    strip_sphinx_roles,
)


class RawParamSpec(TypedDict, total=False):
    """Typed specification for a raw canonical parameter entry."""

    name: str
    kind: str
    default: Optional[str]
    annotation: Optional[str]
    type: Optional[str]
    description: Optional[str]
    direction: Optional[str]
    role: Optional[str]


class CanonicalOpSpec(TypedDict, total=False):
    """Typed specification for a canonical fallback operation or instruction."""

    name: str
    mnemonic: str
    api_path: str
    kind: str
    params: List[RawParamSpec]
    inputs: List[str]
    outputs: List[str]
    attributes: List[Dict[str, Any]]
    docstring: str
    description: str
    category: str
    submodule: str
    address_space: str


__all__ = [
    # --- Core Schema Models ---
    "CanonicalOpSpec",
    "ExtendedGhostParam",
    "ExtendedGhostRef",
    "GhostInstructionRef",
    "GhostIsaRef",
    "GhostMlirRef",
    "GhostOperationRef",
    "GhostParam",
    "GhostPythonRef",
    "GhostRef",
    "GhostResult",
    "IRParameterRole",
    "OperandDirection",
    "ParameterKind",
    "RawParamSpec",
    "SnapshotEnvelope",
    # --- Backward-Compatibility Introspection & Hydration Re-Exports ---
    "FRAMEWORK_CAPABILITIES",
    "GhostInspector",
    "STANDARD_ARG_MAP",
    "STANDARD_ENUM_MAP",
    "_GRIFFE_CACHE",
    "extract_accepted_kwargs_from_ast",
    "extract_c_extension_signature",
    "extract_griffe_docstring_metadata",
    "get_framework_docstring_parser",
    "hydrate_ghost_ref",
    "migrate_ghost_ref_v2",
    "preload_griffe_cache",
    "resolve_griffe_parser",
    "sanitize_param_default",
    "sanitize_type_str",
    "strip_sphinx_roles",
    "to_parameter_kind",
]
