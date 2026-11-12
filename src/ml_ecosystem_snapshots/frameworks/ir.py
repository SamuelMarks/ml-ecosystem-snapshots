"""ML-Switcheroo IR API Snapshot Extractor.

Dynamically introspects live `ml_switcheroo_ir` public classes, protocols,
exceptions, enums, and graph transformation functions so the IR snapshot stays
in strict 1-to-1 parity with the installed `ml_switcheroo_ir` package.
"""

from __future__ import annotations

import enum
import inspect

import ml_switcheroo_ir as ir
from ml_switcheroo_ir.schema.ghost import (
    GhostParam,
    GhostRef,
    ParameterKind,
    SemanticTier,
)

from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    LivePythonTarget,
    get_member_dict,
    get_runtime_signature,
)
from ml_ecosystem_snapshots.introspection.sanitize import (
    sanitize_param_default,
    sanitize_type_str,
)

IR_EXPORTED_SYMBOLS: tuple[str, ...] = (
    "LogicalGraph",
    "LogicalNode",
    "TensorSpec",
    "TensorShape",
    "LogicalEdge",
    "LogicalMesh",
    "LogicalAxis",
    "PartitionSpec",
    "NodeDict",
    "DType",
    "EdgeList",
    "VersionedOutputList",
    "AutodiffSentinelProtocol",
    "ZeroTangent",
    "NoTangent",
    "SymNode",
    "Polynomial",
    "SymInt",
    "SymVar",
    "SymConst",
    "SymBinaryOp",
    "SymUnaryOp",
    "SymPiecewise",
    "WebRTCSignalingTopology",
    "WebRTCPeerConfig",
    "PipelineTopologyConfig",
    "MicrobatchSplittingConfig",
    "MeshMappingConfig",
    "StageCommunicationConfig",
    "DependencyConfig",
    "PipelineScheduleConfig",
    "SchedulePhaseConfig",
    "SymbolicSolver",
    "SymbolicConstraintTracker",
    "ShapeTracker",
    "ParameterTranslationEngine",
    "BaseFrontend",
    "GraphFrontend",
    "CompilerBackend",
    "CyclicGraphError",
    "ShapeMismatchError",
    "DiagnosticSeverity",
    "GroundingDiagnostic",
    "GroundingReport",
    "eliminate_dead_nodes",
    "eliminate_common_subexpressions",
    "propagate_shapes_and_constants",
    "estimate_communication_volume",
    "estimate_graph_communication_volume",
    "topological_sort",
    "export_to_python",
    "export_schemas",
    "get_json_schema",
    "generate_typescript_definitions",
    "get_abstract_op",
    "broadcast_shapes",
    "broadcast_dimension",
    "matmul_shape",
    "normalize_axis",
)

_NON_SIGNATURE_TYPES: frozenset[str] = frozenset({"DType", "DiagnosticSeverity", "CyclicGraphError"})
_INITIAL_IR_MEMBERS: dict[str, LivePythonTarget] = get_member_dict(ir)


def _build_ir_ghost_param(param: inspect.Parameter) -> GhostParam:
    """Convert a live `inspect.Parameter` from `ml_switcheroo_ir` into a `GhostParam`.

    Args:
        param: Live runtime `inspect.Parameter` descriptor.

    Returns:
        Populated `GhostParam` preserving kind, annotation, and default presence.
    """
    kind = ParameterKind[param.kind.name]
    anno: str | None = None
    if param.annotation is not inspect.Parameter.empty:
        raw_ann = param.annotation if isinstance(param.annotation, str) else str(param.annotation)
        anno = sanitize_type_str(raw_ann)

    default_str: str | None = None
    if param.default is not inspect.Parameter.empty:
        sanitized_def, _factory, _mandatory = sanitize_param_default(param.default)
        default_str = sanitized_def if sanitized_def is not None else "..."

    return GhostParam(
        name=param.name,
        kind=kind,
        default=default_str,
        annotation=anno,
    )


def _inspect_ir_symbol(name: str, target: LivePythonTarget) -> GhostRef:
    """Build a `GhostRef` for a single exported `ml_switcheroo_ir` symbol.

    Args:
        name: Public symbol name in `ml_switcheroo_ir`.
        target: Runtime class, enum, protocol, or function object.

    Returns:
        Constructed `GhostRef` with live parameters, return type, and docstring.
    """
    is_cls = isinstance(target, type) or isinstance(target, enum.EnumMeta)
    kind = "class" if is_cls else "function"
    raw_doc = inspect.getdoc(target) if (is_cls or callable(target)) else None
    summary_doc = raw_doc.splitlines()[0].strip() if raw_doc else f"IR symbol {name}."

    params: list[GhostParam] = []
    returns_type: str | None = None
    if name not in _NON_SIGNATURE_TYPES and callable(target):
        sig = get_runtime_signature(target)
        params = [_build_ir_ghost_param(p) for p in sig.parameters.values()]
        if not is_cls and sig.return_annotation is not inspect.Parameter.empty:
            raw_ret = sig.return_annotation if isinstance(sig.return_annotation, str) else str(sig.return_annotation)
            returns_type = sanitize_type_str(raw_ret)

    return GhostRef(
        name=name,
        kind=kind,
        api_path=f"ml_switcheroo_ir.{name}",
        docstring=summary_doc,
        params=params,
        returns_type=returns_type,
    )


def collect_api(category: SemanticTier, include_nonpublic: bool = False) -> list[GhostRef]:
    """Collect `ml_switcheroo_ir` API symbols dynamically for supported semantic tiers.

    Args:
        category: Target `SemanticTier` category.
        include_nonpublic: Whether to include private symbols (unused for canonical IR exports).

    Returns:
        List of dynamically introspected `GhostRef` definitions.
    """
    del include_nonpublic
    if category not in (SemanticTier.ARRAY_API, SemanticTier.NEURAL, SemanticTier.UTIL):
        return []
    members = {**_INITIAL_IR_MEMBERS, **get_member_dict(ir)}
    return [_inspect_ir_symbol(sym, members[sym]) for sym in IR_EXPORTED_SYMBOLS]


__all__ = ["IR_EXPORTED_SYMBOLS", "collect_api"]
