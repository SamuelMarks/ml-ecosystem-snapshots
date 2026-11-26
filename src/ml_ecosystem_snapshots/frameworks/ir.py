"""ML-Switcheroo IR API Snapshot Extractor.

Dynamically introspects live `ml_switcheroo_ir` public classes, protocols,
typeclasses, identifiers, proofs, exceptions, enums, constants, and graph
transformation functions so the IR snapshot stays in strict 1-to-1 parity
with the installed `ml_switcheroo_ir` package.
"""

from __future__ import annotations

import enum
import inspect
import typing

import ml_switcheroo_ir as ir
import pydantic
import typing_extensions
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
    default_value_or_none,
    sanitize_param_default,
    sanitize_type_str,
)

IR_EXPORTED_SYMBOLS: tuple[str, ...] = (
    "ACYCLIC_CANONICAL_OPTIMIZE_PIPELINE",
    "ACYCLIC_CSE_PASS",
    "ACYCLIC_DCE_PASS",
    "ACYCLIC_SHAPE_CONSTANT_PROP_PASS",
    "ALL_DIALECT_CATALOG_WITNESSES",
    "ARRAY_API_REGISTRY",
    "ATEN_REGISTRY",
    "AcyclicAndSchemaPhase",
    "AcyclicGraphPass",
    "AcyclicSchemaShardingPhase",
    "AcyclicSortedPhase",
    "ArrayBufferProtocol",
    "AttrTerm",
    "AttributeValue",
    "AutodiffSentinelProtocol",
    "BaseFrontend",
    "BoundedLattice",
    "CANONICAL_OPTIMIZE_PIPELINE",
    "CHEX_REGISTRY",
    "CSE_PASS",
    "Codec",
    "CodecError",
    "CompilerBackend",
    "CustomRuleHandleProtocol",
    "CyclicGraphError",
    "DCE_PASS",
    "DEFAULT_SHAPE_RULE_REGISTRY",
    "DType",
    "DTypeLikeProtocol",
    "DependencyConfig",
    "DeviceLikeProtocol",
    "DiagnosticSeverity",
    "DialectCatalogWitness",
    "DialectDomain",
    "DimensionType",
    "DimensionValue",
    "EdgeList",
    "Endo",
    "EndoRewrite",
    "Eq",
    "Err",
    "ExtendedGhostRef",
    "FROZEN_LOGICAL_GRAPH_EQ",
    "FROZEN_LOGICAL_GRAPH_HASH",
    "FrozenLogicalEdge",
    "FrozenLogicalGraph",
    "FrozenLogicalNode",
    "FullyVerifiedPhase",
    "GRAIN_REGISTRY",
    "GhostOverload",
    "GhostParam",
    "GraphCycleDiagnosticError",
    "GraphFrontend",
    "GraphTransformPass",
    "GroundedPhase",
    "GroundingAuditBatchError",
    "GroundingDiagnostic",
    "GroundingReport",
    "HashableWitness",
    "IRGraph",
    "IRNode",
    "IR_DOMAIN_ERROR_EQ",
    "IR_DOMAIN_ERROR_PRETTY",
    "IdentifierKind",
    "IdentifierParseError",
    "InitializerValue",
    "Invalid",
    "IrBoundaryException",
    "IrCodecDecodeError",
    "IrDomainError",
    "JsonCodec",
    "LOGICAL_GRAPH_EQ",
    "LOGICAL_GRAPH_PRETTY",
    "LOGICAL_NODE_EQ",
    "LOGICAL_NODE_REWRITE",
    "LogicalAxis",
    "LogicalEdge",
    "LogicalGraph",
    "LogicalMesh",
    "LogicalNode",
    "LoopOptionsProtocol",
    "MLIR_REGISTRY",
    "MeshAxisName",
    "MeshLikeProtocol",
    "MeshMappingConfig",
    "MicrobatchSplittingConfig",
    "Monoid",
    "NoTangent",
    "NodeDict",
    "NodeId",
    "NonBoolInt",
    "NonEmpty",
    "NonNegativeInt",
    "ODL_CATALOG",
    "OPTAX_REGISTRY",
    "ORBAX_REGISTRY",
    "Ok",
    "OpName",
    "OpSchemaConfig",
    "Ord",
    "Ordering",
    "ParameterTranslationEngine",
    "PartialIso",
    "PartitionSpec",
    "PipelineScheduleConfig",
    "PipelineTopologyConfig",
    "Polynomial",
    "PortRef",
    "PositiveInt",
    "Pretty",
    "ProvenGraph",
    "PythonExprEmit",
    "QualifiedOpKey",
    "RecursiveFold",
    "Result",
    "Ring",
    "SHAPE_CONSTANT_PROP_PASS",
    "SHAPE_RULE_REGISTRY_MONOID",
    "SchedulePhaseConfig",
    "SchemaValidPhase",
    "SchemaValidationBatchError",
    "Semigroup",
    "Semiring",
    "ShapeInferenceRuleRegistry",
    "ShapeMetadataType",
    "ShapeMismatchError",
    "ShapeTracker",
    "ShapeTransferWitness",
    "ShapeUnificationError",
    "ShardingValidPhase",
    "ShardingValidationBatchError",
    "SsaValueName",
    "StageCommunicationConfig",
    "SubgraphLikeProtocol",
    "SymBinaryOp",
    "SymCondition",
    "SymConst",
    "SymExpr",
    "SymInt",
    "SymNode",
    "SymPiecewise",
    "SymUnaryOp",
    "SymVar",
    "SymbolicConstraintTracker",
    "SymbolicDimProtocol",
    "SymbolicIntWrapperProtocol",
    "SymbolicNodeProtocol",
    "SymbolicSolver",
    "TensorShape",
    "TensorSpec",
    "TranslationIsoError",
    "Valid",
    "Validation",
    "ValidatorConfig",
    "VersionedOutputList",
    "WebRTCPeerConfig",
    "WebRTCSignalingTopology",
    "ZeroTangent",
    "broadcast_dimension",
    "broadcast_shapes",
    "collect_unit_validations_ne",
    "compose_acyclic_passes",
    "compose_passes",
    "eliminate_common_subexpressions",
    "eliminate_dead_nodes",
    "embed_attribute_value",
    "estimate_communication_volume",
    "estimate_graph_communication_volume",
    "export_schemas",
    "export_to_python",
    "fixpoint_acyclic_pass",
    "fixpoint_graph_pass",
    "fold_logical_graph",
    "format_ir_domain_error",
    "freeze_graph",
    "from_error_sequence",
    "generate_typescript_definitions",
    "get_abstract_op",
    "get_catalog_path",
    "get_json_schema",
    "graph_from_dict_result",
    "graph_from_json_result",
    "invalid_one",
    "is_dynamic_dimension",
    "is_non_bool_int",
    "is_symbolic_dimension",
    "lift_acyclic_pass",
    "matmul_shape",
    "nonempty_eq",
    "nonempty_hash",
    "nonempty_ord",
    "nonempty_pretty",
    "nonempty_semigroup",
    "normalize_axis",
    "normalize_dimension_tuple",
    "normalize_dtype",
    "normalize_optional_dtype",
    "project_attribute_value",
    "propagate_shapes_and_constants",
    "prove_acyclic_sorted",
    "prove_fully_verified",
    "prove_grounded",
    "prove_schema_valid",
    "prove_sharding_valid",
    "rewrite_logical_graph_bottom_up",
    "thaw_graph",
    "to_boundary_exception",
    "topological_sort",
    "traverse_validation_ne",
    "upgrade_to_fully_verified",
    "upgrade_to_schema_valid",
    "upgrade_to_sharding_valid",
    "validation_ne_to_tuple",
    "validation_tuple_to_ne",
)

_NON_SIGNATURE_TYPES: frozenset[str] = frozenset(
    {"DType", "DiagnosticSeverity", "IdentifierKind", "Ordering", "CyclicGraphError"}
)
_INITIAL_IR_MEMBERS: dict[str, LivePythonTarget] = get_member_dict(ir)


class _IrSymbolVariant(pydantic.BaseModel):
    """Frozen base model for `ml_switcheroo_ir` symbol classification variants."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    target: LivePythonTarget


class _IrClassSymbol(_IrSymbolVariant):
    """Exported `ml_switcheroo_ir` class or protocol with an inspectable constructor signature."""


class _IrFunctionSymbol(_IrSymbolVariant):
    """Exported `ml_switcheroo_ir` function or callable `GraphTransformPass` instance."""


class _IrEnumOrExceptionSymbol(_IrSymbolVariant):
    """Exported `ml_switcheroo_ir` enum or C-backed exception without an explicit Python constructor."""


class _IrTypeAliasOrNewTypeSymbol(_IrSymbolVariant):
    """Exported `ml_switcheroo_ir` `TypeAlias` union, generic alias, or `NewType` wrapper."""


class _IrConstantWitnessSymbol(_IrSymbolVariant):
    """Exported `ml_switcheroo_ir` module-level catalog registry or frozen witness constant."""


_IrSymbolClassification: typing_extensions.TypeAlias = (
    "_IrClassSymbol"
    " | _IrFunctionSymbol"
    " | _IrEnumOrExceptionSymbol"
    " | _IrTypeAliasOrNewTypeSymbol"
    " | _IrConstantWitnessSymbol"
)


def _classify_ir_symbol(name: str, target: LivePythonTarget) -> _IrSymbolClassification:
    """Classify a live `ml_switcheroo_ir` export into a closed `_IrSymbolClassification` variant.

    Args:
        name: Public symbol name in `ml_switcheroo_ir`.
        target: Runtime symbol value retrieved from `ml_switcheroo_ir`.

    Returns:
        Disjoint `_IrSymbolClassification` sum-type instance.
    """
    if typing.get_origin(target) is not None or "__supertype__" in get_member_dict(target):
        return _IrTypeAliasOrNewTypeSymbol.model_construct(target=target)
    if isinstance(target, enum.EnumMeta) or name in _NON_SIGNATURE_TYPES:
        return _IrEnumOrExceptionSymbol.model_construct(target=target)
    if isinstance(target, type):
        return _IrClassSymbol.model_construct(target=target)
    if callable(target):
        return _IrFunctionSymbol.model_construct(target=target)
    return _IrConstantWitnessSymbol.model_construct(target=target)


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
        default_str = default_value_or_none(sanitize_param_default(param.default), unrepresentable_token="...")

    return GhostParam(
        name=param.name,
        kind=kind,
        default=default_str,
        annotation=anno,
    )


def _extract_summary_doc(name: str, target: LivePythonTarget, fallback_prefix: str) -> str:
    """Extract a single-line summary docstring for `target` with a deterministic fallback.

    Args:
        name: Public symbol name.
        target: Runtime symbol object.
        fallback_prefix: Prefix used when `inspect.getdoc(target)` is unavailable.

    Returns:
        Non-empty summary documentation line.
    """
    raw_doc = inspect.getdoc(target)
    if raw_doc:
        return raw_doc.splitlines()[0].strip()
    return f"{fallback_prefix} {name}."


def _inspect_ir_symbol(name: str, target: LivePythonTarget) -> GhostRef:
    """Build a `GhostRef` for a single exported `ml_switcheroo_ir` symbol.

    Args:
        name: Public symbol name in `ml_switcheroo_ir`.
        target: Runtime class, enum, protocol, type alias, constant, or function object.

    Returns:
        Constructed `GhostRef` with live parameters, return type, and docstring.
    """
    classified = _classify_ir_symbol(name, target)
    kind: str
    summary_doc: str
    params: list[GhostParam] = []
    returns_type: str | None = None

    if isinstance(classified, _IrClassSymbol):
        cls_target = classified.target
        kind = "class"
        summary_doc = _extract_summary_doc(name, cls_target, "IR class")
        sig = get_runtime_signature(cls_target)
        params = [_build_ir_ghost_param(p) for p in sig.parameters.values()]
    elif isinstance(classified, _IrFunctionSymbol):
        fn_target = classified.target
        kind = "function"
        summary_doc = _extract_summary_doc(name, fn_target, "IR function")
        sig = get_runtime_signature(fn_target)
        params = [_build_ir_ghost_param(p) for p in sig.parameters.values()]
        if sig.return_annotation is not inspect.Parameter.empty:
            raw_ret = sig.return_annotation if isinstance(sig.return_annotation, str) else str(sig.return_annotation)
            returns_type = sanitize_type_str(raw_ret)
    elif isinstance(classified, _IrEnumOrExceptionSymbol):
        kind = "class"
        summary_doc = _extract_summary_doc(name, classified.target, "IR symbol")
    elif isinstance(classified, _IrTypeAliasOrNewTypeSymbol):
        kind = "class"
        summary_doc = f"IR type definition {name}."
    elif isinstance(classified, _IrConstantWitnessSymbol):
        kind = "class"
        summary_doc = _extract_summary_doc(name, classified.target, "IR constant")
    else:
        typing_extensions.assert_never(classified)

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


extract_ir_summary_doc = _extract_summary_doc
inspect_ir_symbol = _inspect_ir_symbol

__all__ = [
    "IR_EXPORTED_SYMBOLS",
    "collect_api",
    "extract_ir_summary_doc",
    "inspect_ir_symbol",
]
