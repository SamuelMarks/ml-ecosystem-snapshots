"""Shared declarative collector utilities for ML ecosystem framework extractors.

Consolidates canonical-or-live Python symbol collection, submodule-backed
canonical augmentation, and SSA/operation specification hydration across
framework collector modules.
"""

from __future__ import annotations

import enum
import logging
import types
import typing

import griffe
import pydantic
import typing_extensions
from ml_switcheroo_ir.schema.ghost import (
    ExtendedGhostParam,
    GhostOperationRef,
    GhostParam,
    GhostPythonRef,
    GhostRef,
    GhostResult,
    IRParameterRole,
    OperandDirection,
    ParameterKind,
)

from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    STATIC_GRIFFE_TYPES,
    LivePythonTarget,
    StaticGriffeNode,
    get_member_dict,
)
from ml_ecosystem_snapshots.introspection.static_parsers import (
    load_griffe_module_safely,
)
from ml_ecosystem_snapshots.models import JsonValue, ScalarValue, SymbolKind
from ml_ecosystem_snapshots.network_env import is_offline_mode

_LOGGER = logging.getLogger(__name__)


class CollectorDomain(str, enum.Enum):
    """Domain classification for snapshot collectors across Python, compiler IR, hardware ISA, C++, and DSL targets."""

    PYTHON_FRAMEWORK = "python_framework"
    COMPILER_IR = "compiler_ir"
    HARDWARE_ISA = "hardware_isa"
    CPP_RUNTIME = "cpp_runtime"
    DOCUMENT_DSL = "document_dsl"


@typing.runtime_checkable
class DomainSnapshotCollector(typing.Protocol):
    """Protocol for non-Python domain collectors (`COMPILER_IR`, `HARDWARE_ISA`, `CPP_RUNTIME`, `DOCUMENT_DSL`)."""

    collector_domain: CollectorDomain
    category_key: str

    def collect_domain(self, include_nonpublic: bool = False) -> list[GhostRef]:
        """Collect domain-native `GhostRef` symbols without `SemanticTier` shoehorning.

        Args:
            include_nonpublic: Whether to include non-public symbols.

        Returns:
            List of collected domain `GhostRef` items.
        """
        ...


class DomainCollectorAdapter:
    """Concrete `DomainSnapshotCollector` wrapping a domain-native collection callable."""

    def __init__(
        self,
        collector_domain: CollectorDomain,
        category_key: str,
        collect_fn: typing.Callable[[bool], list[GhostRef]],
    ) -> None:
        """Initialize the domain collector adapter.

        Args:
            collector_domain: Target `CollectorDomain` classification.
            category_key: Snapshot category key emitted for this domain (e.g. `'operations'`, `'instructions'`).
            collect_fn: Callable accepting `include_nonpublic` and returning `list[GhostRef]`.
        """
        self.collector_domain = collector_domain
        self.category_key = category_key
        self._collect_fn = collect_fn

    def collect_domain(self, include_nonpublic: bool = False) -> list[GhostRef]:
        """Execute domain-native symbol collection.

        Args:
            include_nonpublic: Whether to include non-public symbols.

        Returns:
            List of collected `GhostRef` items.
        """
        return self._collect_fn(include_nonpublic)


class CanonicalParamSpec(pydantic.BaseModel):
    """Pydantic specification for a parameter inside a canonical Python op."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    name: str
    kind: str = "POSITIONAL_OR_KEYWORD"
    default: str | None = None


class CanonicalPythonOpSpec(pydantic.BaseModel):
    """Pydantic specification for a canonical Python operation entry."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    name: str
    api_path: str
    kind: str = "function"
    params: list[CanonicalParamSpec] = pydantic.Field(default_factory=list)
    docstring: str | None = None


class CanonicalSubmoduleOpSpec(CanonicalPythonOpSpec):
    """Pydantic specification for a submodule-backed canonical Python operation entry."""

    submodule: str = ""


CanonicalParamTuple: typing_extensions.TypeAlias = "tuple[str, str, str | None]"
CanonicalPythonOpTuple: typing_extensions.TypeAlias = "tuple[str, str, str, list[CanonicalParamTuple], str]"
CanonicalSubmoduleOpTuple: typing_extensions.TypeAlias = "tuple[str, str, str, str, list[CanonicalParamTuple], str]"


def build_canonical_python_ops(
    defs: typing.Sequence[CanonicalPythonOpTuple],
) -> list[CanonicalPythonOpSpec]:
    """Convert compact `(name, api_path, kind, params, docstring)` tuples into `CanonicalPythonOpSpec` models.

    Args:
        defs: Sequence of `(name, api_path, kind, params, docstring)` tuples.

    Returns:
        List of validated `CanonicalPythonOpSpec` models.
    """
    return [
        CanonicalPythonOpSpec(
            name=name,
            api_path=api_path,
            kind=kind,
            params=[CanonicalParamSpec(name=pn, kind=pk, default=pd) for pn, pk, pd in params],
            docstring=doc,
        )
        for name, api_path, kind, params, doc in defs
    ]


def build_canonical_submodule_ops(
    defs: typing.Sequence[CanonicalSubmoduleOpTuple],
) -> list[CanonicalSubmoduleOpSpec]:
    """Convert compact `(name, api_path, kind, submodule, params, docstring)` tuples into `CanonicalSubmoduleOpSpec` models.

    Args:
        defs: Sequence of `(name, api_path, kind, submodule, params, docstring)` tuples.

    Returns:
        List of validated `CanonicalSubmoduleOpSpec` models.
    """
    return [
        CanonicalSubmoduleOpSpec(
            name=name,
            api_path=api_path,
            kind=kind,
            submodule=submodule,
            params=[CanonicalParamSpec(name=pn, kind=pk, default=pd) for pn, pk, pd in params],
            docstring=doc,
        )
        for name, api_path, kind, submodule, params, doc in defs
    ]


class CanonicalSsaAttrSpec(pydantic.BaseModel):
    """Pydantic specification for an attribute inside a canonical SSA op."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    name: str
    type: str = "any"
    default: ScalarValue | list[ScalarValue] = None
    required: bool = False
    description: str | None = None
    options: list[ScalarValue] | None = None


class CanonicalSsaOpSpec(pydantic.BaseModel):
    """Pydantic specification for a canonical SSA / hardware operation entry."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    name: str = ""
    mnemonic: str = ""
    domain: str = ""
    category: str = ""
    address_space: str = ""
    since_version: int | None = None
    inputs: list[str] = pydantic.Field(default_factory=list)
    outputs: list[str] = pydantic.Field(default_factory=list)
    attributes: list[CanonicalSsaAttrSpec] = pydantic.Field(default_factory=list)
    description: str = ""


def _coerce_python_op(
    op: CanonicalPythonOpSpec | typing.Mapping[str, JsonValue],
) -> CanonicalPythonOpSpec:
    """Coerce a dictionary or `CanonicalPythonOpSpec` into `CanonicalPythonOpSpec`.

    Args:
        op: Operation specification model or mapping.

    Returns:
        Validated `CanonicalPythonOpSpec`.
    """
    return op if isinstance(op, CanonicalPythonOpSpec) else CanonicalPythonOpSpec.model_validate(dict(op))


def _coerce_submodule_op(
    op: CanonicalSubmoduleOpSpec | typing.Mapping[str, JsonValue],
) -> CanonicalSubmoduleOpSpec:
    """Coerce a dictionary or `CanonicalSubmoduleOpSpec` into `CanonicalSubmoduleOpSpec`.

    Args:
        op: Submodule operation specification model or mapping.

    Returns:
        Validated `CanonicalSubmoduleOpSpec`.
    """
    return op if isinstance(op, CanonicalSubmoduleOpSpec) else CanonicalSubmoduleOpSpec.model_validate(dict(op))


def _coerce_ssa_op(
    op: CanonicalSsaOpSpec | typing.Mapping[str, JsonValue],
) -> CanonicalSsaOpSpec:
    """Coerce a dictionary or `CanonicalSsaOpSpec` into `CanonicalSsaOpSpec`.

    Args:
        op: SSA operation specification model or mapping.

    Returns:
        Validated `CanonicalSsaOpSpec`.
    """
    return op if isinstance(op, CanonicalSsaOpSpec) else CanonicalSsaOpSpec.model_validate(dict(op))


def build_canonical_python_ref(
    op: CanonicalPythonOpSpec | typing.Mapping[str, JsonValue],
    default_doc_prefix: str,
    environment_tags: typing.Sequence[str],
    domain_metadata: dict[str, JsonValue] | None = None,
) -> GhostPythonRef:
    """Construct a `GhostPythonRef` from a canonical operation specification.

    Args:
        op: Specification describing `name`, `api_path`, `kind`, `params`, and `docstring`.
        default_doc_prefix: Prefix used when `docstring` is omitted in `op`.
        environment_tags: Sequence of environment tag strings to attach.
        domain_metadata: Optional dictionary of domain-specific metadata.

    Returns:
        Hydrated `GhostPythonRef` instance.
    """
    spec = _coerce_python_op(op)
    params = [
        GhostParam(
            name=p.name,
            kind=ParameterKind[p.kind],
            default=p.default,
        )
        for p in spec.params
    ]
    return GhostPythonRef(
        name=spec.name,
        api_path=spec.api_path,
        kind=spec.kind,
        params=params,
        docstring=spec.docstring or f"{default_doc_prefix} {spec.name} API.",
        environment_tags=list(environment_tags),
        domain_metadata=dict(domain_metadata) if domain_metadata is not None else {},
    )


class StaticInspectorCallable(typing.Protocol):
    """Protocol for static Griffe node or live symbol inspection functions."""

    def __call__(
        self,
        obj: StaticGriffeNode,
        api_path: str,
        is_public: bool | None = None,
    ) -> GhostRef:
        """Inspect a static Griffe AST node into a `GhostRef`.

        Args:
            obj: Static Griffe AST node to inspect.
            api_path: Qualified API path string.
            is_public: Optional public visibility override.

        Returns:
            Extracted `GhostRef` instance.
        """
        ...


class InspectorCallable(typing.Protocol):
    """Protocol for callable symbol inspection functions."""

    def __call__(
        self,
        obj: LivePythonTarget,
        api_path: str,
        is_public: bool | None = None,
    ) -> GhostRef:
        """Inspect a live Python runtime symbol into a `GhostRef`.

        Args:
            obj: Live Python symbol to inspect.
            api_path: Qualified API path string.
            is_public: Optional public visibility override.

        Returns:
            Extracted `GhostRef` record.
        """
        ...


def collect_from_canonical_or_live(
    live_module: types.ModuleType | None,
    module_prefix: str,
    target_names: typing.Sequence[str],
    canonical_ops: typing.Sequence[CanonicalPythonOpSpec | typing.Mapping[str, JsonValue]],
    default_doc_prefix: str,
    environment_tags: typing.Sequence[str],
    inspector_fn: InspectorCallable | None = None,
) -> list[GhostRef]:
    """Collect GhostRefs from a live module when installed, or fall back to canonical specs.

    Args:
        live_module: Imported runtime module or None when unavailable.
        module_prefix: Canonical dot-prefix (e.g. 'awkward', 'dpnp', 'numba').
        target_names: Attribute names to inspect on `live_module`.
        canonical_ops: Canonical fallback operation specifications.
        default_doc_prefix: Fallback docstring label when building canonical refs.
        environment_tags: Environment tags appended to each returned ref.
        inspector_fn: Optional custom inspection callable; defaults to `GhostInspector.inspect`.

    Returns:
        List of `GhostRef` objects.
    """
    if live_module is None:
        return [
            build_canonical_python_ref(
                op,
                default_doc_prefix=default_doc_prefix,
                environment_tags=environment_tags,
            )
            for op in canonical_ops
        ]

    inspect_callable = inspector_fn or GhostInspector.inspect
    results: list[GhostRef] = []
    live_members: dict[str, LivePythonTarget] = get_member_dict(live_module)
    for name in target_names:
        if name in live_members:
            obj: LivePythonTarget = live_members[name]
            try:
                ref = inspect_callable(obj, f"{module_prefix}.{name}", is_public=True)
                ref.environment_tags = list(ref.environment_tags or ()) + list(environment_tags)
                results.append(ref)
            except (
                ImportError,
                AttributeError,
                TypeError,
                ValueError,
                OSError,
                RuntimeError,
                LookupError,
                SyntaxError,
                AssertionError,
                griffe.GriffeError,
            ):
                _LOGGER.debug("Failed to inspect live member %s.%s", module_prefix, name, exc_info=True)
    return results


def collect_from_submodules_or_canonical(
    loaded_submods: typing.Mapping[str, types.ModuleType],
    canonical_ops: typing.Sequence[CanonicalSubmoduleOpSpec | typing.Mapping[str, JsonValue]],
    framework_tag: str,
    default_submodule: str,
    default_doc_prefix: str,
    inspector_fn: InspectorCallable | None = None,
) -> list[GhostRef]:
    """Collect GhostRefs across loaded submodules, backfilling missed ops from canonical specs.

    Args:
        loaded_submods: Mapping from submodule short name to imported module object.
        canonical_ops: Canonical fallback operation specifications.
        framework_tag: Primary environment tag (e.g. 'safetensors', 'scipy', 'torchaudio').
        default_submodule: Default submodule key when `op` omits `'submodule'`.
        default_doc_prefix: Prefix used when `docstring` is omitted in `op`.
        inspector_fn: Optional custom inspection callable; defaults to `GhostInspector.inspect`.

    Returns:
        List of `GhostRef` objects.
    """
    specs = [_coerce_submodule_op(op) for op in canonical_ops]
    if not loaded_submods:
        return [
            build_canonical_python_ref(
                spec,
                default_doc_prefix=default_doc_prefix,
                environment_tags=(
                    framework_tag,
                    spec.submodule or default_submodule,
                ),
                domain_metadata={"submodule": spec.submodule or default_submodule},
            )
            for spec in specs
        ]

    inspect_callable = inspector_fn or GhostInspector.inspect
    results: list[GhostRef] = []
    seen: set[str] = set()

    for spec in specs:
        sub = spec.submodule or default_submodule
        if sub in loaded_submods:
            mod = loaded_submods[sub]
            mod_members: dict[str, LivePythonTarget] = get_member_dict(mod)
            if spec.name in mod_members:
                obj: LivePythonTarget = mod_members[spec.name]
                try:
                    ref = inspect_callable(obj, spec.api_path, is_public=True)
                    ref.environment_tags = list(ref.environment_tags or ()) + [
                        framework_tag,
                        sub,
                    ]
                    results.append(ref)
                    seen.add(spec.api_path)
                except (
                    ImportError,
                    AttributeError,
                    TypeError,
                    ValueError,
                    OSError,
                    RuntimeError,
                    LookupError,
                    SyntaxError,
                    AssertionError,
                    griffe.GriffeError,
                ):
                    _LOGGER.debug("Failed to inspect submodule member %s", spec.api_path, exc_info=True)

    for spec in specs:
        if spec.api_path not in seen:
            results.append(
                build_canonical_python_ref(
                    spec,
                    default_doc_prefix=default_doc_prefix,
                    environment_tags=(
                        framework_tag,
                        spec.submodule or default_submodule,
                    ),
                    domain_metadata={"submodule": spec.submodule or default_submodule},
                )
            )

    return results


class _SsaRefOptions(pydantic.BaseModel):
    """Keyword options for constructing SSA operation `GhostOperationRef` records."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    attr_description_prefix: str
    result_type: str
    empty_return_type: str
    provenance_fn: typing.Callable[[CanonicalSsaOpSpec], typing.Mapping[str, JsonValue]]
    default_doc_fn: typing.Callable[[str], str]
    environment_tags: typing.Sequence[str]
    api_path_fn: typing.Callable[[CanonicalSsaOpSpec, str], str] | None = None


_SsaRefOptionValue: typing_extensions.TypeAlias = "str | typing.Sequence[str] | typing.Callable[[CanonicalSsaOpSpec], typing.Mapping[str, JsonValue]] | typing.Callable[[str], str] | typing.Callable[[CanonicalSsaOpSpec, str], str] | None"


def build_ssa_operation_refs(
    ops: typing.Sequence[CanonicalSsaOpSpec | typing.Mapping[str, JsonValue]],
    api_prefix: str,
    name_key: str,
    operand_annotation_fn: typing.Callable[[CanonicalSsaOpSpec, str], str],
    operand_description_fn: typing.Callable[[CanonicalSsaOpSpec, str], str],
    operand_direction_fn: typing.Callable[[CanonicalSsaOpSpec, str], OperandDirection],
    **options: _SsaRefOptionValue,
) -> list[GhostRef]:
    """Construct `GhostOperationRef` objects from SSA / hardware operation specs.

    Args:
        ops: Sequence of operation records.
        api_prefix: Prefix for `api_path` (e.g. 'cpp', 'metal', 'wasm', 'webgl').
        name_key: Dictionary key for operation identifier ('name' or 'mnemonic').
        operand_annotation_fn: Callback returning type annotation for `(op, inp)`.
        operand_description_fn: Callback returning description for `(op, inp)`.
        operand_direction_fn: Callback returning `OperandDirection` for `(op, inp)`.
        **options: Keyword options validated via `_SsaRefOptions` specifying `attr_description_prefix`,
            `result_type`, `empty_return_type`, `provenance_fn`, `default_doc_fn`,
            `environment_tags`, and optional `api_path_fn`.

    Returns:
        List of populated `GhostOperationRef` items.
    """
    opts = _SsaRefOptions.model_validate(options)
    attr_description_prefix = opts.attr_description_prefix
    result_type = opts.result_type
    empty_return_type = opts.empty_return_type
    provenance_fn = opts.provenance_fn
    default_doc_fn = opts.default_doc_fn
    environment_tags = opts.environment_tags
    api_path_fn = opts.api_path_fn

    refs: list[GhostRef] = []
    for raw_op in ops:
        spec = _coerce_ssa_op(raw_op)
        op_name = spec.mnemonic if name_key == "mnemonic" else spec.name
        api_path = api_path_fn(spec, op_name) if api_path_fn is not None else f"{api_prefix}.{op_name}"
        params: list[ExtendedGhostParam] = []

        for inp_str in spec.inputs:
            params.append(
                ExtendedGhostParam(
                    name=inp_str,
                    kind=ParameterKind.POSITIONAL_ONLY,
                    annotation=operand_annotation_fn(spec, inp_str),
                    standardized_name=inp_str,
                    description=operand_description_fn(spec, inp_str),
                    direction=operand_direction_fn(spec, inp_str),
                    role=IRParameterRole.OPERAND,
                )
            )

        attrs_meta: dict[str, JsonValue] = {}
        for attr in spec.attributes:
            attr_name = attr.name
            attr_type = attr.type
            attr_def = attr.default
            attr_desc = attr.description or f"{attr_description_prefix} {attr_name}"
            params.append(
                ExtendedGhostParam(
                    name=attr_name,
                    kind=ParameterKind.KEYWORD_ONLY,
                    annotation=attr_type,
                    default=str(attr_def) if attr_def is not None else None,
                    standardized_name=attr_name,
                    description=attr_desc,
                    role=IRParameterRole.ATTRIBUTE,
                )
            )
            attrs_meta[attr_name] = attr.model_dump(exclude_none=True)

        returns: list[GhostResult] = [GhostResult(name=out, type=result_type) for out in spec.outputs]
        returns_type = result_type if returns else empty_return_type
        params_list: list[GhostParam] = list(params)

        refs.append(
            GhostOperationRef(
                name=op_name,
                api_path=api_path,
                kind=SymbolKind.OPERATION.value,
                params=params_list,
                operands=[p for p in params if p.role == IRParameterRole.OPERAND],
                returns=returns,
                returns_type=returns_type,
                docstring=spec.description or default_doc_fn(op_name),
                environment_tags=list(environment_tags),
                domain_metadata=dict(provenance_fn(spec)),
                attributes=attrs_meta if attrs_meta else None,
            )
        )
    return refs


def collect_dir_callables(
    mod_obj: types.ModuleType | type,
    api_prefix: str,
    include_nonpublic: bool = False,
    inspector_fn: InspectorCallable | None = None,
) -> list[GhostRef]:
    """Iterate `dir(mod_obj)` and inspect callable attributes into `GhostRef` items.

    Args:
        mod_obj: Target module or namespace object.
        api_prefix: Qualified dot prefix (e.g. 'cupy', 'dask.array').
        include_nonpublic: Whether to include underscore-prefixed symbols.
        inspector_fn: Optional custom inspection callable; defaults to `GhostInspector.inspect`.

    Returns:
        List of `GhostRef` objects.
    """
    inspect_callable = inspector_fn or GhostInspector.inspect
    results: list[GhostRef] = []
    for name, obj in get_member_dict(mod_obj).items():
        if not include_nonpublic and name.startswith("_"):
            continue
        if callable(obj):
            try:
                res = inspect_callable(obj, f"{api_prefix}.{name}", is_public=not name.startswith("_"))
                results.append(res)
            except (
                ImportError,
                AttributeError,
                TypeError,
                ValueError,
                OSError,
                RuntimeError,
                LookupError,
                SyntaxError,
                AssertionError,
                griffe.GriffeError,
            ):
                _LOGGER.debug("Failed to inspect callable member %s.%s", api_prefix, name, exc_info=True)
    return results


def scan_griffe_module_members(
    module_path: str,
    api_prefix: str,
    include_nonpublic: bool = False,
    allowed_kinds: tuple[str, ...] = ("function", "class"),
    block_list: set[str] | None = None,
    environment_tags: tuple[str, ...] = ("cpu",),
    framework_name: str | None = None,
    inspector_fn: StaticInspectorCallable | None = None,
) -> list[GhostRef]:
    """Statically scan a module via the custom Griffe fork and extract `GhostRef` items.

    Args:
        module_path: Dotted Python module path to load via Griffe (e.g. 'chex', 'optax.losses').
        api_prefix: Canonical API prefix for emitted `GhostRef` paths.
        include_nonpublic: Whether to include underscore-prefixed private members.
        allowed_kinds: Tuple of allowed symbol kinds ('function', 'class').
        block_list: Optional set of member names to skip.
        environment_tags: Environment tags attached to each extracted `GhostRef`.
        framework_name: Optional framework identifier for docstring parser selection.
        inspector_fn: Optional custom inspection callable; defaults to `GhostInspector.inspect`.

    Returns:
        List of statically extracted `GhostRef` records.
    """
    loaded = load_griffe_module_safely(module_path, framework_name=framework_name)
    if not isinstance(loaded, STATIC_GRIFFE_TYPES):
        return []

    blocked = block_list if block_list is not None else set()
    inspect_callable = inspector_fn or GhostInspector.inspect
    results: list[GhostRef] = []

    for name, member in loaded.members.items():
        if (not include_nonpublic and name.startswith("_")) or name in blocked:
            continue
        try:
            is_cls = bool(member.is_class)
            is_fn = bool(member.is_function)
        except (
            ImportError,
            AttributeError,
            TypeError,
            ValueError,
            OSError,
            RuntimeError,
            LookupError,
            SyntaxError,
            AssertionError,
            griffe.GriffeError,
        ):
            _LOGGER.debug("Failed to read Griffe member kind for %s.%s", module_path, name, exc_info=True)
            continue

        if ("class" in allowed_kinds and is_cls) or ("function" in allowed_kinds and is_fn):
            try:
                ref = inspect_callable(
                    member,
                    f"{api_prefix}.{name}",
                    is_public=not name.startswith("_"),
                )
                ref.environment_tags = list(ref.environment_tags or ()) + [
                    t for t in environment_tags if t not in (ref.environment_tags or ())
                ]
                results.append(ref)
            except (
                ImportError,
                AttributeError,
                TypeError,
                ValueError,
                OSError,
                RuntimeError,
                LookupError,
                SyntaxError,
                AssertionError,
                griffe.GriffeError,
            ):
                _LOGGER.debug("Failed to inspect Griffe member %s.%s", api_prefix, name, exc_info=True)

    return results


class HybridFrameworkConfig(pydantic.BaseModel):
    """Configuration for the 3-tier (`Live -> Griffe Static AST -> Canonical Fallback`) collector pipeline."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    canonical_ops: list[CanonicalSubmoduleOpSpec]
    default_doc_prefix: str
    default_submodule: str
    environment_tags: tuple[str, ...]
    live_refs: list[GhostRef] = pydantic.Field(default_factory=list)
    griffe_refs: list[GhostRef] = pydantic.Field(default_factory=list)
    postprocess_fn: typing.Callable[[GhostRef], None] | None = None


def collect_hybrid_framework_api(config: HybridFrameworkConfig) -> list[GhostRef]:
    """Execute the standardized 3-tier (`Live -> Griffe Static AST -> Canonical Fallback`) discovery pipeline.

    Args:
        config: Validated `HybridFrameworkConfig` specifying live refs, Griffe refs, and canonical fallbacks.

    Returns:
        Deduplicated list of `GhostRef` items across all three discovery tiers.
    """
    results: list[GhostRef] = []
    seen_paths: set[str] = set()
    env_tags = config.environment_tags

    def _record(ref: GhostRef, merge_env: bool) -> None:
        """Append `ref` if unseen, merging environment tags and applying postprocess hook.

        Args:
            ref: Candidate `GhostRef` to record.
            merge_env: Whether to merge `env_tags` onto `ref.environment_tags`.
        """
        if ref.api_path in seen_paths:
            return
        if merge_env:
            ref.environment_tags = list(ref.environment_tags or ()) + [
                t for t in env_tags if t not in (ref.environment_tags or ())
            ]
        if config.postprocess_fn is not None:
            config.postprocess_fn(ref)
        results.append(ref)
        seen_paths.add(ref.api_path)

    if not is_offline_mode():
        for live_ref in config.live_refs:
            _record(live_ref, merge_env=True)
        for griffe_ref in config.griffe_refs:
            _record(griffe_ref, merge_env=False)

    for spec in config.canonical_ops:
        if spec.api_path not in seen_paths:
            fallback_ref = build_canonical_python_ref(
                spec,
                default_doc_prefix=config.default_doc_prefix,
                environment_tags=env_tags,
                domain_metadata={"submodule": spec.submodule or config.default_submodule},
            )
            _record(fallback_ref, merge_env=False)

    return results


__all__ = [
    "CanonicalParamSpec",
    "CanonicalParamTuple",
    "CanonicalPythonOpSpec",
    "CanonicalPythonOpTuple",
    "CanonicalSsaAttrSpec",
    "CanonicalSsaOpSpec",
    "CanonicalSubmoduleOpSpec",
    "CanonicalSubmoduleOpTuple",
    "CollectorDomain",
    "DomainCollectorAdapter",
    "DomainSnapshotCollector",
    "HybridFrameworkConfig",
    "InspectorCallable",
    "StaticInspectorCallable",
    "SymbolKind",
    "build_canonical_python_ops",
    "build_canonical_python_ref",
    "build_canonical_submodule_ops",
    "build_ssa_operation_refs",
    "collect_hybrid_framework_api",
    "scan_griffe_module_members",
]
