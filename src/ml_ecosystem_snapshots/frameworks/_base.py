"""Shared declarative collector utilities for ML ecosystem framework extractors.

Consolidates canonical-or-live Python symbol collection, submodule-backed
canonical augmentation, and SSA/operation specification hydration across
framework collector modules.
"""

from __future__ import annotations

import types
import typing

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
import pydantic

from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.models import JsonValue, ScalarValue


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

    @typing.overload
    def __getitem__(self, key: typing.Literal["name", "api_path", "kind"]) -> str: ...

    @typing.overload
    def __getitem__(
        self, key: typing.Literal["params"]
    ) -> list[CanonicalParamSpec]: ...

    @typing.overload
    def __getitem__(self, key: str) -> str | list[CanonicalParamSpec] | JsonValue: ...

    def __getitem__(self, key: str) -> str | list[CanonicalParamSpec] | JsonValue:
        """Retrieve a canonical operation field by dictionary key.

        Args:
            key: Field name.

        Returns:
            Field value.
        """
        val: str | list[CanonicalParamSpec] | JsonValue = getattr(self, key)
        return val

    def get(
        self,
        key: str,
        default: str | list[CanonicalParamSpec] | None = None,
    ) -> str | list[CanonicalParamSpec] | None:
        """Retrieve a canonical operation field by key with default.

        Args:
            key: Field name.
            default: Default value when absent.

        Returns:
            Field value or `default`.
        """
        val: str | list[CanonicalParamSpec] | None = getattr(self, key, default)
        return val


class CanonicalSubmoduleOpSpec(CanonicalPythonOpSpec):
    """Pydantic specification for a submodule-backed canonical Python operation entry."""

    submodule: str = ""


class CanonicalSsaAttrSpec(pydantic.BaseModel):
    """Pydantic specification for an attribute inside a canonical SSA op."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    name: str
    type: str = "any"
    default: ScalarValue | list[ScalarValue] = None
    required: bool = False
    description: str | None = None
    options: list[ScalarValue] | None = None

    def __getitem__(self, key: str) -> JsonValue:
        """Retrieve an SSA attribute field by dictionary key.

        Args:
            key: Field name.

        Returns:
            Field value.
        """
        val: JsonValue = getattr(self, key)
        return val


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

    @typing.overload
    def __getitem__(
        self,
        key: typing.Literal[
            "name", "mnemonic", "domain", "category", "address_space", "description"
        ],
    ) -> str: ...

    @typing.overload
    def __getitem__(self, key: typing.Literal["inputs", "outputs"]) -> list[str]: ...

    @typing.overload
    def __getitem__(
        self, key: typing.Literal["attributes"]
    ) -> list[CanonicalSsaAttrSpec]: ...

    @typing.overload
    def __getitem__(
        self, key: str
    ) -> str | list[str] | list[CanonicalSsaAttrSpec] | JsonValue: ...

    def __getitem__(
        self, key: str
    ) -> str | list[str] | list[CanonicalSsaAttrSpec] | JsonValue:
        """Retrieve an SSA operation field by dictionary key.

        Args:
            key: Field name.

        Returns:
            Field value.
        """
        val: str | list[str] | list[CanonicalSsaAttrSpec] | JsonValue = getattr(
            self, key
        )
        return val

    @typing.overload
    def get(
        self,
        key: typing.Literal[
            "name", "mnemonic", "domain", "category", "address_space", "description"
        ],
        default: str | None = None,
    ) -> str: ...

    @typing.overload
    def get(
        self,
        key: typing.Literal["inputs", "outputs"],
        default: list[str] | None = None,
    ) -> list[str]: ...

    @typing.overload
    def get(
        self,
        key: typing.Literal["attributes"],
        default: list[CanonicalSsaAttrSpec] | None = None,
    ) -> list[CanonicalSsaAttrSpec]: ...

    @typing.overload
    def get(
        self,
        key: str,
        default: list[str] | list[CanonicalSsaAttrSpec] | JsonValue = None,
    ) -> list[str] | list[CanonicalSsaAttrSpec] | JsonValue: ...

    def get(
        self,
        key: str,
        default: list[str] | list[CanonicalSsaAttrSpec] | JsonValue = None,
    ) -> list[str] | list[CanonicalSsaAttrSpec] | JsonValue:
        """Retrieve an SSA operation field by key with an optional default.

        Args:
            key: Field name.
            default: Default value if field is empty/None.

        Returns:
            Field value or `default`.
        """
        val: list[str] | list[CanonicalSsaAttrSpec] | JsonValue = getattr(
            self, key, None
        )
        return val if val is not None else default


def _coerce_python_op(
    op: CanonicalPythonOpSpec | typing.Mapping[str, JsonValue],
) -> CanonicalPythonOpSpec:
    """Coerce a dictionary or `CanonicalPythonOpSpec` into `CanonicalPythonOpSpec`.

    Args:
        op: Operation specification model or mapping.

    Returns:
        Validated `CanonicalPythonOpSpec`.
    """
    return (
        op
        if isinstance(op, CanonicalPythonOpSpec)
        else CanonicalPythonOpSpec.model_validate(dict(op))
    )


def _coerce_submodule_op(
    op: CanonicalSubmoduleOpSpec | typing.Mapping[str, JsonValue],
) -> CanonicalSubmoduleOpSpec:
    """Coerce a dictionary or `CanonicalSubmoduleOpSpec` into `CanonicalSubmoduleOpSpec`.

    Args:
        op: Submodule operation specification model or mapping.

    Returns:
        Validated `CanonicalSubmoduleOpSpec`.
    """
    return (
        op
        if isinstance(op, CanonicalSubmoduleOpSpec)
        else CanonicalSubmoduleOpSpec.model_validate(dict(op))
    )


def _coerce_ssa_op(
    op: CanonicalSsaOpSpec | typing.Mapping[str, JsonValue],
) -> CanonicalSsaOpSpec:
    """Coerce a dictionary or `CanonicalSsaOpSpec` into `CanonicalSsaOpSpec`.

    Args:
        op: SSA operation specification model or mapping.

    Returns:
        Validated `CanonicalSsaOpSpec`.
    """
    return (
        op
        if isinstance(op, CanonicalSsaOpSpec)
        else CanonicalSsaOpSpec.model_validate(dict(op))
    )


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
            kind=getattr(ParameterKind, p.kind),
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


class InspectorCallable(typing.Protocol):
    """Protocol for callable symbol inspection functions."""

    def __call__(
        self,
        obj: object,
        api_path: str,
        is_public: bool | None = None,
    ) -> GhostRef:
        """Inspect a live Python object into a `GhostRef`.

        Args:
            obj: Live Python object to inspect.
            api_path: Qualified API path string.
            is_public: Optional public visibility override.

        Returns:
            Extracted `GhostRef` record.
        """
        ...


def collect_from_canonical_or_live(
    live_module: types.ModuleType | object | None,
    module_prefix: str,
    target_names: typing.Sequence[str],
    canonical_ops: typing.Sequence[
        CanonicalPythonOpSpec | typing.Mapping[str, JsonValue]
    ],
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
    for name in target_names:
        if hasattr(live_module, name):
            obj = getattr(live_module, name)
            try:
                ref = inspect_callable(obj, f"{module_prefix}.{name}", is_public=True)
                ref.environment_tags = list(ref.environment_tags or ()) + list(
                    environment_tags
                )
                results.append(ref)
            except Exception:
                pass
    return results


def collect_from_submodules_or_canonical(
    loaded_submods: typing.Mapping[str, types.ModuleType | object],
    canonical_ops: typing.Sequence[
        CanonicalSubmoduleOpSpec | typing.Mapping[str, JsonValue]
    ],
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
            if hasattr(mod, spec.name):
                obj = getattr(mod, spec.name)
                try:
                    ref = inspect_callable(obj, spec.api_path, is_public=True)
                    ref.environment_tags = list(ref.environment_tags or ()) + [
                        framework_tag,
                        sub,
                    ]
                    results.append(ref)
                    seen.add(spec.api_path)
                except Exception:
                    pass

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


def build_ssa_operation_refs(
    ops: typing.Sequence[CanonicalSsaOpSpec | typing.Mapping[str, JsonValue]],
    api_prefix: str,
    name_key: str,
    operand_annotation_fn: typing.Callable[[CanonicalSsaOpSpec, str], str],
    operand_description_fn: typing.Callable[[CanonicalSsaOpSpec, str], str],
    operand_direction_fn: typing.Callable[[CanonicalSsaOpSpec, str], OperandDirection],
    attr_description_prefix: str,
    result_type: str,
    empty_return_type: str,
    provenance_fn: typing.Callable[[CanonicalSsaOpSpec], dict[str, JsonValue]],
    default_doc_fn: typing.Callable[[str], str],
    environment_tags: typing.Sequence[str],
    api_path_fn: typing.Callable[[CanonicalSsaOpSpec, str], str] | None = None,
) -> list[GhostRef]:
    """Construct `GhostOperationRef` objects from SSA / hardware operation specs.

    Args:
        ops: Sequence of operation records.
        api_prefix: Prefix for `api_path` (e.g. 'cpp', 'metal', 'wasm', 'webgl').
        name_key: Dictionary key for operation identifier ('name' or 'mnemonic').
        operand_annotation_fn: Callback returning type annotation for `(op, inp)`.
        operand_description_fn: Callback returning description for `(op, inp)`.
        operand_direction_fn: Callback returning `OperandDirection` for `(op, inp)`.
        attr_description_prefix: Prefix for attribute descriptions (e.g. 'C++ attribute').
        result_type: Result type string when outputs exist (e.g. 'auto', 'T', 'v128', 'vec4').
        empty_return_type: Return type string when no outputs exist (e.g. 'void', 'None').
        provenance_fn: Callback returning provenance metadata dictionary for `op`.
        default_doc_fn: Callback returning fallback docstring for `op_name`.
        environment_tags: Sequence of environment tag strings.
        api_path_fn: Optional callback returning `api_path` for `(op, op_name)`.

    Returns:
        List of populated `GhostOperationRef` items.
    """
    refs: list[GhostRef] = []
    for raw_op in ops:
        spec = _coerce_ssa_op(raw_op)
        op_name = spec.mnemonic if name_key == "mnemonic" else spec.name
        api_path = (
            api_path_fn(spec, op_name)
            if api_path_fn is not None
            else f"{api_prefix}.{op_name}"
        )
        params: list[GhostParam] = []

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

        returns: list[GhostResult] = [
            GhostResult(name=out, type=result_type) for out in spec.outputs
        ]
        returns_type = result_type if returns else empty_return_type

        refs.append(
            GhostOperationRef(
                name=op_name,
                api_path=api_path,
                kind="function",
                params=params,
                operands=[
                    p
                    for p in params
                    if getattr(p, "role", None) == IRParameterRole.OPERAND
                ],
                returns=returns,
                returns_type=returns_type,
                docstring=spec.description or default_doc_fn(op_name),
                environment_tags=list(environment_tags),
                domain_metadata=provenance_fn(spec),
                attributes=attrs_meta if attrs_meta else None,
            )
        )
    return refs


def collect_dir_callables(
    mod_obj: types.ModuleType | object,
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
    for name in dir(mod_obj):
        if not include_nonpublic and name.startswith("_"):
            continue
        obj = getattr(mod_obj, name)
        if callable(obj):
            try:
                res = inspect_callable(
                    obj, f"{api_prefix}.{name}", is_public=not name.startswith("_")
                )
                results.append(res)
            except Exception:
                pass
    return results
