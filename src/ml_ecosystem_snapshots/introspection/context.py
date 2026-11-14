"""Shared data models and parameter extraction contexts for `GhostInspector`."""

from __future__ import annotations

import ast
import inspect
import types
import typing

import pydantic
import typing_extensions
from ml_switcheroo_ir.schema.ghost import ParameterKind

from ml_ecosystem_snapshots.c_extension_sig import (
    CExtensionSignature,
)
from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    LivePythonTarget,
    StaticGriffeNode,
    get_member_dict,
)
from ml_ecosystem_snapshots.introspection.sanitize import sanitize_type_str
from ml_ecosystem_snapshots.models import JsonValue, ScalarValue


def _get_target_module_name(target: LivePythonTarget | StaticGriffeNode) -> str:
    """Return the `__module__` string of a runtime target or empty string.

    Args:
        target: Runtime object or callable.

    Returns:
        Module name string or `""`.
    """
    mod = get_member_dict(target).get("__module__")
    return str(mod) if mod else ""


RawParamTuple: typing_extensions.TypeAlias = tuple[str, str, "str | None", "str | None"]

ParamDefaultAtom: typing_extensions.TypeAlias = (
    "JsonValue | ast.AST | bytes | set[ScalarValue] | frozenset[ScalarValue] | tuple[ScalarValue, ...]"
)
ParamDefaultValue: typing_extensions.TypeAlias = (
    "ParamDefaultAtom | tuple[ParamDefaultAtom, ...] | list[ParamDefaultAtom] | dict[str, ParamDefaultAtom]"
)


class ExtractedParamRecord(pydantic.BaseModel):
    """Intermediate extracted parameter metadata from CDD or Griffe docstrings/AST."""

    model_config = pydantic.ConfigDict(extra="allow", arbitrary_types_allowed=True)

    doc: str | None = None
    typ: str | None = None
    default: (
        JsonValue
        | ast.AST
        | bytes
        | set[ScalarValue]
        | frozenset[ScalarValue]
        | tuple[ScalarValue, ...]
        | tuple[
            JsonValue | ast.AST | bytes | set[ScalarValue] | frozenset[ScalarValue] | tuple[ScalarValue, ...],
            ...,
        ]
        | list[JsonValue | ast.AST | bytes | set[ScalarValue] | frozenset[ScalarValue] | tuple[ScalarValue, ...]]
        | dict[str, JsonValue | ast.AST | bytes | set[ScalarValue] | frozenset[ScalarValue] | tuple[ScalarValue, ...]]
    ) = None

    def get(self, key: str, default: str | None = None) -> str | None:
        """Retrieve a string field (`doc`, `typ`, or `default`) by key.

        Args:
            key: Field name.
            default: Default value when field is None or absent.

        Returns:
            String representation or `default`.
        """
        field_map: dict[str, ParamDefaultValue] = {
            "doc": self.doc,
            "typ": self.typ,
            "default": self.default,
        }
        val = field_map.get(key)
        return str(val) if val is not None else default


class DocstringIrResult(pydantic.BaseModel):
    """Structured result returned by `_extract_docstring_ir`."""

    model_config = pydantic.ConfigDict(extra="forbid")

    cdd_params: dict[str, ExtractedParamRecord] = pydantic.Field(default_factory=dict)
    returns_type: str | None = None
    returns_description: str | None = None
    raises: list[str] = pydantic.Field(default_factory=list)


class ParamExtractionContext:
    """Bundled metadata and helper inputs for `_extract_raw_parameters`.

    Attributes:
        name: Unqualified symbol name.
        kind: Symbol kind ('class', 'function', 'method').
        api_path: Canonical dotted API path.
        cdd_ast_params: CDD AST parameter dictionary.
        griffe_node: Optional Griffe AST node.
        resolved_hints: Resolved `typing.get_type_hints` dictionary.
        returns_type: Current inferred return type string.
        environment_tags: Optional explicit execution environment tags.
        active_ast_mod: Active `ast` module reference.
        active_sanitize_type_str: Type sanitizer callable.
        active_extract_c_ext_sig: C-extension signature extractor callable.
    """

    name: str
    kind: str
    api_path: str
    cdd_ast_params: typing.Mapping[str, ExtractedParamRecord]
    griffe_node: StaticGriffeNode | None
    resolved_hints: typing.Mapping[str, type | types.GenericAlias | str]
    returns_type: str | None
    environment_tags: list[str] | None
    active_ast_mod: types.ModuleType
    active_sanitize_type_str: typing.Callable[[str | None], str | None]
    active_extract_c_ext_sig: typing.Callable[[LivePythonTarget | StaticGriffeNode, str], CExtensionSignature | None]

    def __init__(
        self,
        name: str,
        kind: str,
        api_path: str,
        cdd_ast_params: typing.Mapping[str, ExtractedParamRecord],
        griffe_node: StaticGriffeNode | None,
        resolved_hints: typing.Mapping[str, type | types.GenericAlias | str],
        **extra: str | list[str] | None,
    ) -> None:
        """Initialize parameter extraction context.

        Args:
            name: Unqualified symbol name.
            kind: Symbol kind ('class', 'function', 'method').
            api_path: Canonical dotted API path.
            cdd_ast_params: CDD AST parameter dictionary.
            griffe_node: Optional Griffe AST node.
            resolved_hints: Resolved `typing.get_type_hints` dictionary.
            **extra: Additional runtime context (`returns_type`, `environment_tags`).
        """
        self.name = name
        self.kind = kind
        self.api_path = api_path
        self.cdd_ast_params = cdd_ast_params
        self.griffe_node = griffe_node
        self.resolved_hints = resolved_hints
        raw_ret = extra.get("returns_type")
        self.returns_type = str(raw_ret) if isinstance(raw_ret, str) else None
        raw_env = extra.get("environment_tags")
        self.environment_tags = list(raw_env) if isinstance(raw_env, list) else None
        import ml_ecosystem_snapshots.introspection.inspector as _insp_mod

        self.active_ast_mod = _insp_mod.ast
        self.active_sanitize_type_str = sanitize_type_str
        self.active_extract_c_ext_sig = _insp_mod.extract_c_extension_signature


class OverloadExtractionContext:
    """Bundled inputs for extracting function or method overloads in `_extract_overloads`.

    Attributes:
        griffe_node: Optional Griffe AST node.
        c_ext_params: Optional parsed `CExtensionSignature` with `.overloads`.
        name: Unqualified symbol name.
        api_path: Canonical dotted API path.
        kind: Symbol kind ('function', 'class', 'method').
        determined_is_public: Public visibility flag.
        env_tags: Execution environment tags.
        is_torch_target: Whether symbol belongs to PyTorch.
        active_sanitize_type_str: Type annotation string sanitizer callable.
    """

    griffe_node: StaticGriffeNode | None
    c_ext_params: CExtensionSignature | None
    name: str
    api_path: str
    kind: str
    determined_is_public: bool
    env_tags: list[str]
    is_torch_target: bool
    active_sanitize_type_str: typing.Callable[[str | None], str | None]

    def __init__(
        self,
        griffe_node: StaticGriffeNode | None,
        c_ext_params: CExtensionSignature | None,
        name: str,
        api_path: str,
        kind: str,
        determined_is_public: bool,
        *,
        env_tags: list[str],
        is_torch_target: bool = False,
    ) -> None:
        """Initialize overload extraction context.

        Args:
            griffe_node: Optional Griffe AST node.
            c_ext_params: Optional parsed `CExtensionSignature` with `.overloads`.
            name: Unqualified symbol name.
            api_path: Canonical dotted API path.
            kind: Symbol kind ('function', 'class', 'method').
            determined_is_public: Public visibility flag.
            env_tags: List of runtime hardware/environment capability tags.
            is_torch_target: Whether the inspected target belongs to PyTorch.
        """
        self.griffe_node = griffe_node
        self.c_ext_params = c_ext_params
        self.name = name
        self.api_path = api_path
        self.kind = kind
        self.determined_is_public = determined_is_public
        self.env_tags = list(env_tags)
        self.is_torch_target = is_torch_target
        self.active_sanitize_type_str = sanitize_type_str


def to_parameter_kind(
    val: ParameterKind | inspect._ParameterKind | str,
) -> ParameterKind:
    """Normalize a parameter kind string or enum to `ParameterKind`.

    Args:
        val: Parameter kind string, enum, or `inspect.Parameter._ParameterKind`.

    Returns:
        The corresponding `ParameterKind` enum member.
    """
    if isinstance(val, ParameterKind):
        return val
    s = val.name.upper() if isinstance(val, inspect._ParameterKind) else str(val).upper()
    try:
        return ParameterKind[s]
    except KeyError:
        return ParameterKind.POSITIONAL_OR_KEYWORD


__all__ = [
    "DocstringIrResult",
    "ExtractedParamRecord",
    "OverloadExtractionContext",
    "ParamDefaultAtom",
    "ParamDefaultValue",
    "ParamExtractionContext",
    "RawParamTuple",
    "_get_target_module_name",
    "to_parameter_kind",
]
