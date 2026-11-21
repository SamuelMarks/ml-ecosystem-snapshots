"""Default value and type annotation sanitization utilities for Ghost introspection.

Models parameter default extraction as a closed algebraic sum type (`SanitizedDefault`)
and type-annotation normalization as a pure `Endo[str]` monoid pipeline.
"""

from __future__ import annotations

import ast
import inspect
import re
import threading
import typing

import pydantic
import typing_extensions
from ml_switcheroo_ir.typeclasses.algebra import Endo, Monoid, endo_monoid

from ml_ecosystem_snapshots.exceptions import SnapshotError
from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    DefaultValueInput,
    get_member_dict,
)
from ml_ecosystem_snapshots.utils import strip_sphinx_roles


class _SanitizedDefaultBase(pydantic.BaseModel):
    """Frozen base model for `SanitizedDefault` sum-type variants."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)


class MandatoryDefault(_SanitizedDefaultBase):
    """Variant indicating a parameter has no default value (`inspect.Parameter.empty`)."""


class LiteralDefault(_SanitizedDefaultBase):
    """Variant carrying a deterministic literal, class name, or scrubbed string default.

    Attributes:
        value_repr: Deterministic serialized string representation of the default value.
    """

    value_repr: str


class FactoryDefault(_SanitizedDefaultBase):
    """Variant carrying a factory placeholder and scrubbed callable/device descriptor.

    Attributes:
        placeholder: Deterministic placeholder emitted into snapshot parameter defaults.
        factory_repr: Scrubbed factory or constructor representation.
    """

    placeholder: str
    factory_repr: str


class UnrepresentableDefault(_SanitizedDefaultBase):
    """Variant indicating an optional default whose runtime representation failed safely."""


SanitizedDefault: typing_extensions.TypeAlias = (
    "MandatoryDefault | LiteralDefault | FactoryDefault | UnrepresentableDefault"
)


def default_value_or_none(sanitized: SanitizedDefault, unrepresentable_token: str = "<unrepresentable>") -> str | None:
    """Fold a `SanitizedDefault` sum-type value into its snapshot default string or `None`.

    Args:
        sanitized: Closed `SanitizedDefault` sum-type instance.
        unrepresentable_token: String emitted when `sanitized` is `UnrepresentableDefault`.

    Returns:
        `None` for `MandatoryDefault`, otherwise deterministic string representation.
    """
    if isinstance(sanitized, MandatoryDefault):
        return None
    if isinstance(sanitized, LiteralDefault):
        return sanitized.value_repr
    if isinstance(sanitized, FactoryDefault):
        return sanitized.placeholder
    if isinstance(sanitized, UnrepresentableDefault):
        return unrepresentable_token
    typing_extensions.assert_never(sanitized)


def _sanitize_callable_default(val: DefaultValueInput) -> SanitizedDefault:
    """Sanitize a callable parameter default value into `LiteralDefault` or `FactoryDefault`.

    Args:
        val: Callable default value from `inspect.Parameter`.

    Returns:
        `LiteralDefault` for class objects, otherwise `FactoryDefault`.
    """
    val_repr = repr(val)
    if val_repr.startswith("<class '") and val_repr.endswith("'>"):
        cls_name = val_repr[8:-2]
        return LiteralDefault(value_repr=cls_name)
    func_name = get_member_dict(val).get("__name__")
    factory_str = f"<function {func_name}>" if func_name else "<factory_default>"
    return FactoryDefault(placeholder="<factory_default>", factory_repr=factory_str)


def _sanitize_repr_default(val: DefaultValueInput) -> SanitizedDefault:
    """Sanitize a non-primitive, non-callable default value by scrubbing memory addresses.

    Args:
        val: Non-primitive default value object.

    Returns:
        `LiteralDefault` or `FactoryDefault` with memory addresses scrubbed.
    """
    val_repr = repr(val)
    m_dt = re.match(r"^<dtype:\s*'([^']+)'\s*>$", val_repr)
    if m_dt:
        return LiteralDefault(value_repr=f"tf.{m_dt.group(1)}")

    if "device(type=" in val_repr:
        m_dev = re.search(r"type='([^']+)'", val_repr)
        dev_type = m_dev.group(1) if m_dev else "cpu"
        return FactoryDefault(placeholder=repr(dev_type), factory_repr=val_repr)

    if " at 0x" in val_repr or re.search(r"\b0x[0-9a-fA-F]{4,}\b", val_repr):
        scrubbed = re.sub(r"\s*at\s*0x[0-9a-fA-F]+", "", val_repr)
        scrubbed = re.sub(r"\b0x[0-9a-fA-F]{4,}\b", "<addr>", scrubbed)
        if scrubbed.startswith("<") and scrubbed.endswith(">"):
            return FactoryDefault(placeholder="<factory_default>", factory_repr=scrubbed)
        return LiteralDefault(value_repr=scrubbed)

    val_str = str(val)
    if " at 0x" in val_str or re.search(r"\b0x[0-9a-fA-F]{4,}\b", val_str):
        return FactoryDefault(placeholder="<factory_default>", factory_repr="<factory_default>")
    return LiteralDefault(value_repr=val_str)


def sanitize_param_default(val: DefaultValueInput) -> SanitizedDefault:
    """Sanitize parameter default value into a closed `SanitizedDefault` sum-type value.

    Args:
        val: Raw default value from `inspect.Parameter` or AST.

    Returns:
        Closed `SanitizedDefault` variant (`MandatoryDefault`, `LiteralDefault`,
        `FactoryDefault`, or `UnrepresentableDefault`).
    """
    if val is inspect.Parameter.empty:
        return MandatoryDefault()
    if val is Ellipsis or val is None or isinstance(val, (bool, int, float)):
        return LiteralDefault(value_repr="..." if val is Ellipsis else str(val))
    if isinstance(val, str):
        return LiteralDefault(value_repr=repr(val))

    try:
        if callable(val):
            return _sanitize_callable_default(val)
        return _sanitize_repr_default(val)
    except (
        SnapshotError,
        ImportError,
        AttributeError,
        TypeError,
        ValueError,
        LookupError,
        OSError,
        RuntimeError,
        SyntaxError,
    ):
        return UnrepresentableDefault()


_PEP585_TYPE_MAP: typing.Final[dict[str, str]] = {
    "List": "list",
    "Dict": "dict",
    "Tuple": "tuple",
    "Set": "set",
    "Type": "type",
}


class _TypeHintSanitizer(ast.NodeTransformer):
    """AST transformer to normalize `Union`, `Optional`, and `typing.*` nodes to PEP-585/604."""

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        """Transform `Union[...]` and `Optional[...]` subscripts into bitwise-or (`|`) expressions.

        Args:
            node: The AST Subscript node to transform.

        Returns:
            The transformed AST node.
        """
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "Union":
            slice_val = node.slice
            elts = slice_val.elts if isinstance(slice_val, ast.Tuple) else [slice_val]
            if len(elts) >= 2:
                new_node: ast.expr = elts[0]
                for elt in elts[1:]:
                    new_node = ast.BinOp(left=new_node, op=ast.BitOr(), right=elt)
                return ast.copy_location(new_node, node)

        if isinstance(node.value, ast.Name) and node.value.id == "Optional":
            return ast.copy_location(
                ast.BinOp(left=node.slice, op=ast.BitOr(), right=ast.Constant(value=None)),
                node,
            )

        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        """Normalize capitalized typing generic names (`List` -> `list`, `Dict` -> `dict`).

        Args:
            node: The AST Name node to normalize.

        Returns:
            The normalized AST Name node.
        """
        if node.id in _PEP585_TYPE_MAP:
            return ast.copy_location(ast.Name(id=_PEP585_TYPE_MAP[node.id], ctx=node.ctx), node)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        """Flatten `typing.<Name>` attribute accesses into unqualified PEP-585 identifiers.

        Args:
            node: AST Attribute node to normalize.

        Returns:
            The transformed AST node.
        """
        if isinstance(node.value, ast.Name) and node.value.id == "typing":
            target_id = _PEP585_TYPE_MAP.get(node.attr, node.attr)
            return ast.copy_location(ast.Name(id=target_id, ctx=ast.Load()), node)
        self.generic_visit(node)
        return node


def _strip_sphinx_stage(raw: str) -> str:
    """Strip Sphinx markup roles from a type string (`Endo[str]` stage 1).

    Args:
        raw: Input type annotation string.

    Returns:
        String with Sphinx roles stripped.
    """
    return strip_sphinx_roles(raw) or ""


def _unwrap_class_tag_stage(raw: str) -> str:
    """Unwrap `<class 'pkg.Type'>` tags to `pkg.Type` (`Endo[str]` stage 2).

    Args:
        raw: Input type annotation string.

    Returns:
        String with `<class '...'>` wrappers removed.
    """
    if not raw:
        return raw
    return re.sub(r"<class '([^']+)'>", r"\1", raw)


def _scrub_memory_addr_stage(raw: str) -> str:
    """Scrub runtime hex memory addresses (`at 0x...`) from type strings (`Endo[str]` stage 3).

    Args:
        raw: Input type annotation string.

    Returns:
        String with runtime hex memory addresses removed.
    """
    if not raw or "0x" not in raw:
        return raw
    return re.sub(r"\s*at\s*0x[0-9a-fA-F]+", "", raw)


_AST_PARSE_LOCK: typing.Final[threading.Lock] = threading.Lock()


def _normalize_ast_pep604_stage(raw: str) -> str:
    """Normalize `Optional[T]` / `Union[A, B]` AST nodes to PEP-604 `A | B` syntax.

    Args:
        raw: Input type annotation string.

    Returns:
        PEP-585/604 normalized type string, or `raw` unchanged if non-Python syntax.
    """
    if not raw:
        return raw
    try:
        with _AST_PARSE_LOCK:
            node = ast.parse(raw, mode="eval")
        transformed = _TypeHintSanitizer().visit(node)
        return ast.unparse(transformed)
    except (
        SnapshotError,
        ImportError,
        AttributeError,
        TypeError,
        ValueError,
        LookupError,
        OSError,
        RuntimeError,
        SyntaxError,
    ):
        return raw


_SPHINX_STRIP_ENDO: typing.Final[Endo[str]] = Endo(app=_strip_sphinx_stage)
_CLASS_TAG_UNWRAP_ENDO: typing.Final[Endo[str]] = Endo(app=_unwrap_class_tag_stage)
_MEMORY_ADDR_SCRUB_ENDO: typing.Final[Endo[str]] = Endo(app=_scrub_memory_addr_stage)
_AST_PEP604_NORMALIZE_ENDO: typing.Final[Endo[str]] = Endo(app=_normalize_ast_pep604_stage)
_STR_ENDO_MONOID: typing.Final[Monoid[Endo[str]]] = endo_monoid()

TYPE_STR_SANITIZE_ENDO: typing.Final[Endo[str]] = _STR_ENDO_MONOID.mconcat(
    (
        _AST_PEP604_NORMALIZE_ENDO,
        _MEMORY_ADDR_SCRUB_ENDO,
        _CLASS_TAG_UNWRAP_ENDO,
        _SPHINX_STRIP_ENDO,
    )
)


def sanitize_type_str(typ_str: str | None) -> str | None:
    """Normalize type hint strings to PEP-585 and PEP-604 syntax via `TYPE_STR_SANITIZE_ENDO`.

    Args:
        typ_str: Raw type annotation string or None.

    Returns:
        Sanitized type string or None.
    """
    if not typ_str:
        return typ_str
    return TYPE_STR_SANITIZE_ENDO.app(typ_str)


__all__ = [
    "FactoryDefault",
    "LiteralDefault",
    "MandatoryDefault",
    "SanitizedDefault",
    "TYPE_STR_SANITIZE_ENDO",
    "UnrepresentableDefault",
    "default_value_or_none",
    "sanitize_param_default",
    "sanitize_type_str",
    "_AST_PEP604_NORMALIZE_ENDO",
    "_CLASS_TAG_UNWRAP_ENDO",
    "_MEMORY_ADDR_SCRUB_ENDO",
    "_SPHINX_STRIP_ENDO",
]
