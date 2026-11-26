"""C-extension, PyBind11, and ATen docstring signature extraction utilities.

Models extracted C-extension signatures as deeply immutable frozen Pydantic models
and exposes total `Result[CExtensionParseFault, ...]` parsers.
"""

from __future__ import annotations

import ast
import re
import typing

import pydantic
import typing_extensions
from ml_switcheroo_ir.typeclasses.outcome import Err, Ok, Result

from ml_ecosystem_snapshots.errors import CExtensionParseFault
from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    LivePythonTarget,
    StaticGriffeNode,
    get_member_dict,
)

RawParamTuple: typing_extensions.TypeAlias = tuple[str, str, "str | None", "str | None"]


class ParsedParamTuple(pydantic.BaseModel):
    """Structured parameter tuple extracted from C-extension or PyBind11 docstrings."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    name: str
    kind: str
    default: str | None = None
    annotation: str | None = None

    def as_tuple(self) -> RawParamTuple:
        """Return the 4-tuple `(name, kind, default, annotation)` representation.

        Returns:
            Tuple of `(name, kind, default, annotation)`.
        """
        return (self.name, self.kind, self.default, self.annotation)


def to_parsed_param_tuple(item: ParsedParamTuple | RawParamTuple) -> ParsedParamTuple:
    """Coerce a 4-tuple or `ParsedParamTuple` into a `ParsedParamTuple` instance.

    Args:
        item: Parameter tuple `(name, kind, default, annotation)` or `ParsedParamTuple`.

    Returns:
        Frozen `ParsedParamTuple` instance.
    """
    if isinstance(item, ParsedParamTuple):
        return item
    return ParsedParamTuple(name=item[0], kind=item[1], default=item[2], annotation=item[3])


class CExtensionSignature(pydantic.BaseModel):
    """Immutable extracted signature parameter tuple for C-extensions with overloads support.

    Attributes:
        params: Immutable tuple of `ParsedParamTuple` descriptors.
        returns_type: Optional return type annotation string.
        overloads: Immutable tuple of secondary/tertiary overload `CExtensionSignature` entries.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    params: tuple[ParsedParamTuple, ...] = ()
    returns_type: str | None = None
    overloads: tuple[CExtensionSignature, ...] = ()

    @pydantic.field_validator("params", mode="before")
    @classmethod
    def _coerce_raw_params(
        cls,
        raw_params: typing.Sequence[ParsedParamTuple | RawParamTuple],
    ) -> tuple[ParsedParamTuple, ...]:
        """Coerce raw `(name, kind, default, annotation)` 4-tuples into `ParsedParamTuple`s.

        Args:
            raw_params: Sequence of `ParsedParamTuple` or raw 4-tuples.

        Returns:
            Immutable tuple of `ParsedParamTuple` items.
        """
        return tuple(to_parsed_param_tuple(p) for p in raw_params)

    def iter_params(self) -> typing.Iterator[ParsedParamTuple]:
        """Iterate over the immutable parameter tuples.

        Returns:
            Iterator of `ParsedParamTuple` items.
        """
        return iter(self.params)

    def as_raw_tuples(self) -> tuple[RawParamTuple, ...]:
        """Convert all parameters to `(name, kind, default, annotation)` 4-tuples.

        Returns:
            Immutable tuple of `RawParamTuple` items.
        """
        return tuple(p.as_tuple() for p in self.params)


def build_c_extension_signature(
    params: typing.Sequence[ParsedParamTuple | RawParamTuple] = (),
    returns_type: str | None = None,
    overloads: typing.Sequence[CExtensionSignature] = (),
) -> CExtensionSignature:
    """Construct a frozen `CExtensionSignature` coercing raw 4-tuples into `ParsedParamTuple`s.

    Args:
        params: Sequence of `ParsedParamTuple` or `(name, kind, default, annotation)` 4-tuples.
        returns_type: Optional return type annotation string.
        overloads: Sequence of overload `CExtensionSignature` instances.

    Returns:
        Frozen `CExtensionSignature` instance.
    """
    return CExtensionSignature(
        params=tuple(to_parsed_param_tuple(p) for p in params),
        returns_type=returns_type,
        overloads=tuple(overloads),
    )


class ParsedCSignatureLine(pydantic.BaseModel):
    """Immutable parsed representation of a single C-extension signature line.

    Attributes:
        func_name: Unqualified function or operator name.
        params: Immutable tuple of extracted `ParsedParamTuple` descriptors.
        returns_type: Optional return type annotation string.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    func_name: str
    params: tuple[ParsedParamTuple, ...]
    returns_type: str | None = None


def _split_c_sig_tokens(args_str: str) -> tuple[str, ...]:
    """Split a raw C-extension parameter string by top-level commas.

    Args:
        args_str: Raw parameter list string from a C-extension docstring.

    Returns:
        Immutable tuple of stripped parameter token strings.
    """
    tokens: list[str] = []
    depth = 0
    cur: list[str] = []
    for char in args_str:
        if char in "<[(":
            depth += 1
            cur.append(char)
        elif char in ">])":
            depth -= 1
            cur.append(char)
        elif char == "," and depth == 0:
            tokens.append("".join(cur).strip())
            cur = []
        else:
            cur.append(char)
    if cur:
        tokens.append("".join(cur).strip())
    return tuple(tokens)


def _transform_c_sig_token(tok: str) -> str:
    """Convert a single C++/PyBind11 parameter token into Python parameter syntax.

    Args:
        tok: Single parameter token string.

    Returns:
        Normalized Python parameter syntax string.
    """
    if tok in ("*", "/") or tok.startswith("*") or ":" in tok:
        return tok
    m = re.match(
        r"^(?:const\s+)?([a-zA-Z_][\w:]*(?:<[^>]+>)?(?:\s*[*&]+)?)\s+([a-zA-Z_]\w*)(?:\s*=\s*(.+))?$",
        tok,
    )
    if m:
        typ, name, default = m.group(1).strip(), m.group(2).strip(), m.group(3)
        clean_typ = typ.replace("&", "").replace("*", "").strip()
        if default is not None:
            return f'{name}: "{clean_typ}" = {default.strip()}'
        return f'{name}: "{clean_typ}"'
    return tok


def normalize_c_sig_args(args_str: str) -> str:
    """Normalize C++ / PyBind / ATen parameter syntax to Python syntax for AST parsing.

    Converts signatures like `'Tensor self, bool inplace=False'` into `'self: "Tensor" = False'`.

    Args:
        args_str: Raw parameter string from C-extension docstring.

    Returns:
        Normalized Python parameter string.
    """
    transformed = [_transform_c_sig_token(tok) for tok in _split_c_sig_tokens(args_str) if tok]
    return ", ".join(transformed)


def parse_c_sig_ast_arguments(args_str: str) -> Result[CExtensionParseFault, ast.arguments]:
    """Parse raw C-extension parameter text into an `ast.arguments` node without raising.

    Args:
        args_str: Raw parameter list string inside signature parentheses.

    Returns:
        `Ok(ast.arguments)` on success, or `Err(CExtensionParseFault)` if invalid.
    """
    dummy_code = f"def dummy_func({args_str}): pass"
    try:
        tree = ast.parse(dummy_code)
        return Ok([node.args for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)][0])
    except SyntaxError:
        norm_args = normalize_c_sig_args(args_str)
        try:
            tree = ast.parse(f"def dummy_func({norm_args}): pass")
            return Ok([node.args for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)][0])
        except SyntaxError as exc:
            return Err(
                CExtensionParseFault(
                    signature_text=args_str,
                    offset=exc.offset,
                    reason=f"Failed to parse C-extension parameter signature: '{args_str}': {exc}",
                )
            )


def _unparse_c_sig_anno(node: ast.AST | None) -> str | None:
    """Convert an AST annotation node back to a string.

    Args:
        node: AST annotation node or None.

    Returns:
        Unparsed annotation string or None when unannotated.
    """
    if node is None:
        return None
    return ast.unparse(node)


def _unparse_c_sig_default(node: ast.AST) -> str:
    """Convert an AST default value node to a string representation.

    Args:
        node: AST default value node.

    Returns:
        String representation of the default value.
    """
    try:
        val = ast.literal_eval(node)
        if isinstance(val, str):
            return repr(val)
        return str(val)
    except ValueError:
        return ast.unparse(node)


def _extract_params_from_ast_args(args: ast.arguments) -> tuple[ParsedParamTuple, ...]:
    """Extract immutable `ParsedParamTuple` entries from a parsed `ast.arguments` node.

    Args:
        args: Parsed Python `ast.arguments` node.

    Returns:
        Immutable tuple of `ParsedParamTuple` descriptors excluding `self`.
    """
    extracted_params: list[ParsedParamTuple] = []

    num_defaults = len(args.defaults)
    posonly_args = args.posonlyargs
    posonly_count = len(posonly_args)
    combined_args = posonly_args + args.args

    num_combined = len(combined_args)
    default_offset = num_combined - num_defaults

    for i, arg in enumerate(combined_args):
        p_name = arg.arg
        if p_name == "self":
            continue
        p_kind = "POSITIONAL_ONLY" if i < posonly_count else "POSITIONAL_OR_KEYWORD"
        p_anno = _unparse_c_sig_anno(arg.annotation)
        p_def = _unparse_c_sig_default(args.defaults[i - default_offset]) if i >= default_offset else None
        extracted_params.append(ParsedParamTuple(name=p_name, kind=p_kind, default=p_def, annotation=p_anno))

    if args.vararg:
        extracted_params.append(
            ParsedParamTuple(
                name=args.vararg.arg,
                kind="VAR_POSITIONAL",
                default=None,
                annotation=_unparse_c_sig_anno(args.vararg.annotation),
            )
        )

    for i, arg in enumerate(args.kwonlyargs):
        default_node = args.kw_defaults[i]
        p_def = _unparse_c_sig_default(default_node) if default_node else None
        extracted_params.append(
            ParsedParamTuple(
                name=arg.arg,
                kind="KEYWORD_ONLY",
                default=p_def,
                annotation=_unparse_c_sig_anno(arg.annotation),
            )
        )

    if args.kwarg:
        extracted_params.append(
            ParsedParamTuple(
                name=args.kwarg.arg,
                kind="VAR_KEYWORD",
                default=None,
                annotation=_unparse_c_sig_anno(args.kwarg.annotation),
            )
        )

    return tuple(extracted_params)


def parse_c_extension_sig_str(
    sig_line: str,
) -> Result[CExtensionParseFault, ParsedCSignatureLine]:
    """Parse a single signature line into a `ParsedCSignatureLine` wrapped in `Result`.

    Args:
        sig_line: A single signature line (e.g. `'add(input, other) -> Tensor'`).

    Returns:
        `Ok(ParsedCSignatureLine)` if valid, else `Err(CExtensionParseFault)`.
    """
    clean_line = sig_line.strip()
    clean_line = re.sub(r"^(?:\d+[\.\)]\s*)?(?:aten::|c10::|torch\.)", "", clean_line)
    pattern = r"^(?:\d+[\.\)]\s*)?([a-zA-Z0-9_]+)\((.*)\)(?:\s*->\s*(.*))?$"
    matched = re.match(pattern, clean_line)
    if not matched:
        return Err(
            CExtensionParseFault(
                signature_text=sig_line,
                offset=0,
                reason=f"Line does not match C-extension signature syntax: {sig_line!r}",
            )
        )

    func_name = matched.group(1)
    args_str = matched.group(2)
    ret_type_str = matched.group(3).strip() if matched.group(3) else None

    args_res = parse_c_sig_ast_arguments(args_str)
    if isinstance(args_res, Err):
        return Err(args_res.error)
    if isinstance(args_res, Ok):
        return Ok(
            ParsedCSignatureLine(
                func_name=func_name,
                params=_extract_params_from_ast_args(args_res.value),
                returns_type=ret_type_str,
            )
        )
    typing_extensions.assert_never(args_res)


def _collect_docstring_signatures(
    lines: typing.Sequence[str],
    has_numbered: bool,
) -> tuple[tuple[ParsedCSignatureLine, ...], bool]:
    """Scan docstring lines for primary and overloaded C-extension signatures.

    Args:
        lines: Docstring lines to inspect.
        has_numbered: Initial flag indicating whether numbered overloads exist.

    Returns:
        Tuple of `(parsed_signatures, updated_has_numbered)`.
    """
    parsed_sigs: list[ParsedCSignatureLine] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if parsed_sigs and not has_numbered:
                break
        else:
            parsed_res = parse_c_extension_sig_str(stripped)
            if isinstance(parsed_res, Ok):
                parsed_sigs.append(parsed_res.value)
            elif parsed_sigs and not has_numbered:
                if "overloaded function" in stripped.lower():
                    has_numbered = True
                else:
                    break
            elif "overloaded function" in stripped.lower():
                has_numbered = True
    return tuple(parsed_sigs), has_numbered


def _fallback_target_signature(
    lines: typing.Sequence[str],
    target_name: str,
) -> tuple[ParsedCSignatureLine, ...]:
    """Search docstring lines for a fallback call example matching `target_name`.

    Args:
        lines: Docstring lines to scan.
        target_name: Name of the callable target.

    Returns:
        Single-element tuple with `ParsedCSignatureLine` if found, otherwise empty tuple.
    """
    for line in lines:
        stripped = line.strip()
        if "(" in stripped and ")" in stripped:
            clean_cand = re.sub(r"^(?:>>>|\.\.\.)\s*", "", stripped).strip()
            if (
                clean_cand.startswith(f"{target_name}(")
                or f".{target_name}(" in clean_cand
                or target_name in clean_cand
            ):
                parsed_res = parse_c_extension_sig_str(clean_cand)
                if isinstance(parsed_res, Ok):
                    return (parsed_res.value,)
    return ()


def extract_c_extension_signature(
    target: LivePythonTarget | StaticGriffeNode,
    target_name: str = "",
) -> Result[CExtensionParseFault, CExtensionSignature]:
    """Extract signature parameters from C-extension / PyBind11 docstrings into a `Result`.

    Args:
        target: The callable or static target to inspect.
        target_name: The name of the callable.

    Returns:
        `Ok(CExtensionSignature)` with `.params`, `.returns_type`, and `.overloads`
        populated, or `Err(CExtensionParseFault)` if no docstring signature exists.
    """
    docstring = get_member_dict(target).get("__doc__")
    if not docstring or not isinstance(docstring, str) or not docstring.strip():
        return Err(
            CExtensionParseFault(
                signature_text="",
                offset=None,
                reason=f"Target {target_name!r} has no inspectable docstring.",
            )
        )

    lines = tuple(docstring.strip().splitlines())
    has_numbered = any(re.match(r"^\s*\d+[\.\)]", line) for line in lines)
    parsed_sigs, has_numbered = _collect_docstring_signatures(lines, has_numbered)

    if not parsed_sigs:
        parsed_sigs = _fallback_target_signature(lines, target_name)

    if not parsed_sigs:
        return Err(
            CExtensionParseFault(
                signature_text=docstring,
                offset=None,
                reason=f"No C-extension signature found in docstring for {target_name!r}.",
            )
        )

    if (
        len(parsed_sigs) > 1
        and has_numbered
        and len(parsed_sigs[0].params) == 2
        and parsed_sigs[0].params[0].kind == "VAR_POSITIONAL"
        and parsed_sigs[0].params[1].kind == "VAR_KEYWORD"
    ):
        primary = parsed_sigs[1]
        secondary = parsed_sigs[2:]
    else:
        primary = parsed_sigs[0]
        secondary = parsed_sigs[1:]

    overload_objs = tuple(
        CExtensionSignature(params=ov.params, returns_type=ov.returns_type, overloads=()) for ov in secondary
    )

    return Ok(
        CExtensionSignature(
            params=primary.params,
            returns_type=primary.returns_type,
            overloads=overload_objs,
        )
    )


__all__ = [
    "CExtensionSignature",
    "ParsedCSignatureLine",
    "ParsedParamTuple",
    "RawParamTuple",
    "build_c_extension_signature",
    "extract_c_extension_signature",
    "normalize_c_sig_args",
    "parse_c_extension_sig_str",
    "parse_c_sig_ast_arguments",
    "to_parsed_param_tuple",
]
