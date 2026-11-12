"""C-extension, PyBind11, and ATen docstring signature extraction utilities."""

from __future__ import annotations

import ast
import re
import typing

import pydantic
import typing_extensions

from ml_ecosystem_snapshots.exceptions import CExtensionSignatureError
from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    LivePythonTarget,
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


def _to_parsed_param_tuple(
    item: ParsedParamTuple | RawParamTuple,
) -> ParsedParamTuple:
    """Coerce a 4-tuple or `ParsedParamTuple` into a `ParsedParamTuple` instance.

    Args:
        item: Parameter tuple `(name, kind, default, annotation)`.

    Returns:
        `ParsedParamTuple` instance.
    """
    if isinstance(item, ParsedParamTuple):
        return item
    return ParsedParamTuple(name=item[0], kind=item[1], default=item[2], annotation=item[3])


class CExtensionSignature:
    """Extracted signature parameter sequence for C-extensions with overloads support."""

    params: list[ParsedParamTuple]
    returns_type: str | None
    overloads: list[CExtensionSignature]

    def __init__(
        self,
        params: typing.Sequence[ParsedParamTuple | RawParamTuple] = (),
        returns_type: str | None = None,
        overloads: list[CExtensionSignature] | None = None,
    ) -> None:
        """Initialize signature with parameters, return type, and optional overloads.

        Args:
            params: Sequence of parameter tuples `(name, kind, default, annotation)`.
            returns_type: Optional return type annotation string.
            overloads: Optional list of secondary/tertiary overload signatures.
        """
        self.params = [_to_parsed_param_tuple(p) for p in params]
        self.returns_type = returns_type
        self.overloads = list(overloads) if overloads is not None else []

    def __len__(self) -> int:
        """Return the number of extracted parameters.

        Returns:
            Parameter count.
        """
        return len(self.params)

    def __getitem__(self, index: int) -> ParsedParamTuple:
        """Retrieve a parameter tuple by integer index.

        Args:
            index: Integer index.

        Returns:
            Single `ParsedParamTuple` item.
        """
        return self.params[index]

    def __iter__(self) -> typing.Iterator[ParsedParamTuple]:
        """Iterate over the parameter tuples.

        Returns:
            Iterator of `ParsedParamTuple` items.
        """
        return iter(self.params)

    def iter_params(self) -> typing.Iterator[ParsedParamTuple]:
        """Iterate over the parameter tuples.

        Returns:
            Iterator of `ParsedParamTuple` items.
        """
        return iter(self.params)

    def __eq__(
        self,
        other: (CExtensionSignature | typing.Sequence[ParsedParamTuple | RawParamTuple] | LivePythonTarget),
    ) -> bool:
        """Compare equality against another `CExtensionSignature` or `list` of tuples.

        Args:
            other: Signature or sequence to compare with.

        Returns:
            True if parameter sequences are equal, False otherwise.
        """
        if isinstance(other, CExtensionSignature):
            return self.params == other.params
        if isinstance(other, list):
            return list(self.params) == other
        return False

    def __repr__(self) -> str:
        """Return a developer-friendly string representation.

        Returns:
            Formatted representation string.
        """
        return repr([p.as_tuple() for p in self.params])


def _split_c_sig_tokens(args_str: str) -> list[str]:
    """Split a raw C-extension parameter string by top-level commas.

    Args:
        args_str: Raw parameter list string from a C-extension docstring.

    Returns:
        List of individual stripped parameter token strings.
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
    return tokens


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

    Converts signatures like 'Tensor self, bool inplace=False' into 'self: "Tensor" = False'.

    Args:
        args_str: Raw parameter string from C-extension docstring.

    Returns:
        Normalized Python parameter string.
    """
    transformed = [_transform_c_sig_token(tok) for tok in _split_c_sig_tokens(args_str) if tok]
    return ", ".join(transformed)


_normalize_c_sig_args = normalize_c_sig_args


def parse_c_sig_ast_arguments(args_str: str) -> ast.arguments:
    """Parse raw C-extension parameter text into an `ast.arguments` node.

    Args:
        args_str: Raw parameter list string inside signature parentheses.

    Returns:
        Parsed `ast.arguments` AST node.

    Raises:
        CExtensionSignatureError: If neither direct nor normalized syntax parses as valid Python parameters.
    """
    dummy_code = f"def dummy_func({args_str}): pass"
    try:
        tree = ast.parse(dummy_code)
        return [node.args for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)][0]
    except SyntaxError:
        norm_args = normalize_c_sig_args(args_str)
        try:
            tree = ast.parse(f"def dummy_func({norm_args}): pass")
            return [node.args for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)][0]
        except SyntaxError as exc:
            raise CExtensionSignatureError(f"Failed to parse C-extension parameter signature: '{args_str}'") from exc


_parse_c_sig_ast_arguments = parse_c_sig_ast_arguments


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


def parse_c_extension_sig_str(
    sig_line: str,
) -> tuple[str, list[ParsedParamTuple], str | None] | None:
    """Parse a single signature line into function name, params, and return type.

    Args:
        sig_line: A single signature line (e.g. `add(input, other) -> Tensor`).

    Returns:
        A tuple of (func_name, extracted_params, returns_type), or None if parsing fails.
    """
    clean_line = sig_line.strip()
    clean_line = re.sub(r"^(?:\d+[\.\)]\s*)?(?:aten::|c10::|torch\.)", "", clean_line)
    pattern = r"^(?:\d+[\.\)]\s*)?([a-zA-Z0-9_]+)\((.*)\)(?:\s*->\s*(.*))?$"
    match = re.match(pattern, clean_line)
    if not match:
        return None

    func_name = match.group(1)
    args_str = match.group(2)
    ret_type_str = match.group(3).strip() if match.group(3) else None

    try:
        args = _parse_c_sig_ast_arguments(args_str)
    except CExtensionSignatureError:
        return None

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
        p_name = args.vararg.arg
        p_kind = "VAR_POSITIONAL"
        p_anno = _unparse_c_sig_anno(args.vararg.annotation)
        extracted_params.append(ParsedParamTuple(name=p_name, kind=p_kind, default=None, annotation=p_anno))

    for i, arg in enumerate(args.kwonlyargs):
        p_name = arg.arg
        p_kind = "KEYWORD_ONLY"
        p_anno = _unparse_c_sig_anno(arg.annotation)
        default_node = args.kw_defaults[i]
        p_def = _unparse_c_sig_default(default_node) if default_node else None
        extracted_params.append(ParsedParamTuple(name=p_name, kind=p_kind, default=p_def, annotation=p_anno))

    if args.kwarg:
        p_name = args.kwarg.arg
        p_kind = "VAR_KEYWORD"
        p_anno = _unparse_c_sig_anno(args.kwarg.annotation)
        extracted_params.append(ParsedParamTuple(name=p_name, kind=p_kind, default=None, annotation=p_anno))

    return func_name, extracted_params, ret_type_str


_parse_c_extension_sig_str = parse_c_extension_sig_str


def _collect_docstring_signatures(
    lines: list[str],
    has_numbered: bool,
) -> tuple[list[tuple[str, list[ParsedParamTuple], str | None]], bool]:
    """Scan docstring lines for primary and overloaded C-extension signatures.

    Args:
        lines: Stripped docstring lines to inspect.
        has_numbered: Initial flag indicating whether numbered overloads exist.

    Returns:
        Tuple of `(parsed_signatures, updated_has_numbered)`.
    """
    parsed_sigs: list[tuple[str, list[ParsedParamTuple], str | None]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if parsed_sigs and not has_numbered:
                break
        elif (parsed := _parse_c_extension_sig_str(stripped)) is not None:
            parsed_sigs.append(parsed)
        elif parsed_sigs and not has_numbered:
            if "overloaded function" in stripped.lower():
                has_numbered = True
            else:
                break
        elif "overloaded function" in stripped.lower():
            has_numbered = True
    return parsed_sigs, has_numbered


def _fallback_target_signature(
    lines: list[str],
    target_name: str,
) -> list[tuple[str, list[ParsedParamTuple], str | None]]:
    """Search docstring lines for a fallback call example matching `target_name`.

    Args:
        lines: Docstring lines to scan.
        target_name: Name of the callable target.

    Returns:
        Single-element list with parsed signature tuple if found, otherwise empty list.
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
                parsed = _parse_c_extension_sig_str(clean_cand)
                if parsed is not None:
                    return [parsed]
    return []


def extract_c_extension_signature(target: LivePythonTarget, target_name: str) -> CExtensionSignature | None:
    """Extract signature parameters from C-extension / PyBind11 docstrings.

    Args:
        target: The callable target to inspect.
        target_name: The name of the callable.

    Returns:
        A CExtensionSignature sequence of parameter tuples: (name, kind, default_str, annotation_str)
        with .returns_type and .overloads populated, or None if parsing fails.
    """
    docstring = get_member_dict(target).get("__doc__")
    if not docstring or not isinstance(docstring, str) or not docstring.strip():
        return None

    lines = docstring.strip().splitlines()
    has_numbered = any(re.match(r"^\s*\d+[\.\)]", line) for line in lines)
    parsed_sigs, has_numbered = _collect_docstring_signatures(lines, has_numbered)

    if not parsed_sigs:
        parsed_sigs = _fallback_target_signature(lines, target_name)

    if not parsed_sigs:
        return None

    if (
        len(parsed_sigs) > 1
        and has_numbered
        and len(parsed_sigs[0][1]) == 2
        and parsed_sigs[0][1][0].kind == "VAR_POSITIONAL"
        and parsed_sigs[0][1][1].kind == "VAR_KEYWORD"
    ):
        primary = parsed_sigs[1]
        secondary = parsed_sigs[2:]
    else:
        primary = parsed_sigs[0]
        secondary = parsed_sigs[1:]

    overload_objs = [CExtensionSignature(params=ov[1], returns_type=ov[2], overloads=[]) for ov in secondary]

    return CExtensionSignature(
        params=primary[1],
        returns_type=primary[2],
        overloads=overload_objs,
    )


__all__ = [
    "CExtensionSignature",
    "ParsedParamTuple",
    "RawParamTuple",
    "extract_c_extension_signature",
]
