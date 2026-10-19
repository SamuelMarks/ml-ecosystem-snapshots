"""C-extension, PyBind11, and ATen docstring signature extraction utilities."""

from __future__ import annotations

import ast
from collections.abc import Iterable, MutableSequence
import re
from typing import (
    Any,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

from ml_ecosystem_snapshots.exceptions import CExtensionSignatureError


class ParsedParamTuple(NamedTuple):
    """Structured parameter tuple extracted from C-extension or PyBind11 docstrings."""

    name: str
    kind: str
    default: Optional[str] = None
    annotation: Optional[str] = None


def _to_parsed_param_tuple(item: Any) -> ParsedParamTuple:
    """Coerce a 4-tuple or `ParsedParamTuple` into a `ParsedParamTuple` instance.

    Args:
        item: Parameter tuple `(name, kind, default, annotation)`.

    Returns:
        `ParsedParamTuple` instance.
    """
    if isinstance(item, ParsedParamTuple):
        return item
    return ParsedParamTuple(item[0], item[1], item[2], item[3])


class CExtensionSignature(
    MutableSequence[Tuple[str, str, Optional[str], Optional[str]]]
):
    """Extracted signature parameter sequence for C-extensions with overloads support."""

    def __init__(
        self,
        params: Sequence[Tuple[str, str, Optional[str], Optional[str]]],
        returns_type: Optional[str] = None,
        overloads: Optional[List["CExtensionSignature"]] = None,
    ) -> None:
        """Initialize signature with parameters, return type, and optional overloads.

        Args:
            params: Sequence of parameter tuples `(name, kind, default, annotation)`.
            returns_type: Optional return type annotation string.
            overloads: Optional list of secondary/tertiary overload signatures.
        """
        self._params: List[ParsedParamTuple] = [
            _to_parsed_param_tuple(p) for p in params
        ]
        self.returns_type: Optional[str] = returns_type
        self.overloads: List["CExtensionSignature"] = overloads or []

    def __len__(self) -> int:
        """Return the number of extracted parameters.

        Returns:
            Parameter count.
        """
        return len(self._params)

    @overload
    def __getitem__(self, index: int) -> ParsedParamTuple: ...

    @overload
    def __getitem__(
        self, index: slice
    ) -> MutableSequence[Tuple[str, str, Optional[str], Optional[str]]]: ...

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[
        ParsedParamTuple,
        MutableSequence[Tuple[str, str, Optional[str], Optional[str]]],
    ]:
        """Retrieve a parameter tuple or slice of parameter tuples.

        Args:
            index: Integer index or slice.

        Returns:
            Single `ParsedParamTuple` or list of `ParsedParamTuple` items.
        """
        if isinstance(index, slice):
            res_slice: List[Tuple[str, str, Optional[str], Optional[str]]] = list(
                self._params[index]
            )
            return res_slice
        return self._params[index]

    @overload
    def __setitem__(
        self,
        index: int,
        value: Tuple[str, str, Optional[str], Optional[str]],
    ) -> None: ...

    @overload
    def __setitem__(
        self,
        index: slice,
        value: Iterable[Tuple[str, str, Optional[str], Optional[str]]],
    ) -> None: ...

    def __setitem__(
        self,
        index: Union[int, slice],
        value: Union[
            Tuple[str, str, Optional[str], Optional[str]],
            Iterable[Tuple[str, str, Optional[str], Optional[str]]],
        ],
    ) -> None:
        """Assign a parameter tuple or slice of parameter tuples.

        Args:
            index: Integer index or slice.
            value: Replacement tuple or sequence of tuples.
        """
        if isinstance(index, slice):
            self._params[index] = [_to_parsed_param_tuple(item) for item in value]
        else:
            self._params[index] = _to_parsed_param_tuple(value)

    def __delitem__(self, index: Union[int, slice]) -> None:
        """Remove a parameter tuple or slice of parameter tuples.

        Args:
            index: Integer index or slice to delete.
        """
        del self._params[index]

    def insert(
        self,
        index: int,
        value: Tuple[str, str, Optional[str], Optional[str]],
    ) -> None:
        """Insert a parameter tuple at `index`.

        Args:
            index: Target insertion index.
            value: Parameter tuple to insert.
        """
        self._params.insert(index, _to_parsed_param_tuple(value))

    def __iter__(self) -> Iterator[ParsedParamTuple]:
        """Iterate over the parameter tuples.

        Returns:
            Iterator of `ParsedParamTuple` items.
        """
        return iter(self._params)

    def __eq__(self, other: object) -> bool:
        """Compare equality against another `CExtensionSignature` or `list` of tuples.

        Args:
            other: Object to compare with.

        Returns:
            True if parameter sequences are equal, False otherwise.
        """
        if isinstance(other, CExtensionSignature):
            return self._params == other._params
        if isinstance(other, list):
            return list(self._params) == other
        return False

    def __repr__(self) -> str:
        """Return a developer-friendly string representation.

        Returns:
            Formatted representation string.
        """
        return repr(list(self._params))


def _normalize_c_sig_args(args_str: str) -> str:
    """Normalize C++ / PyBind / ATen parameter syntax to Python syntax for AST parsing.

    Converts signatures like 'Tensor self, bool inplace=False' into 'self: "Tensor" = False'.

    Args:
        args_str: Raw parameter string from C-extension docstring.

    Returns:
        Normalized Python parameter string.
    """
    tokens: List[str] = []
    depth = 0
    cur: List[str] = []
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

    transformed: List[str] = []
    for tok in tokens:
        if not tok:
            continue
        if tok in ("*", "/"):
            transformed.append(tok)
            continue
        if tok.startswith("*"):
            transformed.append(tok)
            continue
        if ":" in tok:
            transformed.append(tok)
            continue
        m = re.match(
            r"^(?:const\s+)?([a-zA-Z_][\w:]*(?:<[^>]+>)?(?:\s*[*&]+)?)\s+([a-zA-Z_]\w*)(?:\s*=\s*(.+))?$",
            tok,
        )
        if m:
            typ, name, default = m.group(1).strip(), m.group(2).strip(), m.group(3)
            clean_typ = typ.replace("&", "").replace("*", "").strip()
            if default is not None:
                transformed.append(f'{name}: "{clean_typ}" = {default.strip()}')
            else:
                transformed.append(f'{name}: "{clean_typ}"')
        else:
            transformed.append(tok)
    return ", ".join(transformed)


def _parse_c_sig_ast_arguments(args_str: str) -> ast.arguments:
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
        func_def = tree.body[0]
        assert isinstance(func_def, ast.FunctionDef)
        return func_def.args
    except SyntaxError:
        norm_args = _normalize_c_sig_args(args_str)
        try:
            tree = ast.parse(f"def dummy_func({norm_args}): pass")
            func_def = tree.body[0]
            assert isinstance(func_def, ast.FunctionDef)
            return func_def.args
        except SyntaxError as exc:
            raise CExtensionSignatureError(
                f"Failed to parse C-extension parameter signature: '{args_str}'"
            ) from exc


def _parse_c_extension_sig_str(
    sig_line: str,
) -> Optional[
    Tuple[
        str,
        List[Tuple[str, str, Optional[str], Optional[str]]],
        Optional[str],
    ]
]:
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

    extracted_params: List[Tuple[str, str, Optional[str], Optional[str]]] = []

    def unparse_anno(node: Any) -> Optional[str]:
        """Convert an AST annotation node back to a string.

        Args:
            node: AST node.

        Returns:
            String representation or None.
        """
        if node is None:
            return None
        return ast.unparse(node)

    def unparse_default(node: Any) -> Optional[str]:
        """Convert an AST default value node to a string representation.

        Args:
            node: AST node.

        Returns:
            String representation or None.
        """
        try:
            val = ast.literal_eval(node)
            if isinstance(val, str):
                return repr(val)
            return str(val)
        except ValueError:
            return ast.unparse(node)

    num_defaults = len(args.defaults)
    posonly_args = getattr(args, "posonlyargs", [])
    posonly_count = len(posonly_args)
    combined_args = posonly_args + args.args

    num_combined = len(combined_args)
    default_offset = num_combined - num_defaults

    for i, arg in enumerate(combined_args):
        p_name = arg.arg
        if p_name == "self":
            continue
        p_kind = "POSITIONAL_ONLY" if i < posonly_count else "POSITIONAL_OR_KEYWORD"
        p_anno = unparse_anno(arg.annotation)
        p_def = (
            unparse_default(args.defaults[i - default_offset])
            if i >= default_offset
            else None
        )
        extracted_params.append(ParsedParamTuple(p_name, p_kind, p_def, p_anno))

    if args.vararg:
        p_name = args.vararg.arg
        p_kind = "VAR_POSITIONAL"
        p_anno = unparse_anno(args.vararg.annotation)
        extracted_params.append(ParsedParamTuple(p_name, p_kind, None, p_anno))

    for i, arg in enumerate(args.kwonlyargs):
        p_name = arg.arg
        p_kind = "KEYWORD_ONLY"
        p_anno = unparse_anno(arg.annotation)
        default_node = args.kw_defaults[i]
        p_def = unparse_default(default_node) if default_node else None
        extracted_params.append(ParsedParamTuple(p_name, p_kind, p_def, p_anno))

    if args.kwarg:
        p_name = args.kwarg.arg
        p_kind = "VAR_KEYWORD"
        p_anno = unparse_anno(args.kwarg.annotation)
        extracted_params.append(ParsedParamTuple(p_name, p_kind, None, p_anno))

    return func_name, extracted_params, ret_type_str


def extract_c_extension_signature(
    target: Any, target_name: str
) -> Optional[CExtensionSignature]:
    """Extract signature parameters from C-extension / PyBind11 docstrings.

    Args:
        target: The callable object to inspect.
        target_name: The name of the callable.

    Returns:
        A CExtensionSignature sequence of parameter tuples: (name, kind, default_str, annotation_str)
        with .returns_type and .overloads populated, or None if parsing fails.
    """
    docstring = getattr(target, "__doc__", None)
    if not docstring or not isinstance(docstring, str) or not docstring.strip():
        return None

    lines = docstring.strip().splitlines()

    parsed_sigs: List[
        Tuple[
            str,
            List[Tuple[str, str, Optional[str], Optional[str]]],
            Optional[str],
        ]
    ] = []

    has_numbered = any(re.match(r"^\s*\d+[\.\)]", line) for line in lines)

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

    if not parsed_sigs:
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
                        parsed_sigs.append(parsed)
                        break

    if not parsed_sigs:
        return None

    if (
        len(parsed_sigs) > 1
        and has_numbered
        and len(parsed_sigs[0][1]) == 2
        and parsed_sigs[0][1][0][1] == "VAR_POSITIONAL"
        and parsed_sigs[0][1][1][1] == "VAR_KEYWORD"
    ):
        primary = parsed_sigs[1]
        secondary = parsed_sigs[2:]
    else:
        primary = parsed_sigs[0]
        secondary = parsed_sigs[1:]

    overload_objs = [
        CExtensionSignature(params=ov[1], returns_type=ov[2], overloads=[])
        for ov in secondary
    ]

    return CExtensionSignature(
        params=primary[1],
        returns_type=primary[2],
        overloads=overload_objs,
    )


__all__ = [
    "CExtensionSignature",
    "ParsedParamTuple",
    "_normalize_c_sig_args",
    "_parse_c_extension_sig_str",
    "extract_c_extension_signature",
]
