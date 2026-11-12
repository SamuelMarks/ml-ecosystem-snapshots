"""Utility functions for inspecting and extracting information from Python modules."""

from __future__ import annotations

import enum
import logging
import re
import types
import typing

import pydantic

from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    LivePythonTarget,
    get_member_dict,
)

if typing.TYPE_CHECKING:
    import griffe

_LOGGER = logging.getLogger(__name__)


@typing.runtime_checkable
class _RestParserClassProtocol(typing.Protocol):
    """Protocol for a Griffe Parser class exposing a `.rest` attribute."""

    rest: griffe.Parser


@typing.runtime_checkable
class _DocstringParamItemProtocol(typing.Protocol):
    """Protocol for a Griffe docstring parameter element."""

    name: str | None
    annotation: LivePythonTarget
    value: LivePythonTarget
    description: str | None


@typing.runtime_checkable
class _DocstringReturnItemProtocol(typing.Protocol):
    """Protocol for a Griffe docstring return element."""

    annotation: LivePythonTarget
    description: str | None


@typing.runtime_checkable
class _DocstringRaiseItemProtocol(typing.Protocol):
    """Protocol for a Griffe docstring raise element."""

    annotation: LivePythonTarget


class DocstringParamMeta(pydantic.BaseModel):
    """Extracted parameter metadata from a parsed docstring."""

    model_config = pydantic.ConfigDict(extra="forbid")

    doc: str = ""
    typ: str | None = None
    default: str | None = None


class DocstringReturnMeta(pydantic.BaseModel):
    """Extracted return metadata from a parsed docstring."""

    model_config = pydantic.ConfigDict(extra="forbid")

    doc: str = ""
    typ: str | None = None


class DocstringMetadata(pydantic.BaseModel):
    """Structured metadata extracted from a docstring via Griffe."""

    model_config = pydantic.ConfigDict(extra="forbid")

    params: dict[str, DocstringParamMeta] = pydantic.Field(default_factory=dict)
    returns: DocstringReturnMeta | None = None
    raises: list[str] = pydantic.Field(default_factory=list)


def get_all_members(
    module: types.ModuleType | types.SimpleNamespace | type,
) -> list[tuple[str, LivePythonTarget]]:
    """Safely extract all accessible members from a module or class.

    Args:
        module: The module or class to inspect.

    Returns:
        A list of `(name, member_target)` tuples.
    """
    return list(get_member_dict(module).items())


def _detect_docstring_style_from_text(docstring: str) -> str | None:
    """Detect the docstring parser convention from explicit markers in `docstring`.

    Args:
        docstring: Non-empty docstring text to inspect.

    Returns:
        Detected parser identifier (`'rest'`, `'numpy'`, or `'google'`), or None if inconclusive.
    """
    if any(marker in docstring for marker in (":param ", ":type ", ":return:", ":rtype:", ":raises:")):
        return "rest"
    if re.search(r"(?:^|\s)(Parameters|Returns|Raises)\s*[-=]{3,}", docstring):
        return "numpy"
    if re.search(r"(?:^|\s)(Args|Arguments|Returns|Raises):\s*", docstring):
        return "google"
    return None


def get_framework_docstring_parser(framework_name: str, docstring: str | None = None) -> str:
    """Determine the most appropriate docstring parser for a framework.

    Args:
        framework_name: Name of the framework or top-level module (e.g. 'torch', 'numpy').
        docstring: Optional docstring text to inspect for format heuristics.

    Returns:
        The docstring parser style identifier ('rest', 'numpy', 'google', or 'auto').
    """
    if docstring and isinstance(docstring, str):
        detected = _detect_docstring_style_from_text(docstring)
        if detected is not None:
            return detected

    name_clean = framework_name.lower().replace("-", "_")
    if name_clean in ("numpy", "np"):
        return "numpy"
    if name_clean in (
        "jax",
        "flax",
        "keras",
        "transformers",
        "huggingface",
        "chex",
        "grain",
        "pygrain",
        "orbax",
        "orbax_checkpoint",
        "optax",
    ):
        return "google"
    return "rest"


def resolve_griffe_parser(parser_name: str) -> griffe.Parser | None:
    """Resolve a parser identifier into a valid Griffe Parser instance.

    Args:
        parser_name: The parser name ('rest', 'sphinx', 'google', 'numpy', 'auto').

    Returns:
        A Griffe Parser instance or None.
    """
    try:
        import griffe as _griffe_mod

        if parser_name == "rest":
            parser_cls: LivePythonTarget = _griffe_mod.Parser
            if isinstance(parser_cls, _RestParserClassProtocol):
                rest_val: griffe.Parser = parser_cls.rest
                return rest_val
            sphinx_val: griffe.Parser = _griffe_mod.Parser("sphinx")
            return sphinx_val
        parsed_val: griffe.Parser = _griffe_mod.Parser(parser_name)
        return parsed_val
    except (ImportError, ValueError, KeyError, AttributeError, TypeError, RuntimeError) as exc:
        _LOGGER.debug("Failed to resolve Griffe parser '%s': %s", parser_name, exc)
        return None


def parse_docstring_with_griffe(docstring: str, parser_name: str | None = None) -> list[griffe.DocstringSection]:
    """Parse a docstring using Griffe with dynamic fallback across conventions.

    Args:
        docstring: The docstring text to parse.
        parser_name: Optional starting parser ('rest', 'google', 'numpy').

    Returns:
        A list of parsed Griffe DocstringSection objects.
    """
    try:
        import griffe as _griffe_mod
    except ImportError:
        return []

    initial_parser = parser_name or "rest"
    candidates = [initial_parser]
    for alt in ("rest", "google", "numpy"):
        if alt not in candidates:
            candidates.append(alt)

    best_sections: list[griffe.DocstringSection] = []

    for candidate in candidates:
        resolved = resolve_griffe_parser(candidate)
        if not resolved:
            continue
        try:
            doc_obj = _griffe_mod.Docstring(docstring)
            sections: list[griffe.DocstringSection] = _griffe_mod.parse(doc_obj, resolved)
            has_structured = any(s.kind in ("parameters", "returns", "raises") for s in sections)
            if has_structured:
                return sections
            if not best_sections and sections:
                best_sections = sections
        except (ValueError, KeyError, AttributeError, TypeError, RuntimeError, SyntaxError) as exc:
            _LOGGER.debug("Griffe docstring parse fallback for %s: %s", candidate, exc)

    return best_sections


def strip_sphinx_roles(text: str | None) -> str | None:
    """Strip Sphinx cross-referencing roles from text.

    Args:
        text: The input text containing potential Sphinx roles.

    Returns:
        Cleaned text with Sphinx roles stripped, or None if input is None.
    """
    if text is None:
        return None
    cleaned = str(text)
    cleaned = re.sub(r":[a-zA-Z0-9_:]+:`~?([^`]*)`", r"\1", cleaned)
    return cleaned


def _extract_griffe_params_section(
    items: list[LivePythonTarget],
    params: dict[str, DocstringParamMeta],
) -> None:
    """Populate `params` from a parsed Griffe parameters section list.

    Args:
        items: List of Griffe parameter items from `section.value`.
        params: Mutable dictionary mapping parameter names to `DocstringParamMeta`.
    """
    for item in items:
        if not isinstance(item, _DocstringParamItemProtocol):
            continue
        p_name = item.name
        if p_name:
            annotation = item.annotation
            value = item.value
            desc = item.description
            cleaned_typ = strip_sphinx_roles(str(annotation)) if annotation is not None else None
            params[str(p_name)] = DocstringParamMeta(
                doc=str(desc) if desc else "",
                typ=cleaned_typ,
                default=str(value) if value is not None else None,
            )


def _extract_griffe_returns_section(
    items: list[LivePythonTarget],
) -> DocstringReturnMeta | None:
    """Extract `DocstringReturnMeta` from a parsed Griffe returns section list.

    Args:
        items: List of Griffe return items from `section.value`.

    Returns:
        Last extracted `DocstringReturnMeta` entry, or None if `items` is empty.
    """
    returns: DocstringReturnMeta | None = None
    for item in items:
        if not isinstance(item, _DocstringReturnItemProtocol):
            continue
        annotation = item.annotation
        desc = item.description
        cleaned_typ = strip_sphinx_roles(str(annotation)) if annotation is not None else None
        returns = DocstringReturnMeta(
            doc=str(desc) if desc else "",
            typ=cleaned_typ,
        )
    return returns


def _extract_griffe_raises_section(
    items: list[LivePythonTarget],
    raises: list[str],
) -> None:
    """Populate `raises` from a parsed Griffe raises section list.

    Args:
        items: List of Griffe exception items from `section.value`.
        raises: Mutable list of unique exception type names.
    """
    for item in items:
        if not isinstance(item, _DocstringRaiseItemProtocol):
            continue
        annotation = item.annotation
        if annotation:
            exc_name = strip_sphinx_roles(str(annotation)) or str(annotation)
            if exc_name not in raises:
                raises.append(exc_name)


def extract_griffe_docstring_metadata(docstring: str, parser_name: str | None = None) -> DocstringMetadata:
    """Extract structured parameters, return types, and exceptions from a docstring.

    Args:
        docstring: The docstring text to parse.
        parser_name: Optional parser name to prioritize.

    Returns:
        A `DocstringMetadata` instance containing `params`, `returns`, and `raises`.
    """
    sections = parse_docstring_with_griffe(docstring, parser_name=parser_name)

    params: dict[str, DocstringParamMeta] = {}
    returns: DocstringReturnMeta | None = None
    raises: list[str] = []

    for section in sections:
        raw_kind: LivePythonTarget = section.kind
        kind_name: str = (
            raw_kind
            if isinstance(raw_kind, str)
            else (str(raw_kind.value) if isinstance(raw_kind, enum.Enum) else str(raw_kind))
        )
        if kind_name == "parameters" and isinstance(section.value, list):
            _extract_griffe_params_section(section.value, params)
        elif kind_name == "returns" and isinstance(section.value, list):
            returns = _extract_griffe_returns_section(section.value)
        elif kind_name == "raises" and isinstance(section.value, list):
            _extract_griffe_raises_section(section.value, raises)

    for exc in re.findall(r":(?:raises?|except)\s+([A-Za-z0-9_.]+):", docstring):
        if exc in params:
            del params[exc]
        if exc not in raises:
            raises.append(exc)

    return DocstringMetadata(params=params, returns=returns, raises=raises)


_BACKSLASH = chr(92)
_TABLEGEN_DEPTH_DELTAS: dict[str, tuple[int, int, int]] = {
    "[": (1, 0, 0),
    "]": (-1, 0, 0),
    "<": (0, 1, 0),
    ">": (0, -1, 0),
    "(": (0, 0, 1),
    ")": (0, 0, -1),
}


def _append_non_comment_element(elements: list[str], current: list[str]) -> None:
    """Append the trimmed string formed by `current` to `elements` if non-empty and not a comment.

    Args:
        elements: Mutable list of accumulated TableGen element strings.
        current: Character buffer for the current element.
    """
    elem = "".join(current).strip()
    if elem and not elem.startswith("//"):
        elements.append(elem)


def split_tablegen_list(raw_list_str: str) -> list[str]:
    """Split a comma-separated TableGen list while respecting nested brackets, angles, parens, and quotes.

    Args:
        raw_list_str: Content inside outer brackets [ ... ].

    Returns:
        List of trimmed TableGen element strings.
    """
    elements: list[str] = []
    current: list[str] = []
    depths = (0, 0, 0)
    in_quote = False
    escape = False

    cleaned_str = re.sub(r"/\*.*?\*/", "", raw_list_str, flags=re.DOTALL)
    cleaned_str = re.sub(r"//.*", "", cleaned_str)

    for char in cleaned_str:
        if escape:
            current.append(char)
            escape = False
            continue
        if char == _BACKSLASH:
            current.append(char)
            escape = True
            continue
        if char == '"':
            in_quote = not in_quote
            current.append(char)
            continue
        if in_quote:
            current.append(char)
            continue

        if char in _TABLEGEN_DEPTH_DELTAS:
            db, da, dp = _TABLEGEN_DEPTH_DELTAS[char]
            depths = (depths[0] + db, depths[1] + da, depths[2] + dp)
            current.append(char)
        elif char == "," and depths == (0, 0, 0):
            _append_non_comment_element(elements, current)
            current = []
        else:
            current.append(char)

    if current:
        _append_non_comment_element(elements, current)
    return elements


_TRAIT_INTERFACE_ALIASES: dict[str, str] = {
    '"getAsmResultNames"': "OpAsmOpInterface::getAsmResultNames",
    '["getAsmResultNames"]': "OpAsmOpInterface::getAsmResultNames",
    '"inferReturnTypeComponents"': "InferTypeOpInterface::inferReturnTypeComponents",
    '["inferReturnTypeComponents"': "InferTypeOpInterface::inferReturnTypeComponents",
    '"inferResultRanges"': "InferIntRangeInterface::inferResultRanges",
    '["inferResultRanges"': "InferIntRangeInterface::inferResultRanges",
}


def _find_unquoted_open_bracket(text: str) -> int:
    """Locate the index of the first unquoted `'['` character in `text`.

    Args:
        text: TableGen source text to scan.

    Returns:
        Zero-based character index of `'['`, or `-1` if not found.
    """
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if esc:
            esc = False
        elif ch == _BACKSLASH:
            esc = True
        elif ch == '"':
            in_str = not in_str
        elif not in_str and ch == "[":
            return i
    return -1


def _find_matching_close_bracket(text: str, start: int) -> int:
    """Locate the index of the closing `']'` matching the `'['` at `start`.

    Args:
        text: TableGen source text to scan.
        start: Index of the opening `'['` character.

    Returns:
        Zero-based character index of the matching `']'`, or `-1` if unbalanced.
    """
    depth = 0
    in_quote = False
    escape = False
    for i in range(start, len(text)):
        char = text[i]
        if escape:
            escape = False
        elif char == _BACKSLASH:
            escape = True
        elif char == '"':
            in_quote = not in_quote
        elif not in_quote:
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    return i
    return -1


def extract_tablegen_traits(text: str) -> list[str]:
    """Extract and parse TableGen traits from a definition or arguments string.

    Args:
        text: TableGen source text containing trait definitions (e.g. '[Trait1, Trait2]').

    Returns:
        List of parsed trait names with nested parameters preserved.
    """
    start = _find_unquoted_open_bracket(text)
    if start == -1:
        return []
    end = _find_matching_close_bracket(text, start)
    if end == -1:
        return []

    raw_elems = split_tablegen_list(text[start + 1 : end])
    traits: list[str] = []
    for elem in raw_elems:
        cleaned = elem.strip()
        if cleaned and not cleaned.startswith("//"):
            traits.append(_TRAIT_INTERFACE_ALIASES.get(cleaned, cleaned))
    return traits


__all__ = [
    "DocstringMetadata",
    "DocstringParamMeta",
    "DocstringReturnMeta",
    "extract_griffe_docstring_metadata",
    "extract_tablegen_traits",
    "get_all_members",
    "get_framework_docstring_parser",
    "parse_docstring_with_griffe",
    "resolve_griffe_parser",
    "split_tablegen_list",
    "strip_sphinx_roles",
]
