"""Utility functions for inspecting and extracting information from Python modules."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ml_ecosystem_snapshots.c_extension_sig import (
    CExtensionSignature,
    _normalize_c_sig_args,
    _parse_c_extension_sig_str,
    extract_c_extension_signature,
)
from ml_ecosystem_snapshots.network_env import (
    fetch_url,
    get_custom_snapshots_paths,
    is_offline_mode,
    load_json_data,
    stream_snapshot_items,
)


def get_all_members(module: Any) -> List[Tuple[str, Any]]:
    """Safely extracts all members from a module, bypassing __getattr__ obfuscation.

    Many modern ML frameworks use lazy-loading modules that define __getattr__
    and __all__ but do not correctly implement __dir__. This function combines
    dir() and __all__ to forcefully resolve hidden members.

    Args:
        module: The module to inspect.

    Returns:
        A list of (name, object) tuples.
    """
    members = {}

    for name in dir(module):
        try:
            members[name] = getattr(module, name)
        except (
            AttributeError,
            TypeError,
            ValueError,
            KeyError,
            IndexError,
            SyntaxError,
            OSError,
            ImportError,
            RuntimeError,
            Exception,
        ):
            pass

    if hasattr(module, "__all__"):
        for name in module.__all__:
            if name not in members:
                try:
                    members[name] = getattr(module, name)
                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    KeyError,
                    IndexError,
                    SyntaxError,
                    OSError,
                    ImportError,
                    RuntimeError,
                    Exception,
                ):
                    pass

    return list(members.items())


def get_framework_docstring_parser(
    framework_name: str, docstring: Optional[str] = None
) -> str:
    """Determine the most appropriate docstring parser for a framework.

    Args:
        framework_name: Name of the framework or top-level module (e.g. 'torch', 'numpy').
        docstring: Optional docstring text to inspect for format heuristics.

    Returns:
        The docstring parser style identifier ('rest', 'numpy', 'google', or 'auto').
    """
    nl = chr(10)
    if docstring and isinstance(docstring, str):
        if any(
            marker in docstring
            for marker in (":param ", ":type ", ":return:", ":rtype:", ":raises:")
        ):
            return "rest"
        if re.search(
            nl + r"(Parameters|Returns|Raises)" + nl + r"\s*[-=]{3,}", docstring
        ):
            return "numpy"
        if re.search(nl + r"(Args|Arguments|Returns|Raises):\s*" + nl, docstring):
            return "google"

    name_clean = framework_name.lower().replace("-", "_")
    if name_clean in (
        "torch",
        "pytorch",
        "tensorflow",
        "tf",
        "sklearn",
        "scikit_learn",
        "scipy",
    ):
        return "rest"
    if name_clean in ("numpy", "np"):
        return "numpy"
    if name_clean in ("jax", "flax", "keras", "transformers", "huggingface"):
        return "google"

    return "rest"


def resolve_griffe_parser(parser_name: str) -> Any:
    """Resolve a parser identifier into a valid Griffe Parser instance.

    Args:
        parser_name: The parser name ('rest', 'sphinx', 'google', 'numpy', 'auto').

    Returns:
        A Griffe Parser instance or None.
    """
    try:
        import griffe

        if parser_name == "rest":
            return getattr(griffe.Parser, "rest", griffe.Parser.sphinx)
        return griffe.Parser(parser_name)
    except Exception:
        return None


def parse_docstring_with_griffe(
    docstring: str, parser_name: Optional[str] = None
) -> List[Any]:
    """Parse a docstring using Griffe with dynamic fallback across conventions.

    Args:
        docstring: The docstring text to parse.
        parser_name: Optional starting parser ('rest', 'google', 'numpy').

    Returns:
        A list of parsed Griffe DocstringSection objects.
    """
    try:
        import griffe
    except ImportError:
        return []

    initial_parser = parser_name or "rest"
    candidates = [initial_parser]
    for alt in ("rest", "google", "numpy"):
        if alt not in candidates:
            candidates.append(alt)

    best_sections: List[Any] = []

    for candidate in candidates:
        resolved = resolve_griffe_parser(candidate)
        if not resolved:
            continue
        try:
            doc_obj = griffe.Docstring(docstring)
            sections: List[Any] = griffe.parse(doc_obj, resolved)
            has_structured = any(
                getattr(s.kind, "value", str(s.kind))
                in ("parameters", "returns", "raises")
                for s in sections
            )
            if has_structured:
                return sections
            if not best_sections and sections:
                best_sections = sections
        except (
            AttributeError,
            TypeError,
            ValueError,
            KeyError,
            IndexError,
            SyntaxError,
            OSError,
            ImportError,
            RuntimeError,
            Exception,
        ):
            pass

    return best_sections


def strip_sphinx_roles(text: Optional[str]) -> Optional[str]:
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


def extract_griffe_docstring_metadata(
    docstring: str, parser_name: Optional[str] = None
) -> Dict[str, Any]:
    """Extract structured parameters, return types, and exceptions from a docstring.

    Args:
        docstring: The docstring text to parse.
        parser_name: Optional parser name to prioritize.

    Returns:
        A dictionary containing 'params', 'returns', and 'raises'.
    """
    sections = parse_docstring_with_griffe(docstring, parser_name=parser_name)

    params: Dict[str, Dict[str, Optional[str]]] = {}
    returns: Optional[Dict[str, Optional[str]]] = None
    raises: List[str] = []

    for section in sections:
        kind_name = getattr(section.kind, "value", str(section.kind))
        if (
            kind_name == "parameters"
            and hasattr(section, "value")
            and isinstance(section.value, list)
        ):
            for item in section.value:
                p_name = getattr(item, "name", None)
                if p_name:
                    annotation = getattr(item, "annotation", None)
                    value = getattr(item, "value", None)
                    desc = getattr(item, "description", None)
                    cleaned_typ = (
                        strip_sphinx_roles(str(annotation))
                        if annotation is not None
                        else None
                    )
                    params[p_name] = {
                        "doc": str(desc) if desc else "",
                        "typ": cleaned_typ,
                        "default": str(value) if value is not None else None,
                    }
        elif (
            kind_name == "returns"
            and hasattr(section, "value")
            and isinstance(section.value, list)
        ):
            for item in section.value:
                annotation = getattr(item, "annotation", None)
                desc = getattr(item, "description", None)
                cleaned_typ = (
                    strip_sphinx_roles(str(annotation))
                    if annotation is not None
                    else None
                )
                returns = {
                    "doc": str(desc) if desc else "",
                    "typ": cleaned_typ,
                }
        elif (
            kind_name == "raises"
            and hasattr(section, "value")
            and isinstance(section.value, list)
        ):
            for item in section.value:
                annotation = (
                    getattr(item, "annotation", None)
                    or getattr(item, "name", None)
                    or getattr(item, "value", None)
                )
                if annotation:
                    exc_name = strip_sphinx_roles(str(annotation)) or str(annotation)
                    if exc_name not in raises:
                        raises.append(exc_name)

    for exc in re.findall(r":(?:raises?|except)\s+([A-Za-z0-9_.]+):", docstring):
        if exc in params:
            del params[exc]
        if exc not in raises:
            raises.append(exc)

    return {"params": params, "returns": returns, "raises": raises}


def split_tablegen_list(raw_list_str: str) -> List[str]:
    """Split a comma-separated TableGen list while respecting nested brackets, angles, parens, and quotes.

    Args:
        raw_list_str: Content inside outer brackets [ ... ].

    Returns:
        List of trimmed TableGen element strings.
    """
    bs = chr(92)
    elements: List[str] = []
    current: List[str] = []
    depth_bracket = 0
    depth_angle = 0
    depth_paren = 0
    in_quote = False
    escape = False

    cleaned_str = re.sub(r"/\*.*?\*/", "", raw_list_str, flags=re.DOTALL)
    cleaned_str = re.sub(r"//.*", "", cleaned_str)

    for char in cleaned_str:
        if escape:
            current.append(char)
            escape = False
            continue
        if char == bs:
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

        if char == "[":
            depth_bracket += 1
            current.append(char)
        elif char == "]":
            depth_bracket -= 1
            current.append(char)
        elif char == "<":
            depth_angle += 1
            current.append(char)
        elif char == ">":
            depth_angle -= 1
            current.append(char)
        elif char == "(":
            depth_paren += 1
            current.append(char)
        elif char == ")":
            depth_paren -= 1
            current.append(char)
        elif (
            char == "," and depth_bracket == 0 and depth_angle == 0 and depth_paren == 0
        ):
            elem = "".join(current).strip()
            if elem and not elem.startswith("//"):
                elements.append(elem)
            current = []
        else:
            current.append(char)

    if current:
        elem = "".join(current).strip()
        if elem and not elem.startswith("//"):
            elements.append(elem)
    return elements


def extract_tablegen_traits(text: str) -> List[str]:
    """Extract and parse TableGen traits from a definition or arguments string.

    Args:
        text: TableGen source text containing trait definitions (e.g. '[Trait1, Trait2]').

    Returns:
        List of parsed trait names with nested parameters preserved.
    """
    bs = chr(92)
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if esc:
            esc = False
            continue
        if ch == bs:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if not in_str and ch == "[":
            start = i
            break

    if start == -1:
        return []
    depth = 0
    in_quote = False
    escape = False
    end = -1
    for i in range(start, len(text)):
        char = text[i]
        if escape:
            escape = False
            continue
        if char == bs:
            escape = True
            continue
        if char == '"':
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end != -1:
        inner = text[start + 1 : end]
        raw_elems = split_tablegen_list(inner)
        traits: List[str] = []
        for elem in raw_elems:
            cleaned = elem.strip()
            if cleaned.startswith("//"):
                continue
            if cleaned in ('"getAsmResultNames"', '["getAsmResultNames"]'):
                traits.append("OpAsmOpInterface::getAsmResultNames")
            elif cleaned in (
                '"inferReturnTypeComponents"',
                '["inferReturnTypeComponents"',
            ):
                traits.append("InferTypeOpInterface::inferReturnTypeComponents")
            elif cleaned in ('"inferResultRanges"', '["inferResultRanges"'):
                traits.append("InferIntRangeInterface::inferResultRanges")
            elif cleaned:
                traits.append(cleaned)
        return traits
    return []


__all__ = [
    "CExtensionSignature",
    "_normalize_c_sig_args",
    "_parse_c_extension_sig_str",
    "extract_c_extension_signature",
    "extract_griffe_docstring_metadata",
    "extract_tablegen_traits",
    "fetch_url",
    "get_all_members",
    "get_custom_snapshots_paths",
    "get_framework_docstring_parser",
    "is_offline_mode",
    "load_json_data",
    "parse_docstring_with_griffe",
    "resolve_griffe_parser",
    "split_tablegen_list",
    "stream_snapshot_items",
    "strip_sphinx_roles",
]
