"""Utility functions for inspecting and extracting information from Python modules."""

from __future__ import annotations

import logging
import re
import types
import typing

import griffe
import pydantic

_LOGGER = logging.getLogger(__name__)


class DocstringParamMeta(pydantic.BaseModel):
    """Extracted parameter metadata from a parsed docstring."""

    model_config = pydantic.ConfigDict(extra="forbid")

    doc: str = ""
    typ: str | None = None
    default: str | None = None

    @typing.overload
    def __getitem__(self, key: typing.Literal["doc"]) -> str: ...

    @typing.overload
    def __getitem__(self, key: typing.Literal["typ", "default"]) -> str | None: ...

    @typing.overload
    def __getitem__(self, key: str) -> str | None: ...

    def __getitem__(self, key: str) -> str | None:
        """Retrieve a docstring parameter field by key.

        Args:
            key: Field name ('doc', 'typ', or 'default').

        Returns:
            Field value.
        """
        val: str | None = getattr(self, key)
        return val


class DocstringReturnMeta(pydantic.BaseModel):
    """Extracted return metadata from a parsed docstring."""

    model_config = pydantic.ConfigDict(extra="forbid")

    doc: str = ""
    typ: str | None = None

    @typing.overload
    def __getitem__(self, key: typing.Literal["doc"]) -> str: ...

    @typing.overload
    def __getitem__(self, key: typing.Literal["typ"]) -> str | None: ...

    @typing.overload
    def __getitem__(self, key: str) -> str | None: ...

    def __getitem__(self, key: str) -> str | None:
        """Retrieve a docstring return field by key.

        Args:
            key: Field name ('doc' or 'typ').

        Returns:
            Field value.
        """
        val: str | None = getattr(self, key)
        return val


class DocstringMetadata(pydantic.BaseModel):
    """Structured metadata extracted from a docstring via Griffe."""

    model_config = pydantic.ConfigDict(extra="forbid")

    params: dict[str, DocstringParamMeta] = pydantic.Field(default_factory=dict)
    returns: DocstringReturnMeta | None = None
    raises: list[str] = pydantic.Field(default_factory=list)

    @typing.overload
    def __getitem__(
        self, key: typing.Literal["params"]
    ) -> dict[str, DocstringParamMeta]: ...

    @typing.overload
    def __getitem__(
        self, key: typing.Literal["returns"]
    ) -> DocstringReturnMeta | None: ...

    @typing.overload
    def __getitem__(self, key: typing.Literal["raises"]) -> list[str]: ...

    @typing.overload
    def __getitem__(
        self, key: str
    ) -> dict[str, DocstringParamMeta] | DocstringReturnMeta | list[str] | None: ...

    def __getitem__(
        self, key: str
    ) -> dict[str, DocstringParamMeta] | DocstringReturnMeta | list[str] | None:
        """Retrieve a docstring metadata field by key.

        Args:
            key: Field name ('params', 'returns', or 'raises').

        Returns:
            Field value.
        """
        val: dict[str, DocstringParamMeta] | DocstringReturnMeta | list[str] | None = (
            getattr(self, key)
        )
        return val

    @typing.overload
    def get(
        self,
        key: typing.Literal["params"],
        default: dict[str, DocstringParamMeta] | None = None,
    ) -> dict[str, DocstringParamMeta]: ...

    @typing.overload
    def get(
        self,
        key: typing.Literal["returns"],
        default: DocstringReturnMeta | None = None,
    ) -> DocstringReturnMeta | None: ...

    @typing.overload
    def get(
        self,
        key: typing.Literal["raises"],
        default: list[str] | None = None,
    ) -> list[str]: ...

    def get(
        self,
        key: str,
        default: (
            dict[str, DocstringParamMeta] | DocstringReturnMeta | list[str] | None
        ) = None,
    ) -> dict[str, DocstringParamMeta] | DocstringReturnMeta | list[str] | None:
        """Retrieve a docstring metadata field by key with default.

        Args:
            key: Field name.
            default: Default value when absent.

        Returns:
            Field value or `default`.
        """
        val: dict[str, DocstringParamMeta] | DocstringReturnMeta | list[str] | None = (
            getattr(self, key, default)
        )
        return val


def get_all_members(
    module: types.ModuleType | type,
) -> list[tuple[str, object]]:
    """Safely extract all members from a module, bypassing __getattr__ obfuscation.

    Many modern ML frameworks use lazy-loading modules that define __getattr__
    and __all__ but do not correctly implement __dir__. This function combines
    dir() and __all__ to forcefully resolve hidden members.

    Args:
        module: The module to inspect.

    Returns:
        A list of (name, object) tuples.
    """
    members: list[tuple[str, object]] = []
    seen: set[str] = set()

    for name in dir(module):
        try:
            members.append((name, getattr(module, name)))
            seen.add(name)
        except Exception as exc:
            _LOGGER.debug("Skipping inaccessible member %s: %s", name, exc)

    if hasattr(module, "__all__"):
        for name in getattr(module, "__all__"):
            if name not in seen:
                try:
                    members.append((name, getattr(module, name)))
                    seen.add(name)
                except Exception as exc:
                    _LOGGER.debug(
                        "Skipping inaccessible __all__ member %s: %s", name, exc
                    )

    return members


def get_framework_docstring_parser(
    framework_name: str, docstring: str | None = None
) -> str:
    """Determine the most appropriate docstring parser for a framework.

    Args:
        framework_name: Name of the framework or top-level module (e.g. 'torch', 'numpy').
        docstring: Optional docstring text to inspect for format heuristics.

    Returns:
        The docstring parser style identifier ('rest', 'numpy', 'google', or 'auto').
    """
    if docstring and isinstance(docstring, str):
        if any(
            marker in docstring
            for marker in (":param ", ":type ", ":return:", ":rtype:", ":raises:")
        ):
            return "rest"
        if re.search(r"(?:^|\s)(Parameters|Returns|Raises)\s*[-=]{3,}", docstring):
            return "numpy"
        if re.search(r"(?:^|\s)(Args|Arguments|Returns|Raises):\s*", docstring):
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
            rest_parser: griffe.Parser = getattr(
                _griffe_mod.Parser, "rest", _griffe_mod.Parser.sphinx
            )
            return rest_parser
        return _griffe_mod.Parser(parser_name)
    except Exception:
        return None


def parse_docstring_with_griffe(
    docstring: str, parser_name: str | None = None
) -> list[griffe.DocstringSection]:
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
            sections: list[griffe.DocstringSection] = _griffe_mod.parse(
                doc_obj, resolved
            )
            has_structured = any(
                getattr(s.kind, "value", str(s.kind))
                in ("parameters", "returns", "raises")
                for s in sections
            )
            if has_structured:
                return sections
            if not best_sections and sections:
                best_sections = sections
        except Exception as exc:
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


def extract_griffe_docstring_metadata(
    docstring: str, parser_name: str | None = None
) -> DocstringMetadata:
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
                    params[str(p_name)] = DocstringParamMeta(
                        doc=str(desc) if desc else "",
                        typ=cleaned_typ,
                        default=str(value) if value is not None else None,
                    )
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
                returns = DocstringReturnMeta(
                    doc=str(desc) if desc else "",
                    typ=cleaned_typ,
                )
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

    return DocstringMetadata(params=params, returns=returns, raises=raises)


_BACKSLASH = chr(92)


def split_tablegen_list(raw_list_str: str) -> list[str]:
    """Split a comma-separated TableGen list while respecting nested brackets, angles, parens, and quotes.

    Args:
        raw_list_str: Content inside outer brackets [ ... ].

    Returns:
        List of trimmed TableGen element strings.
    """
    elements: list[str] = []
    current: list[str] = []
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


def extract_tablegen_traits(text: str) -> list[str]:
    """Extract and parse TableGen traits from a definition or arguments string.

    Args:
        text: TableGen source text containing trait definitions (e.g. '[Trait1, Trait2]').

    Returns:
        List of parsed trait names with nested parameters preserved.
    """
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if esc:
            esc = False
            continue
        if ch == _BACKSLASH:
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
        if char == _BACKSLASH:
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
        traits: list[str] = []
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
