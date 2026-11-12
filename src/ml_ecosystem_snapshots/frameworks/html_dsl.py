"""HTML DSL API Snapshot Extractor.

Provides a static snapshot of standard HTML tags and attributes for the HTML DSL.
"""

from __future__ import annotations

from ml_switcheroo_ir.schema.ghost import (
    GhostParam,
    GhostRef,
    ParameterKind,
    SemanticTier,
)

from ml_ecosystem_snapshots.frameworks._base import (
    CollectorDomain,
    DomainCollectorAdapter,
    SymbolKind,
)

_HTML_TAGS = [
    "div",
    "span",
    "a",
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "ul",
    "ol",
    "li",
    "table",
    "tr",
    "td",
    "th",
    "thead",
    "tbody",
    "img",
    "form",
    "input",
    "button",
    "select",
    "option",
    "textarea",
    "label",
    "br",
    "hr",
    "strong",
    "em",
    "head",
    "body",
    "html",
    "title",
    "meta",
    "link",
    "style",
    "script",
    "nav",
    "header",
    "footer",
    "main",
    "section",
    "article",
    "aside",
    "figure",
    "figcaption",
]


def collect_domain(include_nonpublic: bool = False) -> list[GhostRef]:
    """Collect HTML DSL element signatures without `SemanticTier` shoehorning.

    Args:
        include_nonpublic: Whether to include non-public elements.

    Returns:
        List of `GhostRef` items representing HTML DSL elements.
    """
    del include_nonpublic
    refs: list[GhostRef] = []
    for tag in _HTML_TAGS:
        params = [
            GhostParam(name="children", kind=ParameterKind.POSITIONAL_OR_KEYWORD),
            GhostParam(name="id", kind=ParameterKind.KEYWORD_ONLY),
            GhostParam(name="class_name", kind=ParameterKind.KEYWORD_ONLY),
            GhostParam(name="style", kind=ParameterKind.KEYWORD_ONLY),
            GhostParam(name="kwargs", kind=ParameterKind.VAR_KEYWORD),
        ]

        refs.append(
            GhostRef(
                name=tag,
                api_path=f"html.{tag}",
                kind=SymbolKind.ELEMENT.value,
                params=params,
                docstring=f"HTML <{tag}> element.",
            )
        )
    return refs


DOMAIN_COLLECTOR = DomainCollectorAdapter(CollectorDomain.DOCUMENT_DSL, "elements", collect_domain)


def collect_api(category: SemanticTier, include_nonpublic: bool = False) -> list[GhostRef]:
    """Entrypoint to collect the HTML DSL API signature.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered.
    """
    if category != SemanticTier.UTIL:
        return []
    return collect_domain(include_nonpublic)
