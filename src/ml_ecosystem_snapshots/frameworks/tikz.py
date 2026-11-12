"""TikZ DSL API Snapshot Extractor.

Provides a static snapshot of standard TikZ drawing commands and node shapes.
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

_TIKZ_COMMANDS = [
    "draw",
    "fill",
    "filldraw",
    "pattern",
    "shade",
    "shadedraw",
    "clip",
    "useasboundingbox",
    "node",
    "coordinate",
    "matrix",
    "pic",
    "path",
]

_TIKZ_SHAPES = [
    "circle",
    "rectangle",
    "ellipse",
    "diamond",
    "triangle",
    "star",
    "regular polygon",
    "cylinder",
    "kite",
    "dart",
    "trapezium",
]


def collect_domain(include_nonpublic: bool = False) -> list[GhostRef]:
    """Collect TikZ DSL command and shape node signatures without `SemanticTier` shoehorning.

    Args:
        include_nonpublic: Whether to include non-public symbols.

    Returns:
        List of `GhostRef` items representing TikZ commands and shape nodes.
    """
    del include_nonpublic
    refs: list[GhostRef] = []

    for cmd in _TIKZ_COMMANDS:
        params = [
            GhostParam(name="options", kind=ParameterKind.KEYWORD_ONLY),
            GhostParam(name="path", kind=ParameterKind.POSITIONAL_OR_KEYWORD),
        ]
        refs.append(
            GhostRef(
                name=cmd,
                api_path=f"tikz.{cmd}",
                kind=SymbolKind.FUNCTION.value,
                params=params,
                docstring=f"TikZ \\{cmd} command.",
            )
        )

    for shape in _TIKZ_SHAPES:
        clean_name = shape.replace(" ", "_")
        params = [
            GhostParam(name="name", kind=ParameterKind.POSITIONAL_OR_KEYWORD),
            GhostParam(name="at", kind=ParameterKind.KEYWORD_ONLY),
            GhostParam(name="options", kind=ParameterKind.KEYWORD_ONLY),
        ]
        refs.append(
            GhostRef(
                name=clean_name,
                api_path=f"tikz.shape.{clean_name}",
                kind=SymbolKind.ELEMENT.value,
                params=params,
                docstring=f"TikZ {shape} shape node.",
            )
        )

    return refs


DOMAIN_COLLECTOR = DomainCollectorAdapter(CollectorDomain.DOCUMENT_DSL, "elements", collect_domain)


def collect_api(category: SemanticTier, include_nonpublic: bool = False) -> list[GhostRef]:
    """Entrypoint to collect the TikZ DSL API signature.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered.
    """
    if category != SemanticTier.UTIL:
        return []
    return collect_domain(include_nonpublic)
