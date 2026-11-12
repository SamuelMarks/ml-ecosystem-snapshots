"""Thin MCP adapter for cross-framework concept argument translation."""

from __future__ import annotations

from ml_ecosystem_snapshots.grounding.concepts import (
    CONCEPT_ALIAS_MAP,
    DEFAULT_CONCEPT_MAP,
    load_concept_map,
    translate_concept_arguments,
)

__all__ = [
    "CONCEPT_ALIAS_MAP",
    "DEFAULT_CONCEPT_MAP",
    "load_concept_map",
    "translate_concept_arguments",
]
