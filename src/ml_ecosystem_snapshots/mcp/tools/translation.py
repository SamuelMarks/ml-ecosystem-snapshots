"""MCP tool module for cross-framework concept argument translation."""

from __future__ import annotations

from typing import Any, Dict
from ml_ecosystem_snapshots.mcp_server import (
    CONCEPT_ALIAS_MAP,
    translate_concept_arguments,
)

CONCEPT_ARG_MAPPINGS: Dict[str, Any] = CONCEPT_ALIAS_MAP

__all__ = ["CONCEPT_ALIAS_MAP", "CONCEPT_ARG_MAPPINGS", "translate_concept_arguments"]
