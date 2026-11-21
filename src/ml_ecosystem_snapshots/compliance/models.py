"""Immutable Pydantic report schemas for compliance scoring and snippet verification."""

from __future__ import annotations

import typing

import pydantic
import typing_extensions

ParamSignatureTuple: typing_extensions.TypeAlias = tuple[str, str, "str | None", "str | None"]


class NamespaceAlignmentRule(pydantic.BaseModel):
    """Immutable rule set for normalizing and aligning target symbol paths to reference namespaces."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    prefix_map: tuple[tuple[str, str], ...] = ()
    subpackage_roots: tuple[str, ...] = ()
    node_prefix_rewrites: tuple[tuple[str, str], ...] = ()
    path_rewrites: tuple[tuple[str, str], ...] = ()

    @pydantic.field_validator("prefix_map", "node_prefix_rewrites", mode="before")
    @classmethod
    def _coerce_string_pairs(
        cls,
        raw: typing.Mapping[str, str] | typing.Sequence[tuple[str, str]],
    ) -> tuple[tuple[str, str], ...]:
        """Coerce mapping or pair sequence inputs into an immutable tuple of `(key, value)` pairs.

        Args:
            raw: Mapping or sequence of `(str, str)` pairs.

        Returns:
            Immutable tuple of `(str, str)` pairs.
        """
        if isinstance(raw, typing.Mapping):
            return tuple((str(k), str(v)) for k, v in raw.items())
        return tuple((str(k), str(v)) for k, v in raw)

    def lookup_prefix(self, key: str) -> str | None:
        """Look up `key` inside `prefix_map`.

        Args:
            key: Exact symbol or namespace prefix key.

        Returns:
            Mapped canonical namespace string if present, else `None`.
        """
        for src_key, dst_val in self.prefix_map:
            if src_key == key:
                return dst_val
        return None

    def lookup_node_rewrite(self, key: str) -> str | None:
        """Look up `key` inside `node_prefix_rewrites`.

        Args:
            key: Node prefix string to match.

        Returns:
            Replacement format string if present, else `None`.
        """
        for src_key, dst_val in self.node_prefix_rewrites:
            if src_key == key:
                return dst_val
        return None


class ComplianceMismatchEntry(pydantic.BaseModel):
    """Immutable mismatch entry within `ComplianceScoreReport`."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    api_path: str
    expected: tuple[ParamSignatureTuple, ...] = ()
    actual: tuple[ParamSignatureTuple, ...] = ()
    reason: str | None = None


class ComplianceScoreReport(pydantic.BaseModel):
    """Immutable compliance report returned by `score_compliance`."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    score_percentage: float = 0.0
    total_endpoints: int = 0
    matched: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    mismatched: tuple[ComplianceMismatchEntry, ...] = ()
    opaque_signatures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class SnippetComplianceReport(pydantic.BaseModel):
    """Immutable compliance report returned by MLIR, WGSL, SASS, and RDNA snippet verifiers."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    is_compliant: bool = False
    total_ops: int = 0
    verified_ops: int = 0
    total_instructions: int = 0
    verified_instructions: int = 0
    errors: tuple[str, ...] = ()


__all__ = [
    "ComplianceMismatchEntry",
    "ComplianceScoreReport",
    "NamespaceAlignmentRule",
    "ParamSignatureTuple",
    "SnippetComplianceReport",
]
