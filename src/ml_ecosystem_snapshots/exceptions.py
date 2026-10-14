"""Structured domain exception hierarchy for ml_ecosystem_snapshots.

Provides a unified exception tree rooted at `SnapshotError` so callers and
internal subsystems can catch and classify specific failure modes without
resorting to broad `Exception` handlers.
"""

from __future__ import annotations


class SnapshotError(Exception):
    """Root base exception for all ml_ecosystem_snapshots domain errors."""


class SnapshotNotFoundError(SnapshotError, FileNotFoundError):
    """Raised when a requested framework or hardware snapshot cannot be located."""


class SnapshotSchemaError(SnapshotError, ValueError):
    """Raised when a snapshot payload violates the expected envelope or GhostRef schema."""


class IntrospectionError(SnapshotError):
    """Raised when live or static introspection of a symbol fails."""


class CExtensionSignatureError(IntrospectionError):
    """Raised when parsing a C-extension or pybind11 docstring signature fails."""


class ComplianceVerificationError(SnapshotError):
    """Raised when a target source tree fails API compliance verification against a snapshot."""


class GroundingValidationError(SnapshotError):
    """Raised when an instruction, operation, or API call fails grounding validation."""


class OfflineModeNetworkError(SnapshotError, RuntimeError):
    """Raised when an outbound network operation is attempted while offline mode is active."""


__all__ = [
    "CExtensionSignatureError",
    "ComplianceVerificationError",
    "GroundingValidationError",
    "IntrospectionError",
    "OfflineModeNetworkError",
    "SnapshotError",
    "SnapshotNotFoundError",
    "SnapshotSchemaError",
]
