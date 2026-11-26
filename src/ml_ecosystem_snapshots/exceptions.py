"""Imperative CLI / MCP boundary exception wrapper around `SnapshotDomainError`.

Pure functional core subsystems in `ml_ecosystem_snapshots` communicate failures
exclusively through closed `SnapshotDomainError` algebraic data types inside
`Result` and `Validation`. At imperative CLI, MCP, or stateful repository edges,
`raise_snapshot_boundary` and `raise_validation_boundary` project `Err` / `Invalid`
outcomes into a single `SnapshotError(Exception)` carrier preserving `.domain_error`.
"""

from __future__ import annotations

import typing

import typing_extensions
from ml_switcheroo_ir.typeclasses.algebra import NonEmpty
from ml_switcheroo_ir.typeclasses.outcome import Err, Invalid, Ok, Result, Valid, Validation

from ml_ecosystem_snapshots.errors import (
    SnapshotDomainError,
    format_snapshot_error,
)

_T = typing.TypeVar("_T")


class SnapshotError(Exception):
    """Imperative boundary exception wrapping a typed `SnapshotDomainError` value.

    Attributes:
        domain_error: Structured closed `SnapshotDomainError` ADT payload.
    """

    domain_error: SnapshotDomainError

    def __init__(self, domain_error: SnapshotDomainError) -> None:
        """Initialize `SnapshotError` from a `SnapshotDomainError` variant.

        Args:
            domain_error: Closed `SnapshotDomainError` ADT instance describing the failure.
        """
        self.domain_error = domain_error
        self.args = (format_snapshot_error(domain_error),)


def to_snapshot_exception(domain_error: SnapshotDomainError) -> SnapshotError:
    """Wrap a pure `SnapshotDomainError` value inside an imperative `SnapshotError`.

    Args:
        domain_error: Closed `SnapshotDomainError` variant.

    Returns:
        Constructed `SnapshotError` carrying `domain_error`.
    """
    return SnapshotError(domain_error)


def raise_snapshot_boundary(res: Result[SnapshotDomainError, _T]) -> _T:
    """Unwrap `Ok(value)` or raise `SnapshotError(res.error)` at an imperative boundary.

    Args:
        res: Total `Result[SnapshotDomainError, _T]` computation outcome.

    Returns:
        Unwrapped success value of type `_T`.

    Raises:
        SnapshotError: When `res` is `Err(domain_error)`.
    """
    if isinstance(res, Ok):
        return res.value
    if isinstance(res, Err):
        raise SnapshotError(res.error)
    typing_extensions.assert_never(res)


def raise_validation_boundary(val: Validation[NonEmpty[SnapshotDomainError], _T]) -> _T:
    """Unwrap `Valid(value)` or raise `SnapshotError(val.errors.head)` at an imperative boundary.

    Args:
        val: Applicative `Validation[NonEmpty[SnapshotDomainError], _T]` outcome.

    Returns:
        Unwrapped validated value of type `_T`.

    Raises:
        SnapshotError: When `val` is `Invalid(errors)`, wrapping `errors.head`.
    """
    if isinstance(val, Valid):
        return val.value
    if isinstance(val, Invalid):
        raise SnapshotError(val.errors.head)
    typing_extensions.assert_never(val)


__all__ = [
    "SnapshotError",
    "raise_snapshot_boundary",
    "raise_validation_boundary",
    "to_snapshot_exception",
]
