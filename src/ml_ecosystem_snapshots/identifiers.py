"""Domain `NewType` identifiers, total smart constructors, and algebraic witnesses.

Provides zero-cost static `NewType` distinctions for distinct string and integer
domain concepts (`FrameworkId`, `SnapshotVersion`, `ApiPath`, `SymbolPath`,
`CategoryName`, `Sha256Hex`, plus re-exported `DialectDomain`, `OpName`,
`PositiveInt`, `NonNegativeInt`, `NonBoolInt` from `ml_switcheroo_ir.identifiers`),
paired with total `Result[IdentifierValidationFault, T]` smart constructors and
explicit `Eq`, `Ord`, and `HashableWitness` typeclass instances.
"""

from __future__ import annotations

import hashlib
import re
import typing

from ml_switcheroo_ir.identifiers import (
    DIALECT_DOMAIN_EQ as DIALECT_DOMAIN_EQ,
)
from ml_switcheroo_ir.identifiers import (
    DIALECT_DOMAIN_HASH as DIALECT_DOMAIN_HASH,
)
from ml_switcheroo_ir.identifiers import (
    DIALECT_DOMAIN_ORD as DIALECT_DOMAIN_ORD,
)
from ml_switcheroo_ir.identifiers import (
    OP_NAME_EQ as OP_NAME_EQ,
)
from ml_switcheroo_ir.identifiers import (
    OP_NAME_HASH as OP_NAME_HASH,
)
from ml_switcheroo_ir.identifiers import (
    OP_NAME_ORD as OP_NAME_ORD,
)
from ml_switcheroo_ir.identifiers import (
    DialectDomain as DialectDomain,
)
from ml_switcheroo_ir.identifiers import (
    IdentifierKind as IdentifierKind,
)
from ml_switcheroo_ir.identifiers import (
    IdentifierParseError as IdentifierParseError,
)
from ml_switcheroo_ir.identifiers import (
    NonBoolInt as NonBoolInt,
)
from ml_switcheroo_ir.identifiers import (
    NonNegativeInt as NonNegativeInt,
)
from ml_switcheroo_ir.identifiers import (
    OpName as OpName,
)
from ml_switcheroo_ir.identifiers import (
    PositiveInt as PositiveInt,
)
from ml_switcheroo_ir.identifiers import (
    is_non_bool_int as is_non_bool_int,
)
from ml_switcheroo_ir.identifiers import (
    parse_dialect_domain as parse_dialect_domain,
)
from ml_switcheroo_ir.identifiers import (
    parse_non_negative_int as parse_non_negative_int,
)
from ml_switcheroo_ir.identifiers import (
    parse_op_name as parse_op_name,
)
from ml_switcheroo_ir.identifiers import (
    parse_positive_int as parse_positive_int,
)
from ml_switcheroo_ir.typeclasses.algebra import (
    INT_EQ,
    INT_HASH,
    INT_ORD,
    STR_EQ,
    STR_HASH,
    STR_ORD,
    Eq,
    HashableWitness,
    Ord,
    contramap_eq,
    contramap_hash,
    contramap_ord,
    lexical_pair_ord,
)
from ml_switcheroo_ir.typeclasses.outcome import Err, Ok, Result

from ml_ecosystem_snapshots.errors import IdentifierValidationFault

FrameworkId = typing.NewType("FrameworkId", str)
SnapshotVersion = typing.NewType("SnapshotVersion", str)
ApiPath = typing.NewType("ApiPath", str)
SymbolPath = typing.NewType("SymbolPath", str)
CategoryName = typing.NewType("CategoryName", str)
Sha256Hex = typing.NewType("Sha256Hex", str)

_FRAMEWORK_ID_RE: typing.Final[re.Pattern[str]] = re.compile(r"^[a-z0-9_]+$")
_CATEGORY_NAME_RE: typing.Final[re.Pattern[str]] = re.compile(r"^[a-zA-Z0-9_.-]+$")
_SHA256_HEX_RE: typing.Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_SYMBOL_PATH_RE: typing.Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:[.][A-Za-z_][A-Za-z0-9_]*)*$")


def _framework_id_to_str(value: FrameworkId) -> str:
    """Project `FrameworkId` to its underlying `str` representation.

    Args:
        value: Validated `FrameworkId`.

    Returns:
        Underlying normalized framework identifier string.
    """
    return str(value)


def _snapshot_version_to_str(value: SnapshotVersion) -> str:
    """Project `SnapshotVersion` to its underlying `str` representation.

    Args:
        value: Validated `SnapshotVersion`.

    Returns:
        Underlying version string.
    """
    return str(value)


def _api_path_to_str(value: ApiPath) -> str:
    """Project `ApiPath` to its underlying `str` representation.

    Args:
        value: Validated `ApiPath`.

    Returns:
        Underlying dotted API path string.
    """
    return str(value)


def _api_path_to_len_and_str(value: ApiPath) -> tuple[int, str]:
    """Project `ApiPath` into `(len(path), path)` for canonical alias selection.

    Args:
        value: Validated `ApiPath`.

    Returns:
        Pair `(len(value), str(value))`.
    """
    raw = str(value)
    return (len(raw), raw)


def _symbol_path_to_str(value: SymbolPath) -> str:
    """Project `SymbolPath` to its underlying `str` representation.

    Args:
        value: Validated `SymbolPath`.

    Returns:
        Underlying dotted symbol path string.
    """
    return str(value)


def _category_name_to_str(value: CategoryName) -> str:
    """Project `CategoryName` to its underlying `str` representation.

    Args:
        value: Validated `CategoryName`.

    Returns:
        Underlying category string.
    """
    return str(value)


def _sha256_hex_to_str(value: Sha256Hex) -> str:
    """Project `Sha256Hex` to its underlying `str` representation.

    Args:
        value: Validated `Sha256Hex`.

    Returns:
        Underlying 64-character lowercase hexadecimal digest.
    """
    return str(value)


def parse_framework_id(raw: str) -> Result[IdentifierValidationFault, FrameworkId]:
    """Validate and normalize a raw framework name into a canonical `FrameworkId`.

    Normalizes hyphens (`-`) to underscores (`_`) and lowercases ASCII characters.

    Args:
        raw: Candidate framework name string (e.g. `'torch'`, `'ml-switcheroo-ir'`).

    Returns:
        `Ok(FrameworkId)` if non-empty and matching `^[a-z0-9_]+$`, else `Err(IdentifierValidationFault)`.
    """
    normalized = raw.strip().lower().replace("-", "_")
    if not normalized:
        return Err(
            IdentifierValidationFault(
                field="FrameworkId",
                raw_value=raw,
                reason="FrameworkId must be non-empty.",
            )
        )
    if not _FRAMEWORK_ID_RE.match(normalized):
        return Err(
            IdentifierValidationFault(
                field="FrameworkId",
                raw_value=raw,
                reason="expected lowercase alphanumeric or underscore characters.",
            )
        )
    return Ok(FrameworkId(normalized))


def parse_snapshot_version(raw: str) -> Result[IdentifierValidationFault, SnapshotVersion]:
    """Validate and trim a raw version string into a `SnapshotVersion`.

    Args:
        raw: Candidate version tag (e.g. `'2.4.0'`, `'0.0.9'`, `'latest'`).

    Returns:
        `Ok(SnapshotVersion)` if non-empty and free of path separators, else `Err(IdentifierValidationFault)`.
    """
    cleaned = raw.strip()
    if not cleaned:
        return Err(
            IdentifierValidationFault(
                field="SnapshotVersion",
                raw_value=raw,
                reason="SnapshotVersion must be non-empty.",
            )
        )
    if "/" in cleaned or chr(92) in cleaned or ".." in cleaned:
        return Err(
            IdentifierValidationFault(
                field="SnapshotVersion",
                raw_value=raw,
                reason="path separators and traversal segments are forbidden.",
            )
        )
    return Ok(SnapshotVersion(cleaned))


def parse_api_path(raw: str) -> Result[IdentifierValidationFault, ApiPath]:
    """Validate and trim a dotted or qualified symbol path into an `ApiPath`.

    Args:
        raw: Candidate API path (e.g. `'torch.nn.Linear'`, `'arith.addi'`, `'FADD'`).

    Returns:
        `Ok(ApiPath)` if non-empty and free of whitespace, else `Err(IdentifierValidationFault)`.
    """
    cleaned = raw.strip()
    if not cleaned:
        return Err(
            IdentifierValidationFault(
                field="ApiPath",
                raw_value=raw,
                reason="ApiPath must be non-empty.",
            )
        )
    if any(ch.isspace() for ch in cleaned):
        return Err(
            IdentifierValidationFault(
                field="ApiPath",
                raw_value=raw,
                reason="whitespace is not allowed.",
            )
        )
    return Ok(ApiPath(cleaned))


def parse_symbol_path(raw: str) -> Result[IdentifierValidationFault, SymbolPath]:
    """Validate and trim a dotted Python/IR identifier path into a `SymbolPath`.

    Args:
        raw: Candidate dotted Python or dialect symbol path (e.g. `'torch.nn.Linear'`).

    Returns:
        `Ok(SymbolPath)` if every segment is a valid identifier, else `Err(IdentifierValidationFault)`.
    """
    cleaned = raw.strip()
    if not cleaned:
        return Err(
            IdentifierValidationFault(
                field="SymbolPath",
                raw_value=raw,
                reason="SymbolPath must be non-empty.",
            )
        )
    if not _SYMBOL_PATH_RE.match(cleaned):
        return Err(
            IdentifierValidationFault(
                field="SymbolPath",
                raw_value=raw,
                reason="expected dot-separated valid Python/IR identifier segments.",
            )
        )
    return Ok(SymbolPath(cleaned))


def parse_category_name(raw: str) -> Result[IdentifierValidationFault, CategoryName]:
    """Validate a snapshot category key into a `CategoryName`.

    Args:
        raw: Candidate category key (e.g. `'neural'`, `'array'`, `'operations'`).

    Returns:
        `Ok(CategoryName)` if non-empty and valid, else `Err(IdentifierValidationFault)`.
    """
    cleaned = raw.strip()
    if not cleaned:
        return Err(
            IdentifierValidationFault(
                field="CategoryName",
                raw_value=raw,
                reason="CategoryName must be non-empty.",
            )
        )
    if not _CATEGORY_NAME_RE.match(cleaned):
        return Err(
            IdentifierValidationFault(
                field="CategoryName",
                raw_value=raw,
                reason="expected alphanumeric, '_', '.', or '-'.",
            )
        )
    return Ok(CategoryName(cleaned))


def parse_sha256_hex(raw: str) -> Result[IdentifierValidationFault, Sha256Hex]:
    """Validate and normalize a 64-character SHA-256 hex digest into `Sha256Hex`.

    Args:
        raw: Candidate 64-character hex string.

    Returns:
        `Ok(Sha256Hex)` if valid 64-char hex, else `Err(IdentifierValidationFault)`.
    """
    cleaned = raw.strip().lower()
    if not _SHA256_HEX_RE.match(cleaned):
        return Err(
            IdentifierValidationFault(
                field="Sha256Hex",
                raw_value=raw,
                reason="expected 64 hexadecimal characters.",
            )
        )
    return Ok(Sha256Hex(cleaned))


def compute_sha256_hex(payload: bytes) -> Sha256Hex:
    """Compute a guaranteed-valid `Sha256Hex` digest from raw bytes.

    Args:
        payload: Byte buffer to hash.

    Returns:
        Validated 64-character lowercase `Sha256Hex`.
    """
    return Sha256Hex(hashlib.sha256(payload).hexdigest())


FRAMEWORK_ID_EQ: Eq[FrameworkId] = contramap_eq(STR_EQ, _framework_id_to_str)
FRAMEWORK_ID_ORD: Ord[FrameworkId] = contramap_ord(STR_ORD, _framework_id_to_str)
FRAMEWORK_ID_HASH: HashableWitness[FrameworkId] = contramap_hash(STR_HASH, _framework_id_to_str)

SNAPSHOT_VERSION_EQ: Eq[SnapshotVersion] = contramap_eq(STR_EQ, _snapshot_version_to_str)
SNAPSHOT_VERSION_ORD: Ord[SnapshotVersion] = contramap_ord(STR_ORD, _snapshot_version_to_str)
SNAPSHOT_VERSION_HASH: HashableWitness[SnapshotVersion] = contramap_hash(STR_HASH, _snapshot_version_to_str)

API_PATH_EQ: Eq[ApiPath] = contramap_eq(STR_EQ, _api_path_to_str)
API_PATH_ORD: Ord[ApiPath] = contramap_ord(STR_ORD, _api_path_to_str)
API_PATH_LEN_LEX_ORD: Ord[ApiPath] = contramap_ord(lexical_pair_ord(INT_ORD, STR_ORD), _api_path_to_len_and_str)
API_PATH_HASH: HashableWitness[ApiPath] = contramap_hash(STR_HASH, _api_path_to_str)

SYMBOL_PATH_EQ: Eq[SymbolPath] = contramap_eq(STR_EQ, _symbol_path_to_str)
SYMBOL_PATH_ORD: Ord[SymbolPath] = contramap_ord(STR_ORD, _symbol_path_to_str)
SYMBOL_PATH_HASH: HashableWitness[SymbolPath] = contramap_hash(STR_HASH, _symbol_path_to_str)

CATEGORY_NAME_EQ: Eq[CategoryName] = contramap_eq(STR_EQ, _category_name_to_str)
CATEGORY_NAME_ORD: Ord[CategoryName] = contramap_ord(STR_ORD, _category_name_to_str)
CATEGORY_NAME_HASH: HashableWitness[CategoryName] = contramap_hash(STR_HASH, _category_name_to_str)

SHA256_HEX_EQ: Eq[Sha256Hex] = contramap_eq(STR_EQ, _sha256_hex_to_str)
SHA256_HEX_ORD: Ord[Sha256Hex] = contramap_ord(STR_ORD, _sha256_hex_to_str)
SHA256_HEX_HASH: HashableWitness[Sha256Hex] = contramap_hash(STR_HASH, _sha256_hex_to_str)

POSITIVE_INT_EQ: Eq[PositiveInt] = contramap_eq(INT_EQ, int)
POSITIVE_INT_ORD: Ord[PositiveInt] = contramap_ord(INT_ORD, int)
POSITIVE_INT_HASH: HashableWitness[PositiveInt] = contramap_hash(INT_HASH, int)

NON_NEGATIVE_INT_EQ: Eq[NonNegativeInt] = contramap_eq(INT_EQ, int)
NON_NEGATIVE_INT_ORD: Ord[NonNegativeInt] = contramap_ord(INT_ORD, int)
NON_NEGATIVE_INT_HASH: HashableWitness[NonNegativeInt] = contramap_hash(INT_HASH, int)

__all__ = [
    "API_PATH_EQ",
    "API_PATH_HASH",
    "API_PATH_LEN_LEX_ORD",
    "API_PATH_ORD",
    "ApiPath",
    "CATEGORY_NAME_EQ",
    "CATEGORY_NAME_HASH",
    "CATEGORY_NAME_ORD",
    "CategoryName",
    "DIALECT_DOMAIN_EQ",
    "DIALECT_DOMAIN_HASH",
    "DIALECT_DOMAIN_ORD",
    "DialectDomain",
    "FRAMEWORK_ID_EQ",
    "FRAMEWORK_ID_HASH",
    "FRAMEWORK_ID_ORD",
    "FrameworkId",
    "IdentifierKind",
    "IdentifierParseError",
    "NON_NEGATIVE_INT_EQ",
    "NON_NEGATIVE_INT_HASH",
    "NON_NEGATIVE_INT_ORD",
    "NonBoolInt",
    "NonNegativeInt",
    "OP_NAME_EQ",
    "OP_NAME_HASH",
    "OP_NAME_ORD",
    "OpName",
    "POSITIVE_INT_EQ",
    "POSITIVE_INT_HASH",
    "POSITIVE_INT_ORD",
    "PositiveInt",
    "SHA256_HEX_EQ",
    "SHA256_HEX_HASH",
    "SHA256_HEX_ORD",
    "SNAPSHOT_VERSION_EQ",
    "SNAPSHOT_VERSION_HASH",
    "SNAPSHOT_VERSION_ORD",
    "SYMBOL_PATH_EQ",
    "SYMBOL_PATH_HASH",
    "SYMBOL_PATH_ORD",
    "Sha256Hex",
    "SnapshotVersion",
    "SymbolPath",
    "compute_sha256_hex",
    "is_non_bool_int",
    "parse_api_path",
    "parse_category_name",
    "parse_dialect_domain",
    "parse_framework_id",
    "parse_non_negative_int",
    "parse_op_name",
    "parse_positive_int",
    "parse_sha256_hex",
    "parse_snapshot_version",
    "parse_symbol_path",
]
