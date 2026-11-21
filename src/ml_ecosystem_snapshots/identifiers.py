"""Domain `NewType` identifiers, total smart constructors, and algebraic witnesses.

Provides zero-cost static `NewType` distinctions for distinct string and integer
domain concepts (`FrameworkId`, `SnapshotVersion`, `ApiPath`, `CategoryName`,
`Sha256Hex`) alongside upstream `ml_switcheroo_ir.identifiers` refined scalars,
paired with total `Result[str, T]` smart constructors and explicit `Eq`, `Ord`,
and `HashableWitness` dictionaries.
"""

from __future__ import annotations

import hashlib
import re
import typing

from ml_switcheroo_ir.identifiers import (
    DialectDomain as DialectDomain,
)
from ml_switcheroo_ir.identifiers import (
    MeshAxisName as MeshAxisName,
)
from ml_switcheroo_ir.identifiers import (
    NodeId as NodeId,
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
    PortRef as PortRef,
)
from ml_switcheroo_ir.identifiers import (
    PositiveInt as PositiveInt,
)
from ml_switcheroo_ir.identifiers import (
    QualifiedOpKey as QualifiedOpKey,
)
from ml_switcheroo_ir.identifiers import (
    SsaValueName as SsaValueName,
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

FrameworkId = typing.NewType("FrameworkId", str)
SnapshotVersion = typing.NewType("SnapshotVersion", str)
ApiPath = typing.NewType("ApiPath", str)
CategoryName = typing.NewType("CategoryName", str)
Sha256Hex = typing.NewType("Sha256Hex", str)

_FRAMEWORK_ID_RE: typing.Final[re.Pattern[str]] = re.compile(r"^[a-z0-9_]+$")
_SHA256_HEX_RE: typing.Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")


def _framework_id_to_str(value: FrameworkId) -> str:
    """Project `FrameworkId` to its underlying `str` representation.

    Args:
        value: Validated `FrameworkId`.

    Returns:
        Underlying string identifier.
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
        Underlying symbol path string.
    """
    return str(value)


def _api_path_to_len_pair(value: ApiPath) -> tuple[int, str]:
    """Project `ApiPath` to `(len(path), path)` for shortest-canonical-path ordering.

    Args:
        value: Validated `ApiPath`.

    Returns:
        Tuple of `(character_length, path_string)`.
    """
    raw = str(value)
    return (len(raw), raw)


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


def parse_framework_id(raw: str) -> Result[str, FrameworkId]:
    """Validate and normalize a raw framework name into a canonical `FrameworkId`.

    Normalizes hyphens (`-`) to underscores (`_`) and lowercases ASCII characters.

    Args:
        raw: Candidate framework name string (e.g. `'torch'`, `'ml-switcheroo-ir'`).

    Returns:
        `Ok(FrameworkId)` if non-empty and matching `^[a-z0-9_]+$`, else `Err(reason)`.
    """
    normalized = raw.strip().lower().replace("-", "_")
    if not normalized:
        return Err("FrameworkId must be non-empty.")
    if not _FRAMEWORK_ID_RE.match(normalized):
        return Err(f"Invalid FrameworkId {raw!r}: expected lowercase alphanumeric or underscore characters.")
    return Ok(FrameworkId(normalized))


def parse_snapshot_version(raw: str) -> Result[str, SnapshotVersion]:
    """Validate and trim a raw version string into a `SnapshotVersion`.

    Args:
        raw: Candidate version tag (e.g. `'2.4.0'`, `'0.0.8'`, `'latest'`).

    Returns:
        `Ok(SnapshotVersion)` if non-empty and free of path separators, else `Err(reason)`.
    """
    cleaned = raw.strip()
    if not cleaned:
        return Err("SnapshotVersion must be non-empty.")
    if "/" in cleaned or chr(92) in cleaned:
        return Err(f"Invalid SnapshotVersion {raw!r}: path separators are forbidden.")
    return Ok(SnapshotVersion(cleaned))


def parse_api_path(raw: str) -> Result[str, ApiPath]:
    """Validate and trim a raw symbol path or mnemonic into an `ApiPath`.

    Args:
        raw: Candidate dotted Python API path, IR op name, or ISA mnemonic.

    Returns:
        `Ok(ApiPath)` if non-empty after trimming, else `Err(reason)`.
    """
    cleaned = raw.strip()
    if not cleaned:
        return Err("ApiPath must be non-empty.")
    return Ok(ApiPath(cleaned))


def parse_category_name(raw: str) -> Result[str, CategoryName]:
    """Validate and trim a raw semantic category label into a `CategoryName`.

    Args:
        raw: Candidate category name (e.g. `'neural'`, `'array'`, `'operations'`).

    Returns:
        `Ok(CategoryName)` if non-empty after trimming, else `Err(reason)`.
    """
    cleaned = raw.strip()
    if not cleaned:
        return Err("CategoryName must be non-empty.")
    return Ok(CategoryName(cleaned))


def parse_sha256_hex(raw: str) -> Result[str, Sha256Hex]:
    """Validate and normalize a raw hexadecimal string into a 64-char lowercase `Sha256Hex`.

    Args:
        raw: Candidate SHA-256 hex digest string.

    Returns:
        `Ok(Sha256Hex)` if 64 hexadecimal characters, else `Err(reason)`.
    """
    normalized = raw.strip().lower()
    if not _SHA256_HEX_RE.match(normalized):
        return Err(f"Invalid Sha256Hex {raw!r}: expected exactly 64 hexadecimal characters.")
    return Ok(Sha256Hex(normalized))


def compute_sha256_hex(payload: bytes) -> Sha256Hex:
    """Compute a guaranteed-valid 64-character lowercase `Sha256Hex` from raw bytes.

    Args:
        payload: Raw byte buffer to hash.

    Returns:
        Deterministic `Sha256Hex` digest.
    """
    return Sha256Hex(hashlib.sha256(payload).hexdigest())


FRAMEWORK_ID_EQ: typing.Final[Eq[FrameworkId]] = contramap_eq(STR_EQ, _framework_id_to_str)
FRAMEWORK_ID_ORD: typing.Final[Ord[FrameworkId]] = contramap_ord(STR_ORD, _framework_id_to_str)
FRAMEWORK_ID_HASH: typing.Final[HashableWitness[FrameworkId]] = contramap_hash(STR_HASH, _framework_id_to_str)

SNAPSHOT_VERSION_EQ: typing.Final[Eq[SnapshotVersion]] = contramap_eq(STR_EQ, _snapshot_version_to_str)
SNAPSHOT_VERSION_ORD: typing.Final[Ord[SnapshotVersion]] = contramap_ord(STR_ORD, _snapshot_version_to_str)
SNAPSHOT_VERSION_HASH: typing.Final[HashableWitness[SnapshotVersion]] = contramap_hash(
    STR_HASH, _snapshot_version_to_str
)

API_PATH_EQ: typing.Final[Eq[ApiPath]] = contramap_eq(STR_EQ, _api_path_to_str)
API_PATH_ORD: typing.Final[Ord[ApiPath]] = contramap_ord(STR_ORD, _api_path_to_str)
API_PATH_LEN_LEX_ORD: typing.Final[Ord[ApiPath]] = contramap_ord(
    lexical_pair_ord(INT_ORD, STR_ORD),
    _api_path_to_len_pair,
)
API_PATH_HASH: typing.Final[HashableWitness[ApiPath]] = contramap_hash(STR_HASH, _api_path_to_str)

CATEGORY_NAME_EQ: typing.Final[Eq[CategoryName]] = contramap_eq(STR_EQ, _category_name_to_str)
CATEGORY_NAME_ORD: typing.Final[Ord[CategoryName]] = contramap_ord(STR_ORD, _category_name_to_str)
CATEGORY_NAME_HASH: typing.Final[HashableWitness[CategoryName]] = contramap_hash(STR_HASH, _category_name_to_str)

SHA256_HEX_EQ: typing.Final[Eq[Sha256Hex]] = contramap_eq(STR_EQ, _sha256_hex_to_str)
SHA256_HEX_ORD: typing.Final[Ord[Sha256Hex]] = contramap_ord(STR_ORD, _sha256_hex_to_str)
SHA256_HEX_HASH: typing.Final[HashableWitness[Sha256Hex]] = contramap_hash(STR_HASH, _sha256_hex_to_str)

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
    "DialectDomain",
    "FRAMEWORK_ID_EQ",
    "FRAMEWORK_ID_HASH",
    "FRAMEWORK_ID_ORD",
    "FrameworkId",
    "MeshAxisName",
    "NodeId",
    "NonBoolInt",
    "NonNegativeInt",
    "OpName",
    "PortRef",
    "PositiveInt",
    "QualifiedOpKey",
    "SHA256_HEX_EQ",
    "SHA256_HEX_HASH",
    "SHA256_HEX_ORD",
    "SNAPSHOT_VERSION_EQ",
    "SNAPSHOT_VERSION_HASH",
    "SNAPSHOT_VERSION_ORD",
    "Sha256Hex",
    "SnapshotVersion",
    "SsaValueName",
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
]
