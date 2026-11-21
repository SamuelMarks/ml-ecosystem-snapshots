"""ML Framework Snapshots.

Provides a Haskell-style pure functional SDK and imperative CLI/MCP shell to
introspect ML frameworks, compiler IRs, and hardware ISAs into verified Ghost
API snapshots.
"""

from __future__ import annotations

__version__ = "0.0.8"

try:
    from ml_ecosystem_snapshots.api import (
        consolidate_aliases,
        extract_all_snapshots,
        extract_snapshot,
        extract_snapshot_isolated,
        write_snapshot,
    )
    from ml_ecosystem_snapshots.identifiers import (
        ApiPath,
        CategoryName,
        FrameworkId,
        Sha256Hex,
        SnapshotVersion,
        compute_sha256_hex,
        parse_api_path,
        parse_category_name,
        parse_framework_id,
        parse_sha256_hex,
        parse_snapshot_version,
    )
    from ml_ecosystem_snapshots.proofs import (
        ComplianceVerifiedPhase,
        DeterminismVerifiedPhase,
        EnvelopeValidPhase,
        HydratedRefsPhase,
        ProvenSnapshot,
        prove_compliance_verified,
        prove_determinism_verified,
        prove_envelope_valid,
        prove_hydrated_refs,
    )
    from ml_ecosystem_snapshots.typeclasses import (
        CODE_BLOCK_RESULT_MONOID,
        COMPLIANCE_REPORT_MONOID,
        DIFF_CHANGELOG_PRETTY,
        DIFF_RESULT_MONOID,
        DIFF_RESULT_SEMIGROUP,
        GHOST_REF_CONSOLIDATION_SEMIGROUP,
        GHOST_REF_JSON_CODEC,
        SNAPSHOT_ENVELOPE_JSON_CODEC,
        SNIPPET_REPORT_MONOID,
        STUB_MODULE_PRETTY,
        TYPE_STR_SANITIZE_ENDO,
        Codec,
        CodecError,
        Endo,
        Eq,
        Err,
        Invalid,
        JsonCodec,
        Monoid,
        Ok,
        Ord,
        Ordering,
        Pretty,
        Result,
        Semigroup,
        Valid,
        Validation,
    )

    __all__ = [
        "ApiPath",
        "CODE_BLOCK_RESULT_MONOID",
        "COMPLIANCE_REPORT_MONOID",
        "CategoryName",
        "Codec",
        "CodecError",
        "ComplianceVerifiedPhase",
        "DIFF_CHANGELOG_PRETTY",
        "DIFF_RESULT_MONOID",
        "DIFF_RESULT_SEMIGROUP",
        "DeterminismVerifiedPhase",
        "Endo",
        "EnvelopeValidPhase",
        "Eq",
        "Err",
        "FrameworkId",
        "GHOST_REF_CONSOLIDATION_SEMIGROUP",
        "GHOST_REF_JSON_CODEC",
        "HydratedRefsPhase",
        "Invalid",
        "JsonCodec",
        "Monoid",
        "Ok",
        "Ord",
        "Ordering",
        "Pretty",
        "ProvenSnapshot",
        "Result",
        "SNAPSHOT_ENVELOPE_JSON_CODEC",
        "SNIPPET_REPORT_MONOID",
        "STUB_MODULE_PRETTY",
        "Semigroup",
        "Sha256Hex",
        "SnapshotVersion",
        "TYPE_STR_SANITIZE_ENDO",
        "Valid",
        "Validation",
        "__version__",
        "compute_sha256_hex",
        "consolidate_aliases",
        "extract_all_snapshots",
        "extract_snapshot",
        "extract_snapshot_isolated",
        "parse_api_path",
        "parse_category_name",
        "parse_framework_id",
        "parse_sha256_hex",
        "parse_snapshot_version",
        "prove_compliance_verified",
        "prove_determinism_verified",
        "prove_envelope_valid",
        "prove_hydrated_refs",
        "write_snapshot",
    ]
except ImportError:
    # Handle the case where the package is installed without 'generate' dependencies.
    __all__ = ["__version__"]
