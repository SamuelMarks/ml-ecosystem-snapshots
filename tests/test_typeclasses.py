"""Algebraic law and round-trip unit tests for `ml_ecosystem_snapshots.typeclasses`."""

from __future__ import annotations

from ml_switcheroo_ir.schema.ghost import (
    ExtendedGhostParam,
    ExtendedGhostRef,
    GhostParam,
    GhostRef,
    ParameterKind,
    SnapshotEnvelope,
)

from ml_ecosystem_snapshots.compliance.models import (
    ComplianceMismatchEntry,
    ComplianceScoreReport,
    NamespaceAlignmentRule,
    SnippetComplianceReport,
)
from ml_ecosystem_snapshots.diff import DiffResult
from ml_ecosystem_snapshots.grounding.models import CodeBlockCheckResult, CodeBlockFinding
from ml_ecosystem_snapshots.typeclasses import (
    CODE_BLOCK_RESULT_EQ,
    CODE_BLOCK_RESULT_MONOID,
    COMPLIANCE_REPORT_EQ,
    COMPLIANCE_REPORT_MONOID,
    DIFF_CHANGELOG_PRETTY,
    DIFF_RESULT_EQ,
    DIFF_RESULT_MONOID,
    DIFF_RESULT_SEMIGROUP,
    GHOST_REF_CANONICAL_EQ,
    GHOST_REF_CANONICAL_HASH,
    GHOST_REF_CONSOLIDATION_SEMIGROUP,
    GHOST_REF_JSON_CODEC,
    SNAPSHOT_ENVELOPE_JSON_CODEC,
    SNIPPET_REPORT_EQ,
    SNIPPET_REPORT_MONOID,
    STUB_MODULE_PRETTY,
    TYPE_STR_SANITIZE_ENDO,
    CodecError,
    Endo,
    Err,
    Monoid,
    Ok,
    endo_monoid,
    make_namespace_align_endo,
)


def test_diff_result_monoid_laws() -> None:
    """Verify Eq, Semigroup associativity, and Monoid identity laws on DIFF_RESULT_MONOID."""
    d_a = DiffResult(
        added=("pkg.alpha",),
        metadata_changed={"version": ("0.1.0", "0.2.0"), "target": ("x", "y")},
    )
    d_b = DiffResult(
        removed=("pkg.beta",),
        breaking_signature_changed=("pkg.gamma",),
        signature_changed=("pkg.gamma",),
        metadata_changed={"version": ("0.2.0", "0.3.0")},
    )
    d_c = DiffResult(
        added=("pkg.delta",),
        deprecated_capabilities=("sm_60",),
        overloads_changed=("pkg.gamma",),
        metadata_changed={"version": ("0.3.0", "0.4.0"), "commit": ("a", "b")},
    )

    assert DIFF_RESULT_EQ.eq(d_a, DiffResult(added=("pkg.alpha",), metadata_changed=d_a.metadata_changed))
    assert DIFF_RESULT_EQ.neq(d_a, d_b)

    empty = DIFF_RESULT_MONOID.empty()
    assert DIFF_RESULT_MONOID.combine(empty, d_a) == d_a
    assert DIFF_RESULT_MONOID.combine(d_a, empty) == d_a

    lhs = DIFF_RESULT_SEMIGROUP.combine(DIFF_RESULT_SEMIGROUP.combine(d_a, d_b), d_c)
    rhs = DIFF_RESULT_SEMIGROUP.combine(d_a, DIFF_RESULT_SEMIGROUP.combine(d_b, d_c))
    assert lhs == rhs
    assert lhs.metadata_changed["version"] == ("0.1.0", "0.4.0")
    assert lhs.metadata_changed["target"] == ("x", "y")
    assert lhs.metadata_changed["commit"] == ("a", "b")
    assert "## Added" in DIFF_CHANGELOG_PRETTY.render(lhs)


def test_compliance_snippet_and_code_block_monoid_laws() -> None:
    """Verify associativity and identity laws across compliance, snippet, and code-block monoids."""
    r_a = ComplianceScoreReport(
        score_percentage=50.0,
        total_endpoints=2,
        matched=("a",),
        missing=("b",),
    )
    r_b = ComplianceScoreReport(
        score_percentage=100.0,
        total_endpoints=1,
        matched=("c",),
        opaque_signatures=("c",),
        warnings=("warn_c",),
    )
    r_c = ComplianceScoreReport(
        score_percentage=0.0,
        total_endpoints=1,
        mismatched=(ComplianceMismatchEntry(api_path="d"),),
    )
    assert COMPLIANCE_REPORT_EQ.eq(r_a, r_a)
    assert COMPLIANCE_REPORT_MONOID.combine(COMPLIANCE_REPORT_MONOID.empty(), r_a) == r_a
    assert COMPLIANCE_REPORT_MONOID.combine(r_a, COMPLIANCE_REPORT_MONOID.empty()) == r_a
    assert COMPLIANCE_REPORT_MONOID.combine(
        COMPLIANCE_REPORT_MONOID.combine(r_a, r_b),
        r_c,
    ) == COMPLIANCE_REPORT_MONOID.combine(
        r_a,
        COMPLIANCE_REPORT_MONOID.combine(r_b, r_c),
    )
    assert COMPLIANCE_REPORT_MONOID.combine(COMPLIANCE_REPORT_MONOID.empty(), COMPLIANCE_REPORT_MONOID.empty()) == (
        COMPLIANCE_REPORT_MONOID.empty()
    )

    s_a = SnippetComplianceReport(is_compliant=True, total_ops=2, verified_ops=2)
    s_b = SnippetComplianceReport(is_compliant=False, total_ops=1, verified_ops=0, errors=("bad op",))
    s_c = SnippetComplianceReport(is_compliant=True, total_instructions=3, verified_instructions=3)
    assert SNIPPET_REPORT_EQ.eq(s_a, s_a)
    assert SNIPPET_REPORT_MONOID.combine(SNIPPET_REPORT_MONOID.empty(), s_a) == s_a
    assert SNIPPET_REPORT_MONOID.combine(s_a, SNIPPET_REPORT_MONOID.empty()) == s_a
    assert SNIPPET_REPORT_MONOID.combine(
        SNIPPET_REPORT_MONOID.combine(s_a, s_b),
        s_c,
    ) == SNIPPET_REPORT_MONOID.combine(
        s_a,
        SNIPPET_REPORT_MONOID.combine(s_b, s_c),
    )

    cb_a = CodeBlockCheckResult(total_analyzed=1, hallucinations_detected=0, is_valid=True)
    cb_b = CodeBlockCheckResult(
        total_analyzed=2,
        hallucinations_detected=1,
        is_valid=False,
        findings=(CodeBlockFinding(line=4, type="python", target="torch.fake", reason="missing"),),
    )
    cb_c = CodeBlockCheckResult(total_analyzed=1, hallucinations_detected=0, is_valid=True)
    assert CODE_BLOCK_RESULT_EQ.eq(cb_a, cb_a)
    assert CODE_BLOCK_RESULT_MONOID.combine(CODE_BLOCK_RESULT_MONOID.empty(), cb_a) == cb_a
    assert CODE_BLOCK_RESULT_MONOID.combine(cb_a, CODE_BLOCK_RESULT_MONOID.empty()) == cb_a
    assert CODE_BLOCK_RESULT_MONOID.combine(
        CODE_BLOCK_RESULT_MONOID.combine(cb_a, cb_b),
        cb_c,
    ) == CODE_BLOCK_RESULT_MONOID.combine(
        cb_a,
        CODE_BLOCK_RESULT_MONOID.combine(cb_b, cb_c),
    )


def test_ghost_ref_consolidation_semigroup_and_hash_coherence() -> None:
    """Verify non-mutating GhostRef consolidation semigroup associativity and hash coherence."""
    ext_long = ExtendedGhostRef(
        name="relu",
        api_path="torch.nn.functional.relu",
        kind="function",
        params=[
            ExtendedGhostParam(
                name="x",
                kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                allowed_dtypes=["float32"],
                rank_constraint="2",
            )
        ],
        aliases=["torch._C.relu"],
        is_public=False,
        domain_metadata={"cat": "act"},
        overloads=[GhostRef(name="relu", api_path="torch.relu", kind="function", params=[])],
    )
    ext_short = ExtendedGhostRef(
        name="relu",
        api_path="torch.relu",
        kind="function",
        params=[
            ExtendedGhostParam(
                name="x",
                kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                allowed_dtypes=["float32"],
                rank_constraint="2",
            )
        ],
        aliases=[],
        is_public=True,
        domain_metadata={"cat": "act"},
        overloads=[GhostRef(name="relu", api_path="torch.relu", kind="function", params=[])],
    )
    ext_mid = ExtendedGhostRef(
        name="relu",
        api_path="torch.ops.relu",
        kind="function",
        params=ext_short.params,
        aliases=[],
        is_public=False,
        domain_metadata={"cat": "act"},
        overloads=ext_short.overloads,
    )

    assert GHOST_REF_CANONICAL_EQ.eq(ext_long, ext_short)
    assert GHOST_REF_CANONICAL_HASH.hash_value(ext_long) == GHOST_REF_CANONICAL_HASH.hash_value(ext_short)

    lhs = GHOST_REF_CONSOLIDATION_SEMIGROUP.combine(
        GHOST_REF_CONSOLIDATION_SEMIGROUP.combine(ext_long, ext_short),
        ext_mid,
    )
    rhs = GHOST_REF_CONSOLIDATION_SEMIGROUP.combine(
        ext_long,
        GHOST_REF_CONSOLIDATION_SEMIGROUP.combine(ext_short, ext_mid),
    )
    assert lhs == rhs
    assert lhs.api_path == "torch.relu"
    assert isinstance(lhs, ExtendedGhostRef)
    assert lhs.aliases == ["torch._C.relu", "torch.nn.functional.relu", "torch.ops.relu"]
    assert lhs.is_public is True
    # Input references remain untouched (zero in-place mutation)
    assert ext_short.aliases == []
    assert ext_long.is_public is False

    # Also test ExtendedGhostRef with unset optional metadata/dtypes/rank and plain GhostRef primary branch
    ext_minimal = ExtendedGhostRef(
        name="cos",
        api_path="pkg.cos",
        kind="function",
        params=[ExtendedGhostParam(name="x", kind=ParameterKind.POSITIONAL_OR_KEYWORD)],
    )
    assert GHOST_REF_CANONICAL_HASH.hash_value(ext_minimal) == GHOST_REF_CANONICAL_HASH.hash_value(ext_minimal)

    plain_short = GhostRef(
        name="abs",
        api_path="pkg.abs",
        kind="function",
        params=[GhostParam(name="x", kind=ParameterKind.POSITIONAL_OR_KEYWORD)],
    )
    plain_long = GhostRef(
        name="abs",
        api_path="pkg.internal.abs",
        kind="function",
        params=[GhostParam(name="x", kind=ParameterKind.POSITIONAL_OR_KEYWORD)],
    )
    merged_plain = GHOST_REF_CONSOLIDATION_SEMIGROUP.combine(plain_long, plain_short)
    assert GHOST_REF_CANONICAL_HASH.hash_value(plain_short) == GHOST_REF_CANONICAL_HASH.hash_value(plain_long)
    assert isinstance(merged_plain, GhostRef)
    assert merged_plain.api_path == "pkg.abs"
    assert merged_plain.aliases == ["pkg.internal.abs"]


def test_json_codecs_pretty_and_endo() -> None:
    """Verify SNAPSHOT_ENVELOPE_JSON_CODEC, GHOST_REF_JSON_CODEC, STUB_MODULE_PRETTY, and Endo."""
    ref = ExtendedGhostRef(
        name="sin",
        api_path="pkg.sin",
        kind="function",
        params=[ExtendedGhostParam(name="x", kind=ParameterKind.POSITIONAL_OR_KEYWORD, annotation="float")],
        returns_type="float",
    )
    encoded_ref = GHOST_REF_JSON_CODEC.encode(ref)
    decoded_ref = GHOST_REF_JSON_CODEC.decode(encoded_ref)
    assert isinstance(decoded_ref, Ok)
    assert decoded_ref.value.api_path == "pkg.sin"
    assert isinstance(GHOST_REF_JSON_CODEC.decode("not_an_object"), Err)
    assert isinstance(GHOST_REF_JSON_CODEC.decode({"schema_version": "0.0.1"}), Err)

    env = SnapshotEnvelope(
        target="pkg",
        version="1.0.0",
        generated_at="2026-11-21T00:00:00+00:00",
        categories={"math": [ref]},
    )
    encoded_env = SNAPSHOT_ENVELOPE_JSON_CODEC.encode(env)
    decoded_env = SNAPSHOT_ENVELOPE_JSON_CODEC.decode(encoded_env)
    assert isinstance(decoded_env, Ok)
    assert decoded_env.value.target == "pkg"
    assert isinstance(SNAPSHOT_ENVELOPE_JSON_CODEC.decode(123), Err)
    bad_env = SNAPSHOT_ENVELOPE_JSON_CODEC.decode({"schema_version": "9.9.9"})
    assert isinstance(bad_env, Err)
    assert isinstance(bad_env.error, CodecError)

    stub_text = STUB_MODULE_PRETTY.render(env)
    assert "# module: pkg" in stub_text
    assert "def sin(x: float) -> float: ..." in stub_text

    invalid_stub_env = SnapshotEnvelope(
        target="pkg",
        version="1.0.0",
        categories={"math": [{"name": "bad", "api_path": "pkg.123bad", "kind": "function", "params": []}]},
    )
    assert STUB_MODULE_PRETTY.render(invalid_stub_env).startswith("# stub error:")

    rule = NamespaceAlignmentRule(
        prefix_map=(("zero_jax", "jax"),),
        path_rewrites=((".nn.nn", ".nn"),),
    )
    align_endo = make_namespace_align_endo("zero_jax", "jax", rule)
    assert align_endo.app("zero_jax") == "jax"
    assert align_endo.app("zero_jax.nn.nn.relu") == "jax.nn.relu"
    plain_endo = make_namespace_align_endo("my_pkg", "canon_pkg")
    assert plain_endo.app("my_pkg") == "canon_pkg"
    assert plain_endo.app("other.fn") == "other.fn"
    noop_endo = make_namespace_align_endo("", "")
    assert noop_endo.app("unchanged.fn") == "unchanged.fn"

    str_monoid: Monoid[Endo[str]] = endo_monoid()
    composed = str_monoid.combine(TYPE_STR_SANITIZE_ENDO, Endo(app=lambda s: f"Optional[{s}]"))
    assert composed.app("int") == "int | None"
