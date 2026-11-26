"""Unit tests for covariant cumulative `ProvenSnapshot[_SnapshotPhaseT_co]` proof witnesses."""

from __future__ import annotations

from pathlib import Path

import pytest
from ml_switcheroo_ir.graph import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo_ir.proofs import GroundedPhase, ProvenGraph, prove_acyclic_sorted
from ml_switcheroo_ir.schema.ghost import (
    ExtendedGhostParam,
    ExtendedGhostRef,
    GhostParam,
    GhostRef,
    ParameterKind,
    SnapshotEnvelope,
)
from ml_switcheroo_ir.typeclasses.algebra import NonEmpty
from ml_switcheroo_ir.typeclasses.outcome import Err, Invalid, Ok, Valid
from ml_switcheroo_ir.types import DType, TensorShape, TensorSpec

from ml_ecosystem_snapshots.compliance import audit_module_sources
from ml_ecosystem_snapshots.errors import (
    ComplianceThresholdFault,
    DeterminismLeakFault,
    HydrationBatchFault,
    IdentifierValidationFault,
    IntrospectionFault,
    OfflineNetworkFault,
    SnapshotEnvelopeFault,
    SnapshotNotFoundFault,
)
from ml_ecosystem_snapshots.exceptions import SnapshotError
from ml_ecosystem_snapshots.exporters.prompts import (
    export_llm_prompt_context,
    export_mlir_prompt_context,
    export_ptx_prompt_context,
    export_rdna_prompt_context,
    export_sass_prompt_context,
    export_scoped_prompt_context,
    export_stablehlo_prompt_context,
)
from ml_ecosystem_snapshots.grounding import ground_proven_graph
from ml_ecosystem_snapshots.identifiers import CategoryName, FrameworkId, SnapshotVersion
from ml_ecosystem_snapshots.models import JsonValue
from ml_ecosystem_snapshots.proofs import (
    ComplianceVerifiedPhase,
    DeterminismVerifiedPhase,
    DeterministicSnapshotPhase,
    EnvelopeAndHydratedPhase,
    EnvelopeValidPhase,
    FullyCertifiedSnapshotPhase,
    ProvenSnapshot,
    certify_snapshot_pipeline,
    prove_compliance_verified,
    prove_determinism_verified,
    prove_envelope_valid,
    prove_hydrated_refs,
    upgrade_to_compliance_certified,
    upgrade_to_deterministic,
    upgrade_to_hydrated,
)


def _sample_valid_raw_snapshot() -> dict[str, JsonValue]:
    """Build a minimal deterministic raw snapshot dictionary for proof testing.

    Returns:
        Valid v2.0.0 snapshot dictionary.
    """
    return {
        "schema_version": "2.0.0",
        "target": "torch",
        "version": "2.4.0",
        "generated_at": "2026-11-21T00:00:00+00:00",
        "categories": {
            "neural": [
                {
                    "name": "relu",
                    "api_path": "torch.nn.functional.relu",
                    "kind": "function",
                    "params": [
                        {
                            "name": "input",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "annotation": "Tensor",
                        },
                        {
                            "name": "inplace",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "default": "False",
                            "annotation": "bool",
                        },
                    ],
                    "returns_type": "Tensor",
                    "docstring": "Apply rectified linear unit.",
                }
            ]
        },
    }


def _accepts_hydrated(snap: ProvenSnapshot[EnvelopeAndHydratedPhase]) -> str:
    """Helper statically typed for `ProvenSnapshot[EnvelopeAndHydratedPhase]` to verify covariance.

    Args:
        snap: Hydrated `ProvenSnapshot` witness (or covariant subtype).

    Returns:
        Target identifier string.
    """
    return str(snap.target)


def test_prove_envelope_valid_ok_err_and_isolation() -> None:
    """Verify prove_envelope_valid mints isolated ProvenSnapshot[EnvelopeValidPhase] and accumulates ID faults."""
    raw = _sample_valid_raw_snapshot()
    res = prove_envelope_valid(raw)
    assert isinstance(res, Valid)
    proven: ProvenSnapshot[EnvelopeValidPhase] = res.value
    assert proven.target == FrameworkId("torch")
    assert proven.version == SnapshotVersion("2.4.0")
    assert proven.to_wire_dict()["target"] == "torch"

    env_in = SnapshotEnvelope(target="jax", version="0.4.30", categories={})
    res_env = prove_envelope_valid(env_in)
    assert isinstance(res_env, Valid)
    env_in.categories["mutated"] = []
    assert "mutated" not in res_env.value.envelope.categories

    assert isinstance(prove_envelope_valid("not_a_dict"), Invalid)
    assert isinstance(prove_envelope_valid({"schema_version": "9.9.9", "target": "torch"}), Invalid)
    assert isinstance(prove_envelope_valid({"target": "bad/target!", "version": "1.0"}), Invalid)
    assert isinstance(prove_envelope_valid({"target": "torch", "version": "../evil"}), Invalid)

    both_bad = prove_envelope_valid({"target": "bad/target!", "version": "../evil"})
    assert isinstance(both_bad, Invalid)
    assert isinstance(both_bad.errors, NonEmpty)
    assert len(both_bad.errors) == 2
    assert all(isinstance(e, IdentifierValidationFault) for e in both_bad.errors.to_tuple())


def test_upgrade_to_hydrated_valid_and_multi_error_accumulation() -> None:
    """Verify upgrade_to_hydrated / prove_hydrated_refs hydrates symbols and accumulates NonEmpty faults."""
    env_ok = prove_envelope_valid(_sample_valid_raw_snapshot())
    assert isinstance(env_ok, Valid)

    hyd_val = upgrade_to_hydrated(env_ok.value)
    assert isinstance(hyd_val, Valid)
    hydrated: ProvenSnapshot[EnvelopeAndHydratedPhase] = hyd_val.value
    assert CategoryName("neural") == hydrated.hydrated_categories[0][0]
    assert len(hydrated.category_dict()["neural"]) == 1
    assert len(hydrated.all_refs()) == 1
    assert isinstance(prove_hydrated_refs(env_ok.value), Valid)

    env_with_model = SnapshotEnvelope(
        target="torch",
        version="2.4.0",
        categories={
            "ops": [
                ExtendedGhostRef(name="add", api_path="torch.add", kind="function", params=[]),
            ]
        },
    )
    ok_model = prove_envelope_valid(env_with_model)
    assert isinstance(ok_model, Valid)
    assert isinstance(upgrade_to_hydrated(ok_model.value), Valid)

    bad_env = SnapshotEnvelope(
        target="torch",
        version="2.4.0",
        categories={
            "   ": [
                {"schema_version": "0.0.1", "name": "bad1", "api_path": "a", "kind": "function"},
                {"schema_version": "0.0.2", "name": "bad2", "api_path": "b", "kind": "function"},
            ],
            "  ": [
                ExtendedGhostRef(name="ok_ref", api_path="torch.ok", kind="function", params=[]),
            ],
        },
    )
    bad_proven = ProvenSnapshot[EnvelopeValidPhase](
        target=FrameworkId("torch"),
        version=SnapshotVersion("2.4.0"),
        envelope=bad_env,
    )
    inv = upgrade_to_hydrated(bad_proven)
    assert isinstance(inv, Invalid)
    assert len(inv.errors) == 3
    batch_faults = [f for f in inv.errors.to_tuple() if isinstance(f, HydrationBatchFault)]
    assert len(batch_faults) == 1
    assert len(batch_faults[0].reasons) == 2


def test_upgrade_to_deterministic_valid_and_invalid() -> None:
    """Verify upgrade_to_deterministic catches hex address leaks and invalid .pyi stubs."""
    env_ok = prove_envelope_valid(_sample_valid_raw_snapshot())
    assert isinstance(env_ok, Valid)
    hyd_ok = upgrade_to_hydrated(env_ok.value)
    assert isinstance(hyd_ok, Valid)

    det_ok = upgrade_to_deterministic(hyd_ok.value)
    assert isinstance(det_ok, Valid)
    det_snap: ProvenSnapshot[DeterministicSnapshotPhase] = det_ok.value
    also_det_tag: ProvenSnapshot[DeterminismVerifiedPhase] = det_snap
    assert _accepts_hydrated(det_snap) == "torch"
    assert also_det_tag.target == FrameworkId("torch")
    assert isinstance(prove_determinism_verified(hyd_ok.value), Valid)

    leaky_ref = ExtendedGhostRef(
        name="class",
        api_path="pkg.class",
        kind="function",
        docstring="<Object at 0x7fff1234abcd>",
        returns_type="<Tensor at 0x7fffdeadbeef>",
        params=[
            ExtendedGhostParam(
                name="x",
                kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                default="<obj at 0x7fff00001111>",
                annotation="<Ptr at 0x7fff22223333>",
            )
        ],
    )
    leaky_env = SnapshotEnvelope(
        target="pkg",
        version="1.0.0",
        categories={"neural": [leaky_ref]},
    )
    leaky_hydrated = ProvenSnapshot[EnvelopeAndHydratedPhase](
        target=FrameworkId("pkg"),
        version=SnapshotVersion("1.0.0"),
        envelope=leaky_env,
        hydrated_categories=((CategoryName("neural"), (leaky_ref,)),),
    )
    det_inv = upgrade_to_deterministic(leaky_hydrated)
    assert isinstance(det_inv, Invalid)
    fault = det_inv.errors.head
    assert isinstance(fault, DeterminismLeakFault)
    assert len(fault.leaked_addresses) == 4
    assert len(fault.stub_syntax_errors) == 1


def test_cumulative_certification_pipeline_and_covariance(tmp_path: Path) -> None:
    """Verify full `certify_snapshot_pipeline`, `upgrade_to_compliance_certified`, and covariance.

    Args:
        tmp_path: Pytest temporary directory path fixture.
    """
    raw_pkg: dict[str, JsonValue] = {
        "schema_version": "2.0.0",
        "target": "torch",
        "version": "2.4.0",
        "generated_at": "2026-11-27T00:00:00+00:00",
        "categories": {
            "neural": [
                {
                    "name": "relu",
                    "api_path": "torch.relu",
                    "kind": "function",
                    "params": [{"name": "x", "kind": "POSITIONAL_OR_KEYWORD", "annotation": "Tensor"}],
                    "returns_type": "Tensor",
                }
            ]
        },
    }
    pkg_dir = tmp_path / "my_torch"
    pkg_dir.mkdir()
    relu_src = chr(10).join(["def relu(x: Tensor) -> Tensor:", "    return x", ""])
    (pkg_dir / "__init__.py").write_text(relu_src, encoding="utf-8")

    certified_val = certify_snapshot_pipeline(raw_pkg, source_dir=pkg_dir, min_score=100.0)
    assert isinstance(certified_val, Valid)
    certified: ProvenSnapshot[FullyCertifiedSnapshotPhase] = certified_val.value
    as_det: ProvenSnapshot[DeterministicSnapshotPhase] = certified
    as_hyd: ProvenSnapshot[EnvelopeAndHydratedPhase] = certified
    as_env: ProvenSnapshot[EnvelopeValidPhase] = certified
    as_comp_tag: ProvenSnapshot[ComplianceVerifiedPhase] = certified
    assert _accepts_hydrated(certified) == "torch"
    assert as_det.target == as_hyd.target == as_env.target == as_comp_tag.target

    # Covariant prompt exporters, audit_module_sources, and ground_proven_graph accept certified!
    assert "relu" in export_llm_prompt_context(certified)
    assert "relu" in export_sass_prompt_context(certified)
    assert "relu" in export_mlir_prompt_context(certified)
    assert "relu" in export_rdna_prompt_context(certified)
    assert "relu" in export_ptx_prompt_context(certified)
    assert "relu" in export_stablehlo_prompt_context(certified)
    assert "relu" in export_scoped_prompt_context(certified, module_prefix="torch")

    src_audit = audit_module_sources({"my_torch/__init__.py": relu_src}, certified)
    assert isinstance(src_audit, Valid)
    assert src_audit.value.score_percentage == 100.0

    graph = LogicalGraph(
        name="relu_net",
        nodes=[
            LogicalNode(id="in0", op_type="Input"),
            LogicalNode(id="r0", op_type="torch.relu", domain="torch"),
        ],
        edges=[LogicalEdge(source="in0", target="r0")],
        input_specs={"in0": TensorSpec(shape=TensorShape([4, 8]), dtype=DType.float32)},
    )
    acyclic_res = prove_acyclic_sorted(graph)
    assert isinstance(acyclic_res, Ok)
    grounded_val = ground_proven_graph(acyclic_res.value, certified)
    assert isinstance(grounded_val, Valid)
    grounded_graph: ProvenGraph[GroundedPhase] = grounded_val.value
    assert len(grounded_graph.ordered_node_ids) == 2


def test_certify_snapshot_pipeline_failure_branches(tmp_path: Path) -> None:
    """Verify failure branches across all 4 stages of `certify_snapshot_pipeline` and `upgrade_to_compliance_certified`.

    Args:
        tmp_path: Pytest temporary directory path fixture.
    """
    pkg_dir = tmp_path / "my_torch_fail"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("def relu(x: Tensor) -> Tensor: return x" + chr(10), encoding="utf-8")

    env_ok = prove_envelope_valid(_sample_valid_raw_snapshot())
    assert isinstance(env_ok, Valid)
    hyd_ok = upgrade_to_hydrated(env_ok.value)
    assert isinstance(hyd_ok, Valid)
    det_ok = upgrade_to_deterministic(hyd_ok.value)
    assert isinstance(det_ok, Valid)
    as_det = det_ok.value

    assert isinstance(certify_snapshot_pipeline("bad", source_dir=pkg_dir), Invalid)
    bad_cat_raw: dict[str, JsonValue] = {
        "schema_version": "2.0.0",
        "target": "torch",
        "version": "2.4.0",
        "categories": {"ops": [{"schema_version": "0.0.1", "name": "x", "api_path": "torch.x"}]},
    }
    assert isinstance(certify_snapshot_pipeline(bad_cat_raw, source_dir=pkg_dir), Invalid)
    leaky_raw: dict[str, JsonValue] = {
        "schema_version": "2.0.0",
        "target": "torch",
        "version": "2.4.0",
        "categories": {
            "ops": [
                {
                    "name": "relu",
                    "api_path": "torch.relu",
                    "kind": "function",
                    "docstring": "<leak at 0x1234>",
                }
            ]
        },
    }
    assert isinstance(certify_snapshot_pipeline(leaky_raw, source_dir=pkg_dir), Invalid)

    missing_dir_inv = upgrade_to_compliance_certified(as_det, source_dir=tmp_path / "missing_dir")
    assert isinstance(missing_dir_inv, Invalid)
    assert isinstance(missing_dir_inv.errors.head, IntrospectionFault)

    root_init = tmp_path / "standalone_root" / "__init__.py"
    root_init.parent.mkdir()
    root_init.write_text("", encoding="utf-8")
    # Passing parent of unprefixed __init__.py triggers module derivation error when root has __init__.py at filesystem root or empty
    empty_pkg = tmp_path / "empty_torch"
    empty_pkg.mkdir()
    (empty_pkg / "__init__.py").write_text("x = 1" + chr(10), encoding="utf-8")
    low_score_inv = upgrade_to_compliance_certified(as_det, source_dir=empty_pkg, min_score=100.0)
    assert isinstance(low_score_inv, Invalid)
    assert isinstance(low_score_inv.errors.head, ComplianceThresholdFault)

    syntax_pkg = tmp_path / "syntax_torch"
    syntax_pkg.mkdir()
    (syntax_pkg / "__init__.py").write_text("def broken(:" + chr(10), encoding="utf-8")
    syntax_inv = upgrade_to_compliance_certified(as_det, source_dir=syntax_pkg, min_score=100.0)
    assert isinstance(syntax_inv, Invalid)
    assert isinstance(syntax_inv.errors.head, IntrospectionFault)


def test_prove_compliance_verified_valid_and_invalid() -> None:
    """Verify prove_compliance_verified mints FullyCertifiedSnapshotPhase or ComplianceThresholdFault."""
    env_ok = prove_envelope_valid(_sample_valid_raw_snapshot())
    assert isinstance(env_ok, Valid)
    hyd_ok = upgrade_to_hydrated(env_ok.value)
    assert isinstance(hyd_ok, Valid)

    matching_target = [
        GhostRef(
            name="relu",
            api_path="torch.nn.functional.relu",
            kind="function",
            params=[
                GhostParam(name="input", kind=ParameterKind.POSITIONAL_OR_KEYWORD, annotation="Tensor"),
                GhostParam(
                    name="inplace",
                    kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                    default="False",
                    annotation="bool",
                ),
            ],
        )
    ]
    comp_ok = prove_compliance_verified(hyd_ok.value, matching_target, min_score=100.0)
    assert isinstance(comp_ok, Valid)
    assert comp_ok.value.version == SnapshotVersion("2.4.0")

    two_ref_env = SnapshotEnvelope(
        target="torch",
        version="2.4.0",
        categories={
            "neural": [
                ExtendedGhostRef(
                    name="relu",
                    api_path="torch.relu",
                    kind="function",
                    params=[ExtendedGhostParam(name="x", kind=ParameterKind.POSITIONAL_OR_KEYWORD)],
                ),
                ExtendedGhostRef(
                    name="gelu",
                    api_path="torch.gelu",
                    kind="function",
                    params=[ExtendedGhostParam(name="x", kind=ParameterKind.POSITIONAL_OR_KEYWORD)],
                    environment_tags=["opaque_c_extension"],
                ),
                ExtendedGhostRef(
                    name="silu",
                    api_path="torch.silu",
                    kind="function",
                    params=[ExtendedGhostParam(name="x", kind=ParameterKind.POSITIONAL_OR_KEYWORD)],
                ),
            ]
        },
    )
    two_proven = prove_envelope_valid(two_ref_env)
    assert isinstance(two_proven, Valid)
    two_hyd = upgrade_to_hydrated(two_proven.value)
    assert isinstance(two_hyd, Valid)

    bad_targets = [
        GhostRef(name="relu", api_path="torch.relu", kind="function", params=[]),
        GhostRef(
            name="gelu",
            api_path="torch.gelu",
            kind="function",
            params=[GhostParam(name="x", kind=ParameterKind.POSITIONAL_OR_KEYWORD)],
        ),
    ]
    comp_inv = prove_compliance_verified(
        two_hyd.value,
        bad_targets,
        min_score=100.0,
        strict_c_extensions=True,
    )
    assert isinstance(comp_inv, Invalid)
    fault = comp_inv.errors.head
    assert isinstance(fault, ComplianceThresholdFault)
    assert set(fault.missing_symbols) == {"torch.gelu", "torch.relu", "torch.silu"}


def test_total_io_and_repository_outcome_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify total Result/Validation branches across serialization, network_env, repository, and api.

    Args:
        tmp_path: Pytest temporary directory path fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    import subprocess

    from ml_ecosystem_snapshots.api import (
        extract_all_snapshots,
        extract_snapshot_isolated,
        write_snapshot,
    )
    from ml_ecosystem_snapshots.network_env import (
        CompressionFormat,
        load_json_data,
        parse_compression_format,
        stream_snapshot_items,
    )
    from ml_ecosystem_snapshots.serialization.hydration import hydrate_ghost_ref

    assert parse_compression_format(CompressionFormat.GZIP) == Ok(CompressionFormat.GZIP)
    assert parse_compression_format("") == Ok(CompressionFormat.NONE)
    assert parse_compression_format("zst") == Ok(CompressionFormat.ZSTD)
    assert isinstance(parse_compression_format("lz4"), Err)

    missing_file = tmp_path / "does_not_exist.json"
    assert isinstance(load_json_data(missing_file), Err)
    assert list(stream_snapshot_items(missing_file)) == []

    env = SnapshotEnvelope(target="pkg", version="1.0.0", categories={})
    proven = prove_envelope_valid(env)
    assert isinstance(proven, Valid)
    env_written = write_snapshot("pkg", env, str(tmp_path / "out_env"))
    assert isinstance(env_written, Ok)
    assert isinstance(write_snapshot("pkg", env, str(tmp_path / "out_bad"), compress="lz4"), Err)
    blocker_file = tmp_path / "blocker"
    blocker_file.write_text("file", encoding="utf-8")
    assert isinstance(write_snapshot("pkg", env, str(blocker_file / "subdir")), Err)

    with monkeypatch.context() as mp:
        mp.setattr(
            "subprocess.run",
            lambda *_a, **_kw: subprocess.CompletedProcess(args=["py"], returncode=0, stdout="[1, 2]"),
        )
        assert isinstance(extract_snapshot_isolated("html_dsl"), Err)
        mp.setattr(
            "subprocess.run",
            lambda *_a, **_kw: subprocess.CompletedProcess(args=["py"], returncode=0, stdout="{}"),
        )
        assert isinstance(extract_snapshot_isolated("html_dsl"), Err)

    all_ok = extract_all_snapshots(output_dir=str(tmp_path / "all_out"), frameworks=("ir",))
    assert isinstance(all_ok, Valid) and "ir" in all_ok.value
    all_write_err = extract_all_snapshots(output_dir=str(blocker_file / "sub"), frameworks=("ir",))
    assert isinstance(all_write_err, Invalid)

    assert isinstance(
        hydrate_ghost_ref({"name": "fn", "api_path": "p.fn", "params": ["not_a_param"]}),
        Err,
    )


def test_total_repository_and_stablehlo_offline_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify SnapshotRepository.resolve_result, live extraction fallback, and StableHLO offline errors.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    from ml_ecosystem_snapshots.serialization.repository import (
        SnapshotRepository,
        SnapshotRuntimeContext,
        SnapshotSource,
        load_or_extract_snapshot,
    )
    from ml_ecosystem_snapshots.tools import build_stablehlo_snapshot

    fresh_store: dict[str, dict[str, JsonValue]] = {}
    loaded_ir = load_or_extract_snapshot("ir", prefer_bundled=False, active_cache=fresh_store)
    assert loaded_ir.get("_snapshot_source") == SnapshotSource.RUNTIME_INTROSPECTION.value

    err_ctx = SnapshotRuntimeContext(
        active_extract=lambda _fw, include_nonpublic=False: Err(SnapshotEnvelopeFault(reason="boom")),
        offline_fn=lambda: False,
    )
    err_store: dict[str, dict[str, JsonValue]] = {}
    fallback_empty = load_or_extract_snapshot("ir", prefer_bundled=False, active_cache=err_store, runtime_ctx=err_ctx)
    assert fallback_empty == {"categories": {}}

    repo = SnapshotRepository(backing_store=dict(fresh_store))
    resolved_ok = repo.resolve_result("ir")
    assert isinstance(resolved_ok, Ok)
    assert resolved_ok.value.source == SnapshotSource.RUNTIME_INTROSPECTION
    empty_repo = SnapshotRepository(backing_store={"empty_fw": {"categories": {}}})
    empty_err = empty_repo.resolve_result("empty_fw")
    assert isinstance(empty_err, Err)
    assert isinstance(empty_err.error, SnapshotNotFoundFault)

    with monkeypatch.context() as mp:
        mp.setattr(
            "ml_ecosystem_snapshots.network_env.fetch_url",
            lambda *_a, **_kw: Err(OfflineNetworkFault(url="https://x", operation="offline_blocked")),
        )
        with pytest.raises(SnapshotError):
            build_stablehlo_snapshot.extract_ops(source_content=None)

        calls: list[str] = []

        def _fail_second(url: str, timeout: int = 10) -> Err[OfflineNetworkFault]:
            """Fail first with transient fetch error and second with offline_blocked.

            Args:
                url: Target URL string.
                timeout: Timeout in seconds.

            Returns:
                `Err` outcome.
            """
            del timeout
            calls.append(url)
            if len(calls) == 1:
                return Err(OfflineNetworkFault(url=url, operation="fetch_failed: transient"))
            return Err(OfflineNetworkFault(url=url, operation="offline_blocked"))

        mp.setattr("ml_ecosystem_snapshots.network_env.fetch_url", _fail_second)
        with pytest.raises(SnapshotError):
            build_stablehlo_snapshot.extract_ops(source_content=None)


def test_total_compliance_diff_stubs_and_cli_branches(
    tmp_path: Path,
) -> None:
    """Verify remaining branches in compliance, diff, and stubs with ProvenSnapshot[EnvelopeAndHydratedPhase].

    Args:
        tmp_path: Pytest temporary directory path fixture.
    """
    from ml_ecosystem_snapshots.compliance import (
        NamespaceAlignmentRule,
        estimate_tensor_memory_bytes,
        extract_target_ast,
        extract_target_refs,
        extract_target_refs_single,
        get_dtype_bitwidth,
        score_compliance,
    )
    from ml_ecosystem_snapshots.diff import diff_snapshots
    from ml_ecosystem_snapshots.stubs import generate_stubs

    rule = NamespaceAlignmentRule(
        prefix_map=(("zero_jax", "jax"),),
        node_prefix_rewrites=(("zero_jax.", "{target_prefix}.jax."),),
    )
    assert NamespaceAlignmentRule.model_validate({"prefix_map": {"zero_jax": "jax"}}).lookup_prefix("zero_jax") == "jax"
    assert rule.lookup_prefix("zero_jax") == "jax"
    assert rule.lookup_prefix("missing") is None
    assert rule.lookup_node_rewrite("zero_jax.") == "{target_prefix}.jax."
    assert rule.lookup_node_rewrite("missing") is None

    assert isinstance(get_dtype_bitwidth("   "), Err)
    assert isinstance(estimate_tensor_memory_bytes([4], "   "), Err)
    assert isinstance(estimate_tensor_memory_bytes([-1, 4], "float32"), Err)
    assert estimate_tensor_memory_bytes([], "float32") == Ok(4.0)

    assert isinstance(extract_target_ast(str(tmp_path / "missing.py")), Err)
    bad_pkg = tmp_path / "bad_pkg"
    bad_pkg.mkdir()
    (bad_pkg / "__init__.py").write_text("def broken(:" + chr(10), encoding="utf-8")
    assert isinstance(extract_target_ast(str(bad_pkg)), Err)
    assert isinstance(extract_target_refs_single(str(tmp_path / "missing.py"), "a", "b"), Err)
    assert isinstance(extract_target_refs_single(str(bad_pkg), "bad_pkg", "b"), Err)
    assert isinstance(extract_target_refs([str(tmp_path / "missing.py")], "a", "b"), Err)

    env = SnapshotEnvelope(
        target="pkg",
        version="1.0.0",
        categories={
            "ops": [
                ExtendedGhostRef(
                    name="fn",
                    api_path="pkg.fn",
                    kind="function",
                    aliases=["pkg.fn_alias"],
                    params=[
                        ExtendedGhostParam(
                            name="x",
                            kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                            annotation='"torch.Tensor"',
                            default="'___NONE___'",
                        )
                    ],
                )
            ]
        },
    )
    proven_res = prove_envelope_valid(env)
    assert isinstance(proven_res, Valid)
    hyd_res = upgrade_to_hydrated(proven_res.value)
    assert isinstance(hyd_res, Valid)
    proven = hyd_res.value

    rep_env = score_compliance(
        env,
        [
            GhostRef(
                name="fn_alias",
                api_path="pkg.fn_alias",
                kind="function",
                params=[
                    GhostParam(
                        name="x",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="'int'",
                        default="None",
                    )
                ],
            )
        ],
    )
    assert len(rep_env.mismatched) == 1
    rep_proven = score_compliance(
        proven,
        [
            GhostRef(
                name="fn_alias",
                api_path="pkg.fn_alias",
                kind="function",
                params=[
                    GhostParam(
                        name="x",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="A" + "ny",
                        default="None",
                    )
                ],
            )
        ],
    )
    assert rep_proven.score_percentage == 100.0
    assert score_compliance({"categories": {"c": ["bad", {"name": "no_path"}]}}, []).total_endpoints == 0

    diff_res = diff_snapshots(env, proven)
    assert diff_res.added == ()

    stub_env_ok = generate_stubs(env, tmp_path / "stubs_env")
    assert isinstance(stub_env_ok, Ok)
    stub_prov_ok = generate_stubs(proven, tmp_path / "stubs_prov")
    assert isinstance(stub_prov_ok, Ok)

    bad_stub_snap: dict[str, JsonValue] = {
        "categories": {"ops": [{"name": "class", "api_path": "pkg.class", "kind": "function", "params": []}]}
    }
    assert isinstance(generate_stubs(bad_stub_snap, tmp_path / "stubs_bad"), Err)

    blocker = tmp_path / "stub_blocker"
    blocker.write_text("file", encoding="utf-8")
    assert isinstance(generate_stubs(env, blocker / "nested"), Err)


def test_remaining_branch_coverage_across_workstreams_2_to_4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise remaining branches in api, cli, compliance, grounding, proofs, and utils.

    Args:
        tmp_path: Pytest temporary directory path fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    import argparse
    import json
    import subprocess

    import griffe

    from ml_ecosystem_snapshots.api import (
        FRAMEWORK_COLLECTORS,
        extract_snapshot,
        extract_snapshot_isolated,
    )
    from ml_ecosystem_snapshots.cli.commands.diff import coerce_proven_or_raw_snapshot
    from ml_ecosystem_snapshots.cli.commands.export import (
        hydrate_snapshot_refs,
        promote_hydrated_snapshot,
    )
    from ml_ecosystem_snapshots.cli.commands.verify_compliance import cmd_check
    from ml_ecosystem_snapshots.utils import resolve_griffe_parser

    # 1. api.py invalid identifier branches in extract_snapshot and extract_snapshot_isolated
    with monkeypatch.context() as mp:
        mp.setitem(
            FRAMEWORK_COLLECTORS,
            "bad/fw!",
            lambda _cat, _inc=False: [GhostRef(name="f", api_path="x.f", kind="function")],
        )
        mp.setattr("ml_ecosystem_snapshots.api.get_pkg_version", lambda _fw: "1.0.0")
        assert isinstance(extract_snapshot("bad/fw!"), Err)

        mp.setattr(
            "subprocess.run",
            lambda *_a, **_kw: subprocess.CompletedProcess(
                args=["py"],
                returncode=0,
                stdout='{"schema_version": "2.0.0", "target": "bad/fw!", "version": "1.0", "categories": {}}',
            ),
        )
        assert isinstance(extract_snapshot_isolated("html_dsl"), Err)

    # 2. utils.py resolve_griffe_parser('rest') when Parser lacks .rest attribute
    orig_sphinx = griffe.Parser.sphinx
    with monkeypatch.context() as mp:
        mp.setattr(griffe, "Parser", lambda _name: orig_sphinx)
        assert resolve_griffe_parser("rest") is orig_sphinx

    # 3. cli/commands/{diff,export,verify_compliance}.py promotion & fallback branches
    env_ok = prove_envelope_valid(_sample_valid_raw_snapshot())
    assert isinstance(env_ok, Valid)
    hyd_ok = upgrade_to_hydrated(env_ok.value)
    assert isinstance(hyd_ok, Valid)
    hyd_snap = hyd_ok.value
    det_ok = upgrade_to_deterministic(hyd_snap)
    assert isinstance(det_ok, Valid)

    assert coerce_proven_or_raw_snapshot(hyd_snap) is hyd_snap
    assert isinstance(coerce_proven_or_raw_snapshot({"schema_version": "9.9.9"}), dict)
    unhydratable_raw: dict[str, JsonValue] = {
        "schema_version": "2.0.0",
        "target": "torch",
        "version": "1.0",
        "categories": {"c": [{"schema_version": "0.0.1", "name": "x", "api_path": "t.x", "kind": "function"}]},
    }
    assert isinstance(coerce_proven_or_raw_snapshot(unhydratable_raw), dict)

    assert promote_hydrated_snapshot(hyd_snap) is hyd_snap
    assert promote_hydrated_snapshot([1, 2]) is None
    assert promote_hydrated_snapshot({"schema_version": "9.9.9"}) is None
    assert promote_hydrated_snapshot(unhydratable_raw) is None
    list_refs = hydrate_snapshot_refs([{"name": "fn", "api_path": "pkg.fn", "kind": "function"}])
    assert len(list_refs) == 1
    dict_refs = hydrate_snapshot_refs({"categories": {"c": [{"name": "fn", "api_path": "pkg.fn", "kind": "function"}]}})
    assert len(dict_refs) == 1
    assert hydrate_snapshot_refs(123) == []

    # verify_compliance.py non-envelope, unhydratable, and non-dict reference_snapshot branches
    pkg_dir = tmp_path / "chk_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("def fn() -> None: pass" + chr(10), encoding="utf-8")
    for idx, payload in enumerate(
        (
            {"schema_version": "9.9.9", "categories": {}},
            unhydratable_raw,
            [],
        )
    ):
        sf = tmp_path / f"snap_{idx}.json"
        sf.write_text(json.dumps(payload), encoding="utf-8")
        cmd_check(
            argparse.Namespace(
                snapshot_json=str(sf),
                target_path=str(pkg_dir),
                target_prefix="chk_pkg",
                reference_prefix="torch",
            )
        )

    # 4. proofs.py upgrade_to_compliance_certified explicit target_prefix and root '/' error
    assert isinstance(upgrade_to_compliance_certified(det_ok.value, source_dir=Path("/")), Invalid)
    assert isinstance(
        upgrade_to_compliance_certified(det_ok.value, source_dir=pkg_dir, min_score=0.0, target_prefix="chk_pkg"),
        Valid,
    )


def test_compliance_and_grounding_remaining_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise remaining compliance and grounding branches with `ProvenSnapshot` witnesses.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import pydantic

    from ml_ecosystem_snapshots.compliance import audit_module_sources, score_compliance
    from ml_ecosystem_snapshots.grounding import (
        GroundingEngine,
        coerce_grounding_engine,
        coerce_ir_attribute_to_json,
        ground_proven_graph,
        validate_onnx_op,
        verify_python_call,
    )

    env_ok = prove_envelope_valid(_sample_valid_raw_snapshot())
    assert isinstance(env_ok, Valid)
    hyd_ok = upgrade_to_hydrated(env_ok.value)
    assert isinstance(hyd_ok, Valid)
    hyd_snap = hyd_ok.value

    score_unhydrated = score_compliance(env_ok.value, [])
    assert score_unhydrated.total_endpoints == 1
    empty_path_ref = ExtendedGhostRef(name="anon", api_path="", kind="function")
    empty_path_snap = ProvenSnapshot[EnvelopeAndHydratedPhase](
        target=FrameworkId("torch"),
        version=SnapshotVersion("1.0"),
        envelope=SnapshotEnvelope(target="torch", version="1.0", categories={"c": [empty_path_ref]}),
        hydrated_categories=((CategoryName("c"), (empty_path_ref,)),),
    )
    assert score_compliance(empty_path_snap, []).total_endpoints == 0

    nl = chr(10)
    decorated_src = nl.join(
        [
            "def custom_dec(f):",
            "    return f",
            "",
            "@custom_dec",
            "@ns.other_dec",
            "def fn(x: int) -> int:",
            "    return x",
            "",
            "class MyBox:",
            "    CLASS_CONST = 42",
            "    def run(self) -> int:",
            "        return 1",
            "",
        ]
    )
    audit_root_first = audit_module_sources(
        {"__init__.py": "X = 1" + nl, "my_torch/ops.py": decorated_src},
        hyd_snap,
    )
    assert isinstance(audit_root_first, Valid)

    # 6. grounding/{domain_ops,dtypes,engine,python_fw}.py branches
    assert validate_onnx_op("Relu", engine=GroundingEngine()).is_grounded
    assert isinstance(coerce_grounding_engine(hyd_snap), GroundingEngine)

    class _AttrModel(pydantic.BaseModel):
        """Tiny Pydantic model to exercise BaseModel branch in coerce_ir_attribute_to_json."""

        k: int = 7

    class _CustomDevice:
        """Non-Pydantic `DeviceLikeProtocol` fixture exercising `str(val)` fallback."""

        type: str = "cpu"
        index: int | None = 0

    from ml_switcheroo_ir.validation.grounding import (
        DiagnosticSeverity,
        GroundingDiagnostic,
        GroundingReport,
    )

    assert coerce_ir_attribute_to_json(_AttrModel()) == {"k": 7}
    assert coerce_ir_attribute_to_json((1, "two")) == [1, "two"]
    assert coerce_ir_attribute_to_json(_CustomDevice()) is not None
    assert verify_python_call(
        "torch", "torch.nn.functional.relu", hyd_snap, snapshot_source="custom", args_count=1
    ).api_exists

    # Grounding graph with excess outputs, scalar TensorShape([]), and non-ERROR diagnostic filtering
    edge_case_graph = LogicalGraph(
        name="edge_cases",
        nodes=[
            LogicalNode(
                id="p0",
                op_type="Input",
                outputs=["v_first", "v_extra"],
                output_specs=[TensorSpec(shape=TensorShape([]), dtype=DType.float32)],
            ),
            LogicalNode(
                id="r0",
                op_type="torch.nn.functional.relu",
                domain="torch",
                inputs=["v_first"],
            ),
        ],
    )
    pg_edge = prove_acyclic_sorted(edge_case_graph)
    assert isinstance(pg_edge, Ok)
    assert isinstance(ground_proven_graph(pg_edge.value, hyd_snap), Valid)

    warn_report = GroundingReport(
        is_grounded=True,
        target="torch",
        symbol="torch.nn.functional.relu",
        diagnostics=[
            GroundingDiagnostic(
                field="kwargs",
                message="advisory warning",
                severity=DiagnosticSeverity.WARNING,
            )
        ],
    )
    with monkeypatch.context() as mp:
        mp.setattr("ml_ecosystem_snapshots.grounding.python_fw.validate_python_call", lambda *_a, **_kw: warn_report)
        assert isinstance(ground_proven_graph(pg_edge.value, hyd_snap), Valid)


def test_live_inspector_mro_and_capture_cli_err(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify MRO super().__init__ fallback, GroundingEngine Err branches, and cmd_capture Err handling.

    Args:
        tmp_path: Pytest temporary directory path fixture.
        monkeypatch: Pytest monkeypatch fixture.
        capsys: Pytest output capture fixture.
    """
    import argparse
    import importlib.util
    import inspect
    import sys

    import ml_ecosystem_snapshots.api as api_mod
    import ml_ecosystem_snapshots.introspection.inspector as insp_mod
    from ml_ecosystem_snapshots.cli.commands.capture import cmd_capture
    from ml_ecosystem_snapshots.grounding.engine import GroundingEngine
    from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
    from ml_ecosystem_snapshots.introspection.runtime_boundary import (
        LivePythonTarget,
        get_member_dict,
        get_runtime_signature,
    )
    from ml_ecosystem_snapshots.stubs import render_stub_modules

    mod_file = tmp_path / "mro_c_parent.py"
    mod_file.write_text(
        chr(10).join(
            [
                "class BaseLayer:",
                "    def __init__(self, x: int = 1) -> None:",
                "        self.x = x",
                "",
                "class ChildErr(BaseLayer):",
                "    def __init__(self, code: int = 0, **kwargs: int) -> None:",
                "        self.code = code",
                "        super().__init__(**kwargs)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("mro_c_parent", str(mod_file))
    assert spec is not None and spec.loader is not None
    dyn_mod = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "mro_c_parent", dyn_mod)
        spec.loader.exec_module(dyn_mod)
        base_cls = get_member_dict(dyn_mod)["BaseLayer"]
        child_cls = get_member_dict(dyn_mod)["ChildErr"]
        base_init = get_member_dict(base_cls)["__init__"]

        def _sig_fail_on_parent(target: LivePythonTarget) -> inspect.Signature:
            """Raise ValueError on parent __init__ to exercise MRO exception handler.

            Args:
                target: Runtime callable target.

            Raises:
                ValueError: When `target` is `base_init`.

            Returns:
                Runtime `inspect.Signature` for non-parent targets.
            """
            if target is base_init or "BaseLayer" in str(target):
                raise ValueError("simulated opaque C parent __init__")
            return get_runtime_signature(target)

        mp.setattr(
            insp_mod,
            "get_runtime_source",
            lambda _obj: chr(10).join(
                [
                    "class ChildErr(BaseLayer):",
                    "    def __init__(self, code: int = 0, **kwargs: int) -> None:",
                    "        super().__init__(**kwargs)",
                    "",
                ]
            ),
        )
        mp.setattr(insp_mod, "get_runtime_signature", _sig_fail_on_parent)
        inspected_child = GhostInspector.inspect(child_cls, "mro_c_parent.ChildErr")
        assert inspected_child.name == "ChildErr"

    corrupt_dir = tmp_path / "corrupt_snaps"
    corrupt_dir.mkdir()
    (corrupt_dir / "ir_v0.0.9.json").write_text("{corrupt json", encoding="utf-8")
    with monkeypatch.context() as mp:
        mp.setattr(
            api_mod,
            "extract_snapshot",
            lambda _fw: Err(SnapshotEnvelopeFault(reason="live err")),
        )
        eng = GroundingEngine(base_dirs=[str(corrupt_dir)])
        assert eng.load_target("ir") == {}

    assert render_stub_modules(None) == Ok(())

    blocker_file = tmp_path / "cap_blocker"
    blocker_file.write_text("file", encoding="utf-8")
    args = argparse.Namespace(
        frameworks=["ir"],
        out_dir=str(blocker_file / "cap_out"),
        include_nonpublic=False,
    )
    cmd_capture(args)
    out = capsys.readouterr().out
    assert "Failed to save ir" in out
