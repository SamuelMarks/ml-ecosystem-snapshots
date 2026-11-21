"""Unit tests for phantom-typed `ProvenSnapshot[PhaseT]` proof witnesses."""

from __future__ import annotations

from pathlib import Path

import pytest
from ml_switcheroo_ir.schema.ghost import (
    ExtendedGhostParam,
    ExtendedGhostRef,
    GhostParam,
    GhostRef,
    ParameterKind,
    SnapshotEnvelope,
)
from ml_switcheroo_ir.typeclasses.outcome import Err, Invalid, Ok, Valid

from ml_ecosystem_snapshots.identifiers import CategoryName, FrameworkId, SnapshotVersion
from ml_ecosystem_snapshots.models import JsonValue
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


def test_prove_envelope_valid_ok_err_and_isolation() -> None:
    """Verify prove_envelope_valid mints isolated ProvenSnapshot[EnvelopeValidPhase] and rejects invalid inputs."""
    raw = _sample_valid_raw_snapshot()
    res = prove_envelope_valid(raw)
    assert isinstance(res, Ok)
    proven: ProvenSnapshot[EnvelopeValidPhase] = res.value
    assert proven.target == FrameworkId("torch")
    assert proven.version == SnapshotVersion("2.4.0")
    assert proven.to_wire_dict()["target"] == "torch"

    # Mutating original raw dictionary or passing SnapshotEnvelope cannot mutate proven witness
    env_in = SnapshotEnvelope(target="jax", version="0.4.30", categories={})
    res_env = prove_envelope_valid(env_in)
    assert isinstance(res_env, Ok)
    env_in.categories["mutated"] = []
    assert "mutated" not in res_env.value.envelope.categories

    # Error cases: non-dict, bad schema_version, invalid target ID, path-traversal version
    assert isinstance(prove_envelope_valid("not_a_dict"), Err)
    assert isinstance(prove_envelope_valid({"schema_version": "9.9.9", "target": "torch"}), Err)
    assert isinstance(prove_envelope_valid({"target": "bad/target!", "version": "1.0"}), Err)
    assert isinstance(prove_envelope_valid({"target": "torch", "version": "../evil"}), Err)


def test_prove_hydrated_refs_valid_and_multi_error_accumulation() -> None:
    """Verify prove_hydrated_refs hydrates valid symbols and accumulates multiple schema errors."""
    env_ok = prove_envelope_valid(_sample_valid_raw_snapshot())
    assert isinstance(env_ok, Ok)

    hyd_val = prove_hydrated_refs(env_ok.value)
    assert isinstance(hyd_val, Valid)
    hydrated: ProvenSnapshot[HydratedRefsPhase] = hyd_val.value
    assert CategoryName("neural") == hydrated.hydrated_categories[0][0]
    assert len(hydrated.category_dict()["neural"]) == 1
    assert len(hydrated.all_refs()) == 1

    # Also test ExtendedGhostRef instance in envelope categories + invalid category name + 2 invalid refs
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
    assert isinstance(ok_model, Ok)
    assert isinstance(prove_hydrated_refs(ok_model.value), Valid)

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
    inv = prove_hydrated_refs(bad_proven)
    assert isinstance(inv, Invalid)
    # Accumulates category-name errors AND both ref schema errors (4 total!)
    assert len(inv.errors) == 4


def test_prove_determinism_verified_valid_and_invalid() -> None:
    """Verify prove_determinism_verified catches hex address leaks and invalid .pyi stubs."""
    env_ok = prove_envelope_valid(_sample_valid_raw_snapshot())
    assert isinstance(env_ok, Ok)
    hyd_ok = prove_hydrated_refs(env_ok.value)
    assert isinstance(hyd_ok, Valid)

    det_ok = prove_determinism_verified(hyd_ok.value)
    assert isinstance(det_ok, Valid)
    det_snap: ProvenSnapshot[DeterminismVerifiedPhase] = det_ok.value
    assert det_snap.target == FrameworkId("torch")

    leaky_ref = ExtendedGhostRef(
        name="123invalid_ident",
        api_path="pkg.123invalid_ident",
        kind="function",
        docstring="Object at 0x7fff1234abcd",
        returns_type="Tensor at 0x7fffdeadbeef",
        params=[
            ExtendedGhostParam(
                name="x",
                kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                default="<obj at 0x7fff00001111>",
                annotation=" Ptr<0x7fff22223333>",
            )
        ],
    )
    leaky_env = SnapshotEnvelope(
        target="pkg",
        version="1.0.0",
        categories={"neural": [leaky_ref]},
    )
    leaky_hydrated = ProvenSnapshot[HydratedRefsPhase](
        target=FrameworkId("pkg"),
        version=SnapshotVersion("1.0.0"),
        envelope=leaky_env,
        hydrated_categories=((CategoryName("neural"), (leaky_ref,)),),
    )
    det_inv = prove_determinism_verified(leaky_hydrated)
    assert isinstance(det_inv, Invalid)
    # 4 address leaks (docstring, returns_type, param.default, param.annotation) + 1 .pyi syntax error = 5!
    assert len(det_inv.errors) == 5


def test_prove_compliance_verified_valid_and_invalid() -> None:
    """Verify prove_compliance_verified mints ComplianceVerifiedPhase or accumulates mismatch reasons."""
    env_ok = prove_envelope_valid(_sample_valid_raw_snapshot())
    assert isinstance(env_ok, Ok)
    hyd_ok = prove_hydrated_refs(env_ok.value)
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
    comp_snap: ProvenSnapshot[ComplianceVerifiedPhase] = comp_ok.value
    assert comp_snap.version == SnapshotVersion("2.4.0")

    # Missing endpoint + mismatched endpoint case
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
    assert isinstance(two_proven, Ok)
    two_hyd = prove_hydrated_refs(two_proven.value)
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
    joined = " | ".join(comp_inv.errors)
    assert "Missing endpoint: torch.silu" in joined
    assert "Signature mismatch: torch.relu" in joined
    assert "Opaque C-extension" in joined


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

    # 1. parse_compression_format branches
    assert parse_compression_format(CompressionFormat.GZIP) == Ok(CompressionFormat.GZIP)
    assert parse_compression_format("") == Ok(CompressionFormat.NONE)
    assert parse_compression_format("zst") == Ok(CompressionFormat.ZSTD)
    assert isinstance(parse_compression_format("lz4"), Err)

    # 2. load_json_data & stream_snapshot_items missing file branches
    missing_file = tmp_path / "does_not_exist.json"
    assert isinstance(load_json_data(missing_file), Err)
    assert list(stream_snapshot_items(missing_file)) == []

    # 3. write_snapshot with SnapshotEnvelope, ProvenSnapshot, invalid compress, and OSError
    env = SnapshotEnvelope(target="pkg", version="1.0.0", categories={})
    proven = prove_envelope_valid(env)
    assert isinstance(proven, Ok)
    env_written = write_snapshot("pkg", env, str(tmp_path / "out_env"))
    assert isinstance(env_written, Ok)
    assert isinstance(write_snapshot("pkg", env, str(tmp_path / "out_bad"), compress="lz4"), Err)
    blocker_file = tmp_path / "blocker"
    blocker_file.write_text("file", encoding="utf-8")
    assert isinstance(write_snapshot("pkg", env, str(blocker_file / "subdir")), Err)

    # 4. extract_snapshot_isolated non-dict & empty-dict stdout, and extract_all_snapshots output_dir
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

    # 5. hydrate_ghost_ref error branch on invalid parameter payload
    assert isinstance(
        hydrate_ghost_ref({"name": "fn", "api_path": "p.fn", "params": ["not_a_param"]}),
        Err,
    )


def test_total_repository_and_stablehlo_offline_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify SnapshotRepository.resolve_result, live extraction fallback, and StableHLO offline errors.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    from ml_ecosystem_snapshots.exceptions import OfflineModeNetworkError, SnapshotError
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
        active_extract=lambda _fw, include_nonpublic=False: Err(SnapshotError("boom")),
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
    assert isinstance(empty_repo.resolve_result("empty_fw"), Err)

    with monkeypatch.context() as mp:
        mp.setattr(
            "ml_ecosystem_snapshots.network_env.fetch_url",
            lambda *_a, **_kw: Err(OfflineModeNetworkError("offline td")),
        )
        with pytest.raises(OfflineModeNetworkError):
            build_stablehlo_snapshot.extract_ops(source_content=None)

        calls: list[str] = []

        def _fail_second(url: str, timeout: int = 10) -> Err[SnapshotError]:
            """Fail first with generic SnapshotError and second with OfflineModeNetworkError.

            Args:
                url: Target URL string.
                timeout: Timeout in seconds.

            Returns:
                `Err` outcome.
            """
            del timeout
            calls.append(url)
            if len(calls) == 1:
                return Err(SnapshotError("td transient error"))
            return Err(OfflineModeNetworkError("offline spec"))

        mp.setattr("ml_ecosystem_snapshots.network_env.fetch_url", _fail_second)
        with pytest.raises(OfflineModeNetworkError):
            build_stablehlo_snapshot.extract_ops(source_content=None)


def test_total_compliance_diff_stubs_and_cli_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify remaining branches in compliance, diff, stubs, live_inspector, and capture CLI.

    Args:
        tmp_path: Pytest temporary directory path fixture.
        monkeypatch: Pytest monkeypatch fixture.
        capsys: Pytest output capture fixture.
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

    # 1. NamespaceAlignmentRule lookup helpers
    rule = NamespaceAlignmentRule(
        prefix_map=(("zero_jax", "jax"),),
        node_prefix_rewrites=(("zero_jax.", "{target_prefix}.jax."),),
    )
    assert NamespaceAlignmentRule.model_validate({"prefix_map": {"zero_jax": "jax"}}).lookup_prefix("zero_jax") == "jax"
    assert rule.lookup_prefix("zero_jax") == "jax"
    assert rule.lookup_prefix("missing") is None
    assert rule.lookup_node_rewrite("zero_jax.") == "{target_prefix}.jax."
    assert rule.lookup_node_rewrite("missing") is None

    # 2. dtype_rules empty dtype, negative dimension, and 0D scalar memory estimation
    assert isinstance(get_dtype_bitwidth("   "), Err)
    assert isinstance(estimate_tensor_memory_bytes([4], "   "), Err)
    assert isinstance(estimate_tensor_memory_bytes([-1, 4], "float32"), Err)
    assert estimate_tensor_memory_bytes([], "float32") == Ok(4.0)

    # 3. ast_extractor error propagation branches
    assert isinstance(extract_target_ast(str(tmp_path / "missing.py")), Err)
    bad_pkg = tmp_path / "bad_pkg"
    bad_pkg.mkdir()
    (bad_pkg / "__init__.py").write_text("def broken(:\n", encoding="utf-8")
    assert isinstance(extract_target_ast(str(bad_pkg)), Err)
    assert isinstance(extract_target_refs_single(str(tmp_path / "missing.py"), "a", "b"), Err)
    assert isinstance(extract_target_refs_single(str(bad_pkg), "bad_pkg", "b"), Err)
    assert isinstance(extract_target_refs([str(tmp_path / "missing.py")], "a", "b"), Err)

    # 4. scorer & diff & stubs accepting SnapshotEnvelope and ProvenSnapshot directly
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
    assert isinstance(proven_res, Ok)
    proven = proven_res.value

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
        "categories": {"ops": [{"name": "123bad", "api_path": "pkg.123bad", "kind": "function", "params": []}]}
    }
    assert isinstance(generate_stubs(bad_stub_snap, tmp_path / "stubs_bad"), Err)

    blocker = tmp_path / "stub_blocker"
    blocker.write_text("file", encoding="utf-8")
    assert isinstance(generate_stubs(env, blocker / "nested"), Err)


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
    from ml_ecosystem_snapshots.exceptions import SnapshotError
    from ml_ecosystem_snapshots.grounding.engine import GroundingEngine
    from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
    from ml_ecosystem_snapshots.introspection.runtime_boundary import (
        LivePythonTarget,
        get_member_dict,
        get_runtime_signature,
    )
    from ml_ecosystem_snapshots.stubs import render_stub_modules

    # 1. MRO super().__init__(**kwargs) when parent signature inspection raises ValueError
    mod_file = tmp_path / "mro_c_parent.py"
    mod_file.write_text(
        "class BaseLayer:\n"
        "    def __init__(self, x: int = 1) -> None:\n"
        "        self.x = x\n\n"
        "class ChildErr(BaseLayer):\n"
        "    def __init__(self, code: int = 0, **kwargs: int) -> None:\n"
        "        self.code = code\n"
        "        super().__init__(**kwargs)\n",
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
            lambda _obj: (
                "class ChildErr(BaseLayer):\n"
                "    def __init__(self, code: int = 0, **kwargs: int) -> None:\n"
                "        super().__init__(**kwargs)\n"
            ),
        )
        mp.setattr(insp_mod, "get_runtime_signature", _sig_fail_on_parent)
        inspected_child = GhostInspector.inspect(child_cls, "mro_c_parent.ChildErr")
        assert inspected_child.name == "ChildErr"

    # 2. GroundingEngine with corrupt JSON file and Err extract_snapshot fallback
    corrupt_dir = tmp_path / "corrupt_snaps"
    corrupt_dir.mkdir()
    (corrupt_dir / "ir_v0.0.8.json").write_text("{corrupt json", encoding="utf-8")
    with monkeypatch.context() as mp:
        mp.setattr(api_mod, "extract_snapshot", lambda _fw: Err(SnapshotError("live err")))
        eng = GroundingEngine(base_dirs=[str(corrupt_dir)])
        assert eng.load_target("ir") == {}

    # 3. render_stub_modules non-mapping / non-sequence fallback
    assert render_stub_modules(None) == Ok(())

    # 4. cmd_capture write_snapshot Err branch via file blocker
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
