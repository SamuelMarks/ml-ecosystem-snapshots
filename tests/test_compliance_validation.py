"""Tests for compliance verification, assembly checking (MLIR, SASS, RDNA), and shape validation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import pytest_mock
from ml_switcheroo_ir.typeclasses.outcome import Err, Ok, raise_on_err


def test_align_namespace_exact_mapping() -> None:
    """Test align_namespace exact mapping with configurable NamespaceAlignmentRule preset."""
    from ml_ecosystem_snapshots.cli.commands.verify_compliance import (
        ZERO_ECOSYSTEM_ALIGNMENT_RULE,
        resolve_alignment_preset,
    )
    from ml_ecosystem_snapshots.compliance import align_namespace

    assert resolve_alignment_preset("zero") == ZERO_ECOSYSTEM_ALIGNMENT_RULE
    assert resolve_alignment_preset(None, "zero_jax") == ZERO_ECOSYSTEM_ALIGNMENT_RULE
    assert resolve_alignment_preset("default", "zero_jax").prefix_map == ()
    assert align_namespace("zero_jax", "target_prefix", "ref", alignment_rule=ZERO_ECOSYSTEM_ALIGNMENT_RULE) == "jax"
    assert (
        align_namespace("zero_optax", "target_prefix", "ref", alignment_rule=ZERO_ECOSYSTEM_ALIGNMENT_RULE) == "optax"
    )


def test_score_compliance_sig_tuple_edge_cases() -> None:
    """Test compliance scoring with signature tuple matching and sanitized defaults."""
    from ml_switcheroo_ir.schema.ghost import GhostParam, GhostRef, ParameterKind

    from ml_ecosystem_snapshots.compliance import score_compliance
    from ml_ecosystem_snapshots.models import JsonValue

    ref_snap: dict[str, JsonValue] = {
        "categories": {
            "test": [
                {
                    "api_path": "ref.func1",
                    "name": "func1",
                    "kind": "function",
                    "params": [
                        {
                            "name": "a",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "default": "'```(None)```'",
                            "annotation": '"str"',
                        },
                        {
                            "name": "b",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "default": "```(None)```",
                            "annotation": "'int'",
                        },
                        {
                            "name": "c",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "default": "'___NONE___'",
                            "annotation": "float",
                        },
                    ],
                    "docstring": "",
                    "aliases": [],
                    "returns_type": "None",
                }
            ]
        }
    }
    target_refs = [
        GhostRef(
            api_path="ref.func1",
            name="func1",
            kind="function",
            params=[
                GhostParam(
                    name="a",
                    kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                    default="None",
                    annotation="str",
                ),
                GhostParam(
                    name="b",
                    kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                    default="None",
                    annotation="int",
                ),
                GhostParam(
                    name="c",
                    kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                    default="None",
                    annotation="float",
                ),
            ],
            docstring="",
            aliases=[],
            returns_type="None",
        )
    ]
    res = score_compliance(ref_snap, target_refs)
    assert "ref.func1" in res.matched


def test_score_compliance_varargs_fallback_special_cases() -> None:
    """Test compliance scoring enforces strict parameter comparison."""
    from ml_switcheroo_ir.schema.ghost import GhostParam, GhostRef, ParameterKind

    from ml_ecosystem_snapshots.compliance import score_compliance
    from ml_ecosystem_snapshots.models import JsonValue

    ref_snap: dict[str, JsonValue] = {
        "categories": {
            "test": [
                {
                    "api_path": "chex.assert_shape",
                    "name": "assert_shape",
                    "kind": "function",
                    "params": [
                        {
                            "name": "x",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "default": None,
                            "annotation": None,
                        },
                    ],
                    "docstring": "",
                    "aliases": [],
                    "returns_type": "None",
                },
                {
                    "api_path": "jax.nn.relu",
                    "name": "relu",
                    "kind": "function",
                    "params": [
                        {
                            "name": "x",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "default": None,
                            "annotation": "Array",
                        },
                    ],
                    "docstring": "",
                    "aliases": [],
                    "returns_type": "None",
                },
                {
                    "api_path": "optax.adam",
                    "name": "adam",
                    "kind": "function",
                    "params": [
                        {
                            "name": "learning_rate",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "default": None,
                            "annotation": "float",
                        },
                    ],
                    "docstring": "",
                    "aliases": [],
                    "returns_type": "None",
                },
                {
                    "api_path": "flax.nnx.Linear",
                    "name": "Linear",
                    "kind": "class",
                    "params": [
                        {
                            "name": "in_features",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "default": None,
                            "annotation": "int",
                        },
                    ],
                    "docstring": "",
                    "aliases": [],
                    "returns_type": "None",
                },
            ]
        }
    }

    target_refs = [
        GhostRef(
            api_path="chex.assert_shape",
            name="assert_shape",
            kind="function",
            params=[
                GhostParam(
                    name="x",
                    kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                    default=None,
                    annotation=None,
                ),
                GhostParam(
                    name="args",
                    kind=ParameterKind.VAR_POSITIONAL,
                    default=None,
                    annotation=None,
                ),
            ],
            docstring="",
            aliases=[],
            returns_type="None",
        ),
        GhostRef(
            api_path="jax.nn.relu",
            name="relu",
            kind="function",
            params=[
                GhostParam(
                    name="wrong_param",
                    kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                    default=None,
                    annotation="str",
                ),
            ],
            docstring="",
            aliases=[],
            returns_type="None",
        ),
        GhostRef(
            api_path="optax.adam",
            name="adam",
            kind="function",
            params=[
                GhostParam(
                    name="wrong_lr",
                    kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                    default=None,
                    annotation="str",
                ),
            ],
            docstring="",
            aliases=[],
            returns_type="None",
        ),
        GhostRef(
            api_path="flax.nnx.Linear",
            name="Linear",
            kind="class",
            params=[
                GhostParam(
                    name="wrong_in",
                    kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                    default=None,
                    annotation="str",
                ),
            ],
            docstring="",
            aliases=[],
            returns_type="None",
        ),
    ]

    # Strict parameter comparison rejects mismatched parameter names/types even when len(params) == len(params)
    strict_res = score_compliance(ref_snap, target_refs)
    assert strict_res.matched == ()
    mismatched_paths = {m.api_path for m in strict_res.mismatched}
    assert mismatched_paths == {
        "chex.assert_shape",
        "jax.nn.relu",
        "optax.adam",
        "flax.nnx.Linear",
    }


def test_extract_target_refs_single_sys_path_and_logging(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    """Verify extract_target_refs_single restores sys.path cleanly and logs warnings instead of printing.

    Args:
        tmp_path: Temporary directory path fixture.
        capsys: Pytest stdout/stderr capture fixture.
        caplog: Pytest log capture fixture.
    """
    import logging
    import sys

    from ml_ecosystem_snapshots.compliance import extract_target_refs_single

    pkg_dir = Path(os.path.join(tmp_path, "warn_pkg"))
    pkg_dir.mkdir()
    Path(os.path.join(pkg_dir, "__init__.py")).touch()
    sub_mod = Path(os.path.join(pkg_dir, "broken_mod.py"))
    sub_mod.write_text("import nonexistent_dep_xyz_987\ndef broken_fn(x: int) -> int:\n    return x\n")

    before_sys_path = list(sys.path)
    with caplog.at_level(logging.WARNING, logger="ml_ecosystem_snapshots.compliance"):
        refs = raise_on_err(extract_target_refs_single(str(sub_mod), "warn_pkg.broken_mod", "ref"))

    assert [r.api_path for r in refs] == ["ref.broken_fn"]
    assert list(sys.path) == before_sys_path
    captured = capsys.readouterr()
    assert captured.out == ""
    assert any("Exception in walk for warn_pkg.broken_mod.nonexistent_dep_xyz_987" in r.message for r in caplog.records)


def test_extract_target_refs_string_path(tmp_path: Path) -> None:
    """Verify extract target refs string path behavior.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    from ml_ecosystem_snapshots.compliance import extract_target_refs

    pkg_dir = Path(os.path.join(tmp_path, "str_pkg"))
    pkg_dir.mkdir()
    Path(os.path.join(pkg_dir, "__init__.py")).touch()
    sub_mod = Path(os.path.join(pkg_dir, "api.py"))
    sub_mod.write_text("def my_func(a: int) -> None: pass")

    refs = raise_on_err(extract_target_refs(str(sub_mod), "str_pkg.api", "ref"))
    assert len(refs) == 1
    assert refs[0].api_path == "ref.my_func"


def test_get_module_info_from_path_no_src_dir(tmp_path: Path) -> None:
    """Verify get module info from path no src dir behavior.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    from ml_ecosystem_snapshots.compliance import get_module_info_from_path

    # mod_name will be empty if we point to a root dir without __init__.py
    path, mod = raise_on_err(get_module_info_from_path(str(tmp_path), target_prefix="my_pkg"))
    assert path == str(tmp_path)
    assert mod == "my_pkg"


def test_extract_target_refs_list_path(tmp_path: Path) -> None:
    """Verify extract target refs list path behavior.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    from ml_ecosystem_snapshots.compliance import extract_target_refs

    pkg_dir = Path(os.path.join(tmp_path, "list_pkg"))
    pkg_dir.mkdir()
    Path(os.path.join(pkg_dir, "__init__.py")).touch()
    sub_mod = Path(os.path.join(pkg_dir, "api.py"))
    sub_mod.write_text("def list_func(): pass")

    refs = raise_on_err(extract_target_refs([str(sub_mod)], "list_pkg.api", "ref"))
    assert len(refs) == 1
    assert refs[0].api_path == "ref.list_func"


def test_check_mlir_text_compliance() -> None:
    """Test check_mlir_text_compliance with valid and invalid MLIR snippets."""
    from ml_ecosystem_snapshots.compliance import check_mlir_text_compliance

    # 1. Valid MLIR snippet
    valid_snippet = """
    // Module containing standard arith ops
    module {
        func.func @test(%a: f32, %b: f32) -> f32 {
            %0 = arith.addf(%a, %b) : f32
            return %0 : f32
        }
    }
    """
    res_valid = check_mlir_text_compliance(valid_snippet)
    assert res_valid.is_compliant is True
    assert res_valid.verified_ops >= 1

    # 2. Invalid MLIR snippet with non-existent op
    invalid_snippet = """
    %0 = non_existent_dialect.non_existent_op(%a) : i32
    """
    res_invalid = check_mlir_text_compliance(invalid_snippet)
    assert res_invalid.is_compliant is False
    assert len(res_invalid.errors) > 0

    # 3. Existing op with invalid operand count (branch 404)
    res_bad_count = check_mlir_text_compliance("%0 = arith.addf(%a) : f32")
    assert res_bad_count.is_compliant is False
    assert any("Operand count mismatch" in err for err in res_bad_count.errors)


def test_check_sass_assembly_compliance() -> None:
    """Test check_sass_assembly_compliance with valid, predicated, and invalid SASS instructions."""
    from ml_ecosystem_snapshots.compliance import check_sass_assembly_compliance

    # 1. Valid SASS snippet on sm_80
    valid_sass = """
    /*0010*/ FADD.FTZ R0, R1, R2;
    /*0020*/ @P0 FADD R3, R4, R5;
    """
    res_valid = check_sass_assembly_compliance(valid_sass, sm_arch="sm_80")
    assert res_valid.is_compliant is True
    assert res_valid.verified_instructions == 2

    # 2. Incompatible instruction for architecture: WGMMA on sm_80
    invalid_sass = """
    WGMMA R0, R1, R2, R3;
    """
    res_invalid = check_sass_assembly_compliance(invalid_sass, sm_arch="sm_80")
    assert res_invalid.is_compliant is False
    assert any("WGMMA instructions are strictly supported on sm_90+" in err for err in res_invalid.errors)

    # 3. Comment lines starting with '#' and lines that do not match instruction regex
    sass_comments = "# A comment line\n// Another comment\n@@ invalid label !@#\n\n"
    res_comments = check_sass_assembly_compliance(sass_comments)
    assert res_comments.is_compliant is True
    assert res_comments.total_instructions == 0


def test_check_rdna_assembly_compliance() -> None:
    """Test check_rdna_assembly_compliance with valid and misaligned RDNA instructions."""
    from ml_ecosystem_snapshots.compliance import check_rdna_assembly_compliance

    # 1. Valid RDNA snippet
    valid_rdna = """
    // Vector addition
    v_add_f32 v0, v1, v2
    """
    res_valid = check_rdna_assembly_compliance(valid_rdna)
    assert res_valid.is_compliant is True
    assert res_valid.verified_instructions == 1

    # 2. Misaligned 64-bit register pair (odd start index)
    invalid_rdna = """
    v_add_f32 v[1:2], v0, v1
    """
    res_invalid = check_rdna_assembly_compliance(invalid_rdna)
    assert res_invalid.is_compliant is False
    assert any("Register alignment error" in err for err in res_invalid.errors)

    # 3. Comment lines starting with ';' or '#' and lines that do not match instruction regex
    rdna_comments = "; Semicolon comment\n# Hash comment\n!! invalid rdna line !@#\n\n"
    res_comments = check_rdna_assembly_compliance(rdna_comments)
    assert res_comments.is_compliant is True
    assert res_comments.total_instructions == 0


def test_validate_broadcast_and_matmul_shapes() -> None:
    """Test validate_broadcast_shapes and basic 1D/strict-2D validate_matmul_shapes rules."""
    from ml_ecosystem_snapshots.compliance import (
        validate_broadcast_shapes,
        validate_matmul_shapes,
    )

    # 1. Broadcasting
    assert validate_broadcast_shapes([2, 3], [1, 3]) == Ok((2, 3))

    err2 = validate_broadcast_shapes([2, 3], [3, 2])
    assert isinstance(err2, Err) and "Shape mismatch" in err2.error

    assert validate_broadcast_shapes([1, 4, 1], [2, 1, 5]) == Ok((2, 4, 5))

    # Dynamic / wildcard and non-int dimensions
    assert validate_broadcast_shapes(["?"], [4]) == Ok((-1,))
    assert validate_broadcast_shapes(["invalid_dim"], [4]) == Ok((-1,))

    # 2. Matmul strict 2D
    err_s2d = validate_matmul_shapes([2, 3, 4], [2, 4, 5], strict_2d=True)
    assert isinstance(err_s2d, Err) and "Strict 2D matmul error" in err_s2d.error

    assert validate_matmul_shapes([2, 3], [3, 4], strict_2d=True) == Ok((2, 4))

    # 1D vector dot products
    assert validate_matmul_shapes([4], [4]) == Ok(())
    assert isinstance(validate_matmul_shapes(["?"], [4]), Ok)
    assert isinstance(validate_matmul_shapes([4], ["?"]), Ok)
    assert isinstance(validate_matmul_shapes(["?"], ["?"]), Ok)

    err_1d = validate_matmul_shapes([4], [5])
    assert isinstance(err_1d, Err) and "dot product mismatch" in err_1d.error


def test_validate_matmul_vector_and_batch_shapes() -> None:
    """Test validate_matmul_shapes for vector-matrix, matrix-vector, 2D, and batched shapes."""
    from ml_ecosystem_snapshots.compliance import validate_matmul_shapes

    # Vector-matrix (1D x 2D) and Matrix-vector (2D x 1D)
    assert validate_matmul_shapes([4], [4, 5]) == Ok((5,))
    assert validate_matmul_shapes([3, 4], [4]) == Ok((3,))

    # Batch vector-matrix and matrix-vector
    assert validate_matmul_shapes([4], [2, 4, 5]) == Ok((2, 5))
    assert validate_matmul_shapes([2, 3, 4], [4]) == Ok((2, 3))

    # Symbolic dynamic dimensions in vector-matrix and matrix-vector
    assert validate_matmul_shapes(["?"], ["?", 5]) == Ok((5,))
    assert validate_matmul_shapes([3, "?"], ["?"]) == Ok((3,))

    # Contracting mismatch in vector-matrix / matrix-vector
    err_vm = validate_matmul_shapes([4], [5, 6])
    assert isinstance(err_vm, Err) and "Vector-matrix multiplication contracting dimension mismatch" in err_vm.error

    err_mv = validate_matmul_shapes([3, 4], [5])
    assert isinstance(err_mv, Err) and "Matrix-vector multiplication contracting dimension mismatch" in err_mv.error

    # Rank < 1 error (e.g. scalar operand)
    err_r0 = validate_matmul_shapes([], [4, 5])
    assert isinstance(err_r0, Err) and "requires at least 2D operands" in err_r0.error

    # 2D matmul valid and contracting mismatch
    assert validate_matmul_shapes([2, 3], [3, 4]) == Ok((2, 4))
    assert isinstance(validate_matmul_shapes([2, "?"], [3, 4]), Ok)

    err_k = validate_matmul_shapes([2, 3], [4, 5])
    assert isinstance(err_k, Err) and "contracting dimension mismatch" in err_k.error.lower()

    # Batch matmul valid and batch mismatch
    assert validate_matmul_shapes([2, 3, 4], [2, 4, 5]) == Ok((2, 3, 5))

    err_bm = validate_matmul_shapes([2, 3, 4], [3, 4, 5])
    assert isinstance(err_bm, Err) and "Batch dimension broadcasting error" in err_bm.error


def test_compliance_edge_branches(tmp_path: Path) -> None:
    """Test edge branches in align_namespace and extract_target_refs_single.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    from ml_ecosystem_snapshots.cli.commands.verify_compliance import ZERO_ECOSYSTEM_ALIGNMENT_RULE
    from ml_ecosystem_snapshots.compliance import (
        align_namespace,
        extract_target_refs_single,
    )

    # Test exact mapping match where api_path != target_prefix
    assert (
        align_namespace("zero_chex", "custom_target", "custom_ref", alignment_rule=ZERO_ECOSYSTEM_ALIGNMENT_RULE)
        == "chex"
    )
    assert (
        align_namespace(
            "zero_chex.assert_shape",
            "custom_target",
            "custom_ref",
            alignment_rule=ZERO_ECOSYSTEM_ALIGNMENT_RULE,
        )
        == "chex.assert_shape"
    )
    assert (
        align_namespace("zero_orbax", "custom_target", "custom_ref", alignment_rule=ZERO_ECOSYSTEM_ALIGNMENT_RULE)
        == "orbax"
    )
    assert (
        align_namespace("zero_grain", "custom_target", "custom_ref", alignment_rule=ZERO_ECOSYSTEM_ALIGNMENT_RULE)
        == "grain"
    )

    # Test extract_target_refs_single with zero_jax module and path_rewrites
    pkg_dir = tmp_path / "zero_jax"
    os.makedirs(pkg_dir / "nn" / "nn", exist_ok=True)
    with open(pkg_dir / "__init__.py", "w") as f:
        f.write("")
    with open(pkg_dir / "nn" / "__init__.py", "w") as f:
        f.write("")
    with open(pkg_dir / "nn" / "nn" / "__init__.py", "w") as f:
        f.write("def relu(x):\n    '''Rectified linear unit.'''\n    return x\n")

    refs = raise_on_err(
        extract_target_refs_single(
            str(pkg_dir),
            "zero_jax",
            "jax",
            alignment_rule=ZERO_ECOSYSTEM_ALIGNMENT_RULE,
        )
    )
    assert any(r.api_path == "jax.nn.relu" for r in refs)

    # Also test when node.path does not start with zero_jax. but zero_jax. in current_path
    pkg_dir2 = tmp_path / "sub_zero_jax"
    os.makedirs(pkg_dir2 / "zero_jax", exist_ok=True)
    with open(pkg_dir2 / "__init__.py", "w") as f:
        f.write("")
    with open(pkg_dir2 / "zero_jax" / "__init__.py", "w") as f:
        f.write("def gelu(x):\n    '''Gelu.'''\n    return x\n")
    refs2 = raise_on_err(
        extract_target_refs_single(
            str(pkg_dir2),
            "sub_zero_jax",
            "jax",
            alignment_rule=ZERO_ECOSYSTEM_ALIGNMENT_RULE,
        )
    )
    assert len(refs2) >= 1

    # Test rule where neither node.path nor current_path matches node_prefix_rewrites
    single_file = tmp_path / "single_file.py"
    with open(single_file, "w") as f:
        f.write("def standalone():\n    '''Standalone.'''\n    return 42\n")
    refs3 = raise_on_err(
        extract_target_refs_single(
            str(single_file),
            "single_file",
            "ref_pkg",
            alignment_rule=ZERO_ECOSYSTEM_ALIGNMENT_RULE,
        )
    )
    assert len(refs3) >= 1


def test_score_compliance_opaque_c_extensions() -> None:
    """Test opaque C-extension signature matching and strict rejection."""
    from ml_switcheroo_ir.schema.ghost import GhostRef

    from ml_ecosystem_snapshots.compliance import score_compliance
    from ml_ecosystem_snapshots.models import JsonValue

    ref_snap: dict[str, JsonValue] = {
        "categories": {
            "math": [
                {
                    "api_path": "test.opaque",
                    "name": "opaque",
                    "kind": "function",
                    "environment_tags": ["opaque_c_extension"],
                    "params": [],
                }
            ]
        }
    }
    target_opaque = [
        GhostRef(
            api_path="test.opaque",
            name="opaque",
            kind="function",
            environment_tags=["opaque_c_extension"],
            params=[],
        )
    ]
    # Permissive
    res_perm = score_compliance(ref_snap, target_opaque, strict_c_extensions=False)
    assert "test.opaque" in res_perm.matched

    # Strict
    res_strict = score_compliance(ref_snap, target_opaque, strict_c_extensions=True)
    assert any(m.api_path == "test.opaque" for m in res_strict.mismatched)


def test_extract_target_refs_single_root_function(tmp_path: Path, mocker: pytest_mock.MockerFixture) -> None:
    """Test extract_target_refs_single when root node is a Function.

    Args:
        tmp_path: Temporary directory path fixture.
        mocker: Pytest mocker fixture.
    """
    import griffe

    from ml_ecosystem_snapshots.compliance import extract_target_refs_single

    fpath = os.path.join(str(tmp_path), "standalone.py")
    with open(fpath, "w") as f:
        f.write("def standalone():\n    return 42\n")

    def real_fn() -> int:
        """Standalone.

        Returns:
            Integer constant 42.
        """
        return 42

    fn_node = griffe.Function(name="standalone", parent=None)
    mocker.patch("griffe.load", return_value=fn_node)
    mocker.patch("importlib.import_module", return_value=real_fn)

    refs = raise_on_err(extract_target_refs_single(fpath, "standalone", "ref_pkg"))
    assert len(refs) == 1


def test_compliance_models_typed_attributes() -> None:
    """Test typed attribute access and serialization on ComplianceMismatchEntry, ComplianceScoreReport, and SnippetComplianceReport."""
    from ml_ecosystem_snapshots.compliance import (
        ComplianceMismatchEntry,
        ComplianceScoreReport,
        SnippetComplianceReport,
    )

    mismatch = ComplianceMismatchEntry(
        api_path="pkg.fn",
        expected=(("a", "POSITIONAL_OR_KEYWORD", None, "int"),),
        actual=(),
        reason="Mismatch reason",
    )
    assert mismatch.api_path == "pkg.fn"
    assert len(mismatch.expected) == 1
    assert mismatch.actual == ()
    assert mismatch.reason == "Mismatch reason"

    score_rep = ComplianceScoreReport(
        score_percentage=75.0,
        total_endpoints=4,
        matched=("a", "b", "c"),
        missing=("d",),
        mismatched=(mismatch,),
        opaque_signatures=("op1",),
        warnings=("warn1",),
    )
    assert score_rep.score_percentage == 75.0
    assert score_rep.total_endpoints == 4
    assert score_rep.matched == ("a", "b", "c")
    assert score_rep.missing == ("d",)
    assert score_rep.mismatched == (mismatch,)
    assert score_rep.opaque_signatures == ("op1",)
    assert score_rep.warnings == ("warn1",)

    snip_rep = SnippetComplianceReport(
        is_compliant=True,
        total_ops=3,
        verified_ops=3,
        total_instructions=2,
        verified_instructions=2,
        errors=(),
    )
    assert snip_rep.is_compliant is True
    assert snip_rep.total_ops == 3
    assert snip_rep.verified_ops == 3
    assert snip_rep.total_instructions == 2
    assert snip_rep.verified_instructions == 2
    assert snip_rep.errors == ()
