"""Exhaustive test suite for the Anti-Hallucination Grounding SDK."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import pytest
from ml_switcheroo_ir.validation.core import ValidationError

from ml_ecosystem_snapshots.grounding import (
    DiagnosticSeverity,
    GroundingDiagnostic,
    GroundingEngine,
    GroundingReport,
    compute_levenshtein,
    validate_mlir_op,
    validate_python_call,
    validate_rdna_instruction,
    validate_sass_instruction,
    validate_stablehlo_op,
)
from ml_ecosystem_snapshots.models import GhostRef, JsonValue


def _diag_field(diag: GroundingDiagnostic | ValidationError) -> str:
    """Return the evaluated field or attribute name from a grounding diagnostic item.

    Args:
        diag: Grounding diagnostic model or graph validation error instance.

    Returns:
        Evaluated field or attribute identifier string.
    """
    return diag.field if isinstance(diag, GroundingDiagnostic) else diag.attribute


def test_compute_levenshtein() -> None:
    """Test string distance computation with edge cases."""
    assert compute_levenshtein("", "") == 0
    assert compute_levenshtein("abc", "") == 3
    assert compute_levenshtein("", "xyz") == 3
    assert compute_levenshtein("kitten", "sitting") == 3
    assert compute_levenshtein("stablehlo.matmul", "stablehlo.dot_general") > 0
    assert compute_levenshtein("addf", "addf") == 0


def test_grounding_models() -> None:
    """Test GroundingDiagnostic and GroundingReport models and methods."""
    diag = GroundingDiagnostic(
        field="operands",
        message="Invalid operand count",
        severity=DiagnosticSeverity.ERROR,
        suggested_fix="add",
    )
    assert diag.field == "operands"
    assert _diag_field(diag) == "operands"
    val_err = ValidationError(node_id="n0", attribute="kind", message="Ungrounded symbol")
    assert _diag_field(val_err) == "kind"
    assert diag.severity == DiagnosticSeverity.ERROR
    assert diag.suggested_fix == "add"

    report = GroundingReport(
        is_grounded=True,
        target="stablehlo",
        symbol="stablehlo.add",
    )
    assert [report.has_errors, report.is_grounded] == [False, True]

    report.add_diagnostic("info_field", "Informational note", DiagnosticSeverity.INFO)
    assert [report.has_errors, report.is_grounded] == [False, True]

    report.add_diagnostic("warn_field", "Warning message", DiagnosticSeverity.WARNING)
    assert [report.has_errors, report.is_grounded] == [False, True]

    report.add_diagnostic("err_field", "Error message", DiagnosticSeverity.ERROR)
    assert [report.has_errors, report.is_grounded] == [True, False]


def test_grounding_engine_custom_dir(tmp_path: Path) -> None:
    """Test GroundingEngine loading from custom directory with json and json.gz.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    custom_dir = str(tmp_path / "custom_snaps")
    os.makedirs(custom_dir, exist_ok=True)

    # 1. Plain json snapshot
    plain_data = [
        {
            "name": "custom_op",
            "api_path": "custom.custom_op",
            "kind": "function",
            "params": [{"name": "x", "kind": "POSITIONAL_OR_KEYWORD"}],
        }
    ]
    with open(os.path.join(custom_dir, "custom_exhaustive.json"), "w") as f:
        json.dump(plain_data, f)

    # 2. Gzipped json snapshot with categories layout
    gz_data = {
        "categories": {
            "math": [
                {
                    "name": "gz_op",
                    "api_path": "gz.gz_op",
                    "kind": "function",
                    "params": [],
                }
            ]
        }
    }
    with gzip.open(os.path.join(custom_dir, "gz_exhaustive.json.gz"), "wt", encoding="utf-8") as gf:
        json.dump(gz_data, gf)

    # 3. Snapshot with items list
    items_data = {
        "items": [
            {
                "name": "item_op",
                "api_path": "items.item_op",
                "kind": "function",
                "params": [],
            }
        ]
    }
    with open(os.path.join(custom_dir, "items_v1.0.0.json"), "w") as f:
        json.dump(items_data, f)

    # 4. Single item dictionary and items with mnemonic and malformed entries
    single_data = {
        "name": "single_op",
        "api_path": "single.single_op",
        "kind": "function",
        "params": [],
    }
    with open(os.path.join(custom_dir, "single.json"), "w") as f:
        json.dump(single_data, f)

    # 5. File with mnemonics, non-dict entries, and items causing hydrate errors
    with open(os.path.join(custom_dir, "mixed_exhaustive.json"), "w") as f:
        json.dump(
            [
                "not_a_dict",
                {
                    "mnemonic": "INST_M",
                    "api_path": "isa.inst.INST_M",
                    "operands": [["R1", "R2"]],
                },
                123,
            ],
            f,
        )

    # 6. Non-json file
    with open(os.path.join(custom_dir, "ignored.txt"), "w") as f:
        f.write("ignored")

    engine = GroundingEngine(base_dirs=[custom_dir, "/non_existent_path_xyz"])

    # Verify custom_op
    assert engine.has_symbol("custom", "custom.custom_op")
    assert engine.has_symbol("custom", "custom_op")
    sym = engine.get_symbol("custom", "custom.custom_op")
    assert sym is not None
    assert sym.name == "custom_op"
    # Test short name lookup via get_symbol
    sym_short = engine.get_symbol("custom", "custom_op")
    assert sym_short is not None

    # Test non-existent symbol via get_symbol
    assert engine.get_symbol("custom", "non_existent_sym") is None

    # Verify mixed target with mnemonic
    assert engine.has_symbol("mixed", "INST_M")
    inst_sym = engine.get_symbol("mixed", "INST_M")
    assert inst_sym is not None

    # Verify gz_op
    assert engine.has_symbol("gz", "gz.gz_op")
    assert engine.has_symbol("gz", "gz_op")

    # Verify item_op
    assert engine.has_symbol("items", "items.item_op")

    # Verify single_op
    assert engine.has_symbol("single", "single.single_op")

    # Verify loaded targets
    targets = engine.get_loaded_targets()
    assert "custom" in targets
    assert "gz" in targets
    assert "items" in targets
    assert "single" in targets


def test_grounding_engine_suggestions_and_fuzzy(tmp_path: Path) -> None:
    """Test suggestion and fuzzy searching in GroundingEngine.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    custom_dir = str(tmp_path / "fuzzy_dir")
    os.makedirs(custom_dir, exist_ok=True)

    data = [
        {
            "name": "dot_general",
            "api_path": "stablehlo.dot_general",
            "kind": "function",
            "params": [],
        },
        {
            "name": "add",
            "api_path": "stablehlo.add",
            "kind": "function",
            "params": [],
        },
        {
            "name": "multiply",
            "api_path": "stablehlo.multiply",
            "kind": "function",
            "params": [],
        },
    ]
    with open(os.path.join(custom_dir, "testtarget_exhaustive.json"), "w") as f:
        json.dump(data, f)

    engine = GroundingEngine(base_dirs=[custom_dir])

    # Test suggest_closest_symbol
    suggested = engine.suggest_closest_symbol("testtarget", "stablehlo.dot_genera")
    assert suggested == "stablehlo.dot_general"

    # Distance too large returns None
    assert engine.suggest_closest_symbol("testtarget", "completely_unrelated_xyz", max_distance=2) is None

    # Unknown target returns None
    assert engine.suggest_closest_symbol("unknown_target", "abc") is None

    # Test fuzzy search ranking
    results = engine.fuzzy_search_symbol("testtarget", "add")
    assert "stablehlo.add" in results or "add" in results

    # Exact, prefix, substring, and distance matches
    results_dot = engine.fuzzy_search_symbol("testtarget", "dot")
    assert any("dot_general" in r for r in results_dot)

    # Empty target fuzzy search
    assert engine.fuzzy_search_symbol("nonexistent", "abc") == []


def test_validate_sass_instruction() -> None:
    """Test validate_sass_instruction with valid and invalid combinations."""
    # Test valid SASS instruction
    report_valid = validate_sass_instruction(
        mnemonic="FFMA",
        architecture="sm_80",
        operands=["R1", "R2", "R3", "R4"],
        modifiers=[".F32"],
    )
    assert report_valid.is_grounded
    assert report_valid.symbol == "FFMA"

    # Test unrecognized mnemonic with typo suggestion
    report_invalid_mnem = validate_sass_instruction(
        mnemonic="FFM",
        architecture="sm_80",
        operands=["R1", "R2"],
    )
    assert not report_invalid_mnem.is_grounded
    assert any(_diag_field(d) == "mnemonic" and d.suggested_fix == "FFMA" for d in report_invalid_mnem.diagnostics)

    # Test invalid architecture
    report_bad_arch = validate_sass_instruction(
        mnemonic="WGMMA",
        architecture="sm_70",
        operands=["R1", "R2", "R3", "R4"],
    )
    assert not report_bad_arch.is_grounded
    assert any(_diag_field(d) == "architecture" for d in report_bad_arch.diagnostics)

    # Test unrecognized modifier
    report_bad_mod = validate_sass_instruction(
        mnemonic="FFMA",
        architecture="sm_80",
        operands=["R1", "R2", "R3", "R4"],
        modifiers=[".NONEXISTENT_MODIFIER"],
    )
    assert any(_diag_field(d) == "modifiers" for d in report_bad_mod.diagnostics)

    # Test operand count mismatch
    report_bad_ops = validate_sass_instruction(
        mnemonic="FFMA",
        architecture="sm_80",
        operands=["R1"],
    )
    assert not report_bad_ops.is_grounded
    assert any(_diag_field(d) == "operands" for d in report_bad_ops.diagnostics)


def test_validate_rdna_instruction() -> None:
    """Test validate_rdna_instruction including register alignment checking."""
    # Test valid RDNA instruction with even-aligned register pair
    report_valid = validate_rdna_instruction(
        mnemonic="V_ADD_F32",
        gfx_arch="GFX11",
        operands=["v[0:1]", "v[2:3]"],
    )
    assert report_valid.is_grounded

    # Test unrecognized mnemonic with suggestion
    report_typo = validate_rdna_instruction(
        mnemonic="V_ADD_F3",
        gfx_arch="GFX11",
        operands=["v0", "v1"],
    )
    assert not report_typo.is_grounded
    assert any(_diag_field(d) == "mnemonic" for d in report_typo.diagnostics)

    # Test odd-aligned 64-bit register pair violation
    report_odd_reg = validate_rdna_instruction(
        mnemonic="V_ADD_F32",
        gfx_arch="GFX11",
        operands=["v[1:2]", "v[2:3]"],
    )
    assert not report_odd_reg.is_grounded
    assert any("operands[0]" in _diag_field(d) and "even-aligned" in d.message for d in report_odd_reg.diagnostics)


def test_validate_stablehlo_op() -> None:
    """Test validate_stablehlo_op verifying attribute schemas and operand counts."""
    # Test valid dot_general with dot_dimension_numbers
    report_dot = validate_stablehlo_op(
        op_name="dot_general",
        operand_types=["tensor<128x64xf32>", "tensor<64x256xf32>"],
        attributes={"dot_dimension_numbers": {}},
    )
    assert report_dot.is_grounded

    # Test dot_general missing dot_dimension_numbers
    report_dot_missing = validate_stablehlo_op(
        op_name="stablehlo.dot_general",
        operand_types=["tensor<128x64xf32>", "tensor<64x256xf32>"],
        attributes={},
    )
    assert not report_dot_missing.is_grounded
    assert any(_diag_field(d) == "attributes.dot_dimension_numbers" for d in report_dot_missing.diagnostics)

    # Test convolution missing dimension_numbers
    report_conv_missing = validate_stablehlo_op(
        op_name="stablehlo.convolution",
        operand_types=["tensor<1x28x28x1xf32>", "tensor<3x3x1x32xf32>"],
        attributes={},
    )
    assert not report_conv_missing.is_grounded
    assert any(_diag_field(d) == "attributes.dimension_numbers" for d in report_conv_missing.diagnostics)

    # Test unrecognized op with suggestion
    report_typo = validate_stablehlo_op(
        op_name="stablehlo.matmul",
        operand_types=["t1", "t2"],
        attributes={},
    )
    assert not report_typo.is_grounded
    assert any(_diag_field(d) == "operation" for d in report_typo.diagnostics)

    # Test operand count mismatch
    report_op_count = validate_stablehlo_op(
        op_name="stablehlo.add",
        operand_types=["t1"],  # add requires 2 operands
        attributes={},
    )
    assert not report_op_count.is_grounded
    assert any(_diag_field(d) == "operands" for d in report_op_count.diagnostics)


def test_validate_mlir_op() -> None:
    """Test validate_mlir_op checking dialect operations and operand count."""
    # Test valid arith.addf
    report_valid = validate_mlir_op(
        dialect="arith",
        op_name="addf",
        operand_count=2,
        attributes={},
    )
    assert report_valid.is_grounded

    # Test unrecognized op
    report_unknown = validate_mlir_op(
        dialect="arith",
        op_name="unknown_arith_op_xyz",
        operand_count=2,
        attributes={},
    )
    assert not report_unknown.is_grounded

    # Test operand count mismatch
    report_count = validate_mlir_op(
        dialect="arith",
        op_name="arith.addf",
        operand_count=1,
        attributes={},
    )
    assert not report_count.is_grounded
    assert any(_diag_field(d) == "operand_count" for d in report_count.diagnostics)


def test_validate_python_call(tmp_path: Path) -> None:
    """Test validate_python_call with mock framework snapshot.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    custom_dir = str(tmp_path / "py_dir")
    os.makedirs(custom_dir, exist_ok=True)

    torch_data = {
        "categories": {
            "math": [
                {
                    "name": "sum",
                    "api_path": "torch.sum",
                    "kind": "function",
                    "params": [
                        {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
                        {"name": "dim", "kind": "KEYWORD_ONLY"},
                        {"name": "keepdim", "kind": "KEYWORD_ONLY"},
                    ],
                }
            ]
        }
    }
    with open(os.path.join(custom_dir, "torch_v2.0.0.json"), "w") as f:
        json.dump(torch_data, f)

    numpy_data = {
        "categories": {
            "math": [
                {
                    "name": "sum",
                    "api_path": "numpy.sum",
                    "kind": "function",
                    "params": [
                        {"name": "a", "kind": "POSITIONAL_OR_KEYWORD"},
                        {"name": "axis", "kind": "KEYWORD_ONLY"},
                    ],
                }
            ]
        }
    }
    with open(os.path.join(custom_dir, "numpy_v1.0.0.json"), "w") as f:
        json.dump(numpy_data, f)

    engine = GroundingEngine(base_dirs=[custom_dir])

    # Valid call
    res_valid = validate_python_call(
        framework="torch",
        api_path="torch.sum",
        args=[[1, 2, 3]],
        kwargs={"dim": 0, "keepdim": True},
        engine=engine,
    )
    assert res_valid.is_grounded

    # Unrecognized API path
    res_unrec = validate_python_call(
        framework="torch",
        api_path="torch.summ",
        args=[],
        kwargs={},
        engine=engine,
    )
    assert not res_unrec.is_grounded
    assert any(_diag_field(d) == "api_path" and d.suggested_fix == "torch.sum" for d in res_unrec.diagnostics)

    # Hallucinated axis kwarg on torch -> suggests dim
    res_halluc_axis = validate_python_call(
        framework="torch",
        api_path="torch.sum",
        args=[[1, 2, 3]],
        kwargs={"axis": 0},
        engine=engine,
    )
    assert not res_halluc_axis.is_grounded
    assert any(_diag_field(d) == "kwargs.axis" and d.suggested_fix == "dim" for d in res_halluc_axis.diagnostics)

    # Hallucinated dim kwarg on numpy -> suggests axis
    res_halluc_dim = validate_python_call(
        framework="numpy",
        api_path="numpy.sum",
        args=[[1, 2, 3]],
        kwargs={"dim": 0},
        engine=engine,
    )
    assert not res_halluc_dim.is_grounded
    assert any(_diag_field(d) == "kwargs.dim" and d.suggested_fix == "axis" for d in res_halluc_dim.diagnostics)

    # Hallucinated arbitrary kwarg
    res_rand_kw = validate_python_call(
        framework="torch",
        api_path="torch.sum",
        args=[[1, 2, 3]],
        kwargs={"nonexistent_param": 123},
        engine=engine,
    )
    assert not res_rand_kw.is_grounded

    # Too many positional arguments
    res_too_many_args = validate_python_call(
        framework="torch",
        api_path="torch.sum",
        args=[[1, 2, 3], 4, 5],
        kwargs={},
        engine=engine,
    )
    assert not res_too_many_args.is_grounded
    assert any(_diag_field(d) == "args" for d in res_too_many_args.diagnostics)

    # Overloads (dict & object) and accepted_kwargs coverage
    from ml_switcheroo_ir.schema.ghost import ParameterKind

    from ml_ecosystem_snapshots.models import ExtendedGhostRef, GhostParam

    mock_ref = ExtendedGhostRef.model_validate(
        {
            "name": "fn",
            "api_path": "mock.fn",
            "kind": "function",
            "params": [GhostParam(name="a", kind=ParameterKind.POSITIONAL_OR_KEYWORD)],
            "overloads": [
                {"params": [{"name": "b"}, {"name": ""}]},
                GhostRef(
                    name="fn",
                    api_path="mock.fn",
                    kind="function",
                    params=[GhostParam(name="c", kind=ParameterKind.KEYWORD_ONLY)],
                ),
            ],
            "accepted_kwargs": ["extra_kw"],
        }
    )

    engine._target_cache["mock"] = {"mock.fn": mock_ref}
    res_ov = validate_python_call(
        framework="mock",
        api_path="mock.fn",
        args=[],
        kwargs={"a": 1, "b": 2, "c": 3, "extra_kw": 4},
        engine=engine,
    )
    assert res_ov.is_grounded


def test_grounding_engine_edge_branches(tmp_path: Path) -> None:
    """Test edge branches in compiler, hardware, and engine suggestion tie-breaking.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    # 1. Compiler: convolution with dimension_numbers present and zero expected operands
    conv_report = validate_stablehlo_op(
        op_name="stablehlo.convolution",
        operand_types=["tensor<1x28x28x1xf32>", "tensor<3x3x1x32xf32>"],
        attributes={"dimension_numbers": {}},
    )
    assert conv_report.is_grounded

    # Mock op with no operands field and empty params
    mock_dir = str(tmp_path / "branch_cov")
    os.makedirs(mock_dir, exist_ok=True)
    with open(os.path.join(mock_dir, "stablehlo_exhaustive.json"), "w") as f:
        json.dump(
            [
                {
                    "name": "noop",
                    "api_path": "stablehlo.noop",
                    "kind": "function",
                    "params": [],
                }
            ],
            f,
        )

    eng = GroundingEngine(base_dirs=[mock_dir])
    noop_report = validate_stablehlo_op("stablehlo.noop", [], {}, engine=eng)
    assert noop_report.is_grounded

    # 2. Hardware: SASS with valid modifiers and RDNA operand combinations
    report_sass_mod = validate_sass_instruction(
        mnemonic="FADD",
        architecture="sm_80",
        operands=["R1", "R2", "R3"],
        modifiers=[".FTZ"],
    )
    assert report_sass_mod.is_grounded

    # Mock SASS instruction with no operand_signatures
    no_sig_dir = str(tmp_path / "no_sig_dir")
    os.makedirs(no_sig_dir, exist_ok=True)
    with open(os.path.join(no_sig_dir, "nvidia_sass_exhaustive.json"), "w") as f:
        json.dump(
            [
                {
                    "mnemonic": "NOSIG",
                    "name": "NOSIG",
                    "api_path": "isa.inst.NOSIG",
                    "kind": "function",
                    "domain_metadata": {"modifiers": [".M1"], "operand_signatures": []},
                    "params": [],
                }
            ],
            f,
        )
    no_sig_eng = GroundingEngine(base_dirs=[no_sig_dir])
    report_no_sig = validate_sass_instruction(
        mnemonic="NOSIG",
        architecture="sm_80",
        operands=["R1", "R2"],
        engine=no_sig_eng,
    )
    assert report_no_sig.is_grounded

    # RDNA: test multiple operand forms: scalar v0, 4-reg range v[0:3], even-aligned v[0:1]
    report_rdna_range = validate_rdna_instruction(
        mnemonic="V_ADD_F32",
        gfx_arch="GFX11",
        operands=["v0", "v[0:3]", "v[0:1]"],
    )
    assert report_rdna_range.is_grounded

    # 3. Engine: tie-breaking in suggest_closest_symbol, category edge cases, and empty symbol fields
    tie_dir = str(tmp_path / "tie_dir")
    os.makedirs(tie_dir, exist_ok=True)
    tie_data = {
        "categories": {
            "valid": [
                {"name": "ab", "api_path": "tie.ab", "kind": "function", "params": []},
                {"name": "abc", "api_path": "tie.abc", "kind": "function", "params": []},
                {"name": "only_name", "api_path": "", "kind": "function", "params": []},
                {"name": "", "api_path": "tie.only_api", "kind": "function", "params": []},
                {"name": "aaaaa", "api_path": "tie.aaaaa", "kind": "function", "params": []},
                {"name": "aaa", "api_path": "tie.aaa", "kind": "function", "params": []},
            ],
            "not_a_list": "invalid_value",
        }
    }
    with open(os.path.join(tie_dir, "tie_exhaustive.json"), "w") as f:
        json.dump(tie_data, f)

    tie_eng = GroundingEngine(base_dirs=[tie_dir])
    assert tie_eng.suggest_closest_symbol("tie", "a", max_distance=3) == "ab"
    assert tie_eng.suggest_closest_symbol("tie", "aaaa", max_distance=3) == "aaa"
    assert tie_eng.has_symbol("tie", "AAA") is True
    assert tie_eng.get_symbol("tie", "AAA") is not None
    assert no_sig_eng.suggest_closest_symbol("nvidia_sass", "NOSI", max_distance=2) == "NOSIG"


def test_grounding_engine_tie_break_and_hydrate_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test GroundingEngine candidate tie-breaking, scalar JSON files, and hydrate errors.

    Args:
        tmp_path: Temporary directory path fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    # Test candidate tie-breaking where first candidate is longer and second is shorter
    tie_break_dir = str(tmp_path / "tie_break_dir")
    os.makedirs(tie_break_dir, exist_ok=True)
    with open(os.path.join(tie_break_dir, "tb_exhaustive.json"), "w") as f:
        json.dump(
            [
                {
                    "name": "abcd",
                    "api_path": "tb.abcd",
                    "kind": "function",
                    "params": [],
                },
                {"name": "xy", "api_path": "tb.xy", "kind": "function", "params": []},
            ],
            f,
        )
    tb_eng = GroundingEngine(base_dirs=[tie_break_dir])
    sugg_tb = tb_eng.suggest_closest_symbol("tb", "ab", max_distance=3)
    assert sugg_tb == "xy"

    # Test raw_data not list/dict in file
    primitive_dir = str(tmp_path / "prim_dir")
    os.makedirs(primitive_dir, exist_ok=True)
    with open(os.path.join(primitive_dir, "prim_v1.0.0.json"), "w") as f:
        f.write('"just_a_string"')
    prim_eng = GroundingEngine(base_dirs=[primitive_dir])
    assert prim_eng.load_target("prim") == {}

    # Test hydrate exception handling
    from ml_switcheroo_ir.typeclasses.outcome import Err, Result

    from ml_ecosystem_snapshots.errors import SnapshotEnvelopeFault
    from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
    from ml_ecosystem_snapshots.models import ExtendedGhostRef

    orig_hydrate = GhostInspector.hydrate

    def bad_hydrate(d: dict[str, JsonValue]) -> Result[SnapshotEnvelopeFault, ExtendedGhostRef]:
        """Simulate failure during GhostInspector.hydrate.

        Args:
            d: Dictionary to hydrate.

        Returns:
            `Err(SnapshotEnvelopeFault)` when `name == 'fail'`, else `orig_hydrate(d)`.
        """
        if d.get("name") == "fail":
            return Err(SnapshotEnvelopeFault(reason="simulated hydrate failure"))
        return orig_hydrate(d)

    monkeypatch.setattr(GhostInspector, "hydrate", bad_hydrate)
    fail_dir = str(tmp_path / "fail_dir")
    os.makedirs(fail_dir, exist_ok=True)
    with open(os.path.join(fail_dir, "fail_v1.0.0.json"), "w") as f:
        json.dump([{"name": "fail", "kind": "function"}], f)
    fail_eng = GroundingEngine(base_dirs=[fail_dir])
    assert fail_eng.load_target("fail") == {}


def test_grounding_engine_collector_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test GroundingEngine collector fallback when files are absent on disk.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    empty_dir = str(tmp_path / "empty_dir")
    os.makedirs(empty_dir, exist_ok=True)
    eng = GroundingEngine(base_dirs=[empty_dir])

    # 1. Fallback succeeds for known collector without files on disk
    html_target = eng.load_target("html_dsl")
    assert len(html_target) > 0
    assert "div" in html_target

    # 2. Unknown target returns empty dict
    assert eng.load_target("nonexistent_framework") == {}

    # 3. Exception during extract_snapshot returns empty dict
    import ml_ecosystem_snapshots.api as api_mod

    def mock_extract_fail(target: str) -> dict[str, JsonValue]:
        """Simulate extraction failure.

        Args:
            target: Target framework name.

        Raises:
            RuntimeError: Simulated failure.

        Returns:
            Never returns.
        """
        raise RuntimeError("extraction error")

    monkeypatch.setattr(api_mod, "extract_snapshot", mock_extract_fail)
    eng_fail = GroundingEngine(base_dirs=[empty_dir])
    assert eng_fail.load_target("tikz") == {}

    # 4. Fallback with malformed items and hydrate exception
    def mock_extract_malformed(target: str) -> dict[str, JsonValue]:
        """Return snapshot with malformed items for coverage.

        Args:
            target: Target framework name.

        Returns:
            Snapshot dict with non-dict and faulty items.
        """
        return {
            "categories": {
                "scalar_cat": "not_a_list",
                "tags": [
                    "not_a_dict",
                    {"name": "fail", "kind": "function"},
                    {
                        "name": "valid_tag",
                        "kind": "function",
                        "api_path": "valid_tag",
                        "mnemonic": "VT",
                    },
                    {
                        "name": "",
                        "kind": "function",
                        "api_path": "path_only",
                    },
                    {
                        "name": "name_only",
                        "kind": "function",
                        "api_path": "",
                    },
                ],
            }
        }

    from ml_switcheroo_ir.typeclasses.outcome import Err, Result

    from ml_ecosystem_snapshots.errors import SnapshotEnvelopeFault
    from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
    from ml_ecosystem_snapshots.models import ExtendedGhostRef

    orig_hydrate = GhostInspector.hydrate

    def bad_hydrate(d: dict[str, JsonValue]) -> Result[SnapshotEnvelopeFault, ExtendedGhostRef]:
        """Simulate total hydrate Err for item named fail.

        Args:
            d: Dictionary to hydrate.

        Returns:
            `Err(SnapshotEnvelopeFault)` when `name == 'fail'`, else `orig_hydrate(d)`.
        """
        if d.get("name") == "fail":
            return Err(SnapshotEnvelopeFault(reason="simulated error"))
        return orig_hydrate(d)

    monkeypatch.setattr(GhostInspector, "hydrate", bad_hydrate)
    monkeypatch.setattr(api_mod, "extract_snapshot", mock_extract_malformed)
    eng_malformed = GroundingEngine(base_dirs=[empty_dir])
    res = eng_malformed.load_target("tikz")
    assert "valid_tag" in res
    assert "VT" in res
    assert "path_only" in res
    assert "name_only" in res

    # 5. Fallback when extract_snapshot returns non-dict or missing categories
    monkeypatch.setattr(api_mod, "extract_snapshot", lambda t: {"no_categories": 123})
    eng_non_dict = GroundingEngine(base_dirs=[empty_dir])
    assert eng_non_dict.load_target("latex_dsl") == {}


def test_grounding_new_verifiers() -> None:
    """Test validate_ptx_instruction and edge compiler verifiers in Grounding SDK."""
    from ml_ecosystem_snapshots.grounding import (
        validate_metal_op,
        validate_onnx_op,
        validate_ptx_instruction,
        validate_wasm_op,
        validate_webgl_op,
        validate_wgsl_op,
    )

    # 1. PTX Instruction
    ptx_valid = validate_ptx_instruction("add", sm_arch="sm_80", types=[".s32"])
    assert ptx_valid.is_grounded
    assert not ptx_valid.has_errors

    from unittest.mock import MagicMock

    mock_eng = MagicMock()
    mock_eng.get_symbol.return_value = None
    ptx_no_ref = validate_ptx_instruction("add", engine=mock_eng)
    assert ptx_no_ref.matched_ref is None

    ptx_invalid_type = validate_ptx_instruction("add", types=[".invalid_type"])
    assert ptx_invalid_type.has_errors

    ptx_unknown = validate_ptx_instruction("nonexistent_ptx_op")
    assert ptx_unknown.has_errors
    assert any("Unrecognized NVIDIA PTX" in d.message for d in ptx_unknown.diagnostics)

    # 2. WGSL Builtin
    wgsl_valid = validate_wgsl_op("workgroupBarrier")
    assert wgsl_valid.is_grounded
    assert not wgsl_valid.has_errors

    wgsl_err = validate_wgsl_op("workgroupBarrier", operands_count=99)
    assert wgsl_err.has_errors

    wgsl_unknown = validate_wgsl_op("nonexistent_wgsl_op")
    assert wgsl_unknown.has_errors

    # 3. ONNX Operator
    onnx_valid = validate_onnx_op("MatMul")
    assert onnx_valid.is_grounded
    assert not onnx_valid.has_errors

    onnx_err = validate_onnx_op("MatMul", inputs_count=99)
    assert onnx_err.has_errors

    onnx_unknown = validate_onnx_op("NonexistentOnnxOp")
    assert onnx_unknown.has_errors

    # 4. Metal MSL
    metal_valid = validate_metal_op("threadgroup_barrier")
    assert metal_valid.is_grounded
    assert not metal_valid.has_errors

    metal_err = validate_metal_op("threadgroup_barrier", inputs_count=99)
    assert metal_err.has_errors

    metal_unknown = validate_metal_op("nonexistent_metal_op")
    assert metal_unknown.has_errors

    # 5. WASM SIMD
    wasm_valid = validate_wasm_op("f32x4.add")
    assert wasm_valid.is_grounded
    assert not wasm_valid.has_errors

    wasm_err = validate_wasm_op("f32x4.add", operands_count=99)
    assert wasm_err.has_errors

    wasm_unknown = validate_wasm_op("nonexistent_wasm_op")
    assert wasm_unknown.has_errors

    # 6. WebGL GLSL
    webgl_valid = validate_webgl_op("texture")
    assert webgl_valid.is_grounded
    assert not webgl_valid.has_errors

    webgl_err = validate_webgl_op("texture", inputs_count=99)
    assert webgl_err.has_errors

    webgl_unknown = validate_webgl_op("nonexistent_webgl_op")
    assert webgl_unknown.has_errors


def test_unified_grounding_and_mcp_parity_branches(tmp_path: Path) -> None:
    """Verify unified Grounding SDK and MCP validator branches across Python and ISA.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    from ml_ecosystem_snapshots.grounding.python_fw import verify_python_call
    from ml_ecosystem_snapshots.mcp.cache import SnapshotRepository

    res_cand = verify_python_call(
        framework="torch",
        api_path="torch.add",
        sig={
            "name": "add",
            "params": [{"name": "x", "kind": "POSITIONAL_OR_KEYWORD"}],
            "overloads": [{"name": "add", "params": [{"name": "x", "kind": "POSITIONAL_OR_KEYWORD"}]}],
        },
        snapshot_source="bundled_package",
        args_count=1,
    )
    assert res_cand.is_hallucinated is False

    # 1. GroundingEngine with SnapshotRepository fallback
    repo = SnapshotRepository()
    repo.put(
        "custom_repo_fw",
        {
            "categories": {
                "math": [
                    {
                        "name": "op",
                        "api_path": "custom_repo_fw.op",
                        "kind": "function",
                        "params": [
                            {
                                "name": "x",
                                "kind": "POSITIONAL_OR_KEYWORD",
                                "allowed_dtypes": ["float32"],
                            }
                        ],
                    }
                ]
            }
        },
    )
    empty_dir = str(tmp_path / "empty_repo_dir")
    os.makedirs(empty_dir, exist_ok=True)
    eng_repo = GroundingEngine(base_dirs=[empty_dir], repository=repo)
    assert eng_repo.has_symbol("custom_repo_fw", "custom_repo_fw.op")

    # Python dtype error branch in validate_python_call
    py_dt_err = validate_python_call(
        framework="custom_repo_fw",
        api_path="custom_repo_fw.op",
        args=["x"],
        kwargs={},
        engine=eng_repo,
        arg_dtypes=["int32"],
    )
    assert not py_dt_err.is_grounded
    assert any(_diag_field(d) == "arguments" for d in py_dt_err.diagnostics)

    # Python secondary overload positional arity acceptance
    ov_dir = str(tmp_path / "ov_dir")
    os.makedirs(ov_dir, exist_ok=True)
    with open(os.path.join(ov_dir, "ovfw_v1.0.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "categories": {
                    "math": [
                        {
                            "name": "func",
                            "api_path": "ovfw.func",
                            "kind": "function",
                            "params": [{"name": "a", "kind": "POSITIONAL_OR_KEYWORD"}],
                            "overloads": [
                                {
                                    "name": "func",
                                    "api_path": "ovfw.func",
                                    "kind": "function",
                                    "params": [
                                        {"name": "a", "kind": "POSITIONAL_OR_KEYWORD"},
                                        {"name": "b", "kind": "POSITIONAL_OR_KEYWORD"},
                                    ],
                                },
                                {
                                    "params": [
                                        {"name": "a", "kind": "POSITIONAL_OR_KEYWORD"},
                                        {"name": "alt_kw", "kind": "KEYWORD_ONLY"},
                                    ],
                                    "returns_type": "Tensor",
                                },
                            ],
                        }
                    ]
                }
            },
            f,
        )
    eng_ov = GroundingEngine(base_dirs=[ov_dir])
    res_ov_pos = validate_python_call("ovfw", "ovfw.func", args=[1, 2], kwargs={}, engine=eng_ov)
    assert res_ov_pos.is_grounded

    res_ov_ghost_overload = validate_python_call(
        "ovfw",
        "ovfw.func",
        args=[1],
        kwargs={"alt_kw": 2},
        engine=eng_ov,
    )
    assert res_ov_ghost_overload.is_grounded

    res_ov_pos_bad_kw = validate_python_call(
        "ovfw",
        "ovfw.func",
        args=[1, 2],
        kwargs={"bad_kw": 1},
        engine=eng_ov,
    )
    assert not res_ov_pos_bad_kw.is_grounded

    # 2. SASS: modifier conflict, register error, directionality error, control_codes error
    sass_err_rep = validate_sass_instruction(
        mnemonic="FFMA",
        architecture="sm_80",
        operands=["0x10", "R999", "R2", "R3"],
        modifiers=[".RN", ".RZ"],
        control_codes={"stall_count": 99},
    )
    assert not sass_err_rep.is_grounded
    fields = {_diag_field(d) for d in sass_err_rep.diagnostics}
    assert "modifiers" in fields
    assert "operands[1]" in fields
    assert "operands" in fields
    assert "control_codes" in fields

    # 3. RDNA: unsupported gfx_arch, microarch wave_size error, encoding mismatch, modifier error, operand count mismatch, register error, constant bus error
    rdna_err_rep = validate_rdna_instruction(
        mnemonic="V_DUAL_FMAC_F32",
        gfx_arch="GFX9",
        operands=["v999", "s0", "s1", "s2", "v1"],
        modifiers=["omod:2"],
        encoding="SOP1",
        wave_size=64,
    )
    assert not rdna_err_rep.is_grounded
    rdna_fields = {_diag_field(d) for d in rdna_err_rep.diagnostics}
    assert "gfx_arch" in rdna_fields
    assert "encoding" in rdna_fields
    assert "modifiers" in rdna_fields
    assert "operands" in rdna_fields
    assert "operands[0]" in rdna_fields

    # RDNA constant bus violation on VOP2 instruction in GFX9
    rdna_cbus = validate_rdna_instruction(
        mnemonic="V_ADD_F32",
        gfx_arch="GFX9",
        operands=["v0", "s0", "s1"],
    )
    assert not rdna_cbus.is_grounded
    assert any("Constant bus" in d.message or "constant bus" in d.message.lower() for d in rdna_cbus.diagnostics)


def test_unified_grounding_ir_and_keepdim_branches(tmp_path: Path) -> None:
    """Verify unified Grounding SDK branches for MLIR/StableHLO ops and keepdim suggestions.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    # MLIR variadic op with operand_count != len(expected_operands)
    mlir_var_dir = str(tmp_path / "mlir_var_dir")
    os.makedirs(mlir_var_dir, exist_ok=True)
    with open(os.path.join(mlir_var_dir, "mlir_v1.0.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "categories": {
                    "ops": [
                        {
                            "name": "variadic_op",
                            "api_path": "func.variadic_op",
                            "kind": "operation",
                            "params": [
                                {
                                    "name": "inputs",
                                    "kind": "POSITIONAL_OR_KEYWORD",
                                    "annotation": "Variadic<AnyType>",
                                }
                            ],
                        }
                    ]
                }
            },
            f,
        )
    eng_mlir_var = GroundingEngine(base_dirs=[mlir_var_dir])
    var_rep = validate_mlir_op("func", "variadic_op", operand_count=3, attributes={}, engine=eng_mlir_var)
    assert var_rep.is_grounded

    # Python keepdim / keepdims suggestion branches
    keep_dir = str(tmp_path / "keep_dir")
    os.makedirs(keep_dir, exist_ok=True)
    with open(os.path.join(keep_dir, "torch_v1.0.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "categories": {
                    "math": [
                        {
                            "name": "sum",
                            "api_path": "torch.sum",
                            "kind": "function",
                            "params": [
                                {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
                                {"name": "keepdim", "kind": "KEYWORD_ONLY"},
                            ],
                        }
                    ]
                }
            },
            f,
        )
    with open(os.path.join(keep_dir, "numpy_v1.0.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "categories": {
                    "math": [
                        {
                            "name": "sum",
                            "api_path": "numpy.sum",
                            "kind": "function",
                            "params": [
                                {"name": "a", "kind": "POSITIONAL_OR_KEYWORD"},
                                {"name": "keepdims", "kind": "KEYWORD_ONLY"},
                            ],
                        }
                    ]
                }
            },
            f,
        )
    eng_keep = GroundingEngine(base_dirs=[keep_dir])
    rep_t_kd = validate_python_call("torch", "torch.sum", args=[1], kwargs={"keepdims": True}, engine=eng_keep)
    assert any(d.suggested_fix == "keepdim" for d in rep_t_kd.diagnostics)
    rep_np_kd = validate_python_call("numpy", "numpy.sum", args=[1], kwargs={"keepdim": True}, engine=eng_keep)
    assert any(d.suggested_fix == "keepdims" for d in rep_np_kd.diagnostics)

    # 4. StableHLO & MLIR: type errors, trait errors, structured attribute errors, region errors
    shlo_err = validate_stablehlo_op(
        op_name="stablehlo.add",
        operand_types=["tensor<4xf32>", "tensor<8xi32>"],
        attributes={"comparison_direction": "INVALID_DIR"},
        result_types=["tensor<4xf32>"],
    )
    assert not shlo_err.is_grounded
    shlo_fields = {_diag_field(d) for d in shlo_err.diagnostics}
    assert "traits" in shlo_fields
    assert "attributes" in shlo_fields

    shlo_reg_err = validate_stablehlo_op(
        op_name="stablehlo.reduce",
        operand_types=["tensor<4xf32>", "tensor<f32>"],
        attributes={},
        regions={"body": {"block_arguments": ["tensor<f32>"], "yield_types": []}},
    )
    assert any(_diag_field(d) == "regions.body" for d in shlo_reg_err.diagnostics)

    # RDNA empty gfx_arch branch
    rdna_no_arch = validate_rdna_instruction(
        mnemonic="V_ADD_F32",
        gfx_arch="",
        operands=["v0", "v1", "v2"],
    )
    assert rdna_no_arch.is_grounded

    # Custom MLIR op with explicit type constraint to trigger operand type error
    mlir_custom_dir = str(tmp_path / "mlir_custom")
    os.makedirs(mlir_custom_dir, exist_ok=True)
    with open(os.path.join(mlir_custom_dir, "mlir_exhaustive.json"), "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "name": "arith.custom_op",
                    "api_path": "arith.custom_op",
                    "kind": "operation",
                    "operands": [{"name": "lhs", "type": "AnyFloat"}],
                    "params": [
                        {
                            "name": "lhs",
                            "kind": "POSITIONAL_ONLY",
                            "annotation": "AnyFloat",
                        }
                    ],
                },
                {
                    "name": "stablehlo.custom_op",
                    "api_path": "stablehlo.custom_op",
                    "kind": "operation",
                    "operands": [{"name": "lhs", "type": "AnyFloat"}],
                    "params": [
                        {
                            "name": "lhs",
                            "kind": "POSITIONAL_ONLY",
                            "annotation": "AnyFloat",
                        }
                    ],
                },
            ],
            f,
        )
    with open(
        os.path.join(mlir_custom_dir, "stablehlo_exhaustive.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            [
                {
                    "name": "stablehlo.custom_op",
                    "api_path": "stablehlo.custom_op",
                    "kind": "operation",
                    "operands": [{"name": "lhs", "type": "AnyFloat"}],
                    "params": [
                        {
                            "name": "lhs",
                            "kind": "POSITIONAL_ONLY",
                            "annotation": "AnyFloat",
                        }
                    ],
                }
            ],
            f,
        )
    eng_mlir = GroundingEngine(base_dirs=[mlir_custom_dir])
    shlo_type_err = validate_stablehlo_op(
        op_name="stablehlo.custom_op",
        operand_types=["i32"],
        attributes={},
        engine=eng_mlir,
    )
    assert any(_diag_field(d) == "operands[0]" for d in shlo_type_err.diagnostics)

    mlir_type_err = validate_mlir_op(
        dialect="arith",
        op_name="custom_op",
        operand_count=1,
        attributes={"comparison_direction": "BAD"},
        engine=eng_mlir,
        operand_types=["i32"],
        result_types=["f64"],
    )
    assert not mlir_type_err.is_grounded
    mlir_fields = {_diag_field(d) for d in mlir_type_err.diagnostics}
    assert "operands[0]" in mlir_fields
    assert "attributes" in mlir_fields

    mlir_trait_err = validate_mlir_op(
        dialect="arith",
        op_name="addf",
        operand_count=2,
        attributes={},
        operand_types=["tensor<4xf32>", "tensor<8xf32>"],
        result_types=["tensor<4xf32>"],
    )
    assert any(_diag_field(d) == "traits" for d in mlir_trait_err.diagnostics)


def test_grounding_models_cross_repo_identity() -> None:
    """Verify Grounding models and compute_levenshtein share identity with ml_switcheroo_ir."""
    from ml_switcheroo_ir.validation import core as ir_core
    from ml_switcheroo_ir.validation import grounding as ir_grounding

    from ml_ecosystem_snapshots.grounding import engine as snap_engine
    from ml_ecosystem_snapshots.grounding import models as snap_models

    assert snap_models.DiagnosticSeverity is ir_grounding.DiagnosticSeverity
    assert snap_models.GroundingDiagnostic is ir_grounding.GroundingDiagnostic
    assert snap_models.GroundingReport is ir_grounding.GroundingReport
    assert snap_engine.compute_levenshtein is ir_core.compute_levenshtein


def test_domain_validators_honor_engine_snapshots_dir_and_variadic_mlir(tmp_path: Path) -> None:
    """Verify all domain `validate_*_op` verifiers honor `GroundingEngine.snapshots_dir` and variadic MLIR ops.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    from ml_ecosystem_snapshots.grounding import (
        validate_array_api_op,
        validate_aten_op,
        validate_chex_op,
        validate_cpp_op,
        validate_domain_op,
        validate_flash_attn_op,
        validate_grain_op,
        validate_metal_op,
        validate_mlir_op,
        validate_nccl_op,
        validate_optax_op,
        validate_orbax_op,
        validate_safetensors_op,
        validate_scipy_op,
        validate_torchaudio_op,
        validate_torchvision_op,
    )

    scipy_snap = tmp_path / "scipy_v1.14.0.json"
    scipy_snap.write_text(
        json.dumps(
            {
                "target": "scipy",
                "version": "1.14.0",
                "categories": {
                    "special": [
                        {
                            "name": "custom_bessel",
                            "api_path": "scipy.special.custom_bessel",
                            "kind": "function",
                            "params": [
                                {"name": "v", "kind": "POSITIONAL_ONLY"},
                                {"name": "z", "kind": "POSITIONAL_OR_KEYWORD"},
                                {"name": "scale", "kind": "POSITIONAL_OR_KEYWORD", "default": "1.0"},
                                {"name": "out", "kind": "KEYWORD_ONLY", "default": "None"},
                                {"name": "extra", "kind": "VAR_POSITIONAL"},
                            ],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    metal_snap = tmp_path / "metal_v3.2.json"
    metal_snap.write_text(
        json.dumps(
            {
                "target": "metal",
                "version": "3.2",
                "categories": {
                    "ops": [
                        {
                            "name": "custom_simd_sum",
                            "api_path": "metal.custom_simd_sum",
                            "kind": "operation",
                            "domain_metadata": {"address_space": "threadgroup"},
                            "params": [
                                {"name": "val", "kind": "POSITIONAL_ONLY", "role": "OPERAND"},
                                {"name": "lane", "kind": "POSITIONAL_ONLY", "role": "ATTRIBUTE"},
                                {"name": "rest", "kind": "VAR_POSITIONAL"},
                            ],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    mlir_snap = tmp_path / "mlir_exhaustive.json"
    mlir_snap.write_text(
        json.dumps(
            {
                "target": "mlir",
                "version": "19.1.0",
                "operations": [
                    {
                        "name": "arith.variadic_sum",
                        "api_path": "arith.variadic_sum",
                        "kind": "function",
                        "has_varargs": True,
                        "params": [
                            {"name": "inputs", "kind": "POSITIONAL_ONLY", "annotation": "tensor<*xf32>"},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    eng = GroundingEngine(snapshots_dir=str(tmp_path))
    assert eng.snapshots_dir == str(tmp_path)

    # Custom scipy op resolved from `eng.snapshots_dir`
    ok_scipy = validate_scipy_op("custom_bessel", inputs_count=2, attributes=["scale", "out"], engine=eng)
    assert ok_scipy.is_grounded
    bad_scipy = validate_scipy_op("custom_bessel", inputs_count=1, attributes=["bad_kw"], engine=eng)
    assert bad_scipy.has_errors

    # Custom metal op resolved from `eng.snapshots_dir` with address_space metadata
    ok_metal = validate_metal_op(
        "custom_simd_sum",
        inputs_count=1,
        attributes=["lane"],
        address_space="threadgroup",
        engine=eng,
    )
    assert ok_metal.is_grounded
    bad_metal = validate_metal_op("custom_simd_sum", address_space="device", engine=eng)
    assert bad_metal.has_errors

    # Fallback to canonical catalog when symbol is not overridden in custom snapshots_dir
    fallback_optax = validate_optax_op("adam", engine=eng)
    assert fallback_optax.is_grounded
    missing_optax = validate_optax_op("nonexistent_custom_op", engine=eng)
    assert missing_optax.has_errors

    # Exercise all remaining domain validators
    assert validate_cpp_op("clamp").is_grounded
    assert validate_array_api_op("sum").is_grounded
    assert validate_torchvision_op("nms").is_grounded
    assert validate_torchaudio_op("Spectrogram").is_grounded
    assert validate_safetensors_op("save_file").is_grounded
    assert validate_aten_op("aten.add").is_grounded
    assert validate_nccl_op("all_reduce").is_grounded
    assert validate_flash_attn_op("flash_attn_func").is_grounded
    assert validate_chex_op("assert_shape").is_grounded
    assert validate_grain_op("DataLoader").is_grounded
    assert validate_orbax_op("PyTreeCheckpointer").is_grounded
    assert validate_domain_op("optax", "adamw").is_grounded

    # Variadic MLIR operation in GroundingEngine (`has_varargs=True`)
    var_ok = validate_mlir_op("arith", "variadic_sum", operand_count=3, attributes={}, engine=eng)
    assert var_ok.is_grounded


def test_ground_proven_graph_across_python_mlir_stablehlo_domain_and_subgraphs() -> None:
    """Verify `ground_proven_graph` across Python, StableHLO, MLIR, ONNX, subgraphs, and `NonEmpty` errors."""
    from ml_switcheroo_ir.graph import LogicalEdge, LogicalGraph, LogicalNode
    from ml_switcheroo_ir.proofs import prove_acyclic_sorted
    from ml_switcheroo_ir.schema.ghost import (
        ExtendedGhostParam,
        ExtendedGhostRef,
        ParameterKind,
        SnapshotEnvelope,
    )
    from ml_switcheroo_ir.shapes.ast import SymVar
    from ml_switcheroo_ir.typeclasses.outcome import Invalid, Ok, Valid
    from ml_switcheroo_ir.types import DType, TensorShape, TensorSpec

    from ml_ecosystem_snapshots.grounding import (
        GroundingEngine,
        coerce_ir_attribute_to_json,
        extract_tensor_spec_metadata,
        ground_proven_graph,
        verify_python_call,
    )
    from ml_ecosystem_snapshots.proofs import prove_envelope_valid, upgrade_to_hydrated

    # 1. TensorSpec & AttributeValue coercion branches
    sym_spec = TensorSpec(shape=(2, SymVar("N")), dtype=DType.float32)
    dt_str, rk_val, sh_tup = extract_tensor_spec_metadata(sym_spec)
    assert dt_str == "float32" and rk_val == 2 and sh_tup == (2, "N")
    assert coerce_ir_attribute_to_json(None) is None
    assert coerce_ir_attribute_to_json(b"hello") == "hello"
    assert coerce_ir_attribute_to_json(DType.int64) == "int64"
    assert coerce_ir_attribute_to_json(sym_spec) == {"dtype": "float32", "shape": [2, "N"]}
    assert coerce_ir_attribute_to_json({"a": [1, DType.float16]}) == {"a": [1, "float16"]}
    assert coerce_ir_attribute_to_json(SymVar("B")) is not None

    # 2. Python framework ProvenSnapshot + nested subgraph + multi-error accumulation
    torch_env = SnapshotEnvelope(
        target="torch",
        version="2.4.0",
        categories={
            "neural": [
                ExtendedGhostRef(
                    name="add",
                    api_path="torch.add",
                    kind="function",
                    aliases=["torch.tensor_add"],
                    params=[
                        ExtendedGhostParam(
                            name="input",
                            kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                            allowed_dtypes=["float32"],
                            rank_constraint="2",
                        ),
                        ExtendedGhostParam(
                            name="other",
                            kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                            allowed_dtypes=["float32"],
                            rank_constraint="2",
                        ),
                        ExtendedGhostParam(
                            name="alpha",
                            kind=ParameterKind.KEYWORD_ONLY,
                            default="1",
                        ),
                    ],
                ),
                ExtendedGhostRef(
                    name="",
                    api_path="torch.path_only_op",
                    kind="function",
                    params=[],
                ),
                ExtendedGhostRef.model_validate(
                    {"name": "mnem_only", "api_path": "", "kind": "function", "mnemonic": "MNEM_OP"}
                ),
            ]
        },
    )
    t_env_val = prove_envelope_valid(torch_env)
    assert isinstance(t_env_val, Valid)
    t_hyd_val = upgrade_to_hydrated(t_env_val.value)
    assert isinstance(t_hyd_val, Valid)
    torch_snap = t_hyd_val.value

    eng = GroundingEngine.from_proven_snapshot(torch_snap)
    assert eng.has_proven_target("torch")
    assert verify_python_call("torch", "torch.add", torch_snap, args_count=2).is_hallucinated is False
    assert verify_python_call("torch", "torch.missing_op", torch_snap).is_hallucinated is True

    sub_ok = LogicalGraph(
        name="body",
        nodes=[LogicalNode(id="s1", op_type="add", domain="torch", inputs=["x", "y"])],
        input_specs={
            "x": TensorSpec(shape=TensorShape([4, 4]), dtype=DType.float32),
            "y": TensorSpec(shape=TensorShape([4, 4]), dtype=DType.float32),
        },
    )
    sub_sorted = LogicalGraph(
        name="body_sorted",
        nodes=[LogicalNode(id="s2", op_type="torch.tensor_add", domain="torch", inputs=["x", "y"])],
        sorted_nodes=[LogicalNode(id="s2", op_type="torch.tensor_add", domain="torch", inputs=["x", "y"])],
        input_specs={
            "x": TensorSpec(shape=TensorShape([4, 4]), dtype=DType.float32),
            "y": TensorSpec(shape=TensorShape([4, 4]), dtype=DType.float32),
        },
    )
    valid_graph = LogicalGraph(
        name="valid_torch",
        nodes=[
            LogicalNode(
                id="in0",
                op_type="Input",
                outputs=["v0"],
                output_specs=[TensorSpec(shape=TensorShape([4, 4]), dtype=DType.float32)],
            ),
            LogicalNode(
                id="n_add",
                op_type="torch.add",
                domain="torch",
                inputs=["v0", "in0"],
                attributes={"alpha": 2},
                subgraphs={"a": sub_ok, "b": sub_sorted, "meta": 1},
            ),
        ],
        edges=[LogicalEdge(source="in0", target="n_add")],
    )
    pg_ok = prove_acyclic_sorted(valid_graph)
    assert isinstance(pg_ok, Ok)
    g_res = ground_proven_graph(pg_ok.value, torch_snap)
    assert isinstance(g_res, Valid)

    # Multi-error graph (two hallucinated ops accumulate into NonEmpty of length 2, plus unspec'd edge input)
    bad_graph = LogicalGraph(
        name="bad_torch",
        nodes=[
            LogicalNode(id="in0", op_type="Input"),
            LogicalNode(id="bad1", op_type="torch.fake_one", domain="torch"),
            LogicalNode(id="bad2", op_type="fake_two", domain="torch"),
        ],
        edges=[
            LogicalEdge(source="in0", target="bad1"),
            LogicalEdge(source="bad1", target="bad2"),
        ],
    )
    pg_bad = prove_acyclic_sorted(bad_graph)
    assert isinstance(pg_bad, Ok)
    g_bad = ground_proven_graph(pg_bad.value, torch_snap)
    assert isinstance(g_bad, Invalid)
    assert len(g_bad.errors) == 2
    assert {d.node_id for d in g_bad.errors.to_tuple()} == {"bad1", "bad2"}


def test_ground_proven_graph_across_stablehlo_mlir_and_onnx() -> None:
    """Verify `ground_proven_graph` across StableHLO, MLIR, and ONNX `ProvenSnapshot` targets."""
    from ml_switcheroo_ir.graph import LogicalEdge, LogicalGraph, LogicalNode
    from ml_switcheroo_ir.proofs import prove_acyclic_sorted
    from ml_switcheroo_ir.schema.ghost import (
        ExtendedGhostParam,
        ExtendedGhostRef,
        GhostOperationRef,
        ParameterKind,
        SnapshotEnvelope,
    )
    from ml_switcheroo_ir.shapes.ast import SymVar
    from ml_switcheroo_ir.typeclasses.outcome import Ok, Valid
    from ml_switcheroo_ir.types import DType, TensorSpec

    from ml_ecosystem_snapshots.grounding import (
        ground_proven_graph,
        validate_onnx_op,
        verify_mlir_op,
    )
    from ml_ecosystem_snapshots.proofs import prove_envelope_valid, upgrade_to_hydrated

    shlo_env = SnapshotEnvelope(
        target="stablehlo",
        version="1.9.0",
        categories={
            "operations": [
                GhostOperationRef(
                    name="add",
                    api_path="stablehlo.add",
                    kind="function",
                    params=[
                        ExtendedGhostParam(name="lhs", kind=ParameterKind.POSITIONAL_ONLY),
                        ExtendedGhostParam(name="rhs", kind=ParameterKind.POSITIONAL_ONLY),
                    ],
                )
            ]
        },
    )
    shlo_val = prove_envelope_valid(shlo_env)
    assert isinstance(shlo_val, Valid)
    shlo_hyd = upgrade_to_hydrated(shlo_val.value)
    assert isinstance(shlo_hyd, Valid)
    assert verify_mlir_op("stablehlo.add", shlo_hyd.value, operands_count=2).is_valid
    shlo_graph = LogicalGraph(
        name="shlo_g",
        nodes=[
            LogicalNode(id="a", op_type="Input"),
            LogicalNode(id="b", op_type="Input"),
            LogicalNode(id="op", op_type="stablehlo.add", domain="stablehlo"),
        ],
        edges=[LogicalEdge(source="a", target="op"), LogicalEdge(source="b", target="op")],
        input_specs={
            "a": TensorSpec(shape=(2, SymVar("M")), dtype=DType.float32),
            "b": TensorSpec(shape=(2, SymVar("M")), dtype=DType.float32),
        },
    )
    shlo_pg = prove_acyclic_sorted(shlo_graph)
    assert isinstance(shlo_pg, Ok)
    assert isinstance(ground_proven_graph(shlo_pg.value, shlo_hyd.value), Valid)

    mlir_env = SnapshotEnvelope(
        target="mlir",
        version="19.0",
        categories={
            "operations": [
                GhostOperationRef(
                    name="addf",
                    api_path="arith.addf",
                    kind="function",
                    params=[
                        ExtendedGhostParam(name="lhs", kind=ParameterKind.POSITIONAL_ONLY),
                        ExtendedGhostParam(name="rhs", kind=ParameterKind.POSITIONAL_ONLY),
                    ],
                )
            ]
        },
    )
    mlir_val = prove_envelope_valid(mlir_env)
    assert isinstance(mlir_val, Valid)
    mlir_hyd = upgrade_to_hydrated(mlir_val.value)
    assert isinstance(mlir_hyd, Valid)
    mlir_graph = LogicalGraph(
        name="mlir_g",
        nodes=[
            LogicalNode(id="a", op_type="Input"),
            LogicalNode(id="b", op_type="Input"),
            LogicalNode(id="op", op_type="arith.addf", domain="arith"),
        ],
        edges=[LogicalEdge(source="a", target="op"), LogicalEdge(source="b", target="op")],
    )
    mlir_pg = prove_acyclic_sorted(mlir_graph)
    assert isinstance(mlir_pg, Ok)
    assert isinstance(ground_proven_graph(mlir_pg.value, mlir_hyd.value), Valid)

    onnx_env = SnapshotEnvelope(
        target="onnx",
        version="1.16",
        categories={
            "operations": [
                ExtendedGhostRef(
                    name="Relu",
                    api_path="onnx.Relu",
                    kind="function",
                    params=[ExtendedGhostParam(name="X", kind=ParameterKind.POSITIONAL_ONLY)],
                )
            ]
        },
    )
    onnx_val = prove_envelope_valid(onnx_env)
    assert isinstance(onnx_val, Valid)
    onnx_hyd = upgrade_to_hydrated(onnx_val.value)
    assert isinstance(onnx_hyd, Valid)
    assert validate_onnx_op("Relu", inputs_count=1, engine=onnx_hyd.value).is_grounded
    assert validate_onnx_op("MissingOnnxOp", engine=onnx_hyd.value).has_errors
    onnx_graph = LogicalGraph(
        name="onnx_g",
        nodes=[
            LogicalNode(id="x", op_type="Input"),
            LogicalNode(id="r", op_type="Relu", domain="ai.onnx", inputs=["x"]),
        ],
    )
    onnx_pg = prove_acyclic_sorted(onnx_graph)
    assert isinstance(onnx_pg, Ok)
    assert isinstance(ground_proven_graph(onnx_pg.value, onnx_hyd.value), Valid)
