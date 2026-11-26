"""Tests for the compliance checking module."""

from __future__ import annotations

import os
from pathlib import Path

import griffe
from ml_switcheroo_ir.typeclasses.outcome import Err

from ml_ecosystem_snapshots.compliance import (
    align_namespace,
    extract_target_ast,
    extract_target_refs,
    get_module_info_from_path,
    score_compliance,
)
from ml_ecosystem_snapshots.errors import IntrospectionFault, SnapshotNotFoundFault
from ml_ecosystem_snapshots.exceptions import raise_snapshot_boundary as raise_on_err
from ml_ecosystem_snapshots.models import ExtendedGhostRef, JsonValue


def test_get_module_info_from_path_file(tmp_path: Path) -> None:
    """Test getting module info from a python file.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    pkg_dir = Path(os.path.join(tmp_path, "my_pkg"))
    pkg_dir.mkdir()
    Path(os.path.join(pkg_dir, "__init__.py")).touch()
    sub_mod = Path(os.path.join(pkg_dir, "sub_module.py"))
    sub_mod.write_text("def foo(): pass")

    search_path, mod_name = raise_on_err(get_module_info_from_path(str(sub_mod)))
    assert search_path == str(tmp_path)
    assert mod_name == "my_pkg.sub_module"


def test_get_module_info_from_path_dir(tmp_path: Path) -> None:
    """Test getting module info from a directory package.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    pkg_dir = Path(os.path.join(tmp_path, "my_pkg"))
    pkg_dir.mkdir()
    Path(os.path.join(pkg_dir, "__init__.py")).touch()

    search_path, mod_name = raise_on_err(get_module_info_from_path(str(pkg_dir)))
    assert search_path == str(tmp_path)
    assert mod_name == "my_pkg"


def test_get_module_info_from_path_init_file(tmp_path: Path) -> None:
    """Test getting module info from an __init__.py file.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    pkg_dir = Path(os.path.join(tmp_path, "my_pkg"))
    pkg_dir.mkdir()
    init_file = Path(os.path.join(pkg_dir, "__init__.py"))
    init_file.touch()

    search_path, mod_name = raise_on_err(get_module_info_from_path(str(init_file)))
    assert search_path == str(tmp_path)
    assert mod_name == "my_pkg"


def test_get_module_info_from_path_not_exist() -> None:
    """Test getting module info for non-existent path returns Err(SnapshotNotFoundFault)."""
    res = get_module_info_from_path("does_not_exist_xyz.py")
    assert isinstance(res, Err) and isinstance(res.error, SnapshotNotFoundFault)


def test_get_module_info_from_path_value_error(tmp_path: Path) -> None:
    """Test getting module info returns Err(IntrospectionFault) when module name is empty.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    init_at_root = tmp_path / "__init__.py"
    init_at_root.touch()
    res = get_module_info_from_path(str(init_at_root))
    if isinstance(res, Err):
        assert isinstance(res.error, IntrospectionFault)
    root_res = get_module_info_from_path("/")
    assert isinstance(root_res, Err) and isinstance(root_res.error, IntrospectionFault)


def test_extract_target_ast(tmp_path: Path) -> None:
    """Test extracting target AST via griffe.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    file_path = Path(os.path.join(tmp_path, "simple_mod.py"))
    file_path.write_text("def my_func(): pass")

    ast_node = raise_on_err(extract_target_ast(str(file_path)))
    assert isinstance(ast_node, griffe.Module)
    assert ast_node.name == "simple_mod"
    assert "my_func" in ast_node.members


def test_align_namespace() -> None:
    """Test namespace alignment."""
    assert align_namespace("ml_switcheroo.jax.nn", "ml_switcheroo.jax", "jax") == "jax.nn"
    assert align_namespace("ml_switcheroo.jax", "ml_switcheroo.jax", "jax") == "jax"
    assert align_namespace("other.module", "ml_switcheroo.jax", "jax") == "other.module"


def test_get_module_info_from_path_no_py_ext(tmp_path: Path) -> None:
    """Test getting module info from a file without .py extension.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    pkg_dir = Path(os.path.join(tmp_path, "my_pkg"))
    pkg_dir.mkdir()
    Path(os.path.join(pkg_dir, "__init__.py")).touch()
    sub_mod = Path(os.path.join(pkg_dir, "sub_module"))
    sub_mod.write_text("def foo(): pass")

    search_path, mod_name = raise_on_err(get_module_info_from_path(str(sub_mod)))
    assert search_path == str(tmp_path)
    assert mod_name == "my_pkg.sub_module"


def test_extract_target_refs(tmp_path: Path) -> None:
    """Test extracting target refs dynamically.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    pkg_dir = Path(os.path.join(tmp_path, "mock_pkg"))
    pkg_dir.mkdir()
    Path(os.path.join(pkg_dir, "__init__.py")).touch()
    sub_mod = Path(os.path.join(pkg_dir, "api.py"))
    sub_mod.write_text("def my_func(a: int) -> int: return a")

    refs = raise_on_err(extract_target_refs(str(sub_mod), "mock_pkg.api", "reference"))
    assert len(refs) == 1
    assert refs[0].api_path == "reference.my_func"
    assert refs[0].name == "my_func"
    assert refs[0].params[0].name == "a"


def test_score_compliance_empty() -> None:
    """Test compliance scoring with empty reference."""
    ref_snap: dict[str, JsonValue] = {"categories": {}}
    res = score_compliance(ref_snap, [])
    assert res.score_percentage == 0.0


def test_score_compliance_matches() -> None:
    """Test compliance scoring with matches and missing."""
    from ml_switcheroo_ir.schema.ghost import GhostParam, GhostRef, ParameterKind

    ref_snap: dict[str, JsonValue] = {
        "categories": {
            "LAYER": [
                {
                    "name": "func1",
                    "api_path": "reference.func1",
                    "kind": "function",
                    "params": [
                        {
                            "name": "a",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "annotation": "int",
                        }
                    ],
                    "docstring": "",
                    "aliases": [],
                    "returns_type": "None",
                },
                {
                    "name": "func2",
                    "api_path": "reference.func2",
                    "kind": "function",
                    "params": [],
                    "docstring": "",
                    "aliases": [],
                    "returns_type": "None",
                },
            ]
        }
    }

    target_refs = [
        GhostRef(
            name="func1",
            api_path="reference.func1",
            kind="function",
            params=[
                GhostParam(
                    name="a",
                    kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                    annotation="int",
                )
            ],
            docstring="",
            aliases=[],
            returns_type="None",
        )
    ]

    res = score_compliance(ref_snap, target_refs)
    assert res.total_endpoints == 2
    assert res.score_percentage == 50.0
    assert "reference.func1" in res.matched
    assert "reference.func2" in res.missing


def test_score_compliance_mismatch() -> None:
    """Test compliance scoring with mismatch and varargs fallback."""
    from ml_switcheroo_ir.schema.ghost import GhostParam, GhostRef, ParameterKind

    ref_snap: dict[str, JsonValue] = {
        "categories": {
            "LAYER": [
                {
                    "name": "func1",
                    "api_path": "reference.func1",
                    "kind": "function",
                    "params": [
                        {
                            "name": "a",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "annotation": "int",
                        }
                    ],
                    "docstring": "",
                    "aliases": [],
                    "returns_type": "None",
                },
                {
                    "name": "func2",
                    "api_path": "reference.func2",
                    "kind": "function",
                    "params": [
                        {
                            "name": "b",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "annotation": "int",
                        }
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
            name="func1",
            api_path="reference.func1",
            kind="function",
            params=[
                GhostParam(
                    name="wrong",
                    kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                    annotation="str",
                )
            ],
            docstring="",
            aliases=[],
            returns_type="None",
        ),
        GhostRef(
            name="func2",
            api_path="reference.func2",
            kind="function",
            params=[
                GhostParam(name="args", kind=ParameterKind.VAR_POSITIONAL),
                GhostParam(name="kwargs", kind=ParameterKind.VAR_KEYWORD),
            ],
            docstring="",
            aliases=[],
            returns_type="None",
        ),
    ]

    res = score_compliance(ref_snap, target_refs)
    assert res.total_endpoints == 2
    assert "reference.func1" not in res.matched
    assert "reference.func2" in res.matched  # because varargs fallback
    assert len(res.mismatched) == 1
    assert res.mismatched[0].api_path == "reference.func1"


def test_extract_target_refs_import_error(tmp_path: Path) -> None:
    """Test static extraction extracts defined functions while skipping unresolved imports without executing module code.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    pkg_dir = Path(os.path.join(tmp_path, "bad_pkg"))
    pkg_dir.mkdir()
    Path(os.path.join(pkg_dir, "__init__.py")).touch()
    sub_mod = Path(os.path.join(pkg_dir, "api.py"))
    sub_mod.write_text(
        "import missing_module\nraise RuntimeError('must never execute at import time')\ndef my_func(x: int) -> int:\n    return x\n"
    )

    refs = raise_on_err(extract_target_refs(str(sub_mod), "bad_pkg.api", "reference"))
    assert [r.api_path for r in refs] == ["reference.my_func"]


def test_extract_target_refs_aliases(tmp_path: Path) -> None:
    """Test extraction with aliases mapping correctly.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    from ml_switcheroo_ir.schema.ghost import GhostRef

    ref_snap: dict[str, JsonValue] = {
        "categories": {
            "LAYER": [
                {
                    "name": "func1",
                    "api_path": "reference.func1",
                    "kind": "function",
                    "params": [],
                    "docstring": "",
                    "aliases": ["reference.alias1"],
                    "returns_type": "None",
                }
            ]
        }
    }

    target_refs = [
        GhostRef(
            name="func1",
            api_path="reference.alias1",
            kind="function",
            params=[],
            docstring="",
            aliases=[],
            returns_type="None",
        )
    ]

    res = score_compliance(ref_snap, target_refs)
    assert res.total_endpoints == 1
    assert res.score_percentage == 100.0


def test_extract_target_refs_skip_private(tmp_path: Path) -> None:
    """Test extracting target refs skips private members.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    pkg_dir = Path(os.path.join(tmp_path, "priv_pkg"))
    pkg_dir.mkdir()
    Path(os.path.join(pkg_dir, "__init__.py")).touch()
    sub_mod = Path(os.path.join(pkg_dir, "api.py"))
    sub_mod.write_text("def _private_func(): pass\nclass _PrivateClass: pass")

    refs = raise_on_err(extract_target_refs(str(sub_mod), "priv_pkg.api", "reference"))
    assert len(refs) == 0


def test_extract_target_refs_nested_import_error(tmp_path: Path) -> None:
    """Test extracting target refs fails on nested import cleanly.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    pkg_dir = Path(os.path.join(tmp_path, "nested_pkg"))
    pkg_dir.mkdir()
    Path(os.path.join(pkg_dir, "__init__.py")).touch()
    sub_mod = Path(os.path.join(pkg_dir, "api.py"))
    sub_mod.write_text("import sys\nsys.modules['nested_pkg.api.broken'] = None\ndef broken(): pass")
    refs = raise_on_err(extract_target_refs([str(sub_mod), str(sub_mod)], "nested_pkg.api", "reference"))

    assert [r.name for r in refs] == ["broken"]


def test_extract_target_refs_sys_path(tmp_path: Path) -> None:
    """Test extracting target refs when search path is already in sys.path.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    import sys

    pkg_dir = Path(os.path.join(tmp_path, "sys_pkg"))
    pkg_dir.mkdir()
    Path(os.path.join(pkg_dir, "__init__.py")).touch()
    sub_mod = Path(os.path.join(pkg_dir, "api.py"))
    sub_mod.write_text("def my_func(): pass")

    sys.path.insert(0, str(tmp_path))
    refs = raise_on_err(extract_target_refs(str(sub_mod), "sys_pkg.api", "reference"))
    assert len(refs) == 1
    sys.path.remove(str(tmp_path))


def test_extract_target_refs_empty_members(tmp_path: Path) -> None:
    """Test extracting from a module with no members.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    pkg_dir = Path(os.path.join(tmp_path, "empty_pkg"))
    pkg_dir.mkdir()
    Path(os.path.join(pkg_dir, "__init__.py")).touch()

    refs = raise_on_err(extract_target_refs(str(pkg_dir), "empty_pkg", "reference"))
    assert len(refs) == 0


def test_extract_target_refs_catch_all_exception(tmp_path: Path) -> None:
    """Test extracting target refs skips items that raise general exceptions.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    from ml_ecosystem_snapshots.compliance import extract_target_refs

    pkg_dir = Path(os.path.join(tmp_path, "broken_pkg"))
    pkg_dir.mkdir()
    Path(os.path.join(pkg_dir, "__init__.py")).touch()
    sub_mod = Path(os.path.join(pkg_dir, "api.py"))
    sub_mod.write_text("class BrokenClass:\n    @property\n    def prop(self): raise RuntimeError('Broken')")

    refs = raise_on_err(extract_target_refs(str(sub_mod), "broken_pkg.api", "reference"))
    assert any(r.name == "BrokenClass" for r in refs) or refs == ()


def test_extract_target_refs_catch_inner_exception(tmp_path: Path) -> None:
    """Test extracting target refs fails cleanly on inner node walk.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    from unittest.mock import patch

    from ml_ecosystem_snapshots.compliance import extract_target_refs

    pkg_dir = Path(os.path.join(tmp_path, "inner_broken_pkg"))
    pkg_dir.mkdir()
    Path(os.path.join(pkg_dir, "__init__.py")).touch()
    sub_mod = Path(os.path.join(pkg_dir, "api.py"))
    # Create an invalid state for Griffe so dynamic import fails but walk continues
    sub_mod.write_text("class Outer:\n    class Inner:\n        pass\n")

    import ml_ecosystem_snapshots.introspection.inspector
    from ml_ecosystem_snapshots.introspection.runtime_boundary import (
        LivePythonTarget,
        StaticGriffeNode,
    )

    orig_inspect = ml_ecosystem_snapshots.introspection.inspector.GhostInspector.inspect

    def mock_inspect(
        obj: LivePythonTarget | StaticGriffeNode,
        api_path: str,
        is_public: bool | None = None,
    ) -> ExtendedGhostRef:
        """Mock inspect function.

        Args:
            obj: Target live object to inspect.
            api_path: Dotted canonical API path.
            is_public: Optional public visibility override.

        Returns:
            A mocked GhostRef.

        Raises:
            RuntimeError: When `api_path` references `Inner`.
        """
        if "Inner" in api_path:
            raise RuntimeError("inner broken")
        return orig_inspect(obj, api_path, is_public=bool(is_public))

    with patch.object(
        ml_ecosystem_snapshots.introspection.inspector.GhostInspector,
        "inspect",
        mock_inspect,
    ):
        refs = raise_on_err(extract_target_refs(str(sub_mod), "inner_broken_pkg.api", "reference"))
        # Should have extracted Outer, skipped Inner due to exception
        assert len(refs) == 1


def test_extract_target_refs_no_parts(tmp_path: Path) -> None:
    """Test extracting target refs handles paths with no inner parts (just module).

    Args:
        tmp_path: Temporary directory path fixture.
    """
    from ml_ecosystem_snapshots.compliance import extract_target_refs

    pkg_dir = Path(os.path.join(tmp_path, "top_level_pkg"))
    pkg_dir.mkdir()
    sub_mod = Path(os.path.join(pkg_dir, "__init__.py"))
    sub_mod.write_text("def top_func(): pass")

    refs = raise_on_err(extract_target_refs(str(sub_mod), "top_level_pkg", "reference"))
    assert len(refs) == 1


def test_extract_target_refs_break_loop(tmp_path: Path) -> None:
    """Test extracting target refs loop breaking logic.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    from ml_ecosystem_snapshots.compliance import extract_target_refs

    pkg_dir = Path(os.path.join(tmp_path, "break_pkg"))
    pkg_dir.mkdir()
    Path(os.path.join(pkg_dir, "__init__.py")).touch()
    sub_mod = Path(os.path.join(pkg_dir, "api.py"))
    sub_mod.write_text("class MyClass:\n    pass")

    refs = raise_on_err(extract_target_refs(str(sub_mod), "break_pkg.api", "reference"))
    assert len(refs) == 1


def test_extract_target_refs_continue_loop(tmp_path: Path) -> None:
    """Test extracting target refs handles the loop fully exiting.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    from ml_ecosystem_snapshots.compliance import extract_target_refs

    pkg_dir = Path(os.path.join(tmp_path, "cont_pkg"))
    pkg_dir.mkdir()
    Path(os.path.join(pkg_dir, "__init__.py")).touch()
    sub_mod = Path(os.path.join(pkg_dir, "api.py"))
    sub_mod.write_text("import sys\ndef func(): pass")

    refs = raise_on_err(extract_target_refs(str(sub_mod), "cont_pkg.api", "reference"))
    assert any(r.name == "func" or r.name == "sys" for r in refs)


def test_derive_base_path_src(tmp_path: Path) -> None:
    """Verify derive base path src behavior.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    import os

    from ml_ecosystem_snapshots.compliance import get_module_info_from_path

    src_dir = Path(os.path.join(tmp_path, "src"))
    src_dir.mkdir()
    setup_file = Path(os.path.join(tmp_path, "setup.py"))
    setup_file.touch()

    init_dir = Path(os.path.join(tmp_path, "my_pkg"))
    init_dir.mkdir()
    Path(os.path.join(init_dir, "__init__.py")).touch()
    py_file = Path(os.path.join(init_dir, "my_mod.py"))
    py_file.touch()

    orig = os.getcwd()
    os.chdir(tmp_path)
    try:
        path, mod = raise_on_err(get_module_info_from_path(str(tmp_path), "my_mod"))
        assert path == str(src_dir)
        assert mod == "my_mod"

        path, mod = raise_on_err(get_module_info_from_path(str(py_file), ""))
        assert path == str(tmp_path)
        assert mod == "my_pkg.my_mod"
    finally:
        os.chdir(orig)


def test_map_target_to_reference_path_extra() -> None:
    """Verify map target to reference path extra behavior."""
    from ml_ecosystem_snapshots.cli.commands.verify_compliance import ZERO_ECOSYSTEM_ALIGNMENT_RULE
    from ml_ecosystem_snapshots.compliance import align_namespace

    assert (
        align_namespace(
            "zero_torch.jax.numpy.array", "zero_torch", "torch", alignment_rule=ZERO_ECOSYSTEM_ALIGNMENT_RULE
        )
        == "jax.numpy.array"
    )
    assert (
        align_namespace("zero_torch.optax.adam", "zero_torch", "torch", alignment_rule=ZERO_ECOSYSTEM_ALIGNMENT_RULE)
        == "optax.adam"
    )
    assert align_namespace("zero_torch", "zero_torch", "torch", alignment_rule=ZERO_ECOSYSTEM_ALIGNMENT_RULE) == "torch"
    assert align_namespace("zero.array", "zero", "np") == "np.array"


def test_audit_module_sources_pure_monoid_and_ast_branches() -> None:
    """Verify pure `audit_module_sources` AST extraction, `FILE_COMPLIANCE_MONOID` folding, and error accumulation."""
    from ml_switcheroo_ir.schema.ghost import (
        ExtendedGhostParam,
        ParameterKind,
        SnapshotEnvelope,
    )
    from ml_switcheroo_ir.typeclasses.outcome import Invalid, Valid

    from ml_ecosystem_snapshots.compliance import (
        audit_module_sources,
        canonicalize_ir_dtype_name,
        normalize_source_module_name,
    )
    from ml_ecosystem_snapshots.proofs import prove_envelope_valid, upgrade_to_hydrated
    from ml_ecosystem_snapshots.typeclasses import COMPLIANCE_REPORT_MONOID

    assert normalize_source_module_name("./src/zero_jax/nn/__init__.py") == "zero_jax.nn"
    assert normalize_source_module_name("zero_jax/lax/ops") == "zero_jax.lax.ops"
    assert normalize_source_module_name("zero_jax.nn") == "zero_jax.nn"
    assert normalize_source_module_name("__init__.py", target_prefix="zero_jax") == "zero_jax"
    assert normalize_source_module_name("__init__.py") == "module"
    assert canonicalize_ir_dtype_name("float32") == "float32"
    assert canonicalize_ir_dtype_name("custom_weird_dtype") == "custom_weird_dtype"

    env = SnapshotEnvelope(
        target="jax",
        version="0.4.30",
        categories={
            "array": [
                ExtendedGhostRef(
                    name="add",
                    api_path="jax.add",
                    kind="function",
                    params=[
                        ExtendedGhostParam(name="x", kind=ParameterKind.POSITIONAL_ONLY, annotation="Array"),
                        ExtendedGhostParam(
                            name="y",
                            kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                            default="None",
                            annotation="Array",
                        ),
                        ExtendedGhostParam(name="args", kind=ParameterKind.VAR_POSITIONAL, annotation="int"),
                        ExtendedGhostParam(
                            name="alpha",
                            kind=ParameterKind.KEYWORD_ONLY,
                            default="1",
                            annotation="int",
                        ),
                        ExtendedGhostParam(name="kwargs", kind=ParameterKind.VAR_KEYWORD, annotation="int"),
                    ],
                ),
                ExtendedGhostRef(
                    name="Dense",
                    api_path="jax.Dense",
                    kind="class",
                    params=[
                        ExtendedGhostParam(
                            name="features",
                            kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                            annotation="int",
                        )
                    ],
                ),
                ExtendedGhostRef(
                    name="apply",
                    api_path="jax.Dense.apply",
                    kind="method",
                    params=[
                        ExtendedGhostParam(
                            name="x",
                            kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                            annotation="Array",
                        )
                    ],
                ),
            ]
        },
    )
    env_val = prove_envelope_valid(env)
    assert isinstance(env_val, Valid)
    hyd_val = upgrade_to_hydrated(env_val.value)
    assert isinstance(hyd_val, Valid)
    proven_snap = hyd_val.value

    nl = chr(10)
    mod_a = nl.join(
        [
            "import typing",
            "from typing import overload",
            "",
            "def _private_helper() -> None:",
            "    pass",
            "",
            "@overload",
            "def add(x: Array, /) -> Array: ...",
            "@typing.overload",
            "def add(x: Array, /, y: Array = None) -> Array: ...",
            "def add(x: Array, /, y: Array = None, *args: int, alpha: int = 1, **kwargs: int) -> Array:",
            '    """Add arrays."""',
            "    return x",
            "",
        ]
    )
    mod_b = nl.join(
        [
            "import typing",
            "",
            "class _PrivateCls:",
            "    pass",
            "",
            "class Dense:",
            "    @typing.overload",
            "    def __init__(self) -> None: ...",
            "    def __init__(self, features: int) -> None:",
            "        self.features = features",
            "    def _hidden(self) -> None:",
            "        pass",
            "    @typing.overload",
            "    def apply(self, x: Array) -> Array: ...",
            "    def apply(self, x: Array) -> Array:",
            "        return x",
            "",
            "class JaxBackend(CompilerBackend):",
            "    def to_ir(self) -> None: pass",
            "    def compile(self) -> None: pass",
            "",
            "class JaxAdapter(FrameworkAdapter):",
            "    def convert_tensor(self) -> None: pass",
            "",
            "OPS = {'Add': add}",
            "",
        ]
    )
    # Duplicate definition in a second module also tests seen_paths deduplication
    mod_dup = "def add(x: Array, /, y: Array = None, *args: int, alpha: int = 1, **kwargs: int) -> Array: return x"

    audit_ok = audit_module_sources(
        {
            "src/zero_jax/__init__.py": mod_a,
            "zero_jax.py": mod_dup,
            "zero_jax/__init__.py": mod_b,
        },
        proven_snap,
    )
    assert isinstance(audit_ok, Valid)
    rep = audit_ok.value
    assert rep.score_percentage == 100.0
    assert rep.found_ir is True
    assert rep.ir_score == (2, 5)
    assert rep.found_fw is True
    assert rep.fw_score == (1, 4)

    # Monoid identity and combination laws on ComplianceReport
    empty_rep = COMPLIANCE_REPORT_MONOID.empty()
    assert COMPLIANCE_REPORT_MONOID.semigroup.combine(empty_rep, rep) == rep
    assert COMPLIANCE_REPORT_MONOID.semigroup.combine(rep, empty_rep) == rep

    # Fallback prefix when sources is empty
    empty_sources_ok = audit_module_sources({}, proven_snap)
    assert isinstance(empty_sources_ok, Valid)
    assert empty_sources_ok.value.score_percentage == 0.0

    # Accumulates multiple SyntaxErrors across sources into NonEmpty[SnapshotDomainError]
    bad_audit = audit_module_sources(
        {
            "zero_jax/a.py": "def broken1(:" + nl,
            "zero_jax/b.py": "def broken2(:" + nl,
        },
        proven_snap,
        target_prefix="zero_jax",
    )
    assert isinstance(bad_audit, Invalid)
    assert len(bad_audit.errors) == 2
