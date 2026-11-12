"""Anti-regression test suite for ML framework snapshots.

Enforces:
1. Zero unparsed regex tokens (e.g. '???0', 'SNOWFLAKE') in bundled snapshots.
2. No decay to (*args, **kwargs) for top PyTorch operators.
3. Zero duplicate placeholder operand signatures in AMD RDNA and NVIDIA SASS.
4. Complete separation of SSA operands from buildable attributes in MLIR/StableHLO.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest

from ml_ecosystem_snapshots.models import JsonValue


def _load_fallback_bundled_data(filename: str) -> list[dict[str, JsonValue]]:
    """Load canonical fallback records when an exhaustive JSON file is not on disk.

    Args:
        filename: Name of the JSON file in frameworks directory.

    Returns:
        Canonical fallback records or empty list.
    """
    if "sass" in filename:
        from ml_ecosystem_snapshots.frameworks.nvidia_sass import (
            load_exhaustive_sass,
        )

        return [r.model_dump() for r in load_exhaustive_sass()]
    if "rdna" in filename:
        from ml_ecosystem_snapshots.frameworks.amd_rdna import load_exhaustive_rdna

        return [dict(r) for r in load_exhaustive_rdna()]
    if "mlir" in filename:
        from ml_ecosystem_snapshots.frameworks.mlir import (
            get_canonical_mlir_records,
        )

        return [dict(r) for r in get_canonical_mlir_records()]
    if "stablehlo" in filename:
        from ml_ecosystem_snapshots.frameworks.stablehlo import (
            get_canonical_stablehlo_records,
        )

        return [dict(r) for r in get_canonical_stablehlo_records()]
    return []


def _extract_dict_records(data: JsonValue | list[dict[str, JsonValue]]) -> list[dict[str, JsonValue]]:
    """Normalize loaded JSON data into a flat list of dictionary records.

    Args:
        data: Raw decoded JSON value.

    Returns:
        Flat list of dictionary records.
    """
    if isinstance(data, dict) and "categories" in data:
        categories = data["categories"]
        if isinstance(categories, dict):
            return [
                dict(item)
                for cat_items in categories.values()
                if isinstance(cat_items, list)
                for item in cat_items
                if isinstance(item, dict)
            ]
        return []
    if isinstance(data, list):
        return [dict(item) for item in data if isinstance(item, dict)]
    return []


def get_bundled_json(filename: str) -> list[dict[str, JsonValue]]:
    """Load bundled JSON file from frameworks directory.

    Args:
        filename: Name of the JSON file in frameworks directory.

    Returns:
        Parsed JSON content list.
    """
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fpath = os.path.join(pkg_dir, "src", "ml_ecosystem_snapshots", "frameworks", filename)
    if os.path.isfile(fpath):
        with open(fpath, encoding="utf-8") as f:
            data: JsonValue | list[dict[str, JsonValue]] = json.load(f)
    else:
        data = _load_fallback_bundled_data(filename)
    return _extract_dict_records(data)


def test_zero_unparsed_regex_tokens_in_bundled_snapshots() -> None:
    """Verify zero unparsed regex or snowflake placeholder tokens in any bundled snapshot."""
    forbidden_tokens = ["???0", "???1", "SNOWFLAKE", "<placeholder>", "UNKNOWN_OPERAND"]
    snapshots = [
        "nvidia_sass_exhaustive.json",
        "amd_rdna_exhaustive.json",
        "mlir_exhaustive.json",
        "stablehlo_exhaustive.json",
    ]

    for sname in snapshots:
        data = get_bundled_json(sname)
        raw_text = json.dumps(data)
        for tok in forbidden_tokens:
            assert tok not in raw_text, f"Forbidden unparsed token '{tok}' found in {sname}"


def test_no_decay_to_varargs_for_top_pytorch_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify verbatim PyTorch C-extension `__doc__` signatures never decay to `(*args, **kwargs)`.

    Args:
        monkeypatch: Pytest monkeypatch fixture used to attach verbatim C++ docstrings.
    """
    from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
    from ml_ecosystem_snapshots.introspection.runtime_boundary import LivePythonTarget

    verbatim_c_docs: list[tuple[str, str, list[str]]] = [
        ("torch.relu", "relu(input: Tensor) -> Tensor", ["input"]),
        (
            "torch.matmul",
            "matmul(input: Tensor, other: Tensor, *, out: Optional[Tensor] = None) -> Tensor",
            ["input", "other", "out"],
        ),
        (
            "torch.add",
            "add(input: Tensor, other: Tensor, *, alpha: Number = 1, out: Optional[Tensor] = None) -> Tensor\n"
            "add(input: Tensor, other: Number, alpha: Number = 1) -> Tensor",
            ["input", "other", "alpha", "out"],
        ),
        (
            "torch.sum",
            "sum(input: Tensor, dim: Union[int, Tuple[int, ...]], keepdim: bool = False, *, dtype: Optional[_dtype] = None) -> Tensor",
            ["input", "dim", "keepdim", "dtype"],
        ),
        (
            "torch.cat",
            "cat(tensors: Tuple[Tensor, ...], dim: int = 0, *, out: Optional[Tensor] = None) -> Tensor",
            ["tensors", "dim", "out"],
        ),
        (
            "torch.mm",
            "mm(input: Tensor, mat2: Tensor, *, out: Optional[Tensor] = None) -> Tensor",
            ["input", "mat2", "out"],
        ),
        (
            "torch.bmm",
            "bmm(input: Tensor, mat2: Tensor, *, out: Optional[Tensor] = None) -> Tensor",
            ["input", "mat2", "out"],
        ),
    ]

    import inspect

    def raise_no_c_sig(_target: LivePythonTarget) -> inspect.Signature:
        """Simulate CPython `ValueError` when `inspect.signature` inspects a pybind11 C builtin.

        Args:
            _target: Inspected callable target.

        Raises:
            ValueError: Always raised for compiled C builtins without `__text_signature__`.

        Returns:
            Never returns.
        """
        raise ValueError("no signature found for builtin")

    monkeypatch.setattr("ml_ecosystem_snapshots.introspection.inspector.get_runtime_signature", raise_no_c_sig)
    monkeypatch.setattr("inspect.signature", raise_no_c_sig)
    monkeypatch.setattr("ml_ecosystem_snapshots.introspection.inspector.get_runtime_source", lambda _t: "")

    for api_path, raw_c_doc, expected_param_names in verbatim_c_docs:

        def opaque_c_builtin(*args: LivePythonTarget, **kwargs: LivePythonTarget) -> LivePythonTarget:
            """Opaque variadic C-extension stub replaced with verbatim PyTorch `__doc__`.

            Args:
                *args: Positional C-extension arguments.
                **kwargs: Keyword C-extension arguments.

            Returns:
                First argument or None.
            """
            del kwargs
            return args[0] if args else None

        monkeypatch.setattr(opaque_c_builtin, "__doc__", raw_c_doc)
        monkeypatch.setattr(opaque_c_builtin, "__name__", api_path.split(".")[-1])
        ref = GhostInspector.inspect(opaque_c_builtin, api_path)
        assert [p.name for p in ref.params] == expected_param_names, f"Unexpected params for {api_path}"
        assert "inexact_signature" not in (ref.environment_tags or [])
        if api_path == "torch.add":
            from ml_switcheroo_ir.schema.ghost import GhostRef

            assert ref.overloads is not None and len(ref.overloads) == 1
            ov0 = ref.overloads[0]
            assert isinstance(ov0, GhostRef)
            assert [p.name for p in ov0.params] == ["input", "other", "alpha"]


@pytest.mark.frameworks
def test_live_pytorch_top_ops_no_varargs_decay() -> None:
    """Verify live installed PyTorch C-extension operators never decay to `(*args, **kwargs)`."""
    torch_live = pytest.importorskip("torch")
    from ml_ecosystem_snapshots.introspection.inspector import GhostInspector

    for op_fn, api_path in (
        (torch_live.relu, "torch.relu"),
        (torch_live.matmul, "torch.matmul"),
        (torch_live.add, "torch.add"),
        (torch_live.sum, "torch.sum"),
        (torch_live.cat, "torch.cat"),
        (torch_live.mm, "torch.mm"),
        (torch_live.bmm, "torch.bmm"),
    ):
        ref = GhostInspector.inspect(op_fn, api_path)
        assert [p.name for p in ref.params] != ["args", "kwargs"], f"{api_path} decayed to (*args, **kwargs)"
        assert "inexact_signature" not in (ref.environment_tags or [])


def test_zero_duplicate_placeholder_operand_signatures() -> None:
    """Verify zero duplicate placeholder operand signatures in AMD RDNA and NVIDIA SASS dumps."""
    # 1. NVIDIA SASS
    sass_data = get_bundled_json("nvidia_sass_exhaustive.json")
    for item in sass_data:
        op_sigs = item.get("operand_signatures", [])
        assert isinstance(op_sigs, list)
        sig_tuples = [tuple(sig) for sig in op_sigs if isinstance(sig, list)]
        # No duplicate operand signatures for the same instruction
        assert len(sig_tuples) == len(set(sig_tuples)), (
            f"Duplicate operand signatures found in SASS mnemonic {item.get('mnemonic')}"
        )

    # 2. AMD RDNA
    rdna_data = get_bundled_json("amd_rdna_exhaustive.json")
    for item in rdna_data:
        op_sigs = item.get("operand_signatures", [])
        assert isinstance(op_sigs, list)
        sig_tuples = [tuple(sig) for sig in op_sigs if isinstance(sig, list)]
        assert len(sig_tuples) == len(set(sig_tuples)), (
            f"Duplicate operand signatures found in RDNA mnemonic {item.get('mnemonic')}"
        )


def test_separation_of_ssa_operands_from_attributes_in_dialects() -> None:
    """Verify complete separation of SSA operands from buildable attributes in MLIR and StableHLO."""
    # 1. MLIR
    mlir_ops = get_bundled_json("mlir_exhaustive.json")
    for op in mlir_ops:
        operands = op.get("operands", [])
        attributes = op.get("attributes", [])
        assert isinstance(operands, list)
        assert isinstance(attributes, list)

        op_names: set[str] = {str(o.get("name") if isinstance(o, dict) else o) for o in operands}
        attr_names: set[str] = {str(a.get("name") if isinstance(a, dict) else a) for a in attributes}

        # Intersection between operands and attributes should be empty
        overlap = op_names.intersection(attr_names)
        assert len(overlap) == 0, f"Overlap between SSA operands and attributes in MLIR op {op.get('name')}: {overlap}"

    # 2. StableHLO
    hlo_ops = get_bundled_json("stablehlo_exhaustive.json")
    for op in hlo_ops:
        operands = op.get("operands", [])
        attributes = op.get("attributes", [])
        assert isinstance(operands, list)
        assert isinstance(attributes, list)

        op_names = {str(o.get("name") if isinstance(o, dict) else o) for o in operands}
        attr_names = {str(a.get("name") if isinstance(a, dict) else a) for a in attributes}

        overlap = op_names.intersection(attr_names)
        assert len(overlap) == 0, (
            f"Overlap between SSA operands and attributes in StableHLO op {op.get('name')}: {overlap}"
        )


def test_dask_collector_filtering_and_submodule_traversal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Dask collector filters non-callables/privates and traverses `linalg`, `fft`, and top-level `dask`.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import sys
    import types

    from ml_switcheroo_ir.schema.ghost import SemanticTier

    import ml_ecosystem_snapshots.frameworks.dask as dask_mod

    fake_da = types.ModuleType("dask.array")
    fake_linalg = types.ModuleType("dask.array.linalg")
    fake_fft = types.ModuleType("dask.array.fft")
    fake_top = types.ModuleType("dask")

    def sample_reduction(x: list[float], axis: int = 0, keepdims: bool = False) -> list[float]:
        """Compute array reduction along axis.

        Args:
            x: Input dask array.
            axis: Reduction axis index.
            keepdims: Whether to preserve reduced dimensions.

        Returns:
            Reduced dask array.
        """
        del axis, keepdims
        return x

    def _private_helper(x: list[float]) -> list[float]:
        """Internal Dask helper excluded unless `include_nonpublic=True`.

        Args:
            x: Input dask array.

        Returns:
            Unmodified array.
        """
        return x

    monkeypatch.setattr(fake_da, "mean", sample_reduction, raising=False)
    monkeypatch.setattr(fake_da, "_internal", _private_helper, raising=False)
    monkeypatch.setattr(fake_da, "CONSTANT_PI", 3.14159, raising=False)
    monkeypatch.setattr(fake_linalg, "svd", sample_reduction, raising=False)
    monkeypatch.setattr(fake_fft, "fftn", sample_reduction, raising=False)
    monkeypatch.setattr(fake_da, "linalg", fake_linalg, raising=False)
    monkeypatch.setattr(fake_da, "fft", fake_fft, raising=False)
    monkeypatch.setattr(fake_top, "compute", sample_reduction, raising=False)
    monkeypatch.setattr(fake_top, "_secret", _private_helper, raising=False)

    monkeypatch.setattr(dask_mod, "da", fake_da)
    monkeypatch.setitem(sys.modules, "dask", fake_top)

    arr_public = dask_mod.collect_api(SemanticTier.ARRAY_API, include_nonpublic=False)
    arr_paths = {r.api_path for r in arr_public}
    assert arr_paths == {"dask.array.mean", "dask.array.linalg.svd", "dask.array.fft.fftn"}
    mean_ref = next(r for r in arr_public if r.api_path == "dask.array.mean")
    assert [p.name for p in mean_ref.params] == ["x", "axis", "keepdims"]

    arr_all = dask_mod.collect_api(SemanticTier.ARRAY_API, include_nonpublic=True)
    assert "dask.array._internal" in {r.api_path for r in arr_all}

    util_public = dask_mod.collect_api(SemanticTier.UTIL, include_nonpublic=False)
    assert {r.api_path for r in util_public} == {"dask.compute"}


@pytest.mark.frameworks
def test_dask_minimum_symbol_threshold() -> None:
    """Verify live Dask installation exposes >= 200 public endpoints across array and util tiers."""
    pytest.importorskip("dask.array")
    from ml_switcheroo_ir.schema.ghost import SemanticTier

    import ml_ecosystem_snapshots.frameworks.dask as dask_mod

    arr_refs = dask_mod.collect_api(SemanticTier.ARRAY_API)
    util_refs = dask_mod.collect_api(SemanticTier.UTIL)
    total_count = len(arr_refs) + len(util_refs)
    assert total_count >= 200, f"Live Dask symbol count too low: {total_count}"


def test_all_new_snapshots_exist_and_nonempty(tmp_path: Path) -> None:
    """Verify that all target snapshots can be extracted or loaded as valid non-empty JSON envelopes.

    Args:
        tmp_path: Pytest temporary directory fixture used for isolated snapshot serialization.
    """
    from ml_ecosystem_snapshots.api import extract_snapshot, write_snapshot

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    snap_dir = os.path.join(repo_root, "src", "ml_ecosystem_snapshots", "snapshots")

    expected_prefixes = [
        "wgsl",
        "ir",
        "onnx",
        "metal",
        "numba",
        "sparse",
        "dpnp",
        "awkward",
        "pyarrow_compute",
        "bohrium",
        "wasm_simd",
        "webgl",
        "cpp_runtime",
    ]

    for prefix in expected_prefixes:
        matching: list[str] = []
        if os.path.isdir(snap_dir):
            matching = [
                os.path.join(snap_dir, f)
                for f in os.listdir(snap_dir)
                if f.startswith(f"{prefix}_")
                and f.endswith(".json")
                and not f.endswith(("_ops.json", "_exhaustive.json"))
            ]
        if not matching:
            snap = extract_snapshot(prefix)
            assert snap, f"Failed to extract snapshot for target: {prefix}"
            written_path = write_snapshot(prefix, snap, str(tmp_path))
            matching = [written_path]

        assert len(matching) > 0, f"Missing snapshot for target: {prefix}"
        for fpath in matching:
            size = os.path.getsize(fpath)
            assert size > 500, f"Snapshot {fpath} is too small ({size} bytes)"
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
                assert "categories" in data, f"Snapshot {fpath} missing 'categories' envelope key"


def test_no_mock_introspection_or_type_ignores_in_src() -> None:
    """Verify production code under src/ contains zero mock-sniffing and codebase has zero # type: ignore directives."""
    import glob

    src_files = sorted(glob.glob("src/ml_ecosystem_snapshots/**/*.py", recursive=True))
    banned_tokens = [
        "_mock_return_value",
        "MagicMock",
        "unittest.mock",
        "type: ignore",
        "@dataclass",
        "TypedDict",
        "NamedTuple",
        "from typing",
        "Mapping[str, object]",
        "dict[str, object]",
        "list[object]",
        "Sequence[object]",
        "typing.Optional",
        "typing.Union",
        "typing.Any",
        "typing.cast",
    ]
    for fpath in src_files:
        with open(fpath, encoding="utf-8") as fp:
            content = fp.read()
        for token in banned_tokens:
            assert token not in content, f"Banned token '{token}' found in production module {fpath}"
        assert ": object" not in content and "-> object" not in content, (
            f"Banned 'object' type annotation found in production module {fpath}"
        )

    all_py_files = sorted(
        glob.glob("src/ml_ecosystem_snapshots/**/*.py", recursive=True)
        + glob.glob("tests/**/*.py", recursive=True)
        + glob.glob("scripts/**/*.py", recursive=True)
    )
    for fpath in all_py_files:
        if fpath.endswith("test_anti_regression.py"):
            continue
        with open(fpath, encoding="utf-8") as fp:
            content = fp.read()
        assert "# type: ignore" not in content, f"Banned '# type: ignore' found in {fpath}"
        assert "len(captured.out) >= 0" not in content, f"Tautological assertion found in {fpath}"


def test_every_test_function_has_assertions() -> None:
    """Verify every test_* function in tests/ contains at least one explicit assertion or raises check."""
    import ast
    import glob

    test_files = sorted(glob.glob("tests/**/*.py", recursive=True))
    for fpath in test_files:
        with open(fpath, encoding="utf-8") as fp:
            tree = ast.parse(fp.read(), filename=fpath)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                has_assert = False
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Assert):
                        has_assert = True
                        break
                    if isinstance(sub, ast.Call):
                        call_str = ast.unparse(sub.func)
                        if any(k in call_str for k in ("raises", "assert", "warns", "fail")):
                            has_assert = True
                            break
                assert has_assert, f"Zero-assertion test detected: {fpath}:{node.lineno} ({node.name})"


def _is_overload_func(node: ast.AST) -> bool:
    """Check whether an AST node is an @overload-decorated function.

    Args:
        node: AST definition node to inspect.

    Returns:
        True if decorated with @overload, False otherwise.
    """
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    return any(
        (isinstance(d, ast.Name) and d.id == "overload") or (isinstance(d, ast.Attribute) and d.attr == "overload")
        for d in node.decorator_list
    )


def _collect_func_arg_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Collect all non-self/cls parameter names from a function definition.

    Args:
        node: AST function definition node.

    Returns:
        List of parameter identifier names.
    """
    args = [a.arg for a in node.args.args if a.arg not in ("self", "cls")]
    args += [a.arg for a in node.args.kwonlyargs]
    if node.args.vararg:
        args.append(node.args.vararg.arg)
    if node.args.kwarg:
        args.append(node.args.kwarg.arg)
    return args


def _assert_no_banned_placeholders(doc: str, banned_placeholders: tuple[str, ...], target_label: str) -> None:
    """Assert that a docstring contains none of the banned placeholder substrings.

    Args:
        doc: Docstring content to check.
        banned_placeholders: Disallowed placeholder substrings.
        target_label: Human-readable location label for error reporting.
    """
    for bp in banned_placeholders:
        assert bp not in doc, f"Placeholder '{bp}' in docstring of {target_label}"


def test_codebase_100_percent_docstring_and_arg_coverage() -> None:
    """Verify 100% module, class, function, and parameter docstring coverage with zero placeholders."""
    import glob
    import re

    files = sorted(
        glob.glob("src/ml_ecosystem_snapshots/**/*.py", recursive=True)
        + glob.glob("tests/**/*.py", recursive=True)
        + glob.glob("scripts/**/*.py", recursive=True)
    )
    banned_placeholders = (
        "Function docstring.",
        "Class docstring.",
        "Module docstring.",
        ": Parameter.",
        ": description",
        "Return value.",
        "Input parameter value.",
        "cat: category",
        "node: node",
        "current_path: path",
        "p: param",
        ": mock\n",
        ": node\n",
        ": path\n",
        ": param\n",
        ": category\n",
        "Tuple.",
    )
    for fpath in files:
        with open(fpath, encoding="utf-8") as fp:
            src = fp.read()
        tree = ast.parse(src, filename=fpath)
        mod_doc = ast.get_docstring(tree)
        assert mod_doc, f"Missing module docstring in {fpath}"
        _assert_no_banned_placeholders(mod_doc, banned_placeholders, fpath)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not _is_overload_func(node):
                doc = ast.get_docstring(node)
                assert doc, f"Missing docstring on {node.name} at {fpath}:{node.lineno}"
                _assert_no_banned_placeholders(doc, banned_placeholders, f"{node.name} at {fpath}:{node.lineno}")
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for arg_name in _collect_func_arg_names(node):
                        assert re.search(rf"\b{re.escape(arg_name)}\b\s*(\([^)]*\))?\s*:", doc), (
                            f"Missing arg '{arg_name}' in docstring of {node.name} at {fpath}:{node.lineno}"
                        )


def _check_test_ast_node_hygiene(node: ast.AST, fpath: str) -> None:
    """Assert a single AST node in `tests/` avoids private imports and unscoped `sys.modules` writes.

    Args:
        node: AST node from a test module.
        fpath: Path of the test file being inspected.
    """
    if (
        isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith(("ml_ecosystem_snapshots", "scripts"))
    ):
        for alias in node.names:
            is_dunder = alias.name.startswith("__") and alias.name.endswith("__")
            assert not alias.name.startswith("_") or is_dunder, (
                f"Private symbol import '{alias.name}' from '{node.module}' at {fpath}:{node.lineno}"
            )
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "modules":
        assert not isinstance(node.ctx, (ast.Store, ast.Del)), f"Unscoped sys.modules mutation at {fpath}:{node.lineno}"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "dict" and node.args:
        assert "sys.modules" not in ast.unparse(node.args[0]), (
            f"Non-monkeypatch patch.dict(sys.modules) at {fpath}:{node.lineno}"
        )


def test_zero_private_imports_and_unscoped_sys_modules_in_tests() -> None:
    """Enforce zero private `_` imports from `src/`/`scripts/` and zero unscoped `sys.modules` writes in `tests/`."""
    import glob

    test_files = sorted(glob.glob("tests/**/*.py", recursive=True))
    assert len(test_files) > 0
    for fpath in test_files:
        with open(fpath, encoding="utf-8") as fp:
            tree = ast.parse(fp.read(), filename=fpath)
        for node in ast.walk(tree):
            _check_test_ast_node_hygiene(node, fpath)
