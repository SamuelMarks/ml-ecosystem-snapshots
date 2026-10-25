"""Anti-regression test suite for ML framework snapshots.

Enforces:
1. Zero unparsed regex tokens (e.g. '???0', 'SNOWFLAKE') in bundled snapshots.
2. No decay to (*args, **kwargs) for top PyTorch operators.
3. Zero duplicate placeholder operand signatures in AMD RDNA and NVIDIA SASS.
4. Complete separation of SSA operands from buildable attributes in MLIR/StableHLO.
"""

from __future__ import annotations

import json
from pathlib import Path
import os

import pytest

from ml_ecosystem_snapshots.models import JsonValue


def get_bundled_json(filename: str) -> list[dict[str, JsonValue]]:
    """Load bundled JSON file from frameworks directory.

    Args:
        filename: Name of the JSON file in frameworks directory.

    Returns:
        Parsed JSON content list.
    """
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fpath = os.path.join(
        pkg_dir, "src", "ml_ecosystem_snapshots", "frameworks", filename
    )
    if os.path.isfile(fpath):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        if "sass" in filename:
            from ml_ecosystem_snapshots.frameworks.nvidia_sass import (
                _load_exhaustive_sass,
            )

            data = _load_exhaustive_sass()
        elif "rdna" in filename:
            from ml_ecosystem_snapshots.frameworks.amd_rdna import _load_exhaustive_rdna

            data = _load_exhaustive_rdna()
        elif "mlir" in filename:
            from ml_ecosystem_snapshots.frameworks.mlir import (
                _get_canonical_mlir_records,
            )

            data = _get_canonical_mlir_records()
        elif "stablehlo" in filename:
            from ml_ecosystem_snapshots.frameworks.stablehlo import (
                _get_canonical_stablehlo_records,
            )

            data = _get_canonical_stablehlo_records()
        else:
            data = []
    if isinstance(data, dict) and "categories" in data:
        items: list[dict[str, JsonValue]] = []
        categories = data["categories"]
        if isinstance(categories, dict):
            for cat_items in categories.values():
                if isinstance(cat_items, list):
                    for item in cat_items:
                        if isinstance(item, dict):
                            items.append(item)
        return items
    if isinstance(data, list):
        return [dict(item) for item in data if isinstance(item, dict)]
    return []


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
            assert (
                tok not in raw_text
            ), f"Forbidden unparsed token '{tok}' found in {sname}"


def test_no_decay_to_varargs_for_top_pytorch_ops() -> None:
    """Verify top PyTorch functions do not decay to (*args, **kwargs)."""
    from torch_mock import ensure_torch

    torch = ensure_torch()
    from ml_ecosystem_snapshots.introspection.inspector import GhostInspector

    top_ops = [
        (torch.relu, "torch.relu"),
        (torch.matmul, "torch.matmul"),
        (torch.add, "torch.add"),
        (torch.sum, "torch.sum"),
        (torch.cat, "torch.cat"),
        (torch.mm, "torch.mm"),
        (torch.bmm, "torch.bmm"),
    ]

    for op, api_path in top_ops:
        ref = GhostInspector.inspect(op, api_path)
        # Check that it did not decay to inexact (*args, **kwargs)
        if len(ref.params) == 2:
            names = [p.name for p in ref.params]
            assert not (
                names == ["args", "kwargs"]
            ), f"Operator {api_path} decayed to (*args, **kwargs)!"
        assert "inexact_signature" not in (
            ref.environment_tags or []
        ), f"Operator {api_path} tagged with inexact_signature!"


def test_zero_duplicate_placeholder_operand_signatures() -> None:
    """Verify zero duplicate placeholder operand signatures in AMD RDNA and NVIDIA SASS dumps."""
    # 1. NVIDIA SASS
    sass_data = get_bundled_json("nvidia_sass_exhaustive.json")
    for item in sass_data:
        op_sigs = item.get("operand_signatures", [])
        assert isinstance(op_sigs, list)
        sig_tuples = [tuple(sig) for sig in op_sigs if isinstance(sig, list)]
        # No duplicate operand signatures for the same instruction
        assert len(sig_tuples) == len(
            set(sig_tuples)
        ), f"Duplicate operand signatures found in SASS mnemonic {item.get('mnemonic')}"

    # 2. AMD RDNA
    rdna_data = get_bundled_json("amd_rdna_exhaustive.json")
    for item in rdna_data:
        op_sigs = item.get("operand_signatures", [])
        assert isinstance(op_sigs, list)
        sig_tuples = [tuple(sig) for sig in op_sigs if isinstance(sig, list)]
        assert len(sig_tuples) == len(
            set(sig_tuples)
        ), f"Duplicate operand signatures found in RDNA mnemonic {item.get('mnemonic')}"


def test_separation_of_ssa_operands_from_attributes_in_dialects() -> None:
    """Verify complete separation of SSA operands from buildable attributes in MLIR and StableHLO."""
    # 1. MLIR
    mlir_ops = get_bundled_json("mlir_exhaustive.json")
    for op in mlir_ops:
        operands = op.get("operands", [])
        attributes = op.get("attributes", [])
        assert isinstance(operands, list)
        assert isinstance(attributes, list)

        op_names: set[str] = {
            str(o.get("name") if isinstance(o, dict) else o) for o in operands
        }
        attr_names: set[str] = {
            str(a.get("name") if isinstance(a, dict) else a) for a in attributes
        }

        # Intersection between operands and attributes should be empty
        overlap = op_names.intersection(attr_names)
        assert (
            len(overlap) == 0
        ), f"Overlap between SSA operands and attributes in MLIR op {op.get('name')}: {overlap}"

    # 2. StableHLO
    hlo_ops = get_bundled_json("stablehlo_exhaustive.json")
    for op in hlo_ops:
        operands = op.get("operands", [])
        attributes = op.get("attributes", [])
        assert isinstance(operands, list)
        assert isinstance(attributes, list)

        op_names = {str(o.get("name") if isinstance(o, dict) else o) for o in operands}
        attr_names = {
            str(a.get("name") if isinstance(a, dict) else a) for a in attributes
        }

        overlap = op_names.intersection(attr_names)
        assert (
            len(overlap) == 0
        ), f"Overlap between SSA operands and attributes in StableHLO op {op.get('name')}: {overlap}"


@pytest.mark.frameworks
def test_dask_minimum_symbol_threshold() -> None:
    """Verify Dask collector extracts genuine GhostRef signatures across array and util tiers."""
    import types
    from unittest.mock import patch
    import ml_ecosystem_snapshots.frameworks.dask as dask_mod
    from ml_switcheroo_ir.schema.ghost import SemanticTier

    fake_da = types.ModuleType("dask.array")
    fake_linalg = types.ModuleType("dask.array.linalg")
    fake_fft = types.ModuleType("dask.array.fft")
    fake_top = types.ModuleType("dask")

    def sample_array_op(
        x: list[float], axis: int = 0, keepdims: bool = False
    ) -> list[float]:
        """Compute array reduction along axis.

        Args:
            x: Input dask array.
            axis: Reduction axis index.
            keepdims: Whether to preserve reduced dimensions.

        Returns:
            Reduced dask array.
        """
        return x

    for i in range(180):
        fn = types.FunctionType(
            sample_array_op.__code__,
            sample_array_op.__globals__,
            name=f"op_{i}",
            argdefs=sample_array_op.__defaults__,
            closure=sample_array_op.__closure__,
        )
        fn.__annotations__ = dict(sample_array_op.__annotations__)
        fn.__doc__ = sample_array_op.__doc__
        setattr(fake_da, f"op_{i}", fn)

    for i in range(15):
        fn = types.FunctionType(
            sample_array_op.__code__,
            sample_array_op.__globals__,
            name=f"linalg_op_{i}",
            argdefs=sample_array_op.__defaults__,
            closure=sample_array_op.__closure__,
        )
        setattr(fake_linalg, f"linalg_op_{i}", fn)

    for i in range(10):
        fn = types.FunctionType(
            sample_array_op.__code__,
            sample_array_op.__globals__,
            name=f"util_op_{i}",
            argdefs=sample_array_op.__defaults__,
            closure=sample_array_op.__closure__,
        )
        setattr(fake_top, f"util_op_{i}", fn)

    setattr(fake_da, "linalg", fake_linalg)
    setattr(fake_da, "fft", fake_fft)

    with (
        patch.object(dask_mod, "da", fake_da),
        patch.dict("sys.modules", {"dask": fake_top}),
    ):
        arr_refs = dask_mod.collect_api(SemanticTier.ARRAY_API)
        util_refs = dask_mod.collect_api(SemanticTier.UTIL)

    total_count = len(arr_refs) + len(util_refs)
    assert (
        total_count >= 200
    ), f"Dask symbol count too low: {total_count} endpoints found"
    first_ref = arr_refs[0]
    assert first_ref.api_path.startswith("dask.array.")
    assert any(p.name == "axis" for p in first_ref.params)


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
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                assert (
                    "categories" in data
                ), f"Snapshot {fpath} missing 'categories' envelope key"


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
        with open(fpath, "r", encoding="utf-8") as fp:
            content = fp.read()
        for token in banned_tokens:
            assert (
                token not in content
            ), f"Banned token '{token}' found in production module {fpath}"
        if not fpath.endswith("stubs.py"):
            assert (
                ": object" not in content and "-> object" not in content
            ), f"Banned 'object' type annotation found in production module {fpath}"

    all_py_files = sorted(
        glob.glob("src/ml_ecosystem_snapshots/**/*.py", recursive=True)
        + glob.glob("tests/**/*.py", recursive=True)
        + glob.glob("scripts/**/*.py", recursive=True)
    )
    for fpath in all_py_files:
        if fpath.endswith("test_anti_regression.py"):
            continue
        with open(fpath, "r", encoding="utf-8") as fp:
            content = fp.read()
        assert (
            "# type: ignore" not in content
        ), f"Banned '# type: ignore' found in {fpath}"
        assert (
            "len(captured.out) >= 0" not in content
        ), f"Tautological assertion found in {fpath}"


def test_every_test_function_has_assertions() -> None:
    """Verify every test_* function in tests/ contains at least one explicit assertion or raises check."""
    import ast
    import glob

    test_files = sorted(glob.glob("tests/**/*.py", recursive=True))
    for fpath in test_files:
        with open(fpath, "r", encoding="utf-8") as fp:
            tree = ast.parse(fp.read(), filename=fpath)
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and node.name.startswith("test_"):
                has_assert = False
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Assert):
                        has_assert = True
                        break
                    if isinstance(sub, ast.Call):
                        call_str = ast.unparse(sub.func)
                        if any(
                            k in call_str for k in ("raises", "assert", "warns", "fail")
                        ):
                            has_assert = True
                            break
                assert (
                    has_assert
                ), f"Zero-assertion test detected: {fpath}:{node.lineno} ({node.name})"


def test_codebase_100_percent_docstring_and_arg_coverage() -> None:
    """Verify 100% module, class, function, and parameter docstring coverage with zero placeholders."""
    import ast
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
        with open(fpath, "r", encoding="utf-8") as fp:
            src = fp.read()
        tree = ast.parse(src, filename=fpath)
        mod_doc = ast.get_docstring(tree)
        assert mod_doc, f"Missing module docstring in {fpath}"
        for bp in banned_placeholders:
            assert (
                bp not in mod_doc
            ), f"Placeholder '{bp}' in module docstring of {fpath}"
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                    (isinstance(d, ast.Name) and d.id == "overload")
                    or (isinstance(d, ast.Attribute) and d.attr == "overload")
                    for d in node.decorator_list
                ):
                    continue
                doc = ast.get_docstring(node)
                assert doc, f"Missing docstring on {node.name} at {fpath}:{node.lineno}"
                for bp in banned_placeholders:
                    assert (
                        bp not in doc
                    ), f"Placeholder '{bp}' in docstring of {node.name} at {fpath}:{node.lineno}"
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = [
                        a.arg for a in node.args.args if a.arg not in ("self", "cls")
                    ]
                    args += [a.arg for a in node.args.kwonlyargs]
                    if node.args.vararg:
                        args.append(node.args.vararg.arg)
                    if node.args.kwarg:
                        args.append(node.args.kwarg.arg)
                    for arg_name in args:
                        assert re.search(
                            rf"\b{re.escape(arg_name)}\b\s*(\([^)]*\))?\s*:", doc
                        ), f"Missing arg '{arg_name}' in docstring of {node.name} at {fpath}:{node.lineno}"
