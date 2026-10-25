"""Chex API Snapshot Extractor.

Extracts shape, rank, scalar, type, PyTree, device, and trace assertions,
JAX-compatible PyTree dataclasses, fake JIT/pmap testing contexts, and
parameterized test variant decorators from Google DeepMind's Chex library.
"""

from __future__ import annotations

import importlib
import types
import typing

from ml_switcheroo_ir.schema.ghost import (
    GhostRef,
    SemanticTier,
)

from ml_ecosystem_snapshots.frameworks._base import (
    CanonicalParamSpec,
    CanonicalSubmoduleOpSpec,
    build_canonical_python_ref,
    collect_dir_callables,
    scan_griffe_module_members,
)
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.static_parsers import (
    GriffeCacheEntry,
    load_griffe_module_safely,
)
from ml_ecosystem_snapshots.network_env import is_offline_mode


_CHEX_OP_DEFS: list[
    tuple[
        str,
        str,
        str,
        str,
        list[tuple[str, str, str | None]],
        str,
    ]
] = [
    # Shape, Rank & Dimension Assertions (asserts)
    (
        "assert_shape",
        "chex.assert_shape",
        "function",
        "asserts",
        [
            ("inputs", "POSITIONAL_OR_KEYWORD", None),
            ("expected_shapes", "POSITIONAL_OR_KEYWORD", None),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that all input arrays have the expected shapes.",
    ),
    (
        "assert_equal_shape",
        "chex.assert_equal_shape",
        "function",
        "asserts",
        [
            ("inputs", "POSITIONAL_OR_KEYWORD", None),
            ("dims", "KEYWORD_ONLY", "None"),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that all input arrays have identical shapes (or along specified dims).",
    ),
    (
        "assert_equal_shape_prefix",
        "chex.assert_equal_shape_prefix",
        "function",
        "asserts",
        [
            ("inputs", "POSITIONAL_OR_KEYWORD", None),
            ("prefix_len", "POSITIONAL_OR_KEYWORD", None),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that leading prefix dimensions of all input arrays match.",
    ),
    (
        "assert_equal_shape_suffix",
        "chex.assert_equal_shape_suffix",
        "function",
        "asserts",
        [
            ("inputs", "POSITIONAL_OR_KEYWORD", None),
            ("suffix_len", "POSITIONAL_OR_KEYWORD", None),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that trailing suffix dimensions of all input arrays match.",
    ),
    (
        "assert_rank",
        "chex.assert_rank",
        "function",
        "asserts",
        [
            ("inputs", "POSITIONAL_OR_KEYWORD", None),
            ("expected_ranks", "POSITIONAL_OR_KEYWORD", None),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that all input arrays have the expected ranks (number of dimensions).",
    ),
    (
        "assert_equal_rank",
        "chex.assert_equal_rank",
        "function",
        "asserts",
        [
            ("inputs", "POSITIONAL_OR_KEYWORD", None),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that all input arrays have the same rank.",
    ),
    (
        "assert_axis_dimension",
        "chex.assert_axis_dimension",
        "function",
        "asserts",
        [
            ("tensor", "POSITIONAL_OR_KEYWORD", None),
            ("axis", "POSITIONAL_OR_KEYWORD", None),
            ("expected", "POSITIONAL_OR_KEYWORD", None),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that a specific axis of a tensor has size `expected`.",
    ),
    (
        "assert_axis_dimension_gt",
        "chex.assert_axis_dimension_gt",
        "function",
        "asserts",
        [
            ("tensor", "POSITIONAL_OR_KEYWORD", None),
            ("axis", "POSITIONAL_OR_KEYWORD", None),
            ("val", "POSITIONAL_OR_KEYWORD", None),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that a specific axis of a tensor is strictly greater than `val`.",
    ),
    (
        "assert_axis_dimension_gteq",
        "chex.assert_axis_dimension_gteq",
        "function",
        "asserts",
        [
            ("tensor", "POSITIONAL_OR_KEYWORD", None),
            ("axis", "POSITIONAL_OR_KEYWORD", None),
            ("val", "POSITIONAL_OR_KEYWORD", None),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that a specific axis of a tensor is greater than or equal to `val`.",
    ),
    (
        "assert_axis_dimension_lt",
        "chex.assert_axis_dimension_lt",
        "function",
        "asserts",
        [
            ("tensor", "POSITIONAL_OR_KEYWORD", None),
            ("axis", "POSITIONAL_OR_KEYWORD", None),
            ("val", "POSITIONAL_OR_KEYWORD", None),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that a specific axis of a tensor is strictly less than `val`.",
    ),
    (
        "assert_axis_dimension_lteq",
        "chex.assert_axis_dimension_lteq",
        "function",
        "asserts",
        [
            ("tensor", "POSITIONAL_OR_KEYWORD", None),
            ("axis", "POSITIONAL_OR_KEYWORD", None),
            ("val", "POSITIONAL_OR_KEYWORD", None),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that a specific axis of a tensor is less than or equal to `val`.",
    ),
    (
        "assert_is_broadcastable",
        "chex.assert_is_broadcastable",
        "function",
        "asserts",
        [
            ("shape_a", "POSITIONAL_OR_KEYWORD", None),
            ("shape_b", "POSITIONAL_OR_KEYWORD", None),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that two array shapes are NumPy/JAX broadcast-compatible.",
    ),
    # Scalar Assertions
    (
        "assert_scalar",
        "chex.assert_scalar",
        "function",
        "asserts",
        [
            ("x", "POSITIONAL_OR_KEYWORD", None),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that input `x` is a Python scalar (int, float, complex, bool).",
    ),
    (
        "assert_scalar_in",
        "chex.assert_scalar_in",
        "function",
        "asserts",
        [
            ("x", "POSITIONAL_OR_KEYWORD", None),
            ("min_", "POSITIONAL_OR_KEYWORD", None),
            ("max_", "POSITIONAL_OR_KEYWORD", None),
            ("included", "POSITIONAL_OR_KEYWORD", "True"),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that scalar `x` lies within the interval `[min_, max_]`.",
    ),
    (
        "assert_scalar_non_negative",
        "chex.assert_scalar_non_negative",
        "function",
        "asserts",
        [
            ("x", "POSITIONAL_OR_KEYWORD", None),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that scalar `x` is greater than or equal to zero.",
    ),
    (
        "assert_scalar_negative",
        "chex.assert_scalar_negative",
        "function",
        "asserts",
        [
            ("x", "POSITIONAL_OR_KEYWORD", None),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that scalar `x` is strictly negative.",
    ),
    (
        "assert_scalar_positive",
        "chex.assert_scalar_positive",
        "function",
        "asserts",
        [
            ("x", "POSITIONAL_OR_KEYWORD", None),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that scalar `x` is strictly positive.",
    ),
    # Type, Equality, Value & Numerical Assertions
    (
        "assert_type",
        "chex.assert_type",
        "function",
        "asserts",
        [
            ("inputs", "POSITIONAL_OR_KEYWORD", None),
            ("expected_types", "POSITIONAL_OR_KEYWORD", None),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that all input arrays or scalars match the expected dtypes or Python types.",
    ),
    (
        "assert_equal",
        "chex.assert_equal",
        "function",
        "asserts",
        [
            ("first", "POSITIONAL_OR_KEYWORD", None),
            ("second", "POSITIONAL_OR_KEYWORD", None),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that two objects are equal via `==`.",
    ),
    (
        "assert_equal_size",
        "chex.assert_equal_size",
        "function",
        "asserts",
        [
            ("inputs", "POSITIONAL_OR_KEYWORD", None),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that all input arrays have the same total number of elements.",
    ),
    (
        "assert_not_both_none",
        "chex.assert_not_both_none",
        "function",
        "asserts",
        [
            ("first", "POSITIONAL_OR_KEYWORD", None),
            ("second", "POSITIONAL_OR_KEYWORD", None),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that at least one of `first` or `second` is not `None`.",
    ),
    (
        "assert_exactly_one_is_none",
        "chex.assert_exactly_one_is_none",
        "function",
        "asserts",
        [
            ("first", "POSITIONAL_OR_KEYWORD", None),
            ("second", "POSITIONAL_OR_KEYWORD", None),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that exactly one of `first` or `second` is `None`.",
    ),
    (
        "assert_is_divisible",
        "chex.assert_is_divisible",
        "function",
        "asserts",
        [
            ("numerator", "POSITIONAL_OR_KEYWORD", None),
            ("denominator", "POSITIONAL_OR_KEYWORD", None),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that `numerator` is evenly divisible by `denominator`.",
    ),
    (
        "assert_numerical_grads",
        "chex.assert_numerical_grads",
        "function",
        "asserts",
        [
            ("f", "POSITIONAL_OR_KEYWORD", None),
            ("f_args", "POSITIONAL_OR_KEYWORD", None),
            ("order", "POSITIONAL_OR_KEYWORD", "1"),
            ("atol", "POSITIONAL_OR_KEYWORD", "0.01"),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
            ("seed", "KEYWORD_ONLY", "None"),
        ],
        "Assert that analytical JAX gradients of `f` match finite-difference numerical gradients.",
    ),
    # PyTree Assertions
    (
        "assert_trees_all_close",
        "chex.assert_trees_all_close",
        "function",
        "asserts",
        [
            ("trees", "VAR_POSITIONAL", None),
            ("rtol", "KEYWORD_ONLY", "1e-06"),
            ("atol", "KEYWORD_ONLY", "0.0"),
            ("ignore_nans", "KEYWORD_ONLY", "False"),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
            ("strict", "KEYWORD_ONLY", "False"),
        ],
        "Assert that all passed PyTrees have identical structure and numerically close leaves.",
    ),
    (
        "assert_trees_all_close_ulp",
        "chex.assert_trees_all_close_ulp",
        "function",
        "asserts",
        [
            ("trees", "VAR_POSITIONAL", None),
            ("maxulp", "KEYWORD_ONLY", "1"),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that all passed PyTrees have leaves within `maxulp` units in the last place.",
    ),
    (
        "assert_trees_all_equal",
        "chex.assert_trees_all_equal",
        "function",
        "asserts",
        [
            ("trees", "VAR_POSITIONAL", None),
            ("ignore_nans", "KEYWORD_ONLY", "False"),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
            ("strict", "KEYWORD_ONLY", "False"),
        ],
        "Assert that all passed PyTrees have identical structure and exactly equal leaves.",
    ),
    (
        "assert_trees_all_equal_dtypes",
        "chex.assert_trees_all_equal_dtypes",
        "function",
        "asserts",
        [
            ("trees", "VAR_POSITIONAL", None),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that all passed PyTrees have leaves with identical data types.",
    ),
    (
        "assert_trees_all_equal_shapes",
        "chex.assert_trees_all_equal_shapes",
        "function",
        "asserts",
        [
            ("trees", "VAR_POSITIONAL", None),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that all passed PyTrees have leaves with identical shapes.",
    ),
    (
        "assert_trees_all_equal_shapes_and_dtypes",
        "chex.assert_trees_all_equal_shapes_and_dtypes",
        "function",
        "asserts",
        [
            ("trees", "VAR_POSITIONAL", None),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that all passed PyTrees have leaves with identical shapes and dtypes.",
    ),
    (
        "assert_trees_all_equal_sizes",
        "chex.assert_trees_all_equal_sizes",
        "function",
        "asserts",
        [
            ("trees", "VAR_POSITIONAL", None),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that all passed PyTrees have leaves with identical element counts.",
    ),
    (
        "assert_trees_all_equal_structs",
        "chex.assert_trees_all_equal_structs",
        "function",
        "asserts",
        [
            ("trees", "VAR_POSITIONAL", None),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that all passed PyTrees have the exact same tree definition structure.",
    ),
    (
        "assert_tree_all_finite",
        "chex.assert_tree_all_finite",
        "function",
        "asserts",
        [
            ("tree_like", "POSITIONAL_OR_KEYWORD", None),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that all leaves in a PyTree contain only finite values (no NaN or Inf).",
    ),
    (
        "assert_tree_no_nans",
        "chex.assert_tree_no_nans",
        "function",
        "asserts",
        [
            ("tree_like", "POSITIONAL_OR_KEYWORD", None),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that no leaf in a PyTree contains NaN values.",
    ),
    (
        "assert_tree_is_on_host",
        "chex.assert_tree_is_on_host",
        "function",
        "asserts",
        [
            ("tree_like", "POSITIONAL_OR_KEYWORD", None),
            ("allow_cpu_device", "KEYWORD_ONLY", "True"),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that all leaves in a PyTree reside in host memory.",
    ),
    (
        "assert_tree_is_on_device",
        "chex.assert_tree_is_on_device",
        "function",
        "asserts",
        [
            ("tree_like", "POSITIONAL_OR_KEYWORD", None),
            ("platform", "KEYWORD_ONLY", "None"),
            ("device", "KEYWORD_ONLY", "None"),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that all leaves in a PyTree reside on the specified accelerator device/platform.",
    ),
    (
        "assert_tree_has_only_ndarrays",
        "chex.assert_tree_has_only_ndarrays",
        "function",
        "asserts",
        [
            ("tree_like", "POSITIONAL_OR_KEYWORD", None),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that all leaves in a PyTree are NumPy or JAX ndarrays.",
    ),
    (
        "assert_tree_shape_prefix",
        "chex.assert_tree_shape_prefix",
        "function",
        "asserts",
        [
            ("tree_like", "POSITIONAL_OR_KEYWORD", None),
            ("shape_prefix", "POSITIONAL_OR_KEYWORD", None),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that all leaves in a PyTree share the specified leading shape prefix.",
    ),
    (
        "assert_tree_shape_suffix",
        "chex.assert_tree_shape_suffix",
        "function",
        "asserts",
        [
            ("tree_like", "POSITIONAL_OR_KEYWORD", None),
            ("shape_suffix", "POSITIONAL_OR_KEYWORD", None),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that all leaves in a PyTree share the specified trailing shape suffix.",
    ),
    # Device, Trace & Backend Controls
    (
        "assert_devices_available",
        "chex.assert_devices_available",
        "function",
        "asserts",
        [
            ("n", "POSITIONAL_OR_KEYWORD", None),
            ("devtype", "POSITIONAL_OR_KEYWORD", None),
            ("backend", "POSITIONAL_OR_KEYWORD", "None"),
            ("not_less_than", "POSITIONAL_OR_KEYWORD", "False"),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that `n` JAX devices of type `devtype` are available.",
    ),
    (
        "assert_tpu_available",
        "chex.assert_tpu_available",
        "function",
        "asserts",
        [
            ("backend", "POSITIONAL_OR_KEYWORD", "None"),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that at least one TPU device is available.",
    ),
    (
        "assert_gpu_available",
        "chex.assert_gpu_available",
        "function",
        "asserts",
        [
            ("backend", "POSITIONAL_OR_KEYWORD", "None"),
            ("custom_message", "KEYWORD_ONLY", "None"),
            ("include_default_message", "KEYWORD_ONLY", "True"),
        ],
        "Assert that at least one GPU device is available.",
    ),
    (
        "assert_max_traces",
        "chex.assert_max_traces",
        "function",
        "asserts",
        [
            ("fn", "POSITIONAL_OR_KEYWORD", "None"),
            ("n", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Decorator asserting that a JIT-compiled function is traced at most `n` times.",
    ),
    (
        "clear_trace_counter",
        "chex.clear_trace_counter",
        "function",
        "asserts",
        [],
        "Reset the global `@chex.assert_max_traces` trace counter to zero.",
    ),
    (
        "disable_asserts",
        "chex.disable_asserts",
        "function",
        "asserts",
        [],
        "Context manager that temporarily disables all Chex assertions.",
    ),
    (
        "if_args_not_none",
        "chex.if_args_not_none",
        "function",
        "asserts",
        [
            ("fn", "POSITIONAL_OR_KEYWORD", None),
            ("args", "VAR_POSITIONAL", None),
        ],
        "Execute assertion `fn(*args)` only when none of `args` are `None`.",
    ),
    (
        "ify",
        "chex.ify",
        "function",
        "asserts",
        [
            ("fn", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Wrap a Chex assertion function for `jax.experimental.checkify` compatibility.",
    ),
    (
        "restrict_backends",
        "chex.restrict_backends",
        "function",
        "asserts",
        [
            ("whitelist", "POSITIONAL_OR_KEYWORD", "None"),
            ("blacklist", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Context manager restricting JAX compilation to whitelisted/non-blacklisted backends.",
    ),
    # JAX PyTree Dataclasses (dataclass)
    (
        "dataclass",
        "chex.dataclass",
        "function",
        "dataclass",
        [
            ("clz", "POSITIONAL_OR_KEYWORD", "None"),
            ("mappable_dataclass", "KEYWORD_ONLY", "True"),
            ("frozen", "KEYWORD_ONLY", "False"),
            ("kw_only", "KEYWORD_ONLY", "False"),
            ("match_args", "KEYWORD_ONLY", "True"),
            ("slots", "KEYWORD_ONLY", "False"),
        ],
        "Transform a class into a Python dataclass registered as a JAX PyTree node.",
    ),
    (
        "mappable_dataclass",
        "chex.mappable_dataclass",
        "function",
        "dataclass",
        [
            ("clz", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Expose `collections.abc.Mapping` methods on a dataclass for dictionary-style access.",
    ),
    (
        "register_dataclass_type_with_jax_tree_util",
        "chex.register_dataclass_type_with_jax_tree_util",
        "function",
        "dataclass",
        [
            ("data_clz", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Register an existing dataclass type with `jax.tree_util.register_pytree_node`.",
    ),
    # Fake JIT, Fake pmap & Multi-Device CPU Simulation (fake)
    (
        "fake_jit",
        "chex.fake_jit",
        "function",
        "fake",
        [
            ("enable_patching", "POSITIONAL_OR_KEYWORD", "True"),
        ],
        "Context manager that replaces `jax.jit` with a no-op identity wrapper for debugging.",
    ),
    (
        "fake_pmap",
        "chex.fake_pmap",
        "function",
        "fake",
        [
            ("enable_patching", "POSITIONAL_OR_KEYWORD", "True"),
            ("ignore_unmapped_dims", "POSITIONAL_OR_KEYWORD", "False"),
        ],
        "Context manager that replaces `jax.pmap` with `jax.vmap` for single-device debugging.",
    ),
    (
        "fake_pmap_and_jit",
        "chex.fake_pmap_and_jit",
        "function",
        "fake",
        [
            ("enable_patching_jit", "POSITIONAL_OR_KEYWORD", "True"),
            ("enable_patching_pmap", "POSITIONAL_OR_KEYWORD", "True"),
        ],
        "Context manager that simultaneously patches both `jax.pmap` and `jax.jit`.",
    ),
    (
        "set_n_cpu_devices",
        "chex.set_n_cpu_devices",
        "function",
        "fake",
        [
            ("n", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Configure XLA flags to force `n` virtual CPU devices for multi-device testing.",
    ),
    # Parameterized JAX Test Variants (variants)
    (
        "TestCase",
        "chex.TestCase",
        "class",
        "variants",
        [
            ("methodName", "POSITIONAL_OR_KEYWORD", "'runTest'"),
        ],
        "Parameterized `absltest.TestCase` subclass supporting `@chex.variants`.",
    ),
    (
        "variants",
        "chex.variants",
        "function",
        "variants",
        [
            ("params_list", "VAR_POSITIONAL", None),
        ],
        "Decorator running a test method across specified combinations of `with_jit`, `without_jit`, `with_pmap`, and `with_device`.",
    ),
    (
        "all_variants",
        "chex.all_variants",
        "function",
        "variants",
        [
            ("with_jit", "KEYWORD_ONLY", "True"),
            ("with_pmap", "KEYWORD_ONLY", "True"),
            ("with_device", "KEYWORD_ONLY", "True"),
            ("without_jit", "KEYWORD_ONLY", "True"),
            ("without_device", "KEYWORD_ONLY", "True"),
        ],
        "Decorator running a test method across all enabled JAX execution variants.",
    ),
    # Dimensions & Deprecation Warnings (dimensions, warnings)
    (
        "Dimensions",
        "chex.Dimensions",
        "class",
        "dimensions",
        [
            ("dim_sizes", "VAR_KEYWORD", None),
        ],
        "Symbolic named dimension mapping for concise shape pattern assertions.",
    ),
    (
        "warn_deprecated_function",
        "chex.warn_deprecated_function",
        "function",
        "warnings",
        [
            ("fun", "POSITIONAL_OR_KEYWORD", None),
            ("replacement", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Decorator emitting a deprecation warning pointing to `replacement`.",
    ),
    (
        "warn_only_n_pos_args_in_future",
        "chex.warn_only_n_pos_args_in_future",
        "function",
        "warnings",
        [
            ("fun", "POSITIONAL_OR_KEYWORD", None),
            ("n", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Decorator warning when more than `n` positional arguments are passed.",
    ),
    (
        "warn_keyword_args_only_in_future",
        "chex.warn_keyword_args_only_in_future",
        "function",
        "warnings",
        [
            ("fun", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Decorator warning when positional arguments are passed to a future keyword-only function.",
    ),
    (
        "create_deprecated_function_alias",
        "chex.create_deprecated_function_alias",
        "function",
        "warnings",
        [
            ("fun", "POSITIONAL_OR_KEYWORD", None),
            ("name", "POSITIONAL_OR_KEYWORD", None),
            ("path", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Create a deprecated alias wrapper for `fun` at `path.name`.",
    ),
]


def _build_chex_canonical_ops(
    defs: typing.Sequence[
        tuple[
            str,
            str,
            str,
            str,
            list[tuple[str, str, str | None]],
            str,
        ]
    ],
) -> list[CanonicalSubmoduleOpSpec]:
    """Convert compact Chex operation tuples into `CanonicalSubmoduleOpSpec` models.

    Args:
        defs: Sequence of `(name, api_path, kind, submodule, params, docstring)` tuples.

    Returns:
        List of validated `CanonicalSubmoduleOpSpec` models.
    """
    return [
        CanonicalSubmoduleOpSpec(
            name=name,
            api_path=api_path,
            kind=kind,
            submodule=submodule,
            params=[
                CanonicalParamSpec(name=pn, kind=pk, default=pd)
                for pn, pk, pd in params
            ],
            docstring=doc,
        )
        for name, api_path, kind, submodule, params, doc in defs
    ]


CANONICAL_CHEX_OPS: list[CanonicalSubmoduleOpSpec] = _build_chex_canonical_ops(
    _CHEX_OP_DEFS
)


def _get_chex_module() -> types.ModuleType | None:
    """Safely import the runtime `chex` module when not in offline mode.

    Returns:
        Imported `chex` module or `None` if unavailable or in offline mode.
    """
    if is_offline_mode():
        return None
    try:
        return importlib.import_module("chex")
    except Exception:
        return None


def _get_chex_griffe_module() -> GriffeCacheEntry:
    """Statically load `chex` via the custom Griffe fork when not in offline mode.

    Returns:
        Loaded Griffe AST node for `chex` or `None` if unavailable or in offline mode.
    """
    if is_offline_mode():
        return None
    return load_griffe_module_safely("chex", framework_name="chex")


def collect_chex_api(include_nonpublic: bool = False) -> list[GhostRef]:
    """Collect Chex assertion, dataclass, fake context, and test variant APIs.

    Args:
        include_nonpublic: Whether to include underscore-prefixed private members.

    Returns:
        List of `GhostRef` items representing the Chex API surface.
    """
    env_tags = ("chex", "jax", "testing", "assertions")
    results: list[GhostRef] = []
    seen_paths: set[str] = set()

    if not is_offline_mode():
        live_mod = _get_chex_module()
        if live_mod is not None:
            for ref in collect_dir_callables(
                live_mod,
                api_prefix="chex",
                include_nonpublic=include_nonpublic,
                inspector_fn=GhostInspector.inspect,
            ):
                ref.environment_tags = list(ref.environment_tags or ()) + [
                    t for t in env_tags if t not in (ref.environment_tags or ())
                ]
                results.append(ref)
                seen_paths.add(ref.api_path)

        if _get_chex_griffe_module() is not None:
            for ref in scan_griffe_module_members(
                module_path="chex",
                api_prefix="chex",
                include_nonpublic=include_nonpublic,
                environment_tags=env_tags,
                framework_name="chex",
                inspector_fn=GhostInspector.inspect,
            ):
                if ref.api_path not in seen_paths:
                    results.append(ref)
                    seen_paths.add(ref.api_path)

    for spec in CANONICAL_CHEX_OPS:
        if spec.api_path not in seen_paths:
            results.append(
                build_canonical_python_ref(
                    spec,
                    default_doc_prefix="Chex",
                    environment_tags=env_tags,
                    domain_metadata={"submodule": spec.submodule or "asserts"},
                )
            )
            seen_paths.add(spec.api_path)

    return results


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> list[GhostRef]:
    """Collect Chex API definitions for the requested semantic tier.

    Args:
        category: Target `SemanticTier` category (`ARRAY_API`).
        include_nonpublic: Whether to include non-public symbols.

    Returns:
        List of `GhostRef` objects for `SemanticTier.ARRAY_API`, or `[]` otherwise.
    """
    if category == SemanticTier.ARRAY_API:
        return collect_chex_api(include_nonpublic=include_nonpublic)
    return []


__all__ = [
    "CANONICAL_CHEX_OPS",
    "_CHEX_OP_DEFS",
    "_build_chex_canonical_ops",
    "_get_chex_griffe_module",
    "_get_chex_module",
    "collect_api",
    "collect_chex_api",
]
