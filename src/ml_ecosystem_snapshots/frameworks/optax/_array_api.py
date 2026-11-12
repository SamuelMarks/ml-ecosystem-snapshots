"""Canonical Optax projection, tree-math, linear-algebra, and perturbation definitions."""

from __future__ import annotations

from ml_ecosystem_snapshots.frameworks._base import (
    CanonicalPythonOpSpec,
    CanonicalPythonOpTuple,
    build_canonical_python_ops,
)

_OPTAX_ARRAY_API_DEFS: list[CanonicalPythonOpTuple] = [
    (
        "apply_updates",
        "optax.apply_updates",
        "function",
        [
            ("params", "POSITIONAL_OR_KEYWORD", None),
            ("updates", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Apply transformed gradient `updates` to `params` by computing `params + updates` leaf-wise.",
    ),
    (
        "incremental_update",
        "optax.incremental_update",
        "function",
        [
            ("new_tensors", "POSITIONAL_OR_KEYWORD", None),
            ("old_tensors", "POSITIONAL_OR_KEYWORD", None),
            ("step_size", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Polyak/EMA target network update `step_size * new_tensors + (1 - step_size) * old_tensors`.",
    ),
    (
        "periodic_update",
        "optax.periodic_update",
        "function",
        [
            ("new_tensors", "POSITIONAL_OR_KEYWORD", None),
            ("old_tensors", "POSITIONAL_OR_KEYWORD", None),
            ("steps", "POSITIONAL_OR_KEYWORD", None),
            ("update_period", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Periodically copy `new_tensors` into `old_tensors` every `update_period` steps.",
    ),
    (
        "global_norm",
        "optax.global_norm",
        "function",
        [
            ("updates", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Compute the global L2 norm across all leaves in `updates`.",
    ),
    (
        "value_and_grad_from_state",
        "optax.value_and_grad_from_state",
        "function",
        [
            ("fun", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Decorator reusing cached value and gradient stored in line-search optimizer state when available.",
    ),
    # Projections (optax.projections)
    (
        "projection_non_negative",
        "optax.projections.projection_non_negative",
        "function",
        [
            ("tree", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Euclidean projection onto the non-negative orthant `x >= 0`.",
    ),
    (
        "projection_box",
        "optax.projections.projection_box",
        "function",
        [
            ("tree", "POSITIONAL_OR_KEYWORD", None),
            ("lower", "POSITIONAL_OR_KEYWORD", None),
            ("upper", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Euclidean projection onto box constraints `[lower, upper]`.",
    ),
    (
        "projection_hypercube",
        "optax.projections.projection_hypercube",
        "function",
        [
            ("tree", "POSITIONAL_OR_KEYWORD", None),
            ("scale", "POSITIONAL_OR_KEYWORD", "1.0"),
        ],
        "Euclidean projection onto the `[0, scale]^n` unit hypercube.",
    ),
    (
        "projection_l1_sphere",
        "optax.projections.projection_l1_sphere",
        "function",
        [
            ("tree", "POSITIONAL_OR_KEYWORD", None),
            ("scale", "POSITIONAL_OR_KEYWORD", "1.0"),
        ],
        "Euclidean projection onto the L1 sphere `||x||_1 == scale`.",
    ),
    (
        "projection_l1_ball",
        "optax.projections.projection_l1_ball",
        "function",
        [
            ("tree", "POSITIONAL_OR_KEYWORD", None),
            ("scale", "POSITIONAL_OR_KEYWORD", "1.0"),
        ],
        "Euclidean projection onto the L1 ball `||x||_1 <= scale`.",
    ),
    (
        "projection_l2_sphere",
        "optax.projections.projection_l2_sphere",
        "function",
        [
            ("tree", "POSITIONAL_OR_KEYWORD", None),
            ("scale", "POSITIONAL_OR_KEYWORD", "1.0"),
        ],
        "Euclidean projection onto the L2 sphere `||x||_2 == scale`.",
    ),
    (
        "projection_l2_ball",
        "optax.projections.projection_l2_ball",
        "function",
        [
            ("tree", "POSITIONAL_OR_KEYWORD", None),
            ("scale", "POSITIONAL_OR_KEYWORD", "1.0"),
        ],
        "Euclidean projection onto the L2 ball `||x||_2 <= scale`.",
    ),
    (
        "projection_simplex",
        "optax.projections.projection_simplex",
        "function",
        [
            ("tree", "POSITIONAL_OR_KEYWORD", None),
            ("scale", "POSITIONAL_OR_KEYWORD", "1.0"),
        ],
        "Euclidean projection onto the probability simplex `{x >= 0, sum(x) == scale}`.",
    ),
    # Tree Utilities (optax.tree_utils)
    (
        "tree_add",
        "optax.tree_utils.tree_add",
        "function",
        [
            ("first", "POSITIONAL_OR_KEYWORD", None),
            ("rest", "VAR_POSITIONAL", None),
        ],
        "Add two or more PyTrees leaf-wise.",
    ),
    (
        "tree_sub",
        "optax.tree_utils.tree_sub",
        "function",
        [
            ("first", "POSITIONAL_OR_KEYWORD", None),
            ("second", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Subtract `second` PyTree from `first` leaf-wise.",
    ),
    (
        "tree_mul",
        "optax.tree_utils.tree_mul",
        "function",
        [
            ("first", "POSITIONAL_OR_KEYWORD", None),
            ("second", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Multiply two PyTrees element-wise across leaves.",
    ),
    (
        "tree_scalar_mul",
        "optax.tree_utils.tree_scalar_mul",
        "function",
        [
            ("scalar", "POSITIONAL_OR_KEYWORD", None),
            ("tree", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Multiply all leaves of `tree` by `scalar`.",
    ),
    (
        "tree_vdot",
        "optax.tree_utils.tree_vdot",
        "function",
        [
            ("first", "POSITIONAL_OR_KEYWORD", None),
            ("second", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Compute the inner product between two PyTrees.",
    ),
    (
        "tree_sum",
        "optax.tree_utils.tree_sum",
        "function",
        [
            ("tree", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Compute the sum of all elements across all leaves in `tree`.",
    ),
    (
        "tree_l1_norm",
        "optax.tree_utils.tree_l1_norm",
        "function",
        [
            ("tree", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Compute the L1 norm across all leaves in `tree`.",
    ),
    (
        "tree_l2_norm",
        "optax.tree_utils.tree_l2_norm",
        "function",
        [
            ("tree", "POSITIONAL_OR_KEYWORD", None),
            ("squared", "POSITIONAL_OR_KEYWORD", "False"),
        ],
        "Compute the L2 norm (or squared L2 norm) across all leaves in `tree`.",
    ),
    (
        "tree_zeros_like",
        "optax.tree_utils.tree_zeros_like",
        "function",
        [
            ("tree", "POSITIONAL_OR_KEYWORD", None),
            ("dtype", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Create a zero-filled PyTree with the same structure and shapes as `tree`.",
    ),
    (
        "tree_ones_like",
        "optax.tree_utils.tree_ones_like",
        "function",
        [
            ("tree", "POSITIONAL_OR_KEYWORD", None),
            ("dtype", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Create a ones-filled PyTree with the same structure and shapes as `tree`.",
    ),
    # Second-Order, Assignment & Perturbations
    (
        "hessian_diag",
        "optax.second_order.hessian_diag",
        "function",
        [
            ("fun", "POSITIONAL_OR_KEYWORD", None),
            ("params", "POSITIONAL_OR_KEYWORD", None),
            ("inputs", "POSITIONAL_OR_KEYWORD", None),
            ("targets", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Compute the diagonal of the Hessian of `fun(params, inputs, targets)`.",
    ),
    (
        "fisher_diag",
        "optax.second_order.fisher_diag",
        "function",
        [
            ("negative_log_likelihood", "POSITIONAL_OR_KEYWORD", None),
            ("params", "POSITIONAL_OR_KEYWORD", None),
            ("inputs", "POSITIONAL_OR_KEYWORD", None),
            ("targets", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Compute the empirical Fisher information diagonal.",
    ),
    (
        "hungarian_algorithm",
        "optax.assignment.hungarian_algorithm",
        "function",
        [
            ("cost_matrix", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Solve the linear sum assignment problem for `cost_matrix` using the Hungarian algorithm.",
    ),
    (
        "base_hungarian_algorithm",
        "optax.assignment.base_hungarian_algorithm",
        "function",
        [
            ("cost_matrix", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Unbatched Hungarian algorithm solver for a 2D `cost_matrix`.",
    ),
    (
        "make_perturbed_fun",
        "optax.perturbations.make_perturbed_fun",
        "function",
        [
            ("fun", "POSITIONAL_OR_KEYWORD", None),
            ("num_samples", "POSITIONAL_OR_KEYWORD", "1000"),
            ("sigma", "POSITIONAL_OR_KEYWORD", "0.1"),
            ("noise", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Transform a non-differentiable function `fun` into a smooth differentiable approximation via random input perturbations.",
    ),
]

CANONICAL_OPTAX_ARRAY_API: list[CanonicalPythonOpSpec] = build_canonical_python_ops(_OPTAX_ARRAY_API_DEFS)
KNOWN_OPTAX_ARRAY_API: set[str] = {spec.name.lower() for spec in CANONICAL_OPTAX_ARRAY_API}

__all__ = ["CANONICAL_OPTAX_ARRAY_API", "KNOWN_OPTAX_ARRAY_API"]
