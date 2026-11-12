"""Canonical Optax learning-rate schedule operation definitions."""

from __future__ import annotations

from ml_ecosystem_snapshots.frameworks._base import (
    CanonicalPythonOpSpec,
    CanonicalPythonOpTuple,
    build_canonical_python_ops,
)

_OPTAX_SCHEDULER_DEFS: list[CanonicalPythonOpTuple] = [
    (
        "constant_schedule",
        "optax.schedules.constant_schedule",
        "function",
        [
            ("value", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Return a schedule callable yielding constant `value` at every step.",
    ),
    (
        "polynomial_schedule",
        "optax.schedules.polynomial_schedule",
        "function",
        [
            ("init_value", "POSITIONAL_OR_KEYWORD", None),
            ("end_value", "POSITIONAL_OR_KEYWORD", None),
            ("power", "POSITIONAL_OR_KEYWORD", None),
            ("transition_steps", "POSITIONAL_OR_KEYWORD", None),
            ("transition_begin", "POSITIONAL_OR_KEYWORD", "0"),
        ],
        "Construct a polynomial decay/growth schedule between `init_value` and `end_value`.",
    ),
    (
        "linear_schedule",
        "optax.schedules.linear_schedule",
        "function",
        [
            ("init_value", "POSITIONAL_OR_KEYWORD", None),
            ("end_value", "POSITIONAL_OR_KEYWORD", None),
            ("transition_steps", "POSITIONAL_OR_KEYWORD", None),
            ("transition_begin", "POSITIONAL_OR_KEYWORD", "0"),
        ],
        "Construct a linear interpolation schedule between `init_value` and `end_value`.",
    ),
    (
        "piecewise_constant_schedule",
        "optax.schedules.piecewise_constant_schedule",
        "function",
        [
            ("init_value", "POSITIONAL_OR_KEYWORD", None),
            ("boundaries_and_scales", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Multiply `init_value` by step-indexed scaling factors at specified step boundaries.",
    ),
    (
        "piecewise_interpolate_schedule",
        "optax.schedules.piecewise_interpolate_schedule",
        "function",
        [
            ("interpolate_type", "POSITIONAL_OR_KEYWORD", None),
            ("init_value", "POSITIONAL_OR_KEYWORD", None),
            ("boundaries_and_scales", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Interpolate (`linear` or `cosine`) between scaled values at specified step boundaries.",
    ),
    (
        "exponential_decay",
        "optax.schedules.exponential_decay",
        "function",
        [
            ("init_value", "POSITIONAL_OR_KEYWORD", None),
            ("transition_steps", "POSITIONAL_OR_KEYWORD", None),
            ("decay_rate", "POSITIONAL_OR_KEYWORD", None),
            ("transition_begin", "POSITIONAL_OR_KEYWORD", "0"),
            ("staircase", "POSITIONAL_OR_KEYWORD", "False"),
            ("end_value", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Construct an exponential decay learning rate schedule.",
    ),
    (
        "cosine_decay_schedule",
        "optax.schedules.cosine_decay_schedule",
        "function",
        [
            ("init_value", "POSITIONAL_OR_KEYWORD", None),
            ("decay_steps", "POSITIONAL_OR_KEYWORD", None),
            ("alpha", "POSITIONAL_OR_KEYWORD", "0.0"),
            ("exponent", "POSITIONAL_OR_KEYWORD", "1.0"),
        ],
        "Construct a cosine annealing decay schedule from `init_value` to `alpha * init_value`.",
    ),
    (
        "cosine_onecycle_schedule",
        "optax.schedules.cosine_onecycle_schedule",
        "function",
        [
            ("transition_steps", "POSITIONAL_OR_KEYWORD", None),
            ("peak_value", "POSITIONAL_OR_KEYWORD", None),
            ("pct_start", "POSITIONAL_OR_KEYWORD", "0.3"),
            ("div_factor", "POSITIONAL_OR_KEYWORD", "25.0"),
            ("final_div_factor", "POSITIONAL_OR_KEYWORD", "10000.0"),
        ],
        "Construct a 1cycle cosine warmup and decay schedule.",
    ),
    (
        "linear_onecycle_schedule",
        "optax.schedules.linear_onecycle_schedule",
        "function",
        [
            ("transition_steps", "POSITIONAL_OR_KEYWORD", None),
            ("peak_value", "POSITIONAL_OR_KEYWORD", None),
            ("pct_start", "POSITIONAL_OR_KEYWORD", "0.3"),
            ("pct_final", "POSITIONAL_OR_KEYWORD", "0.85"),
            ("div_factor", "POSITIONAL_OR_KEYWORD", "25.0"),
            ("final_div_factor", "POSITIONAL_OR_KEYWORD", "10000.0"),
        ],
        "Construct a three-phase linear 1cycle schedule.",
    ),
    (
        "warmup_cosine_decay_schedule",
        "optax.schedules.warmup_cosine_decay_schedule",
        "function",
        [
            ("init_value", "POSITIONAL_OR_KEYWORD", None),
            ("peak_value", "POSITIONAL_OR_KEYWORD", None),
            ("warmup_steps", "POSITIONAL_OR_KEYWORD", None),
            ("decay_steps", "POSITIONAL_OR_KEYWORD", None),
            ("end_value", "POSITIONAL_OR_KEYWORD", "0.0"),
            ("exponent", "POSITIONAL_OR_KEYWORD", "1.0"),
        ],
        "Linear warmup from `init_value` to `peak_value` followed by cosine decay to `end_value`.",
    ),
    (
        "warmup_exponential_decay_schedule",
        "optax.schedules.warmup_exponential_decay_schedule",
        "function",
        [
            ("init_value", "POSITIONAL_OR_KEYWORD", None),
            ("peak_value", "POSITIONAL_OR_KEYWORD", None),
            ("warmup_steps", "POSITIONAL_OR_KEYWORD", None),
            ("transition_steps", "POSITIONAL_OR_KEYWORD", None),
            ("decay_rate", "POSITIONAL_OR_KEYWORD", None),
            ("transition_begin", "POSITIONAL_OR_KEYWORD", "0"),
            ("staircase", "POSITIONAL_OR_KEYWORD", "False"),
            ("end_value", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Linear warmup followed by exponential decay.",
    ),
    (
        "warmup_constant_schedule",
        "optax.schedules.warmup_constant_schedule",
        "function",
        [
            ("init_value", "POSITIONAL_OR_KEYWORD", None),
            ("peak_value", "POSITIONAL_OR_KEYWORD", None),
            ("warmup_steps", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Linear warmup from `init_value` to `peak_value` followed by constant `peak_value`.",
    ),
    (
        "sgdr_schedule",
        "optax.schedules.sgdr_schedule",
        "function",
        [
            ("cosine_kwargs", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Stochastic Gradient Descent with Warm Restarts (SGDR) periodic cosine schedule.",
    ),
    (
        "join_schedules",
        "optax.schedules.join_schedules",
        "function",
        [
            ("schedules", "POSITIONAL_OR_KEYWORD", None),
            ("boundaries", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Piecewise-join a sequence of schedule callables at step `boundaries`.",
    ),
    (
        "inject_hyperparams",
        "optax.schedules.inject_hyperparams",
        "function",
        [
            ("inner_factory", "POSITIONAL_OR_KEYWORD", None),
            ("static_args", "POSITIONAL_OR_KEYWORD", "()"),
            ("hyperparam_dtype", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Wrap an optimizer factory so scheduled hyperparameters are tracked in optimizer state.",
    ),
]

CANONICAL_OPTAX_SCHEDULERS: list[CanonicalPythonOpSpec] = build_canonical_python_ops(_OPTAX_SCHEDULER_DEFS)

__all__ = ["CANONICAL_OPTAX_SCHEDULERS"]
