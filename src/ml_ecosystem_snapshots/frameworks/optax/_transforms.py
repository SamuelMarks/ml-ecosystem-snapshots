"""Canonical Optax composable gradient transformation and wrapper definitions."""

from __future__ import annotations

from ml_ecosystem_snapshots.frameworks._base import (
    CanonicalPythonOpSpec,
    CanonicalPythonOpTuple,
    build_canonical_python_ops,
)
from ml_ecosystem_snapshots.frameworks.optax._optimizers import OPTAX_CORE_OPTIMIZER_DEFS

OPTAX_TRANSFORM_DEFS: list[CanonicalPythonOpTuple] = [
    # Composable Gradient Transformations & Wrappers
    (
        "GradientTransformation",
        "optax.GradientTransformation",
        "class",
        [
            ("init", "POSITIONAL_OR_KEYWORD", None),
            ("update", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Structured pair of pure `(init, update)` functions defining an Optax gradient transformation.",
    ),
    (
        "GradientTransformationExtraArgs",
        "optax.GradientTransformationExtraArgs",
        "class",
        [
            ("init", "POSITIONAL_OR_KEYWORD", None),
            ("update", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Extended `GradientTransformation` accepting extra keyword arguments in `update`.",
    ),
    (
        "chain",
        "optax.chain",
        "function",
        [
            ("args", "VAR_POSITIONAL", None),
        ],
        "Chain a sequence of `GradientTransformation` instances sequentially.",
    ),
    (
        "named_chain",
        "optax.named_chain",
        "function",
        [
            ("transforms", "VAR_POSITIONAL", None),
        ],
        "Chain `(name, GradientTransformation)` pairs into a composite transformation with named state.",
    ),
    (
        "partition",
        "optax.partition",
        "function",
        [
            ("transforms", "POSITIONAL_OR_KEYWORD", None),
            ("param_labels", "POSITIONAL_OR_KEYWORD", None),
            ("mask_compatible_extra_args", "KEYWORD_ONLY", "False"),
        ],
        "Route distinct parameter subtrees to separate gradient transformations using `param_labels`.",
    ),
    (
        "multi_transform",
        "optax.multi_transform",
        "function",
        [
            ("transforms", "POSITIONAL_OR_KEYWORD", None),
            ("param_labels", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Apply different gradient transformations to labeled partitions of a parameter PyTree.",
    ),
    (
        "MultiSteps",
        "optax.MultiSteps",
        "class",
        [
            ("opt", "POSITIONAL_OR_KEYWORD", None),
            ("every_k_schedule", "POSITIONAL_OR_KEYWORD", None),
            ("use_grad_mean", "POSITIONAL_OR_KEYWORD", "True"),
            ("should_skip_update_fn", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Gradient accumulation wrapper accumulating micro-step gradients over `every_k_schedule` steps.",
    ),
    (
        "Lookahead",
        "optax.Lookahead",
        "class",
        [
            ("fast_optimizer", "POSITIONAL_OR_KEYWORD", None),
            ("sync_period", "POSITIONAL_OR_KEYWORD", None),
            ("slow_step_size", "POSITIONAL_OR_KEYWORD", None),
            ("reset_state", "POSITIONAL_OR_KEYWORD", "False"),
        ],
        "Lookahead meta-optimizer synchronizing fast and slow parameter weights every `sync_period` steps.",
    ),
    (
        "clip",
        "optax.clip",
        "function",
        [
            ("max_delta", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Clip gradient elements element-wise to `[-max_delta, +max_delta]`.",
    ),
    (
        "clip_by_global_norm",
        "optax.clip_by_global_norm",
        "function",
        [
            ("max_norm", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Clip gradients whenever their global L2 norm across the PyTree exceeds `max_norm`.",
    ),
    (
        "clip_by_block_rms",
        "optax.clip_by_block_rms",
        "function",
        [
            ("threshold", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Clip gradients per parameter leaf when leaf root-mean-square exceeds `threshold`.",
    ),
    (
        "adaptive_grad_clip",
        "optax.adaptive_grad_clip",
        "function",
        [
            ("clipping", "POSITIONAL_OR_KEYWORD", None),
            ("eps", "POSITIONAL_OR_KEYWORD", "0.001"),
        ],
        "Adaptive Gradient Clipping (AGC) clipping unit-wise gradient norms relative to parameter norms.",
    ),
    (
        "per_example_global_norm_clip",
        "optax.per_example_global_norm_clip",
        "function",
        [
            ("grads", "POSITIONAL_OR_KEYWORD", None),
            ("l2_norm_clip", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Clip per-example gradients by their global L2 norm for differential privacy.",
    ),
    (
        "centralize",
        "optax.centralize",
        "function",
        [],
        "Centralize gradients by subtracting their mean across non-batch axes.",
    ),
    (
        "scale",
        "optax.scale",
        "function",
        [
            ("step_size", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Multiply all updates in a PyTree by a scalar `step_size`.",
    ),
    (
        "scale_by_adam",
        "optax.scale_by_adam",
        "function",
        [
            ("b1", "POSITIONAL_OR_KEYWORD", "0.9"),
            ("b2", "POSITIONAL_OR_KEYWORD", "0.999"),
            ("eps", "POSITIONAL_OR_KEYWORD", "1e-08"),
            ("eps_root", "POSITIONAL_OR_KEYWORD", "0.0"),
            ("mu_dtype", "POSITIONAL_OR_KEYWORD", "None"),
            ("nesterov", "KEYWORD_ONLY", "False"),
        ],
        "Rescale updates according to the Adam first- and second-moment rule.",
    ),
    (
        "scale_by_amsgrad",
        "optax.scale_by_amsgrad",
        "function",
        [
            ("b1", "POSITIONAL_OR_KEYWORD", "0.9"),
            ("b2", "POSITIONAL_OR_KEYWORD", "0.999"),
            ("eps", "POSITIONAL_OR_KEYWORD", "1e-08"),
            ("eps_root", "POSITIONAL_OR_KEYWORD", "0.0"),
            ("mu_dtype", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Rescale updates according to the AMSGrad algorithm.",
    ),
    (
        "scale_by_belief",
        "optax.scale_by_belief",
        "function",
        [
            ("b1", "POSITIONAL_OR_KEYWORD", "0.9"),
            ("b2", "POSITIONAL_OR_KEYWORD", "0.999"),
            ("eps", "POSITIONAL_OR_KEYWORD", "1e-16"),
            ("eps_root", "POSITIONAL_OR_KEYWORD", "1e-16"),
            ("nesterov", "KEYWORD_ONLY", "False"),
        ],
        "Rescale updates according to the AdaBelief algorithm.",
    ),
    (
        "scale_by_lion",
        "optax.scale_by_lion",
        "function",
        [
            ("b1", "POSITIONAL_OR_KEYWORD", "0.9"),
            ("b2", "POSITIONAL_OR_KEYWORD", "0.99"),
            ("mu_dtype", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Rescale updates according to the Lion sign-momentum rule.",
    ),
    (
        "scale_by_learning_rate",
        "optax.scale_by_learning_rate",
        "function",
        [
            ("learning_rate", "POSITIONAL_OR_KEYWORD", None),
            ("flip_sign", "POSITIONAL_OR_KEYWORD", "True"),
        ],
        "Scale updates by a scalar or scheduled `learning_rate` (negating when `flip_sign=True`).",
    ),
    (
        "scale_by_schedule",
        "optax.scale_by_schedule",
        "function",
        [
            ("step_size_fn", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Scale updates using a callable `step_size_fn(step_count)` schedule.",
    ),
    (
        "scale_by_rms",
        "optax.scale_by_rms",
        "function",
        [
            ("decay", "POSITIONAL_OR_KEYWORD", "0.9"),
            ("eps", "POSITIONAL_OR_KEYWORD", "1e-08"),
            ("initial_scale", "POSITIONAL_OR_KEYWORD", "0.0"),
            ("bias_correction", "POSITIONAL_OR_KEYWORD", "False"),
        ],
        "Scale updates by the root-mean-square of the exponential moving average of squared gradients.",
    ),
    (
        "scale_by_stddev",
        "optax.scale_by_stddev",
        "function",
        [
            ("decay", "POSITIONAL_OR_KEYWORD", "0.9"),
            ("eps", "POSITIONAL_OR_KEYWORD", "1e-08"),
            ("initial_scale", "POSITIONAL_OR_KEYWORD", "0.0"),
        ],
        "Scale updates by the centered standard deviation of the gradient moving average.",
    ),
    (
        "scale_by_trust_ratio",
        "optax.scale_by_trust_ratio",
        "function",
        [
            ("min_norm", "POSITIONAL_OR_KEYWORD", "0.0"),
            ("trust_coefficient", "POSITIONAL_OR_KEYWORD", "1.0"),
            ("eps", "POSITIONAL_OR_KEYWORD", "0.0"),
        ],
        "Scale updates by the layer-wise ratio of parameter norm to update norm (LAMB/LARS).",
    ),
    (
        "scale_by_param_block_norm",
        "optax.scale_by_param_block_norm",
        "function",
        [
            ("min_scale", "POSITIONAL_OR_KEYWORD", "0.001"),
        ],
        "Scale updates for each parameter block by the L2 norm of that block.",
    ),
    (
        "scale_by_param_block_rms",
        "optax.scale_by_param_block_rms",
        "function",
        [
            ("min_scale", "POSITIONAL_OR_KEYWORD", "0.001"),
        ],
        "Scale updates for each parameter block by the root-mean-square of that block.",
    ),
    (
        "scale_by_backtracking_linesearch",
        "optax.scale_by_backtracking_linesearch",
        "function",
        [
            ("max_backtracking_steps", "POSITIONAL_OR_KEYWORD", None),
            ("slope_rtol", "POSITIONAL_OR_KEYWORD", "0.0001"),
            ("decrease_factor", "POSITIONAL_OR_KEYWORD", "0.8"),
            ("increase_factor", "POSITIONAL_OR_KEYWORD", "1.5"),
            ("max_learning_rate", "POSITIONAL_OR_KEYWORD", "1.0"),
            ("store_grad", "POSITIONAL_OR_KEYWORD", "False"),
        ],
        "Scale updates using Armijo backtracking line search.",
    ),
    (
        "scale_by_zoom_linesearch",
        "optax.scale_by_zoom_linesearch",
        "function",
        [
            ("max_linesearch_steps", "POSITIONAL_OR_KEYWORD", None),
            ("slope_rtol", "POSITIONAL_OR_KEYWORD", "0.0001"),
            ("curv_rtol", "POSITIONAL_OR_KEYWORD", "0.9"),
            ("tol", "POSITIONAL_OR_KEYWORD", "0.0"),
            ("max_learning_rate", "POSITIONAL_OR_KEYWORD", "None"),
            ("initial_guess_strategy", "POSITIONAL_OR_KEYWORD", "None"),
            ("store_grad", "POSITIONAL_OR_KEYWORD", "False"),
        ],
        "Scale updates using strong Wolfe zoom line search (commonly paired with `lbfgs`).",
    ),
    (
        "trace",
        "optax.trace",
        "function",
        [
            ("decay", "POSITIONAL_OR_KEYWORD", None),
            ("nesterov", "POSITIONAL_OR_KEYWORD", "False"),
            ("accumulator_dtype", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Compute Classical or Nesterov momentum trace of past updates.",
    ),
    (
        "ema",
        "optax.ema",
        "function",
        [
            ("decay", "POSITIONAL_OR_KEYWORD", None),
            ("debias", "POSITIONAL_OR_KEYWORD", "True"),
            ("accumulator_dtype", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Compute an exponential moving average (EMA) of past updates.",
    ),
    (
        "add_decayed_weights",
        "optax.add_decayed_weights",
        "function",
        [
            ("weight_decay", "POSITIONAL_OR_KEYWORD", "0.0"),
            ("mask", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Add decoupled weight decay (`weight_decay * params`) to updates.",
    ),
    (
        "add_noise",
        "optax.add_noise",
        "function",
        [
            ("eta", "POSITIONAL_OR_KEYWORD", None),
            ("gamma", "POSITIONAL_OR_KEYWORD", None),
            ("seed", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Add annealed Gaussian noise to updates.",
    ),
    (
        "masked",
        "optax.masked",
        "function",
        [
            ("inner", "POSITIONAL_OR_KEYWORD", None),
            ("mask", "POSITIONAL_OR_KEYWORD", None),
            ("mask_compatible_extra_args", "KEYWORD_ONLY", "False"),
        ],
        "Apply `inner` gradient transformation only to PyTree leaves where `mask` is True.",
    ),
    (
        "flatten",
        "optax.flatten",
        "function",
        [
            ("inner", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Flatten all parameters and gradients into a single 1D vector before applying `inner`.",
    ),
    (
        "set_to_zero",
        "optax.set_to_zero",
        "function",
        [],
        "Stateless transformation replacing all updates with zeros (freezing parameters).",
    ),
    (
        "stateless",
        "optax.stateless",
        "function",
        [
            ("update", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Wrap a stateless `update(updates, params)` function into a `GradientTransformation`.",
    ),
    (
        "zero_nans",
        "optax.zero_nans",
        "function",
        [],
        "Replace any NaN values in gradient updates with zeros.",
    ),
    (
        "keep_params_nonnegative",
        "optax.keep_params_nonnegative",
        "function",
        [],
        "Clamp updates so `params + updates >= 0` everywhere.",
    ),
    (
        "freeze",
        "optax.freeze",
        "function",
        [
            ("mask", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Zero out updates for parameters selected by `mask`.",
    ),
]

_OPTAX_OPTIMIZER_DEFS: list[CanonicalPythonOpTuple] = OPTAX_CORE_OPTIMIZER_DEFS + OPTAX_TRANSFORM_DEFS
CANONICAL_OPTAX_OPTIMIZERS: list[CanonicalPythonOpSpec] = build_canonical_python_ops(_OPTAX_OPTIMIZER_DEFS)
KNOWN_OPTAX_OPTIMIZERS: set[str] = {spec.name.lower() for spec in CANONICAL_OPTAX_OPTIMIZERS}

__all__ = [
    "CANONICAL_OPTAX_OPTIMIZERS",
    "KNOWN_OPTAX_OPTIMIZERS",
    "OPTAX_TRANSFORM_DEFS",
]
