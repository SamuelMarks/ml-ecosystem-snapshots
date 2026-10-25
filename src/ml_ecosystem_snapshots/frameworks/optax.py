"""Optax Scanner & Canonical API Snapshot Extractor.

Provides live introspection, static AST/docstring extraction via the custom Griffe
fork, and canonical offline specifications for Google DeepMind's Optax library
across four semantic tiers:
1. `SemanticTier.OPTIMIZER`: First-order, adaptive, and quasi-Newton optimizers plus composable gradient transformations (`chain`, `clip_by_global_norm`, `scale_by_adam`, `MultiSteps`, `Lookahead`, `ema`, etc.).
2. `SemanticTier.LOSS`: Classification, regression, ranking, contrastive, and divergence loss functions (`optax.losses.*`).
3. `SemanticTier.SCHEDULER`: Learning-rate and hyperparameter schedules (`optax.schedules.*`).
4. `SemanticTier.ARRAY_API`: Parameter update helpers (`apply_updates`, `incremental_update`, `periodic_update`, `global_norm`), projections (`optax.projections.*`), PyTree arithmetic (`optax.tree_utils.*`), second-order curvature (`optax.second_order.*`), assignment (`optax.assignment.*`), and perturbations (`optax.perturbations.*`).
"""

from __future__ import annotations

import inspect
import types
import typing

from ml_switcheroo_ir.schema.ghost import (
    GhostRef,
    SemanticTier,
)

from ml_ecosystem_snapshots.frameworks._base import (
    CanonicalParamSpec,
    CanonicalPythonOpSpec,
    build_canonical_python_ref,
    scan_griffe_module_members,
)
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.static_parsers import (
    GriffeCacheEntry,
    load_griffe_module_safely,
)
from ml_ecosystem_snapshots.network_env import is_offline_mode
from ml_ecosystem_snapshots.utils import get_all_members

optax: types.ModuleType | None = None
try:
    import optax as _optax_mod

    optax = _optax_mod
except Exception:
    pass


_OPTAX_OPTIMIZER_DEFS: list[
    tuple[
        str,
        str,
        str,
        list[tuple[str, str, str | None]],
        str,
    ]
] = [
    # First-Order, Adaptive & Quasi-Newton Optimizers
    (
        "adabelief",
        "optax.adabelief",
        "function",
        [
            ("learning_rate", "POSITIONAL_OR_KEYWORD", None),
            ("b1", "POSITIONAL_OR_KEYWORD", "0.9"),
            ("b2", "POSITIONAL_OR_KEYWORD", "0.999"),
            ("eps", "POSITIONAL_OR_KEYWORD", "1e-16"),
            ("eps_root", "POSITIONAL_OR_KEYWORD", "1e-16"),
            ("nesterov", "KEYWORD_ONLY", "False"),
        ],
        "AdaBelief optimizer adapting step size according to belief in observed gradients.",
    ),
    (
        "adafactor",
        "optax.adafactor",
        "function",
        [
            ("learning_rate", "POSITIONAL_OR_KEYWORD", "None"),
            ("min_dim_size_to_factor", "POSITIONAL_OR_KEYWORD", "128"),
            ("decay_rate", "POSITIONAL_OR_KEYWORD", "0.8"),
            ("decay_offset", "POSITIONAL_OR_KEYWORD", "0"),
            ("multiply_by_parameter_scale", "POSITIONAL_OR_KEYWORD", "True"),
            ("clipping_threshold", "POSITIONAL_OR_KEYWORD", "1.0"),
            ("momentum", "POSITIONAL_OR_KEYWORD", "None"),
            ("dtype_momentum", "POSITIONAL_OR_KEYWORD", "jnp.float32"),
            ("weight_decay_rate", "POSITIONAL_OR_KEYWORD", "None"),
            ("eps", "POSITIONAL_OR_KEYWORD", "1e-30"),
            ("factored", "POSITIONAL_OR_KEYWORD", "True"),
            ("weight_decay_mask", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Adafactor optimizer with factored second-moment estimation for sublinear memory overhead.",
    ),
    (
        "adagrad",
        "optax.adagrad",
        "function",
        [
            ("learning_rate", "POSITIONAL_OR_KEYWORD", None),
            ("initial_accumulator_value", "POSITIONAL_OR_KEYWORD", "0.1"),
            ("eps", "POSITIONAL_OR_KEYWORD", "1e-07"),
        ],
        "AdaGrad optimizer scaling learning rates inversely with accumulated squared gradients.",
    ),
    (
        "adam",
        "optax.adam",
        "function",
        [
            ("learning_rate", "POSITIONAL_OR_KEYWORD", None),
            ("b1", "POSITIONAL_OR_KEYWORD", "0.9"),
            ("b2", "POSITIONAL_OR_KEYWORD", "0.999"),
            ("eps", "POSITIONAL_OR_KEYWORD", "1e-08"),
            ("eps_root", "POSITIONAL_OR_KEYWORD", "0.0"),
            ("mu_dtype", "POSITIONAL_OR_KEYWORD", "None"),
            ("nesterov", "KEYWORD_ONLY", "False"),
        ],
        "Adam optimizer combining exponential moving averages of first and second gradient moments.",
    ),
    (
        "adamax",
        "optax.adamax",
        "function",
        [
            ("learning_rate", "POSITIONAL_OR_KEYWORD", None),
            ("b1", "POSITIONAL_OR_KEYWORD", "0.9"),
            ("b2", "POSITIONAL_OR_KEYWORD", "0.999"),
            ("eps", "POSITIONAL_OR_KEYWORD", "1e-08"),
        ],
        "AdaMax variant of Adam based on the infinity norm.",
    ),
    (
        "adamaxw",
        "optax.adamaxw",
        "function",
        [
            ("learning_rate", "POSITIONAL_OR_KEYWORD", None),
            ("b1", "POSITIONAL_OR_KEYWORD", "0.9"),
            ("b2", "POSITIONAL_OR_KEYWORD", "0.999"),
            ("eps", "POSITIONAL_OR_KEYWORD", "1e-08"),
            ("weight_decay", "POSITIONAL_OR_KEYWORD", "0.0001"),
            ("mask", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "AdaMaxW optimizer with decoupled weight decay.",
    ),
    (
        "adamw",
        "optax.adamw",
        "function",
        [
            ("learning_rate", "POSITIONAL_OR_KEYWORD", None),
            ("b1", "POSITIONAL_OR_KEYWORD", "0.9"),
            ("b2", "POSITIONAL_OR_KEYWORD", "0.999"),
            ("eps", "POSITIONAL_OR_KEYWORD", "1e-08"),
            ("eps_root", "POSITIONAL_OR_KEYWORD", "0.0"),
            ("mu_dtype", "POSITIONAL_OR_KEYWORD", "None"),
            ("weight_decay", "POSITIONAL_OR_KEYWORD", "0.0001"),
            ("mask", "POSITIONAL_OR_KEYWORD", "None"),
            ("nesterov", "KEYWORD_ONLY", "False"),
        ],
        "AdamW optimizer with decoupled weight decay regularization.",
    ),
    (
        "amsgrad",
        "optax.amsgrad",
        "function",
        [
            ("learning_rate", "POSITIONAL_OR_KEYWORD", None),
            ("b1", "POSITIONAL_OR_KEYWORD", "0.9"),
            ("b2", "POSITIONAL_OR_KEYWORD", "0.999"),
            ("eps", "POSITIONAL_OR_KEYWORD", "1e-08"),
            ("eps_root", "POSITIONAL_OR_KEYWORD", "0.0"),
            ("mu_dtype", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "AMSGrad optimizer maintaining the maximum of past squared gradient moving averages.",
    ),
    (
        "fromage",
        "optax.fromage",
        "function",
        [
            ("learning_rate", "POSITIONAL_OR_KEYWORD", None),
            ("min_norm", "POSITIONAL_OR_KEYWORD", "1e-06"),
        ],
        "Frobenius Matched Gradient Descent (Fromage) optimizer.",
    ),
    (
        "lamb",
        "optax.lamb",
        "function",
        [
            ("learning_rate", "POSITIONAL_OR_KEYWORD", None),
            ("b1", "POSITIONAL_OR_KEYWORD", "0.9"),
            ("b2", "POSITIONAL_OR_KEYWORD", "0.999"),
            ("eps", "POSITIONAL_OR_KEYWORD", "1e-06"),
            ("eps_root", "POSITIONAL_OR_KEYWORD", "0.0"),
            ("weight_decay", "POSITIONAL_OR_KEYWORD", "0.0"),
            ("mask", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "LAMB (Layer-wise Adaptive Moments for Batch training) optimizer.",
    ),
    (
        "lars",
        "optax.lars",
        "function",
        [
            ("learning_rate", "POSITIONAL_OR_KEYWORD", None),
            ("weight_decay", "POSITIONAL_OR_KEYWORD", "0.0"),
            ("weight_decay_mask", "POSITIONAL_OR_KEYWORD", "True"),
            ("trust_coefficient", "POSITIONAL_OR_KEYWORD", "0.001"),
            ("eps", "POSITIONAL_OR_KEYWORD", "0.0"),
            ("momentum", "POSITIONAL_OR_KEYWORD", "0.9"),
            ("nesterov", "POSITIONAL_OR_KEYWORD", "False"),
            ("trust_ratio_mask", "POSITIONAL_OR_KEYWORD", "True"),
        ],
        "Layer-wise Adaptive Rate Scaling (LARS) optimizer.",
    ),
    (
        "lbfgs",
        "optax.lbfgs",
        "function",
        [
            ("learning_rate", "POSITIONAL_OR_KEYWORD", "None"),
            ("memory_size", "POSITIONAL_OR_KEYWORD", "10"),
            ("scale_init_precond", "POSITIONAL_OR_KEYWORD", "True"),
            ("linesearch", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Limited-memory BFGS (L-BFGS) quasi-Newton optimizer.",
    ),
    (
        "lion",
        "optax.lion",
        "function",
        [
            ("learning_rate", "POSITIONAL_OR_KEYWORD", None),
            ("b1", "POSITIONAL_OR_KEYWORD", "0.9"),
            ("b2", "POSITIONAL_OR_KEYWORD", "0.99"),
            ("mu_dtype", "POSITIONAL_OR_KEYWORD", "None"),
            ("weight_decay", "POSITIONAL_OR_KEYWORD", "0.0"),
            ("mask", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Lion (EvoLved Sign Momentum) optimizer using sign updates on interpolated momentum.",
    ),
    (
        "nadagrad",
        "optax.nadagrad",
        "function",
        [
            ("learning_rate", "POSITIONAL_OR_KEYWORD", None),
            ("initial_accumulator_value", "POSITIONAL_OR_KEYWORD", "0.1"),
            ("eps", "POSITIONAL_OR_KEYWORD", "1e-07"),
            ("momentum", "POSITIONAL_OR_KEYWORD", "0.9"),
        ],
        "Nesterov-accelerated AdaGrad optimizer.",
    ),
    (
        "nadam",
        "optax.nadam",
        "function",
        [
            ("learning_rate", "POSITIONAL_OR_KEYWORD", None),
            ("b1", "POSITIONAL_OR_KEYWORD", "0.9"),
            ("b2", "POSITIONAL_OR_KEYWORD", "0.999"),
            ("eps", "POSITIONAL_OR_KEYWORD", "1e-08"),
            ("eps_root", "POSITIONAL_OR_KEYWORD", "0.0"),
            ("mu_dtype", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "NAdam optimizer (Adam with Nesterov momentum).",
    ),
    (
        "nadamw",
        "optax.nadamw",
        "function",
        [
            ("learning_rate", "POSITIONAL_OR_KEYWORD", None),
            ("b1", "POSITIONAL_OR_KEYWORD", "0.9"),
            ("b2", "POSITIONAL_OR_KEYWORD", "0.999"),
            ("eps", "POSITIONAL_OR_KEYWORD", "1e-08"),
            ("eps_root", "POSITIONAL_OR_KEYWORD", "0.0"),
            ("mu_dtype", "POSITIONAL_OR_KEYWORD", "None"),
            ("weight_decay", "POSITIONAL_OR_KEYWORD", "0.0001"),
            ("mask", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "NAdamW optimizer with Nesterov momentum and decoupled weight decay.",
    ),
    (
        "noisy_sgd",
        "optax.noisy_sgd",
        "function",
        [
            ("learning_rate", "POSITIONAL_OR_KEYWORD", None),
            ("eta", "POSITIONAL_OR_KEYWORD", "0.01"),
            ("gamma", "POSITIONAL_OR_KEYWORD", "0.55"),
            ("seed", "POSITIONAL_OR_KEYWORD", "0"),
        ],
        "Stochastic gradient descent with annealed Gaussian gradient noise.",
    ),
    (
        "novograd",
        "optax.novograd",
        "function",
        [
            ("learning_rate", "POSITIONAL_OR_KEYWORD", None),
            ("b1", "POSITIONAL_OR_KEYWORD", "0.9"),
            ("b2", "POSITIONAL_OR_KEYWORD", "0.25"),
            ("eps", "POSITIONAL_OR_KEYWORD", "1e-06"),
            ("eps_root", "POSITIONAL_OR_KEYWORD", "0.0"),
            ("weight_decay", "POSITIONAL_OR_KEYWORD", "0.0"),
        ],
        "NovoGrad optimizer with layer-wise second-moment normalization.",
    ),
    (
        "optimistic_gradient_descent",
        "optax.optimistic_gradient_descent",
        "function",
        [
            ("learning_rate", "POSITIONAL_OR_KEYWORD", None),
            ("alpha", "POSITIONAL_OR_KEYWORD", "1.0"),
            ("beta", "POSITIONAL_OR_KEYWORD", "1.0"),
        ],
        "Optimistic Gradient Descent (OGD) optimizer for min-max and game optimization.",
    ),
    (
        "polyak_sgd",
        "optax.polyak_sgd",
        "function",
        [
            ("max_learning_rate", "POSITIONAL_OR_KEYWORD", "1.0"),
            ("f_min", "POSITIONAL_OR_KEYWORD", "0.0"),
            ("eps", "POSITIONAL_OR_KEYWORD", "0.0"),
            ("scaling", "POSITIONAL_OR_KEYWORD", "1.0"),
        ],
        "SGD with Polyak adaptive step size.",
    ),
    (
        "radam",
        "optax.radam",
        "function",
        [
            ("learning_rate", "POSITIONAL_OR_KEYWORD", None),
            ("b1", "POSITIONAL_OR_KEYWORD", "0.9"),
            ("b2", "POSITIONAL_OR_KEYWORD", "0.999"),
            ("eps", "POSITIONAL_OR_KEYWORD", "1e-08"),
            ("eps_root", "POSITIONAL_OR_KEYWORD", "0.0"),
            ("threshold", "POSITIONAL_OR_KEYWORD", "5.0"),
            ("nesterov", "KEYWORD_ONLY", "False"),
        ],
        "Rectified Adam (RAdam) optimizer correcting variance of adaptive learning rate.",
    ),
    (
        "rmsprop",
        "optax.rmsprop",
        "function",
        [
            ("learning_rate", "POSITIONAL_OR_KEYWORD", None),
            ("decay", "POSITIONAL_OR_KEYWORD", "0.9"),
            ("eps", "POSITIONAL_OR_KEYWORD", "1e-08"),
            ("initial_scale", "POSITIONAL_OR_KEYWORD", "0.0"),
            ("centered", "POSITIONAL_OR_KEYWORD", "False"),
            ("momentum", "POSITIONAL_OR_KEYWORD", "None"),
            ("nesterov", "POSITIONAL_OR_KEYWORD", "False"),
            ("bias_correction", "POSITIONAL_OR_KEYWORD", "False"),
        ],
        "RMSProp optimizer dividing gradients by root-mean-square of moving average.",
    ),
    (
        "rprop",
        "optax.rprop",
        "function",
        [
            ("learning_rate", "POSITIONAL_OR_KEYWORD", None),
            ("eta_minus", "POSITIONAL_OR_KEYWORD", "0.5"),
            ("eta_plus", "POSITIONAL_OR_KEYWORD", "1.2"),
            ("min_step_size", "POSITIONAL_OR_KEYWORD", "1e-06"),
            ("max_step_size", "POSITIONAL_OR_KEYWORD", "50.0"),
        ],
        "Resilient backpropagation (Rprop) sign-based step size optimizer.",
    ),
    (
        "sgd",
        "optax.sgd",
        "function",
        [
            ("learning_rate", "POSITIONAL_OR_KEYWORD", None),
            ("momentum", "POSITIONAL_OR_KEYWORD", "None"),
            ("nesterov", "POSITIONAL_OR_KEYWORD", "False"),
            ("accumulator_dtype", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Canonical Stochastic Gradient Descent (SGD) optimizer with optional momentum.",
    ),
    (
        "sign_sgd",
        "optax.sign_sgd",
        "function",
        [
            ("learning_rate", "POSITIONAL_OR_KEYWORD", None),
        ],
        "SignSGD optimizer updating parameters using only the sign of the gradient.",
    ),
    (
        "sm3",
        "optax.sm3",
        "function",
        [
            ("learning_rate", "POSITIONAL_OR_KEYWORD", None),
            ("momentum", "POSITIONAL_OR_KEYWORD", "0.9"),
        ],
        "Square-root of Minima of Sums of Maxima of Squared-gradients Method (SM3) memory-efficient optimizer.",
    ),
    (
        "yogi",
        "optax.yogi",
        "function",
        [
            ("learning_rate", "POSITIONAL_OR_KEYWORD", None),
            ("b1", "POSITIONAL_OR_KEYWORD", "0.9"),
            ("b2", "POSITIONAL_OR_KEYWORD", "0.999"),
            ("eps", "POSITIONAL_OR_KEYWORD", "1e-03"),
            ("initial_accumulator_value", "POSITIONAL_OR_KEYWORD", "1e-06"),
        ],
        "Yogi adaptive optimizer controlling effective learning rate increases via additive updates.",
    ),
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


_OPTAX_LOSS_DEFS: list[
    tuple[
        str,
        str,
        str,
        list[tuple[str, str, str | None]],
        str,
    ]
] = [
    (
        "l2_loss",
        "optax.losses.l2_loss",
        "function",
        [
            ("predictions", "POSITIONAL_OR_KEYWORD", None),
            ("targets", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Compute `0.5 * (predictions - targets) ** 2` element-wise.",
    ),
    (
        "squared_error",
        "optax.losses.squared_error",
        "function",
        [
            ("predictions", "POSITIONAL_OR_KEYWORD", None),
            ("targets", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Compute `(predictions - targets) ** 2` element-wise.",
    ),
    (
        "huber_loss",
        "optax.losses.huber_loss",
        "function",
        [
            ("predictions", "POSITIONAL_OR_KEYWORD", None),
            ("targets", "POSITIONAL_OR_KEYWORD", "None"),
            ("delta", "POSITIONAL_OR_KEYWORD", "1.0"),
        ],
        "Compute Huber loss with quadratic region `|x| <= delta` and linear tails.",
    ),
    (
        "smooth_labels",
        "optax.losses.smooth_labels",
        "function",
        [
            ("labels", "POSITIONAL_OR_KEYWORD", None),
            ("alpha", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Apply label smoothing `(1 - alpha) * labels + alpha / num_classes` to one-hot targets.",
    ),
    (
        "softmax_cross_entropy",
        "optax.losses.softmax_cross_entropy",
        "function",
        [
            ("logits", "POSITIONAL_OR_KEYWORD", None),
            ("labels", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Compute cross-entropy loss between unnormalized `logits` and probability/one-hot `labels`.",
    ),
    (
        "softmax_cross_entropy_with_integer_labels",
        "optax.losses.softmax_cross_entropy_with_integer_labels",
        "function",
        [
            ("logits", "POSITIONAL_OR_KEYWORD", None),
            ("labels", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Compute softmax cross-entropy loss between `logits` and integer class index `labels`.",
    ),
    (
        "safe_softmax_cross_entropy",
        "optax.losses.safe_softmax_cross_entropy",
        "function",
        [
            ("logits", "POSITIONAL_OR_KEYWORD", None),
            ("labels", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Compute softmax cross-entropy handling `0 * -inf = 0` safely.",
    ),
    (
        "sigmoid_binary_cross_entropy",
        "optax.losses.sigmoid_binary_cross_entropy",
        "function",
        [
            ("logits", "POSITIONAL_OR_KEYWORD", None),
            ("labels", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Compute binary cross-entropy loss between unnormalized `logits` and binary `labels`.",
    ),
    (
        "sigmoid_focal_loss",
        "optax.losses.sigmoid_focal_loss",
        "function",
        [
            ("logits", "POSITIONAL_OR_KEYWORD", None),
            ("labels", "POSITIONAL_OR_KEYWORD", None),
            ("alpha", "POSITIONAL_OR_KEYWORD", "None"),
            ("gamma", "POSITIONAL_OR_KEYWORD", "2.0"),
        ],
        "Compute sigmoid focal loss for class-imbalanced classification.",
    ),
    (
        "hinge_loss",
        "optax.losses.hinge_loss",
        "function",
        [
            ("predictor_outputs", "POSITIONAL_OR_KEYWORD", None),
            ("targets", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Compute binary hinge loss `max(0, 1 - predictor_outputs * targets)`.",
    ),
    (
        "multiclass_hinge_loss",
        "optax.losses.multiclass_hinge_loss",
        "function",
        [
            ("scores", "POSITIONAL_OR_KEYWORD", None),
            ("labels", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Compute Crammer-Singer multiclass hinge loss.",
    ),
    (
        "multiclass_perceptron_loss",
        "optax.losses.multiclass_perceptron_loss",
        "function",
        [
            ("scores", "POSITIONAL_OR_KEYWORD", None),
            ("labels", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Compute structured multiclass perceptron loss.",
    ),
    (
        "poly_loss_cross_entropy",
        "optax.losses.poly_loss_cross_entropy",
        "function",
        [
            ("logits", "POSITIONAL_OR_KEYWORD", None),
            ("labels", "POSITIONAL_OR_KEYWORD", None),
            ("epsilon", "POSITIONAL_OR_KEYWORD", "2.0"),
        ],
        "Compute PolyLoss polynomial-expansion cross-entropy loss.",
    ),
    (
        "kl_divergence",
        "optax.losses.kl_divergence",
        "function",
        [
            ("log_predictions", "POSITIONAL_OR_KEYWORD", None),
            ("targets", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Compute Kullback-Leibler divergence `targets * (log(targets) - log_predictions)`.",
    ),
    (
        "kl_divergence_with_log_targets",
        "optax.losses.kl_divergence_with_log_targets",
        "function",
        [
            ("log_predictions", "POSITIONAL_OR_KEYWORD", None),
            ("log_targets", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Compute Kullback-Leibler divergence when both predictions and targets are in log-space.",
    ),
    (
        "convex_kl_divergence",
        "optax.losses.convex_kl_divergence",
        "function",
        [
            ("log_predictions", "POSITIONAL_OR_KEYWORD", None),
            ("targets", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Compute convex generalized Kullback-Leibler divergence.",
    ),
    (
        "cosine_distance",
        "optax.losses.cosine_distance",
        "function",
        [
            ("predictions", "POSITIONAL_OR_KEYWORD", None),
            ("targets", "POSITIONAL_OR_KEYWORD", None),
            ("epsilon", "POSITIONAL_OR_KEYWORD", "0.0"),
        ],
        "Compute `1 - cosine_similarity(predictions, targets, epsilon)`.",
    ),
    (
        "cosine_similarity",
        "optax.losses.cosine_similarity",
        "function",
        [
            ("predictions", "POSITIONAL_OR_KEYWORD", None),
            ("targets", "POSITIONAL_OR_KEYWORD", None),
            ("epsilon", "POSITIONAL_OR_KEYWORD", "0.0"),
        ],
        "Compute normalized dot-product cosine similarity between `predictions` and `targets`.",
    ),
    (
        "log_cosh",
        "optax.losses.log_cosh",
        "function",
        [
            ("predictions", "POSITIONAL_OR_KEYWORD", None),
            ("targets", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Compute `log(cosh(predictions - targets))` smooth L1-like regression loss.",
    ),
    (
        "ctc_loss",
        "optax.losses.ctc_loss",
        "function",
        [
            ("logits", "POSITIONAL_OR_KEYWORD", None),
            ("logit_paddings", "POSITIONAL_OR_KEYWORD", None),
            ("labels", "POSITIONAL_OR_KEYWORD", None),
            ("label_paddings", "POSITIONAL_OR_KEYWORD", None),
            ("blank_id", "POSITIONAL_OR_KEYWORD", "0"),
            ("log_epsilon", "POSITIONAL_OR_KEYWORD", "-100000.0"),
        ],
        "Compute Connectionist Temporal Classification (CTC) alignment loss.",
    ),
    (
        "ctc_loss_with_forward_probs",
        "optax.losses.ctc_loss_with_forward_probs",
        "function",
        [
            ("logits", "POSITIONAL_OR_KEYWORD", None),
            ("logit_paddings", "POSITIONAL_OR_KEYWORD", None),
            ("labels", "POSITIONAL_OR_KEYWORD", None),
            ("label_paddings", "POSITIONAL_OR_KEYWORD", None),
            ("blank_id", "POSITIONAL_OR_KEYWORD", "0"),
            ("log_epsilon", "POSITIONAL_OR_KEYWORD", "-100000.0"),
        ],
        "Compute CTC loss and return both loss values and forward dynamic programming log-probabilities.",
    ),
    (
        "ntxent",
        "optax.losses.ntxent",
        "function",
        [
            ("embeddings", "POSITIONAL_OR_KEYWORD", None),
            ("labels", "POSITIONAL_OR_KEYWORD", None),
            ("temperature", "POSITIONAL_OR_KEYWORD", "1.0"),
        ],
        "Compute Normalized Temperature-scaled Cross-Entropy (NT-Xent) contrastive loss.",
    ),
    (
        "triplet_margin_loss",
        "optax.losses.triplet_margin_loss",
        "function",
        [
            ("anchors", "POSITIONAL_OR_KEYWORD", None),
            ("positives", "POSITIONAL_OR_KEYWORD", None),
            ("negatives", "POSITIONAL_OR_KEYWORD", None),
            ("axis", "POSITIONAL_OR_KEYWORD", "-1"),
            ("norm_degree", "POSITIONAL_OR_KEYWORD", "2"),
            ("margin", "POSITIONAL_OR_KEYWORD", "1.0"),
            ("eps", "POSITIONAL_OR_KEYWORD", "1e-12"),
            ("mode", "POSITIONAL_OR_KEYWORD", "'hard'"),
        ],
        "Compute triplet margin metric learning loss across anchor, positive, and negative embeddings.",
    ),
    (
        "ranking_softmax_loss",
        "optax.losses.ranking_softmax_loss",
        "function",
        [
            ("logits", "POSITIONAL_OR_KEYWORD", None),
            ("labels", "POSITIONAL_OR_KEYWORD", None),
            ("where", "KEYWORD_ONLY", "None"),
            ("weights", "KEYWORD_ONLY", "None"),
            ("reduce_fn", "KEYWORD_ONLY", "None"),
        ],
        "Compute listwise ranking softmax cross-entropy loss.",
    ),
    (
        "make_fenchel_young_loss",
        "optax.losses.make_fenchel_young_loss",
        "function",
        [
            ("max_fun", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Construct a Fenchel-Young loss function from a regularized maximum function `max_fun`.",
    ),
]


_OPTAX_SCHEDULER_DEFS: list[
    tuple[
        str,
        str,
        str,
        list[tuple[str, str, str | None]],
        str,
    ]
] = [
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


_OPTAX_ARRAY_API_DEFS: list[
    tuple[
        str,
        str,
        str,
        list[tuple[str, str, str | None]],
        str,
    ]
] = [
    # Top-level parameter update & gradient utilities
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


def _build_optax_canonical_ops(
    defs: typing.Sequence[
        tuple[
            str,
            str,
            str,
            list[tuple[str, str, str | None]],
            str,
        ]
    ],
) -> list[CanonicalPythonOpSpec]:
    """Convert compact Optax operation tuples into `CanonicalPythonOpSpec` models.

    Args:
        defs: Sequence of `(name, api_path, kind, params, docstring)` tuples.

    Returns:
        List of validated `CanonicalPythonOpSpec` models.
    """
    return [
        CanonicalPythonOpSpec(
            name=name,
            api_path=api_path,
            kind=kind,
            params=[
                CanonicalParamSpec(name=pn, kind=pk, default=pd)
                for pn, pk, pd in params
            ],
            docstring=doc,
        )
        for name, api_path, kind, params, doc in defs
    ]


CANONICAL_OPTAX_OPTIMIZERS: list[CanonicalPythonOpSpec] = _build_optax_canonical_ops(
    _OPTAX_OPTIMIZER_DEFS
)
CANONICAL_OPTAX_LOSSES: list[CanonicalPythonOpSpec] = _build_optax_canonical_ops(
    _OPTAX_LOSS_DEFS
)
CANONICAL_OPTAX_SCHEDULERS: list[CanonicalPythonOpSpec] = _build_optax_canonical_ops(
    _OPTAX_SCHEDULER_DEFS
)
CANONICAL_OPTAX_ARRAY_API: list[CanonicalPythonOpSpec] = _build_optax_canonical_ops(
    _OPTAX_ARRAY_API_DEFS
)
CANONICAL_OPTAX_OPS: list[CanonicalPythonOpSpec] = (
    CANONICAL_OPTAX_OPTIMIZERS
    + CANONICAL_OPTAX_LOSSES
    + CANONICAL_OPTAX_SCHEDULERS
    + CANONICAL_OPTAX_ARRAY_API
)

KNOWN_OPTAX_OPTIMIZERS: set[str] = {
    spec.name.lower() for spec in CANONICAL_OPTAX_OPTIMIZERS
}
KNOWN_OPTAX_LOSSES: set[str] = {spec.name.lower() for spec in CANONICAL_OPTAX_LOSSES}
KNOWN_OPTAX_ARRAY_API: set[str] = {
    spec.name.lower() for spec in CANONICAL_OPTAX_ARRAY_API
}


def _get_optax_griffe_module() -> GriffeCacheEntry:
    """Statically load `optax` via the custom Griffe fork when not in offline mode.

    Returns:
        Loaded Griffe AST node for `optax` or `None` if unavailable or in offline mode.
    """
    if is_offline_mode():
        return None
    return load_griffe_module_safely("optax", framework_name="optax")


def _hydrate_canonical_optax_tier(
    specs: typing.Sequence[CanonicalPythonOpSpec],
    attach_root_alias: bool = False,
) -> list[GhostRef]:
    """Hydrate a sequence of `CanonicalPythonOpSpec` items into `GhostRef` records.

    Args:
        specs: Canonical Optax operation specifications.
        attach_root_alias: Whether to attach `optax.<name>` alias for `optax.losses.*` or `optax.schedules.*`.

    Returns:
        List of hydrated `GhostRef` objects.
    """
    refs: list[GhostRef] = []
    for spec in specs:
        ref = build_canonical_python_ref(
            spec,
            default_doc_prefix="Optax",
            environment_tags=("optax", "jax"),
        )
        if attach_root_alias and (
            ref.api_path.startswith("optax.losses.")
            or ref.api_path.startswith("optax.schedules.")
        ):
            ref.aliases = [f"optax.{ref.name}"]
        refs.append(ref)
    return refs


class OptaxScanner:
    """Helper to inspect Optax APIs for the Ghost Protocol discovery system."""

    @staticmethod
    def scan_optimizers(include_nonpublic: bool = False) -> list[GhostRef]:
        """Scan the Optax public API for optimizer and gradient transformation functions.

        Args:
            include_nonpublic: Whether to include non-public APIs.

        Returns:
            A list of `GhostRef` objects representing found optimizers.
        """
        if is_offline_mode():
            return _hydrate_canonical_optax_tier(CANONICAL_OPTAX_OPTIMIZERS)

        if not optax:
            return []

        results: list[GhostRef] = []
        for name, obj in get_all_members(optax):
            if not include_nonpublic and name.startswith("_"):
                continue

            if inspect.isfunction(obj) or inspect.isclass(obj):
                is_known = name.lower() in KNOWN_OPTAX_OPTIMIZERS
                is_suffixed = name.endswith("_optimizer")

                if is_known or is_suffixed:
                    try:
                        ref = GhostInspector.inspect(obj, f"optax.{name}")
                        results.append(ref)
                    except Exception:
                        pass

        return results

    @staticmethod
    def scan_losses(include_nonpublic: bool = False) -> list[GhostRef]:
        """Scan `optax.losses` for loss functions.

        Args:
            include_nonpublic: Whether to include non-public APIs.

        Returns:
            A list of `GhostRef` objects representing found losses.
        """
        if is_offline_mode():
            return _hydrate_canonical_optax_tier(
                CANONICAL_OPTAX_LOSSES, attach_root_alias=True
            )

        if not optax or not hasattr(optax, "losses"):
            return []

        results: list[GhostRef] = []
        for name, obj in get_all_members(optax.losses):
            if not include_nonpublic and name.startswith("_"):
                continue

            if inspect.isfunction(obj):
                name_lower = name.lower()
                if (
                    "loss" in name_lower
                    or "error" in name_lower
                    or "entropy" in name_lower
                    or name_lower in KNOWN_OPTAX_LOSSES
                ):
                    try:
                        ref = GhostInspector.inspect(obj, f"optax.losses.{name}")
                        results.append(ref)
                    except Exception:
                        pass

        return results

    @staticmethod
    def scan_schedulers(include_nonpublic: bool = False) -> list[GhostRef]:
        """Scan `optax.schedules` for scheduler functions.

        Args:
            include_nonpublic: Whether to include non-public APIs.

        Returns:
            A list of `GhostRef` objects representing found schedulers.
        """
        if is_offline_mode():
            return _hydrate_canonical_optax_tier(
                CANONICAL_OPTAX_SCHEDULERS, attach_root_alias=True
            )

        if not optax or not hasattr(optax, "schedules"):
            return []

        results: list[GhostRef] = []
        for name, obj in get_all_members(optax.schedules):
            if not include_nonpublic and name.startswith("_"):
                continue

            if inspect.isfunction(obj):
                try:
                    ref = GhostInspector.inspect(obj, f"optax.schedules.{name}")
                    results.append(ref)
                except Exception:
                    pass

        return results

    @staticmethod
    def scan_array_api(include_nonpublic: bool = False) -> list[GhostRef]:
        """Scan Optax parameter update, projection, tree_utils, second_order, assignment, and perturbation APIs.

        Args:
            include_nonpublic: Whether to include non-public APIs.

        Returns:
            A list of `GhostRef` objects representing Optax utility operations.
        """
        if is_offline_mode():
            return _hydrate_canonical_optax_tier(CANONICAL_OPTAX_ARRAY_API)

        results: list[GhostRef] = []
        seen_paths: set[str] = set()

        if optax is not None:
            for name, obj in get_all_members(optax):
                if not include_nonpublic and name.startswith("_"):
                    continue
                if inspect.isfunction(obj) and name.lower() in KNOWN_OPTAX_ARRAY_API:
                    try:
                        ref = GhostInspector.inspect(obj, f"optax.{name}")
                        results.append(ref)
                        seen_paths.add(ref.api_path)
                    except Exception:
                        pass

            optax_vars = vars(optax)
            for submod_name in (
                "projections",
                "tree_utils",
                "second_order",
                "assignment",
                "perturbations",
            ):
                submod = optax_vars.get(submod_name)
                if isinstance(submod, types.ModuleType):
                    for name, obj in get_all_members(submod):
                        if not include_nonpublic and name.startswith("_"):
                            continue
                        if inspect.isfunction(obj):
                            api_path = f"optax.{submod_name}.{name}"
                            try:
                                ref = GhostInspector.inspect(obj, api_path)
                                results.append(ref)
                                seen_paths.add(api_path)
                            except Exception:
                                pass

        if _get_optax_griffe_module() is not None:
            for griffe_ref in scan_griffe_module_members(
                module_path="optax",
                api_prefix="optax",
                include_nonpublic=include_nonpublic,
                environment_tags=("optax", "jax"),
                framework_name="optax",
                inspector_fn=GhostInspector.inspect,
            ):
                if (
                    griffe_ref.name.lower() in KNOWN_OPTAX_ARRAY_API
                    and griffe_ref.api_path not in seen_paths
                ):
                    results.append(griffe_ref)
                    seen_paths.add(griffe_ref.api_path)

        for spec in CANONICAL_OPTAX_ARRAY_API:
            if spec.api_path not in seen_paths:
                results.append(
                    build_canonical_python_ref(
                        spec,
                        default_doc_prefix="Optax",
                        environment_tags=("optax", "jax"),
                    )
                )
                seen_paths.add(spec.api_path)

        return results


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> list[GhostRef]:
    """Collect Optax APIs for the requested semantic tier.

    Args:
        category: The semantic tier to target (`OPTIMIZER`, `LOSS`, `SCHEDULER`, or `ARRAY_API`).
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        List of gathered `GhostRef` API references.
    """
    if category == SemanticTier.OPTIMIZER:
        return OptaxScanner.scan_optimizers(include_nonpublic)
    elif category == SemanticTier.LOSS:
        return OptaxScanner.scan_losses(include_nonpublic)
    elif category == SemanticTier.SCHEDULER:
        return OptaxScanner.scan_schedulers(include_nonpublic)
    elif category == SemanticTier.ARRAY_API:
        return OptaxScanner.scan_array_api(include_nonpublic)
    return []


__all__ = [
    "CANONICAL_OPTAX_ARRAY_API",
    "CANONICAL_OPTAX_LOSSES",
    "CANONICAL_OPTAX_OPTIMIZERS",
    "CANONICAL_OPTAX_OPS",
    "CANONICAL_OPTAX_SCHEDULERS",
    "KNOWN_OPTAX_ARRAY_API",
    "KNOWN_OPTAX_LOSSES",
    "KNOWN_OPTAX_OPTIMIZERS",
    "OptaxScanner",
    "_OPTAX_ARRAY_API_DEFS",
    "_OPTAX_LOSS_DEFS",
    "_OPTAX_OPTIMIZER_DEFS",
    "_OPTAX_SCHEDULER_DEFS",
    "_build_optax_canonical_ops",
    "_get_optax_griffe_module",
    "_hydrate_canonical_optax_tier",
    "collect_api",
    "optax",
]
