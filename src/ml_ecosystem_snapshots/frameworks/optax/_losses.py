"""Canonical Optax loss and divergence operation definitions."""

from __future__ import annotations

from ml_ecosystem_snapshots.frameworks._base import (
    CanonicalPythonOpSpec,
    CanonicalPythonOpTuple,
    build_canonical_python_ops,
)

_OPTAX_LOSS_DEFS: list[CanonicalPythonOpTuple] = [
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

CANONICAL_OPTAX_LOSSES: list[CanonicalPythonOpSpec] = build_canonical_python_ops(_OPTAX_LOSS_DEFS)
KNOWN_OPTAX_LOSSES: set[str] = {spec.name.lower() for spec in CANONICAL_OPTAX_LOSSES}

__all__ = ["CANONICAL_OPTAX_LOSSES", "KNOWN_OPTAX_LOSSES"]
