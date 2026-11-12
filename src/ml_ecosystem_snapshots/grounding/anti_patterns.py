"""Anti-pattern recommender and cross-framework hallucinated-argument explainer."""

from __future__ import annotations

import typing

from ml_ecosystem_snapshots.grounding.dtypes import _extract_cand_params
from ml_ecosystem_snapshots.grounding.models import AntiPatternExplanationResult
from ml_ecosystem_snapshots.models import JsonValue


def explain_python_anti_pattern(
    framework: str,
    api_path: str,
    hallucinated_argument: str,
    passed_value: JsonValue = None,
    sig_lookup_fn: typing.Callable[[str, str], dict[str, JsonValue] | None] | None = None,
) -> AntiPatternExplanationResult:
    """Provide canonical migration advice and rationale for a flagged anti-pattern or hallucinated argument.

    Args:
        framework: Target ML framework name (e.g. 'torch', 'jax', 'tensorflow').
        api_path: Target API path (e.g. 'torch.sum', 'jax.numpy.mean').
        hallucinated_argument: Name of the flagged/hallucinated keyword argument.
        passed_value: Optional value passed to the argument for contextual advice.
        sig_lookup_fn: Optional callback `(framework, api_path) -> signature_dict | None`.

    Returns:
        Structured `AntiPatternExplanationResult` with canonical replacement, explanation, and code example.
    """
    clean_fw = framework.lower().strip()
    arg_clean = hallucinated_argument.strip().lower()

    known_migrations: dict[str, dict[str, dict[str, str]]] = {
        "torch": {
            "axis": {
                "canonical": "dim",
                "explanation": "PyTorch standardizes dimension specifications on 'dim' rather than 'axis' (which is used by NumPy and JAX).",
                "example": f"{api_path}(..., dim=0)",
            },
            "keepdims": {
                "canonical": "keepdim",
                "explanation": "PyTorch uses the singular 'keepdim' parameter instead of the plural 'keepdims' (used by NumPy, JAX, and TensorFlow).",
                "example": f"{api_path}(..., keepdim=True)",
            },
            "split_size": {
                "canonical": "split_size_or_sections",
                "explanation": "PyTorch torch.split uses 'split_size_or_sections' for chunk sizes or split sections.",
                "example": f"{api_path}(tensor, split_size_or_sections=2)",
            },
            "device": {
                "canonical": "device='cuda'",
                "explanation": "PyTorch device identifiers use 'cuda' (or 'cuda:0') rather than 'gpu'.",
                "example": f"{api_path}(..., device='cuda')",
            },
            "inplace": {
                "canonical": "out-of-place or trailing underscore",
                "explanation": "PyTorch provides inplace operations via a trailing underscore suffix (e.g. 'relu_()' or 'add_()') rather than an 'inplace=True' argument on functional ops.",
                "example": "x.relu_() or torch.relu(x)",
            },
        },
        "jax": {
            "dim": {
                "canonical": "axis",
                "explanation": "JAX follows the NumPy Array API standard, using 'axis' instead of PyTorch's 'dim'.",
                "example": f"{api_path}(..., axis=0)",
            },
            "keepdim": {
                "canonical": "keepdims",
                "explanation": "JAX follows NumPy naming conventions, using 'keepdims' (plural) instead of PyTorch's 'keepdim'.",
                "example": f"{api_path}(..., keepdims=True)",
            },
            "key": {
                "canonical": "jax.random.PRNGKey",
                "explanation": "JAX random operations require an explicit PRNG key passed to 'key' produced via 'jax.random.PRNGKey(seed)' or 'jax.random.key(seed)'.",
                "example": "jax.random.normal(key, shape=(2, 3))",
            },
            "rng": {
                "canonical": "key",
                "explanation": "JAX standardizes random number generator parameters as 'key', not 'rng' or 'seed'.",
                "example": "jax.random.normal(key=key, shape=(2, 3))",
            },
        },
        "tensorflow": {
            "dim": {
                "canonical": "axis",
                "explanation": "TensorFlow operations use 'axis' instead of PyTorch's 'dim'.",
                "example": f"{api_path}(..., axis=0)",
            },
            "keepdim": {
                "canonical": "keepdims",
                "explanation": "TensorFlow uses 'keepdims' rather than PyTorch's 'keepdim'.",
                "example": f"{api_path}(..., keepdims=True)",
            },
        },
        "optax": {
            "lr": {
                "canonical": "learning_rate",
                "explanation": "Optax optimizers standardize the step size parameter as 'learning_rate' rather than PyTorch's 'lr'.",
                "example": f"{api_path}(learning_rate=1e-3)",
            },
            "betas": {
                "canonical": "b1, b2",
                "explanation": "Optax momentum optimizers accept separate scalar arguments 'b1' and 'b2' instead of PyTorch's 'betas=(b1, b2)' tuple.",
                "example": f"{api_path}(learning_rate=1e-3, b1=0.9, b2=0.999)",
            },
            "epsilon": {
                "canonical": "eps",
                "explanation": "Optax optimizers use 'eps' for numerical stability rather than 'epsilon'.",
                "example": f"{api_path}(learning_rate=1e-3, eps=1e-8)",
            },
        },
        "chex": {
            "msg": {
                "canonical": "custom_message",
                "explanation": "Chex assertions accept 'custom_message' (keyword-only) rather than 'msg' or 'err_msg'.",
                "example": f"{api_path}(..., custom_message='Shape mismatch')",
            },
            "message": {
                "canonical": "custom_message",
                "explanation": "Chex assertions accept 'custom_message' (keyword-only) rather than 'message'.",
                "example": f"{api_path}(..., custom_message='Assertion failed')",
            },
            "shapes": {
                "canonical": "expected_shapes",
                "explanation": "chex.assert_shape uses 'expected_shapes' rather than 'shapes' or 'shape'.",
                "example": f"{api_path}(inputs, expected_shapes=(2, 3))",
            },
        },
        "grain": {
            "dataset": {
                "canonical": "data_source",
                "explanation": "grain.DataLoader takes 'data_source' (a RandomAccessDataSource) and 'sampler' as keyword-only arguments rather than PyTorch's 'dataset'.",
                "example": f"{api_path}(data_source=source, sampler=sampler, worker_count=4)",
            },
            "num_workers": {
                "canonical": "worker_count",
                "explanation": "grain.DataLoader uses 'worker_count' rather than PyTorch's 'num_workers'.",
                "example": f"{api_path}(data_source=source, sampler=sampler, worker_count=4)",
            },
        },
        "orbax": {
            "max_checkpoints": {
                "canonical": "max_to_keep",
                "explanation": "orbax.checkpoint.CheckpointManagerOptions uses 'max_to_keep' rather than 'max_checkpoints' or 'keep_last_n'.",
                "example": f"{api_path}(max_to_keep=5, save_interval_steps=100)",
            },
            "keep_last_n": {
                "canonical": "max_to_keep",
                "explanation": "orbax.checkpoint.CheckpointManagerOptions uses 'max_to_keep' to control checkpoint retention.",
                "example": f"{api_path}(max_to_keep=5)",
            },
        },
    }

    fw_rules = known_migrations.get(clean_fw, {})
    if arg_clean in fw_rules:
        rule = fw_rules[arg_clean]
        return AntiPatternExplanationResult(
            framework=framework,
            api_path=api_path,
            hallucinated_argument=hallucinated_argument,
            is_known_anti_pattern=True,
            canonical_argument=rule["canonical"],
            explanation=rule["explanation"],
            canonical_example=rule["example"],
        )

    sig = sig_lookup_fn(framework, api_path) if sig_lookup_fn is not None else None
    canonical_params: list[str] = []
    if sig:
        canonical_params = [str(p["name"]) for p in _extract_cand_params(sig) if "name" in p]

    return AntiPatternExplanationResult(
        framework=framework,
        api_path=api_path,
        hallucinated_argument=hallucinated_argument,
        is_known_anti_pattern=False,
        canonical_argument=None,
        explanation=f"Argument '{hallucinated_argument}' is not recognized in {framework} for '{api_path}'.",
        canonical_params=canonical_params,
        canonical_example=f"{api_path}(...)",
    )


__all__ = ["explain_python_anti_pattern"]
