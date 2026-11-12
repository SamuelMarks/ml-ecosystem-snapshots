"""PyTorch ATen schema, C-extension signature, and hardware environment-tag helpers."""

from __future__ import annotations

import logging
import typing

from ml_switcheroo_ir.schema.ghost import ExtendedGhostRef

from ml_ecosystem_snapshots.c_extension_sig import CExtensionSignature
from ml_ecosystem_snapshots.exceptions import SnapshotError
from ml_ecosystem_snapshots.introspection.context import (
    OverloadExtractionContext,
    ParamExtractionContext,
    RawParamTuple,
    _get_target_module_name,
)
from ml_ecosystem_snapshots.introspection.live_inspector import (
    _build_extended_params,
)
from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    LivePythonTarget,
    StaticGriffeNode,
)

_LOGGER = logging.getLogger(__name__)


STANDARD_ARG_MAP: dict[str, str] = {
    "x": "input",
    "inputs": "input",
    "input_tensor": "input",
    "y": "other",
    "other_tensor": "other",
    "dim": "dim",
    "axis": "dim",
    "keepdim": "keepdims",
    "keep_dims": "keepdims",
    "keepdims": "keepdims",
}


FRAMEWORK_CAPABILITIES: dict[str, list[str]] = {
    "torch": ["cpu", "cuda", "rocm", "metal"],
    "tensorflow": ["cpu", "cuda", "rocm", "tpu", "metal"],
    "tf": ["cpu", "cuda", "rocm", "tpu", "metal"],
    "jax": ["cpu", "cuda", "rocm", "tpu", "metal"],
    "chex": ["cpu", "cuda", "rocm", "tpu", "metal"],
    "grain": ["cpu", "cuda", "rocm", "tpu"],
    "pygrain": ["cpu", "cuda", "rocm", "tpu"],
    "orbax": ["cpu", "cuda", "rocm", "tpu"],
    "orbax_checkpoint": ["cpu", "cuda", "rocm", "tpu"],
    "optax": ["cpu", "cuda", "rocm", "tpu", "metal"],
    "mlx": ["cpu", "metal"],
    "triton": ["cuda", "rocm"],
    "cupy": ["cuda", "rocm"],
    "numpy": ["cpu"],
    "sklearn": ["cpu"],
    "scikit_learn": ["cpu"],
    "scipy": ["cpu"],
    "deepspeed": ["cpu", "cuda", "rocm"],
    "onnxruntime": ["cpu", "cuda", "rocm", "metal"],
    "nvidia_sass": ["cuda"],
    "amd_rdna": ["rocm"],
    "mlir": ["cpu", "cuda", "rocm", "tpu"],
    "stablehlo": ["cpu", "cuda", "rocm", "tpu"],
    "ir": ["cpu", "cuda", "rocm", "tpu", "metal", "webgpu"],
    "wgsl": ["webgpu"],
}


STANDARD_ENUM_MAP: dict[str, list[str]] = {
    "reduction": ["none", "mean", "sum"],
    "mode": [
        "nearest",
        "linear",
        "bilinear",
        "bicubic",
        "trilinear",
        "area",
        "nearest-exact",
    ],
    "padding": [
        "valid",
        "same",
        "zeros",
        "reflect",
        "replicate",
        "circular",
    ],
    "layout": ["NCHW", "NHWC", "NCDHW", "NDHWC"],
}


def _is_torch_symbol(name: str, target: LivePythonTarget | StaticGriffeNode, api_path: str) -> bool:
    """Determine whether the inspected target belongs to the PyTorch namespace.

    Args:
        name: Unqualified or partially qualified symbol name.
        target: Runtime callable target.
        api_path: Canonical dotted API path.

    Returns:
        True if the symbol belongs to `torch`, False otherwise.
    """
    return bool(
        (name and name.startswith("torch."))
        or _get_target_module_name(target).startswith("torch")
        or api_path.startswith("torch.")
    )


def _resolve_c_ext_and_aten_sig(
    target: LivePythonTarget | StaticGriffeNode,
    ctx: ParamExtractionContext,
) -> tuple[CExtensionSignature | None, bool]:
    """Extract C-extension signature and merge PyTorch ATen schema when applicable.

    Args:
        target: Callable target (`obj` or `obj.__init__`).
        ctx: Bundled parameter extraction context.

    Returns:
        Tuple of `(c_ext_params, is_exact_aten)`.
    """
    c_ext_params = ctx.active_extract_c_ext_sig(target, ctx.name)
    is_exact_aten = False
    if _is_torch_symbol(ctx.name, target, ctx.api_path):
        try:
            from ml_ecosystem_snapshots.frameworks.torch import (
                extract_aten_c_extension_signature,
            )

            aten_sig = extract_aten_c_extension_signature(
                target,
                ctx.name,
                is_method=(ctx.kind == "method" or ".Tensor." in ctx.name),
            )
            if aten_sig is not None:
                if c_ext_params is None:
                    c_ext_params = aten_sig
                    is_exact_aten = True
                elif not c_ext_params.overloads and aten_sig.overloads:
                    c_ext_params.overloads = aten_sig.overloads
                    is_exact_aten = True
        except (
            ImportError,
            AttributeError,
            TypeError,
            ValueError,
            OSError,
            RuntimeError,
            LookupError,
            SyntaxError,
            NameError,
            AssertionError,
            SnapshotError,
        ):
            _LOGGER.debug("Failed to extract ATen C-extension signature for %s", ctx.name, exc_info=True)
    return c_ext_params, is_exact_aten


def _append_opaque_env_tags(environment_tags: list[str] | None) -> list[str]:
    """Append opaque C-extension tags to the environment tag list if not already present.

    Args:
        environment_tags: Existing environment tag list or `None`.

    Returns:
        Updated environment tag list including `'inexact_signature'` and `'opaque_c_extension'`.
    """
    tags = [] if environment_tags is None else environment_tags
    if "inexact_signature" not in tags:
        tags.append("inexact_signature")
    if "opaque_c_extension" not in tags:
        tags.append("opaque_c_extension")
    return tags


def _extract_params_from_c_ext(
    target: LivePythonTarget | StaticGriffeNode,
    ctx: ParamExtractionContext,
) -> tuple[
    list[RawParamTuple],
    CExtensionSignature | None,
    bool,
    typing.Literal["exact", "heuristic", "opaque"],
    str | None,
    list[str] | None,
]:
    """Fallback parameter extractor for C-extensions and opaque builtins.

    Args:
        target: Callable target (`obj` or `obj.__init__`).
        ctx: Bundled parameter extraction context.

    Returns:
        Tuple of `(extracted_params, c_ext_params, has_varargs, sig_completeness,
        returns_type, environment_tags)`.
    """
    extracted_params: list[RawParamTuple] = []
    has_varargs = False
    sig_completeness: typing.Literal["exact", "heuristic", "opaque"] = "exact"
    returns_type = ctx.returns_type
    environment_tags = ctx.environment_tags

    c_ext_params, is_exact_aten = _resolve_c_ext_and_aten_sig(target, ctx)
    if c_ext_params is not None:
        sig_completeness = "exact" if is_exact_aten else "heuristic"
        if (returns_type is None or returns_type == "NoneType") and c_ext_params.returns_type:
            returns_type = ctx.active_sanitize_type_str(c_ext_params.returns_type)
        for idx in range(len(c_ext_params)):
            p_item = c_ext_params[idx]
            pn = p_item.name
            pk = p_item.kind
            pd = p_item.default
            pa = p_item.annotation
            if pk == "VAR_POSITIONAL":
                has_varargs = True
            sanitized_pa = ctx.active_sanitize_type_str(pa) if pa else None
            extracted_params.append((pn, pk, pd, sanitized_pa))
    elif ctx.kind == "function":
        sig_completeness = "opaque"
        has_varargs = True
        extracted_params.append(("args", "VAR_POSITIONAL", None, None))
        extracted_params.append(("kwargs", "VAR_KEYWORD", None, None))
        environment_tags = _append_opaque_env_tags(environment_tags)

    return (
        extracted_params,
        c_ext_params,
        has_varargs,
        sig_completeness,
        returns_type,
        environment_tags,
    )


def _build_c_ext_overload_ref(
    ov: CExtensionSignature,
    ctx: OverloadExtractionContext,
) -> ExtendedGhostRef:
    """Build an `ExtendedGhostRef` from a single C-extension overload signature.

    Args:
        ov: Parsed `CExtensionSignature` overload entry.
        ctx: Bundled overload extraction context.

    Returns:
        Constructed `ExtendedGhostRef` representing the C-extension overload signature.
    """
    ov_raw_tuples: list[RawParamTuple] = []
    ov_has_varargs = False
    for idx in range(len(ov)):
        p_item = ov[idx]
        pn = p_item.name
        pk = p_item.kind
        pd = p_item.default
        pa = p_item.annotation
        if pk == "VAR_POSITIONAL":
            ov_has_varargs = True
        sanitized_pa = ctx.active_sanitize_type_str(pa) if pa else None
        ov_raw_tuples.append((pn, pk, pd, sanitized_pa))

    ov_params = _build_extended_params(
        extracted_params=ov_raw_tuples,
        cdd_params={},
        cdd_ast_params={},
        is_torch_target=ctx.is_torch_target,
        name=ctx.name,
        active_sanitize_type_str=ctx.active_sanitize_type_str,
    )
    ov_ret_val = ov.returns_type
    ov_ret = ctx.active_sanitize_type_str(str(ov_ret_val)) if ov_ret_val else None
    return ExtendedGhostRef(
        name=ctx.name,
        api_path=ctx.api_path,
        kind=ctx.kind,
        params=ov_params,
        docstring=None,
        has_varargs=ov_has_varargs,
        aliases=[],
        is_public=ctx.determined_is_public,
        returns_type=ov_ret,
        returns_description=None,
        raises=[],
        environment_tags=ctx.env_tags,
        overloads=[],
    )


def _resolve_environment_tags(
    environment_tags: list[str] | None,
    api_path: str,
) -> list[str]:
    """Resolve execution environment capability tags for an inspected symbol.

    Args:
        environment_tags: Explicit environment tags list, or `None` for framework defaults.
        api_path: Canonical dotted API path used to infer the top-level package.

    Returns:
        List of execution environment tag strings.
    """
    if environment_tags is not None:
        return list(environment_tags)
    top_pkg = api_path.split(".")[0].lower() if api_path else ""
    return list(FRAMEWORK_CAPABILITIES.get(top_pkg, ["cpu"]))


__all__ = [
    "FRAMEWORK_CAPABILITIES",
    "STANDARD_ARG_MAP",
    "STANDARD_ENUM_MAP",
    "_append_opaque_env_tags",
    "_build_c_ext_overload_ref",
    "_extract_params_from_c_ext",
    "_is_torch_symbol",
    "_resolve_c_ext_and_aten_sig",
    "_resolve_environment_tags",
]
