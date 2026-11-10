"""MCP tool module for hallucination verification, anti-pattern explanation, and AST code-block checking."""

from __future__ import annotations

import ast
import re
import typing

import pydantic
import typing_extensions

from ml_ecosystem_snapshots.introspection.inspector import STANDARD_ENUM_MAP
from ml_ecosystem_snapshots.mcp.cache import SnapshotRepository, resolve_snapshot
from ml_ecosystem_snapshots.mcp.models import (
    AntiPatternExplanationResult,
    CodeBlockCheckResult,
    CodeBlockFinding,
    HallucinationCheckResult,
)
from ml_ecosystem_snapshots.mcp.tools.ir import check_mlir_op
from ml_ecosystem_snapshots.mcp.tools.isa import (
    check_rdna_instruction,
    check_sass_instruction,
)
from ml_ecosystem_snapshots.mcp.tools.signature import get_api_signature
from ml_ecosystem_snapshots.models import JsonValue


class _HallucinationOptions(pydantic.BaseModel):
    """Optional keyword arguments for `check_hallucination`."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    version: str | None = None
    arg_dtypes: typing.Sequence[str] | typing.Mapping[str, str] | JsonValue = None
    kwarg_dtypes: typing.Mapping[str, str] | JsonValue = None
    arg_ranks: typing.Sequence[int | str] | typing.Mapping[str, int | str] | JsonValue = None
    kwarg_ranks: typing.Mapping[str, int | str] | JsonValue = None
    arg_shapes: typing.Sequence[typing.Sequence[int | str]] | JsonValue = None
    kwarg_shapes: typing.Mapping[str, typing.Sequence[int | str]] | JsonValue = None
    strict_c_extensions: bool = False
    repository: SnapshotRepository | None = None


_HallucinationOptionValue: typing_extensions.TypeAlias = (
    "str | bool | SnapshotRepository | typing.Sequence[str] | typing.Mapping[str, str] | typing.Sequence[int | str] | typing.Mapping[str, int | str] | typing.Sequence[typing.Sequence[int | str]] | typing.Mapping[str, typing.Sequence[int | str]] | JsonValue"
)

DTYPE_CANONICAL_ALIASES: dict[str, str] = {
    "str": "string",
    "str_": "string",
    "object_": "object",
    "fp8": "float8_e4m3fn",
    "fp16": "float16",
    "bf16": "bfloat16",
    "fp32": "float32",
    "fp64": "float64",
    "int": "int64",
    "float": "float32",
    "qint4x2": "qint4",
}

KNOWN_ENUM_VALUES: dict[str, list[str]] = {
    **STANDARD_ENUM_MAP,
    "padding": ["valid", "same", "zeros", "reflect", "replicate", "circular"],
}


def normalize_dtype_name(raw_dt: str) -> str:
    """Normalize framework-specific dtype representations into canonical dtype names.

    Args:
        raw_dt: Raw dtype representation (e.g. 'torch.float32', 'jnp.float32', 'tf.float32').

    Returns:
        Canonical dtype name (e.g. 'float32', 'int64', 'qint8', 'float8_e4m3b11fnuz').
    """
    clean = str(raw_dt).lower().strip()
    for prefix in (
        "torch.",
        "jax.numpy.",
        "jnp.",
        "tensorflow.",
        "tf.",
        "numpy.",
        "np.",
        "mlx.core.",
        "mlx.",
    ):
        if clean.startswith(prefix):
            clean = clean[len(prefix) :]
            break
    return DTYPE_CANONICAL_ALIASES.get(clean, clean)


def _matches_rank_constraint(passed_rk: int, constraint: int | str) -> bool:
    """Check whether a passed tensor rank satisfies a rank constraint specification.

    Args:
        passed_rk: Integer tensor rank passed by the user.
        constraint: Rank constraint (e.g. 2, '==2', '>=2', '2D', '2').

    Returns:
        True if valid, False if constraint is violated.
    """
    if isinstance(constraint, int):
        return passed_rk == constraint
    cstr = str(constraint).strip()
    if cstr.startswith("=="):
        target = int(cstr[2:])
        return passed_rk == target
    if cstr.startswith(">="):
        target = int(cstr[2:])
        return passed_rk >= target
    if cstr.endswith(("D", "d")):
        target = int(cstr[:-1])
        return passed_rk == target
    if cstr.isdigit():
        return passed_rk == int(cstr)
    return True


def _extract_cand_params(cand: dict[str, JsonValue]) -> list[dict[str, JsonValue]]:
    """Extract a list of parameter dictionaries from a candidate signature dictionary.

    Args:
        cand: Candidate signature dictionary.

    Returns:
        List of parameter dictionaries.
    """
    raw = cand.get("params") or []
    assert isinstance(raw, list)
    return [dict(p) for p in raw if isinstance(p, dict)]


def _classify_passed_kwargs(
    kwargs: typing.Sequence[str] | None,
    cand_names: set[str],
    accepted_set: set[str] | None,
    cand_has_var_kwargs: bool,
) -> tuple[list[str], list[str]]:
    """Partition passed keyword argument names into invalid and unrecognized sets.

    Args:
        kwargs: Sequence of passed keyword argument names.
        cand_names: Formal parameter names present on the candidate signature.
        accepted_set: Optional AST-discovered accepted kwargs set.
        cand_has_var_kwargs: Whether the candidate signature accepts `**kwargs`.

    Returns:
        Tuple of `(invalid_kwargs, unrecognized_var_kwargs)`.
    """
    invalid: list[str] = []
    unrecognized: list[str] = []
    if not kwargs:
        return invalid, unrecognized
    for kw in kwargs:
        if kw not in cand_names:
            if accepted_set is not None:
                if kw not in accepted_set:
                    invalid.append(kw)
            elif cand_has_var_kwargs:
                unrecognized.append(kw)
            else:
                invalid.append(kw)
    return invalid, unrecognized


def _validate_candidate_kwargs(
    cand: dict[str, JsonValue],
    kwargs: typing.Sequence[str] | None,
    strict_kwargs: bool = True,
    strict_c_extensions: bool = False,
) -> tuple[list[str], list[str], bool, bool, str | None, set[str]]:
    """Validate passed keyword argument names against a candidate signature.

    Args:
        cand: Candidate signature dictionary.
        kwargs: Sequence of passed keyword argument names.
        strict_kwargs: Whether unconstrained `**kwargs` are flagged as invalid.
        strict_c_extensions: Whether opaque C-extension signatures reject unknown kwargs.

    Returns:
        Tuple of (invalid, unrecognized, cand_has_var_kwargs, cand_is_opaque, opaque_warning, cand_names).
    """
    cand_params = _extract_cand_params(cand)
    cand_names = {str(p["name"]) for p in cand_params if "name" in p}
    cand_has_var_kwargs = any(p.get("kind") == "VAR_KEYWORD" for p in cand_params)
    raw_env_tags = cand.get("environment_tags")
    env_tags = [str(t) for t in raw_env_tags] if isinstance(raw_env_tags, list) else []
    cand_is_opaque = cand.get("signature_completeness") == "opaque" or "inexact_signature" in env_tags or "opaque_c_extension" in env_tags

    raw_dom_meta = cand.get("domain_metadata")
    domain_meta = dict(raw_dom_meta) if isinstance(raw_dom_meta, dict) else {}
    raw_accepted = cand.get("accepted_kwargs") or domain_meta.get("accepted_kwargs")
    if isinstance(raw_accepted, list):
        accepted_set: set[str] | None = {str(x) for x in raw_accepted}
        cand_has_var_kwargs = False
    else:
        accepted_set = None

    invalid, unrecognized = _classify_passed_kwargs(kwargs, cand_names, accepted_set, cand_has_var_kwargs)

    opaque_warning: str | None = None
    if cand_is_opaque and kwargs:
        opaque_warning = "WARNING: API has opaque C-extension signature; cannot definitively confirm argument validity"
        if strict_c_extensions and unrecognized:
            invalid.extend(unrecognized)
            unrecognized = []

    if strict_kwargs and unrecognized:
        invalid.extend(unrecognized)

    return (
        invalid,
        unrecognized,
        cand_has_var_kwargs,
        cand_is_opaque,
        opaque_warning,
        cand_names,
    )


def _check_known_and_literal_enums(
    cand_params: list[dict[str, JsonValue]],
    kwarg_values: typing.Mapping[str, JsonValue],
    params_with_allowed: set[str],
    enum_errors: list[str],
) -> None:
    """Append known enum fallback and `Literal[...]` annotation errors to `enum_errors`.

    Args:
        cand_params: Candidate parameter dictionaries.
        kwarg_values: Passed keyword argument values mapping.
        params_with_allowed: Lowercased parameter names already checked via `allowed_values`.
        enum_errors: Mutable list of enum validation error strings.
    """
    for k, val in kwarg_values.items():
        if val is None or not isinstance(val, str):
            continue
        k_norm = k.lower()
        if k_norm in KNOWN_ENUM_VALUES and k_norm not in params_with_allowed and not any(f"parameter '{k}'" in err for err in enum_errors):
            allowed = KNOWN_ENUM_VALUES[k_norm]
            if str(val).lower() not in [a.lower() for a in allowed]:
                enum_errors.append(f"Invalid enum value '{val}' for parameter '{k}'. Expected one of {allowed}")

    for p in cand_params:
        raw_p_name = p.get("name")
        p_name = str(raw_p_name) if raw_p_name else ""
        anno = str(p.get("annotation") or "")
        if p_name in kwarg_values and anno and any(c in anno for c in ("Literal[", "'", '"')) and not any(f"parameter '{p_name}'" in err for err in enum_errors):
            allowed_literals = re.findall(r"['\x22]([^'\x22]+)['\x22]", anno)
            if allowed_literals and str(kwarg_values[p_name]) not in allowed_literals:
                enum_errors.append(f"Invalid literal value '{kwarg_values[p_name]}' for parameter '{p_name}'. Expected one of {allowed_literals}")


def _validate_candidate_enums_and_literals(
    cand: dict[str, JsonValue],
    kwarg_values: typing.Mapping[str, JsonValue] | None,
) -> list[str]:
    """Validate enum and Literal constraints for passed keyword argument values.

    Args:
        cand: Candidate signature dictionary.
        kwarg_values: Optional dictionary mapping keyword names to runtime values.

    Returns:
        List of enum/literal validation error strings.
    """
    enum_errors: list[str] = []
    if not kwarg_values:
        return enum_errors

    cand_params = _extract_cand_params(cand)
    params_with_allowed: set[str] = set()
    for p in cand_params:
        raw_p_name = p.get("name")
        p_name = str(raw_p_name) if raw_p_name else ""
        if not p_name or p_name not in kwarg_values:
            continue
        val = kwarg_values[p_name]
        allowed_vals = p.get("allowed_values")
        if isinstance(allowed_vals, list) and allowed_vals:
            params_with_allowed.add(p_name.lower())
            if str(val).lower() not in [str(a).lower() for a in allowed_vals]:
                enum_errors.append(f"Invalid enum value '{val}' for parameter '{p_name}'. Expected one of {allowed_vals}")

    _check_known_and_literal_enums(cand_params, kwarg_values, params_with_allowed, enum_errors)
    return enum_errors


def _infer_dtypes_from_kwarg_values(
    passed_dtypes: dict[str, str],
    kwarg_values: typing.Mapping[str, JsonValue],
) -> None:
    """Populate `passed_dtypes` from string dtype literals in `kwarg_values`.

    Args:
        passed_dtypes: Mutable dictionary of parameter names to dtype strings.
        kwarg_values: Mapping of passed keyword argument names to runtime values.
    """
    for k, val in kwarg_values.items():
        val_str = str(val).lower().replace("torch.", "")
        if any(k_type in val_str for k_type in ("int", "float", "complex", "bool")) and k != "reduction":
            if k not in passed_dtypes:
                passed_dtypes[k] = val_str


def _collect_passed_dtypes(
    cand_params: list[dict[str, JsonValue]],
    kwarg_values: typing.Mapping[str, JsonValue] | None,
    arg_dtypes: typing.Sequence[str] | typing.Mapping[str, str] | JsonValue,
    kwarg_dtypes: typing.Mapping[str, str] | JsonValue,
) -> dict[str, str]:
    """Collect passed parameter dtype strings from positional, keyword, and value inputs.

    Args:
        cand_params: Candidate parameter dictionaries.
        kwarg_values: Optional dictionary of passed keyword values.
        arg_dtypes: Optional sequence or mapping of positional argument dtypes.
        kwarg_dtypes: Optional mapping of keyword argument dtypes.

    Returns:
        Dictionary mapping parameter names to passed dtype strings.
    """
    passed_dtypes: dict[str, str] = {}
    if isinstance(kwarg_dtypes, typing.Mapping):
        for k, v in kwarg_dtypes.items():
            passed_dtypes[str(k)] = str(v)
    if isinstance(arg_dtypes, typing.Mapping):
        for k, v in arg_dtypes.items():
            passed_dtypes[str(k)] = str(v)
    elif isinstance(arg_dtypes, list):
        for idx, dt in enumerate(arg_dtypes):
            if idx < len(cand_params):
                passed_dtypes[str(cand_params[idx]["name"])] = str(dt)

    if kwarg_values:
        _infer_dtypes_from_kwarg_values(passed_dtypes, kwarg_values)
    return passed_dtypes


def _collect_passed_ranks(
    cand_params: list[dict[str, JsonValue]],
    arg_ranks: typing.Sequence[int | str] | typing.Mapping[str, int | str] | JsonValue,
    kwarg_ranks: typing.Mapping[str, int | str] | JsonValue,
) -> dict[str, int | str]:
    """Collect passed parameter tensor ranks from positional and keyword rank inputs.

    Args:
        cand_params: Candidate parameter dictionaries.
        arg_ranks: Optional sequence or mapping of positional argument ranks.
        kwarg_ranks: Optional mapping of keyword argument ranks.

    Returns:
        Dictionary mapping parameter names to passed tensor ranks.
    """
    passed_ranks: dict[str, int | str] = {}
    if isinstance(kwarg_ranks, typing.Mapping):
        for k, v in kwarg_ranks.items():
            passed_ranks[str(k)] = v if isinstance(v, int) else str(v)
    if isinstance(arg_ranks, typing.Mapping):
        for k, v in arg_ranks.items():
            passed_ranks[str(k)] = v if isinstance(v, int) else str(v)
    elif isinstance(arg_ranks, list):
        for idx, rk in enumerate(arg_ranks):
            if idx < len(cand_params):
                passed_ranks[str(cand_params[idx]["name"])] = rk if isinstance(rk, int) else str(rk)
    return passed_ranks


def _check_param_dtypes_and_ranks(
    cand_params: list[dict[str, JsonValue]],
    api_path: str,
    is_float_complex_api: bool,
    passed_dtypes: dict[str, str],
    passed_ranks: dict[str, int | str],
) -> tuple[list[str], list[str]]:
    """Validate per-parameter allowed dtypes and rank constraints.

    Args:
        cand_params: Candidate parameter dictionaries.
        api_path: Target API path string.
        is_float_complex_api: Whether the API requires floating-point or complex tensors.
        passed_dtypes: Mapping of parameter names to passed dtype strings.
        passed_ranks: Mapping of parameter names to passed tensor ranks.

    Returns:
        Tuple of `(dtype_errors, rank_errors)`.
    """
    dtype_errors: list[str] = []
    rank_errors: list[str] = []
    for p in cand_params:
        p_name = str(p.get("name") or "")
        raw_p_dtypes = p.get("allowed_dtypes") or p.get("dtypes")
        p_dtypes = [str(d) for d in raw_p_dtypes] if isinstance(raw_p_dtypes, list) else None
        p_rank = p.get("rank_constraint") or p.get("rank")

        if p_name in passed_dtypes:
            dt = normalize_dtype_name(passed_dtypes[p_name])
            allowed = p_dtypes
            if not allowed and is_float_complex_api:
                allowed = ["float32", "float64", "complex64", "complex128"]
            if allowed:
                norm_allowed = [normalize_dtype_name(a) for a in allowed]
                if dt not in norm_allowed:
                    dtype_errors.append(f"Dtype '{passed_dtypes[p_name]}' is not supported for parameter '{p_name}' of '{api_path}'. Supported dtypes: {allowed}")

        if p_name in passed_ranks and isinstance(p_rank, (int, str)):
            passed_rk = passed_ranks[p_name]
            if isinstance(passed_rk, int) and not isinstance(passed_rk, bool):
                if not _matches_rank_constraint(passed_rk, p_rank):
                    rank_errors.append(f"Rank {passed_rk} is not supported for parameter '{p_name}' of '{api_path}'. Expected rank: {p_rank}")
    return dtype_errors, rank_errors


_NON_FLOAT_DTYPE_TOKENS: tuple[str, ...] = (
    "int",
    "bool",
    "uint",
    "qint",
    "quint",
    "string",
    "object",
)

_LOGICAL_BITWISE_OPS: tuple[str, ...] = (
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_not",
    "logical_and",
    "logical_or",
    "logical_xor",
    "logical_not",
)

_QUANT_OPS: tuple[str, ...] = (
    "quantize",
    "dequantize",
    "q_per_tensor",
    "q_per_channel",
)

_VALID_QUANT_DTYPES: list[str] = [
    "int8",
    "uint8",
    "qint8",
    "quint8",
    "float8_e4m3fn",
    "float8_e5m2",
]


def _append_unique_dtype_errors(
    passed_dtypes: dict[str, str],
    bad_tokens: tuple[str, ...],
    msg_suffix: str,
    api_path: str,
    dtype_errors: list[str],
) -> None:
    """Append unique dtype error strings when any passed dtype matches `bad_tokens`.

    Args:
        passed_dtypes: Mapping of parameter names to passed dtype strings.
        bad_tokens: Tuple of disallowed dtype substrings.
        msg_suffix: Explanation suffix appended after the unsupported dtype message.
        api_path: Target API path string.
        dtype_errors: Mutable list of dtype validation error strings.
    """
    for dt in passed_dtypes.values():
        if any(bad in dt.lower() for bad in bad_tokens):
            if not any(f"Dtype '{dt}'" in err for err in dtype_errors):
                dtype_errors.append(f"Dtype '{dt}' is not supported for '{api_path}'. {msg_suffix}")


def _check_domain_dtype_rules(
    api_path: str,
    clean_api: str,
    is_float_complex_api: bool,
    passed_dtypes: dict[str, str],
    dtype_errors: list[str],
) -> None:
    """Append domain-wide dtype errors (linear algebra, logical/bitwise, quantization).

    Args:
        api_path: Target API path string.
        clean_api: Unqualified lowercased operation name.
        is_float_complex_api: Whether the operation requires float or complex dtypes.
        passed_dtypes: Mapping of parameter names to passed dtype strings.
        dtype_errors: Mutable list of dtype validation error messages.
    """
    if is_float_complex_api and passed_dtypes:
        _append_unique_dtype_errors(
            passed_dtypes=passed_dtypes,
            bad_tokens=_NON_FLOAT_DTYPE_TOKENS,
            msg_suffix="Floating-point or complex dtype required.",
            api_path=api_path,
            dtype_errors=dtype_errors,
        )

    if any(op in clean_api for op in _LOGICAL_BITWISE_OPS) and passed_dtypes:
        _append_unique_dtype_errors(
            passed_dtypes=passed_dtypes,
            bad_tokens=("float", "complex"),
            msg_suffix="Logical/bitwise operations require integer or boolean types.",
            api_path=api_path,
            dtype_errors=dtype_errors,
        )

    if any(op in clean_api for op in _QUANT_OPS) and "dtype" in passed_dtypes:
        q_dt = passed_dtypes["dtype"].lower()
        if not any(vq in q_dt for vq in _VALID_QUANT_DTYPES):
            dtype_errors.append(f"Dtype '{q_dt}' is not a valid quantization dtype for '{api_path}'. Expected one of {_VALID_QUANT_DTYPES}.")


def _validate_candidate_dtypes_and_ranks(
    cand: dict[str, JsonValue],
    api_path: str,
    framework: str,
    kwarg_values: typing.Mapping[str, JsonValue] | None = None,
    arg_dtypes: typing.Sequence[str] | typing.Mapping[str, str] | JsonValue = None,
    kwarg_dtypes: typing.Mapping[str, str] | JsonValue = None,
    arg_ranks: (typing.Sequence[int | str] | typing.Mapping[str, int | str] | JsonValue) = None,
    kwarg_ranks: typing.Mapping[str, int | str] | JsonValue = None,
) -> tuple[list[str], list[str]]:
    """Validate tensor dtypes and ranks against a candidate signature.

    Args:
        cand: Candidate signature dictionary.
        api_path: Target API path string.
        framework: Target framework identifier.
        kwarg_values: Optional dictionary of passed keyword values.
        arg_dtypes: Optional list or dictionary of positional argument dtypes.
        kwarg_dtypes: Optional dictionary of keyword argument dtypes.
        arg_ranks: Optional list or dictionary of positional argument ranks.
        kwarg_ranks: Optional dictionary of keyword argument ranks.

    Returns:
        Tuple of (dtype_errors, rank_errors).
    """
    cand_params = _extract_cand_params(cand)
    passed_dtypes = _collect_passed_dtypes(cand_params, kwarg_values, arg_dtypes, kwarg_dtypes)
    passed_ranks = _collect_passed_ranks(cand_params, arg_ranks, kwarg_ranks)

    clean_api = api_path.split(".")[-1].lower()
    is_float_complex_api = framework == "torch" and (
        clean_api
        in (
            "inv",
            "linalg_inv",
            "cholesky",
            "linalg_cholesky",
            "det",
            "eig",
            "svd",
            "solve",
        )
        or "linalg.inv" in api_path
        or "linalg.cholesky" in api_path
        or "cholesky" in clean_api
    )

    dtype_errors, rank_errors = _check_param_dtypes_and_ranks(
        cand_params=cand_params,
        api_path=api_path,
        is_float_complex_api=is_float_complex_api,
        passed_dtypes=passed_dtypes,
        passed_ranks=passed_ranks,
    )
    _check_domain_dtype_rules(api_path, clean_api, is_float_complex_api, passed_dtypes, dtype_errors)
    return dtype_errors, rank_errors


def _validate_candidate_shapes(
    cand: dict[str, JsonValue],
    api_path: str,
    clean_api: str,
    arg_shapes: typing.Sequence[typing.Sequence[int | str]] | JsonValue = None,
    kwarg_shapes: typing.Mapping[str, typing.Sequence[int | str]] | JsonValue = None,
) -> list[str]:
    """Validate tensor shapes (broadcasting and matmul contraction) against a candidate.

    Args:
        cand: Candidate signature dictionary.
        api_path: Target API path string.
        clean_api: Unqualified operation name.
        arg_shapes: Optional list of positional argument shapes.
        kwarg_shapes: Optional dictionary of keyword argument shapes.

    Returns:
        List of shape validation error strings.
    """
    shape_errors: list[str] = []
    cand_params = _extract_cand_params(cand)
    passed_shapes: dict[str, typing.Sequence[int | str]] = {}
    if isinstance(kwarg_shapes, typing.Mapping):
        for k, sh_val in kwarg_shapes.items():
            passed_shapes[str(k)] = [d if isinstance(d, int) else str(d) for d in sh_val] if isinstance(sh_val, typing.Sequence) and not isinstance(sh_val, str) else []
    if isinstance(arg_shapes, list):
        for idx, sh in enumerate(arg_shapes):
            norm_sh: list[int | str] = [d if isinstance(d, int) else str(d) for d in sh] if isinstance(sh, typing.Sequence) and not isinstance(sh, str) else []
            if idx < len(cand_params):
                passed_shapes[str(cand_params[idx]["name"])] = norm_sh
            else:
                passed_shapes[f"arg_{idx}"] = norm_sh

    is_elementwise = (
        any(
            op in clean_api
            for op in (
                "add",
                "sub",
                "subtract",
                "mul",
                "multiply",
                "div",
                "divide",
                "maximum",
                "minimum",
                "pow",
            )
        )
        and "matmul" not in clean_api
    )
    if is_elementwise and len(passed_shapes) >= 2:
        from ml_ecosystem_snapshots.compliance import validate_broadcast_shapes

        sh_list = list(passed_shapes.values())
        compat, _, err = validate_broadcast_shapes(sh_list[0], sh_list[1])
        if not compat and err:
            shape_errors.append(f"Broadcasting error in '{api_path}': {err}")

    is_matmul = any(op in clean_api for op in ("matmul", "mm", "bmm")) or clean_api == "dot"
    if is_matmul and len(passed_shapes) >= 2:
        from ml_ecosystem_snapshots.compliance import validate_matmul_shapes

        strict_2d = clean_api == "mm"
        sh_list = list(passed_shapes.values())
        compat, _, err = validate_matmul_shapes(sh_list[0], sh_list[1], strict_2d=strict_2d)
        if not compat and err:
            shape_errors.append(f"Matmul error in '{api_path}': {err}")

    return shape_errors


def _validate_candidate_positional_arity(
    cand: dict[str, JsonValue],
    args_count: int | None,
) -> str | None:
    """Validate positional argument count against a candidate signature.

    Args:
        cand: Candidate signature dictionary.
        args_count: Optional number of positional arguments passed.

    Returns:
        Error string if positional arity is violated, or None if valid.
    """
    if args_count is None:
        return None
    cand_params = _extract_cand_params(cand)
    min_pos = sum(1 for p in cand_params if p.get("default") is None and p.get("kind") in ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD"))
    max_pos = sum(1 for p in cand_params if p.get("kind") in ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD"))
    cand_has_varargs = bool(cand.get("has_varargs", False)) or any(p.get("kind") == "VAR_POSITIONAL" for p in cand_params)
    if args_count < min_pos:
        return f"Missing required positional arguments: expected at least {min_pos}, got {args_count}"
    if not cand_has_varargs and args_count > max_pos:
        return f"Too many positional arguments: expected at most {max_pos}, got {args_count}"
    return None


def _build_candidate_reason(
    cand_hallucinated: bool,
    invalid: list[str],
    unrecognized: list[str],
    enum_errors: list[str],
    dtype_errors: list[str],
    rank_errors: list[str],
    shape_errors: list[str],
    positional_error: str | None,
) -> str:
    """Format the summary validation reason for a candidate overload evaluation.

    Args:
        cand_hallucinated: Whether the candidate failed any validation checks.
        invalid: List of invalid keyword argument names.
        unrecognized: List of unrecognized keyword arguments passed to `**kwargs`.
        enum_errors: List of enum/literal validation error messages.
        dtype_errors: List of tensor dtype validation error messages.
        rank_errors: List of tensor rank validation error messages.
        shape_errors: List of tensor shape/broadcasting validation error messages.
        positional_error: Optional positional arity error string.

    Returns:
        Formatted reason string describing why the candidate passed or failed.
    """
    if cand_hallucinated:
        reasons: list[str] = []
        if invalid:
            reasons.append(f"Invalid/hallucinated keyword arguments: {', '.join(invalid)}")
        for err_list in (enum_errors, dtype_errors, rank_errors, shape_errors):
            if err_list:
                reasons.extend(err_list)
        if positional_error:
            reasons.append(positional_error)
        return "; ".join(reasons)
    if unrecognized:
        return f"Unconstrained keyword arguments passed to **kwargs: {', '.join(unrecognized)}"
    return "Valid API call"


def _evaluate_overload_candidate(
    cand: dict[str, JsonValue],
    framework: str,
    api_path: str,
    kwargs: list[str] | None,
    args_count: int | None,
    strict_kwargs: bool,
    kwarg_values: typing.Mapping[str, JsonValue] | None,
    options: _HallucinationOptions,
) -> tuple[HallucinationCheckResult, int]:
    """Evaluate a single candidate signature or overload against passed call arguments.

    Args:
        cand: Candidate signature dictionary.
        framework: Target framework identifier.
        api_path: Target API path string.
        kwargs: Optional list of keyword argument names.
        args_count: Optional number of positional arguments passed.
        strict_kwargs: Whether unconstrained `**kwargs` are flagged as invalid.
        kwarg_values: Optional mapping of passed keyword argument values.
        options: Additional hallucination validation options (`_HallucinationOptions`).

    Returns:
        Tuple of `(candidate_result, total_error_count)`.
    """
    clean_api = api_path.split(".")[-1].lower()
    (
        invalid,
        unrecognized,
        cand_has_var_kwargs,
        cand_is_opaque,
        opaque_warning,
        cand_names,
    ) = _validate_candidate_kwargs(
        cand=cand,
        kwargs=kwargs,
        strict_kwargs=strict_kwargs,
        strict_c_extensions=options.strict_c_extensions,
    )
    enum_errors = _validate_candidate_enums_and_literals(cand, kwarg_values)
    dtype_errors, rank_errors = _validate_candidate_dtypes_and_ranks(
        cand=cand,
        api_path=api_path,
        framework=framework,
        kwarg_values=kwarg_values,
        arg_dtypes=options.arg_dtypes,
        kwarg_dtypes=options.kwarg_dtypes,
        arg_ranks=options.arg_ranks,
        kwarg_ranks=options.kwarg_ranks,
    )
    shape_errors = _validate_candidate_shapes(
        cand=cand,
        api_path=api_path,
        clean_api=clean_api,
        arg_shapes=options.arg_shapes,
        kwarg_shapes=options.kwarg_shapes,
    )
    positional_error = _validate_candidate_positional_arity(cand, args_count)

    err_count = len(invalid) + len(enum_errors) + len(dtype_errors) + len(rank_errors) + len(shape_errors) + (1 if positional_error else 0)
    cand_hallucinated = err_count > 0
    reason = _build_candidate_reason(
        cand_hallucinated=cand_hallucinated,
        invalid=invalid,
        unrecognized=unrecognized,
        enum_errors=enum_errors,
        dtype_errors=dtype_errors,
        rank_errors=rank_errors,
        shape_errors=shape_errors,
        positional_error=positional_error,
    )
    if opaque_warning and not cand_hallucinated and reason == "Valid API call":
        reason = f"Valid API call with warning: {opaque_warning}"

    res = HallucinationCheckResult(
        api_exists=True,
        is_hallucinated=cand_hallucinated,
        invalid_kwargs=invalid,
        canonical_params=sorted(list(cand_names)),
        has_unconstrained_kwargs=cand_has_var_kwargs,
        reason=reason,
        snapshot_source="bundled",
        signature_completeness=str(cand.get("signature_completeness", "opaque" if cand_is_opaque else "exact")),
        is_c_extension=bool(cand.get("is_c_extension", False) or cand_is_opaque),
        dtype_errors=dtype_errors if dtype_errors else None,
        rank_errors=rank_errors if rank_errors else None,
        shape_errors=shape_errors if shape_errors else None,
        warning=opaque_warning,
    )
    return res, err_count


def check_hallucination(
    framework: str,
    api_path: str,
    kwargs: list[str] | None = None,
    args_count: int | None = None,
    strict_kwargs: bool = True,
    kwarg_values: typing.Mapping[str, JsonValue] | None = None,
    **options: _HallucinationOptionValue,
) -> HallucinationCheckResult:
    """Verify whether an API or specified arguments represent hallucinations.

    Checks:
        - Whether the API path exists in the ground-truth snapshot.
        - Whether any passed keyword arguments are non-existent in the signature.
        - Flags unconstrained keyword arguments when **kwargs is present.
        - Verifies positional parameter arity and counts.
        - Validates string enum parameters against allowed options (e.g. reduction='mean').
        - Suggests canonical argument names (e.g. 'dim' for 'axis' in torch).
        - Validates tensor argument dtypes against allowed dtypes (e.g. float-only for inv/cholesky).
        - Validates tensor argument ranks against expected dimensions (e.g. rank 2 for mm).
        - Validates tensor broadcasting compatibility and matmul contracting dimensions.
        - Flags opaque C-extension signatures to prevent false-negative hallucination passes.

    Args:
        framework: The target framework name.
        api_path: The requested API path.
        kwargs: List of argument names passed to the API.
        args_count: Optional number of positional arguments passed.
        strict_kwargs: Whether to treat unconstrained kwargs as hallucinated.
        kwarg_values: Optional dictionary mapping passed keyword argument names to their values for type and enum validation.
        options: Optional keyword arguments (`version`, `arg_dtypes`, `kwarg_dtypes`, `arg_ranks`, `kwarg_ranks`, `arg_shapes`, `kwarg_shapes`, `strict_c_extensions`, `repository`).

    Returns:
        `HallucinationCheckResult` model containing validation details.
    """
    if kwargs is None and kwarg_values is not None:
        kwargs = list(kwarg_values.keys())

    opts = _HallucinationOptions.model_validate(options)
    version = opts.version
    repository = opts.repository
    sig = get_api_signature(framework, api_path, version=version, repository=repository)
    snap = resolve_snapshot(framework, version=version, repository=repository)
    raw_source = snap.get("_snapshot_source")
    snapshot_source = str(raw_source) if raw_source is not None else ("not_found" if not snap.get("categories") else "unknown")
    if not sig:
        if snapshot_source == "not_found":
            reason = f"API '{api_path}' could not be verified: no local ground-truth snapshot for framework '{framework}' (version '{version or 'any'}') was found in bundled packages or local cache. To ground offline, install '{framework}' or place a versioned snapshot JSON in the cache directory."
        else:
            reason = f"API '{api_path}' does not exist in ground-truth '{framework}' snapshot."
        return HallucinationCheckResult(
            api_exists=False,
            is_hallucinated=True,
            invalid_kwargs=kwargs or [],
            canonical_params=[],
            has_unconstrained_kwargs=False,
            snapshot_source=snapshot_source,
            reason=reason,
        )

    raw_overloads = sig.get("overloads")
    overload_dicts: list[dict[str, JsonValue]] = [dict(ov) for ov in raw_overloads if isinstance(ov, dict)] if isinstance(raw_overloads, list) else []
    candidates = [sig] + overload_dicts

    best_result: HallucinationCheckResult | None = None
    fewest_errors = 999999

    for cand in candidates:
        res, err_count = _evaluate_overload_candidate(
            cand=cand,
            framework=framework,
            api_path=api_path,
            kwargs=kwargs,
            args_count=args_count,
            strict_kwargs=strict_kwargs,
            kwarg_values=kwarg_values,
            options=opts,
        )
        res.snapshot_source = snapshot_source
        if not res["is_hallucinated"]:
            return res
        if err_count < fewest_errors:
            fewest_errors = err_count
            best_result = res

    assert best_result is not None
    return best_result


def explain_anti_pattern(
    framework: str,
    api_path: str,
    hallucinated_argument: str,
    passed_value: JsonValue = None,
    repository: SnapshotRepository | None = None,
) -> AntiPatternExplanationResult:
    """Provide canonical migration advice and rationale for a flagged anti-pattern or hallucinated argument.

    Args:
        framework: Target ML framework name (e.g. 'torch', 'jax', 'tensorflow').
        api_path: Target API path (e.g. 'torch.sum', 'jax.numpy.mean').
        hallucinated_argument: Name of the flagged/hallucinated keyword argument.
        passed_value: Optional value passed to the argument for contextual advice.
        repository: Optional `SnapshotRepository` instance for snapshot lookup.

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

    sig = get_api_signature(framework, api_path, repository=repository)
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


class _CallVisitor(ast.NodeVisitor):
    """AST visitor for tracking module imports and collecting qualified function calls."""

    def __init__(self) -> None:
        """Initialize empty import table and call records list."""
        self.imports: dict[str, str] = {}
        self.calls: list[tuple[int, str, list[str], int, dict[str, JsonValue]]] = []

    def visit_Import(self, node: ast.Import) -> None:
        """Track 'import foo' and 'import foo as bar' statements.

        Args:
            node: The AST Import node.
        """
        for alias in node.names:
            name = alias.name
            asname = alias.asname or name
            self.imports[asname] = name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track 'from foo.bar import baz' statements.

        Args:
            node: The AST ImportFrom node.
        """
        mod = node.module or ""
        for alias in node.names:
            name = alias.name
            asname = alias.asname or name
            full = f"{mod}.{name}" if mod else name
            self.imports[asname] = full
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Inspect Call AST nodes and record resolved target path and constant kwargs.

        Args:
            node: The AST Call node.
        """
        curr: ast.expr = node.func
        attr_parts: list[str] = []
        while isinstance(curr, ast.Attribute):
            attr_parts.append(curr.attr)
            curr = curr.value

        root_id = curr.id if isinstance(curr, ast.Name) else None
        if root_id is not None:
            resolved_base = self.imports.get(root_id, root_id)
            if attr_parts:
                full_path = f"{resolved_base}.{'.'.join(reversed(attr_parts))}"
            else:
                full_path = resolved_base

            kw_names = [kw.arg for kw in node.keywords if kw.arg]
            kw_vals: dict[str, JsonValue] = {}
            for kw in node.keywords:
                if kw.arg and isinstance(kw.value, ast.Constant):
                    val = kw.value.value
                    assert isinstance(val, (str, int, float, bool)) or val is None
                    kw_vals[kw.arg] = val
            self.calls.append((node.lineno, full_path, kw_names, len(node.args), kw_vals))
        self.generic_visit(node)


_CALL_PREFIX_FRAMEWORK_MAP: tuple[tuple[str, str, str | None], ...] = (
    ("torch.", "torch", None),
    ("torch", "torch", None),
    ("jax.", "jax", None),
    ("jax", "jax", None),
    ("tf.", "tensorflow", None),
    ("tensorflow.", "tensorflow", None),
    ("tensorflow", "tensorflow", None),
    ("np.", "numpy", None),
    ("numpy.", "numpy", None),
    ("numpy", "numpy", None),
    ("nn.", "torch", "torch.nn."),
    ("F.", "torch", "torch.F."),
    ("stablehlo.", "stablehlo", None),
    ("chex.", "chex", None),
    ("grain.", "grain", None),
    ("orbax.", "orbax", None),
    ("ocp.", "orbax", "orbax.checkpoint."),
    ("optax.", "optax", None),
)


def _infer_call_framework(
    api_name: str,
    framework_hint: str | None,
) -> tuple[str | None, str]:
    """Infer target framework and normalized API path from a Python call target.

    Args:
        api_name: Dotted API path extracted from the AST Call node.
        framework_hint: Optional caller-supplied framework name.

    Returns:
        Tuple of `(resolved_framework, normalized_api_name)`.
    """
    if framework_hint:
        return framework_hint, api_name
    for prefix, fw, replacement in _CALL_PREFIX_FRAMEWORK_MAP:
        if api_name.startswith(prefix):
            norm_api = f"{replacement}{api_name[len(prefix) :]}" if replacement else api_name
            return fw, norm_api
    return None, api_name


def _check_python_calls(
    calls: list[tuple[int, str, list[str], int, dict[str, JsonValue]]],
    framework: str | None,
    version: str | None,
    repository: SnapshotRepository | None,
) -> tuple[int, list[CodeBlockFinding]]:
    """Validate extracted Python AST calls against ground-truth framework snapshots.

    Args:
        calls: List of `(lineno, api_name, kw_names, args_count, kw_vals)` tuples.
        framework: Optional framework hint.
        version: Optional framework version string.
        repository: Optional `SnapshotRepository` instance.

    Returns:
        Tuple of `(analyzed_call_count, findings_list)`.
    """
    findings: list[CodeBlockFinding] = []
    total_analyzed = 0
    for lineno, raw_api_name, kw_list, arg_cnt, kw_vals in calls:
        fw, api_name = _infer_call_framework(raw_api_name, framework)
        if not fw:
            continue

        total_analyzed += 1
        res = check_hallucination(
            framework=fw,
            api_path=api_name,
            kwargs=kw_list,
            args_count=arg_cnt,
            kwarg_values=kw_vals if kw_vals else None,
            version=version,
            repository=repository,
        )
        if res.get("is_hallucinated"):
            findings.append(
                CodeBlockFinding(
                    line=lineno,
                    type="api_hallucination",
                    framework=fw,
                    target=api_name,
                    reason=str(res.get("reason", "Invalid API call")),
                    invalid_kwargs=list(res.get("invalid_kwargs", [])),
                )
            )
    return total_analyzed, findings


_MLIR_DIALECT_PREFIXES: tuple[str, ...] = (
    "arith.",
    "math.",
    "linalg.",
    "stablehlo.",
    "memref.",
    "func.",
    "scf.",
    "gpu.",
    "vector.",
    "llvm.",
    "nvvm.",
    "rocdl.",
)


def _check_assembly_or_ir_line(
    lineno: int,
    line: str,
    framework: str | None,
    sm_arch: str | None,
    gfx_arch: str | None,
    repository: SnapshotRepository | None,
) -> tuple[int, CodeBlockFinding | None]:
    """Validate a single non-Python assembly or MLIR/StableHLO instruction line.

    Args:
        lineno: 1-based source line number.
        line: Stripped non-empty, non-comment instruction line.
        framework: Optional framework hint ('mlir', 'stablehlo', 'nvidia_sass', 'amd_rdna').
        sm_arch: Optional NVIDIA SM architecture target.
        gfx_arch: Optional AMD RDNA GFX architecture target.
        repository: Optional `SnapshotRepository` instance.

    Returns:
        Tuple of `(analyzed_increment, optional_finding)`.
    """
    mlir_match = re.search(r"(?:%[a-zA-Z0-9_]+\s*=\s*)?([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)", line)
    sass_match = re.match(r"^(?:@!?U?P\d+\s+)?([A-Z0-9_\.]+)(?:\s+(.*))?;?$", line)
    rdna_match = re.match(r"^([vsa]_[a-zA-Z0-9_\.]+)(?:\s+(.*))?$", line)

    if mlir_match and (framework in ("mlir", "stablehlo") or any(d in line for d in _MLIR_DIALECT_PREFIXES)):
        op_name = mlir_match.group(1)
        mlir_res = check_mlir_op(op_name, repository=repository)
        finding = (
            None
            if mlir_res.get("is_valid")
            else CodeBlockFinding(
                line=lineno,
                type="mlir_error",
                target=op_name,
                reason="; ".join(mlir_res.get("errors", [])),
            )
        )
        return 1, finding

    if sass_match and (framework == "nvidia_sass" or any(s in line for s in ("R0", "UR0", "P0", "sm_")) or line.endswith(";")):
        from ml_ecosystem_snapshots.frameworks.nvidia_sass import tokenize_sass_line

        _pred, mnem, mods, ops = tokenize_sass_line(line)
        sass_res = check_sass_instruction(
            mnem,
            operands=ops,
            modifiers=mods,
            sm_arch=sm_arch,
            repository=repository,
        )
        finding = (
            None
            if sass_res.get("is_valid")
            else CodeBlockFinding(
                line=lineno,
                type="sass_error",
                target=mnem,
                reason="; ".join(sass_res.get("errors", [])),
            )
        )
        return 1, finding

    if rdna_match:
        from ml_ecosystem_snapshots.frameworks.amd_rdna import tokenize_rdna_line

        base_mnem, ops, _enc_suf, mods = tokenize_rdna_line(line)
        rdna_res = check_rdna_instruction(
            base_mnem,
            operands=ops,
            modifiers=mods,
            gfx_arch=gfx_arch,
            repository=repository,
        )
        finding = (
            None
            if rdna_res.get("is_valid")
            else CodeBlockFinding(
                line=lineno,
                type="rdna_error",
                target=base_mnem,
                reason="; ".join(rdna_res.get("errors", [])),
            )
        )
        return 1, finding

    return 0, None


def check_code_block(
    code: str,
    framework: str | None = None,
    version: str | None = None,
    sm_arch: str | None = None,
    gfx_arch: str | None = None,
    repository: SnapshotRepository | None = None,
) -> CodeBlockCheckResult:
    """Batch verify a code block for nonexistent APIs, invalid kwargs, and hardware constraints.

    Analyzes Python AST for framework calls or assembly/IR lines for SASS, RDNA, and MLIR
    instructions, returning a comprehensive hallucination and constraint validation report.

    Args:
        code: Python snippet, SASS assembly, RDNA assembly, or MLIR text.
        framework: Optional framework hint ('torch', 'jax', 'nvidia_sass', 'amd_rdna', 'mlir').
        version: Optional framework version string.
        sm_arch: Optional NVIDIA architecture target (e.g. 'sm_80', 'sm_90').
        gfx_arch: Optional AMD RDNA architecture target (e.g. 'GFX11/RDNA3').
        repository: Optional `SnapshotRepository` instance for snapshot lookup.

    Returns:
        `CodeBlockCheckResult` containing 'total_analyzed', 'hallucinations_detected', 'is_valid', and 'findings'.
    """
    findings: list[CodeBlockFinding] = []
    total_analyzed = 0
    is_python = False

    try:
        tree = ast.parse(code)
        visitor = _CallVisitor()
        visitor.visit(tree)
        if visitor.calls:
            is_python = True
            total_analyzed, findings = _check_python_calls(
                calls=visitor.calls,
                framework=framework,
                version=version,
                repository=repository,
            )
    except SyntaxError:
        is_python = False

    if not is_python:
        for lineno, raw_line in enumerate(code.strip().splitlines(), 1):
            line = raw_line.strip()
            if not line or line.startswith(("//", "#", ";")):
                continue
            inc, finding = _check_assembly_or_ir_line(
                lineno=lineno,
                line=line,
                framework=framework,
                sm_arch=sm_arch,
                gfx_arch=gfx_arch,
                repository=repository,
            )
            total_analyzed += inc
            if finding is not None:
                findings.append(finding)

    return CodeBlockCheckResult(
        total_analyzed=total_analyzed,
        hallucinations_detected=len(findings),
        is_valid=len(findings) == 0,
        findings=findings,
    )


__all__ = [
    "DTYPE_CANONICAL_ALIASES",
    "KNOWN_ENUM_VALUES",
    "check_code_block",
    "check_hallucination",
    "explain_anti_pattern",
    "normalize_dtype_name",
]
