"""Canonical dtype aliases, enum/literal validators, and tensor rank/shape constraint matchers."""

from __future__ import annotations

import re
import typing

import pydantic
import typing_extensions

from ml_ecosystem_snapshots.introspection.inspector import STANDARD_ENUM_MAP
from ml_ecosystem_snapshots.models import JsonValue, ensure_json_dict


class CallConstraintOptions(pydantic.BaseModel):
    """Optional tensor dtype, rank, shape, and C-extension validation options."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    arg_dtypes: typing.Sequence[str] | typing.Mapping[str, str] | None = None
    kwarg_dtypes: typing.Mapping[str, str] | None = None
    arg_ranks: typing.Sequence[int | str] | typing.Mapping[str, int | str] | None = None
    kwarg_ranks: typing.Mapping[str, int | str] | None = None
    arg_shapes: typing.Sequence[typing.Sequence[int | str]] | None = None
    kwarg_shapes: typing.Mapping[str, typing.Sequence[int | str]] | None = None
    strict_c_extensions: bool = False


CallOptionValue: typing_extensions.TypeAlias = (
    "typing.Sequence[str]"
    " | typing.Mapping[str, str]"
    " | typing.Sequence[int | str]"
    " | typing.Mapping[str, int | str]"
    " | typing.Sequence[typing.Sequence[int | str]]"
    " | typing.Mapping[str, typing.Sequence[int | str]]"
    " | bool"
    " | None"
)


def _extract_cand_params(cand: typing.Mapping[str, JsonValue]) -> list[dict[str, JsonValue]]:
    """Extract normalized parameter dictionaries from an overload candidate mapping.

    Args:
        cand: Candidate signature dictionary.

    Returns:
        List of parameter dictionaries.
    """
    raw_params = cand.get("params")
    return [ensure_json_dict(p) for p in raw_params] if isinstance(raw_params, list) else []


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


def matches_rank_constraint(passed_rk: int, constraint: int | str) -> bool:
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
        if (
            k_norm in KNOWN_ENUM_VALUES
            and k_norm not in params_with_allowed
            and not any(f"parameter '{k}'" in err for err in enum_errors)
        ):
            allowed = KNOWN_ENUM_VALUES[k_norm]
            if str(val).lower() not in [a.lower() for a in allowed]:
                enum_errors.append(f"Invalid enum value '{val}' for parameter '{k}'. Expected one of {allowed}")

    for p in cand_params:
        raw_p_name = p.get("name")
        p_name = str(raw_p_name) if raw_p_name else ""
        anno = str(p.get("annotation") or "")
        if (
            p_name in kwarg_values
            and anno
            and any(c in anno for c in ("Literal[", "'", '"'))
            and not any(f"parameter '{p_name}'" in err for err in enum_errors)
        ):
            allowed_literals = re.findall(r"['\x22]([^'\x22]+)['\x22]", anno)
            if allowed_literals and str(kwarg_values[p_name]) not in allowed_literals:
                enum_errors.append(
                    f"Invalid literal value '{kwarg_values[p_name]}' for parameter '{p_name}'. Expected one of {allowed_literals}"
                )


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
                enum_errors.append(
                    f"Invalid enum value '{val}' for parameter '{p_name}'. Expected one of {allowed_vals}"
                )

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
                    dtype_errors.append(
                        f"Dtype '{passed_dtypes[p_name]}' is not supported for parameter '{p_name}' of '{api_path}'. Supported dtypes: {allowed}"
                    )

        if p_name in passed_ranks and isinstance(p_rank, (int, str)):
            passed_rk = passed_ranks[p_name]
            if isinstance(passed_rk, int) and not isinstance(passed_rk, bool):
                if not matches_rank_constraint(passed_rk, p_rank):
                    rank_errors.append(
                        f"Rank {passed_rk} is not supported for parameter '{p_name}' of '{api_path}'. Expected rank: {p_rank}"
                    )
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
            dtype_errors.append(
                f"Dtype '{q_dt}' is not a valid quantization dtype for '{api_path}'. Expected one of {_VALID_QUANT_DTYPES}."
            )


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
            passed_shapes[str(k)] = (
                [d if isinstance(d, int) else str(d) for d in sh_val]
                if isinstance(sh_val, typing.Sequence) and not isinstance(sh_val, str)
                else []
            )
    if isinstance(arg_shapes, list):
        for idx, sh in enumerate(arg_shapes):
            norm_sh: list[int | str] = (
                [d if isinstance(d, int) else str(d) for d in sh]
                if isinstance(sh, typing.Sequence) and not isinstance(sh, str)
                else []
            )
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


__all__ = [
    "CallConstraintOptions",
    "CallOptionValue",
    "DTYPE_CANONICAL_ALIASES",
    "KNOWN_ENUM_VALUES",
    "_extract_cand_params",
    "_validate_candidate_dtypes_and_ranks",
    "_validate_candidate_enums_and_literals",
    "_validate_candidate_shapes",
    "matches_rank_constraint",
    "normalize_dtype_name",
]
