"""MCP tool module for hallucination verification, anti-pattern explanation, and AST code-block checking."""

from __future__ import annotations

import ast
import re
import typing

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
    cand_is_opaque = (
        cand.get("signature_completeness") == "opaque"
        or "inexact_signature" in env_tags
        or "opaque_c_extension" in env_tags
    )

    raw_dom_meta = cand.get("domain_metadata")
    domain_meta = dict(raw_dom_meta) if isinstance(raw_dom_meta, dict) else {}
    raw_accepted = cand.get("accepted_kwargs") or domain_meta.get("accepted_kwargs")
    if isinstance(raw_accepted, list):
        accepted_set: set[str] | None = {str(x) for x in raw_accepted}
        cand_has_var_kwargs = False
    else:
        accepted_set = None

    invalid: list[str] = []
    unrecognized: list[str] = []
    if kwargs:
        for kw in kwargs:
            if kw not in cand_names:
                if accepted_set is not None:
                    if kw not in accepted_set:
                        invalid.append(kw)
                elif cand_has_var_kwargs:
                    unrecognized.append(kw)
                else:
                    invalid.append(kw)

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
                enum_errors.append(
                    f"Invalid enum value '{val}' for parameter '{k}'. Expected one of {allowed}"
                )

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

    return enum_errors


def _validate_candidate_dtypes_and_ranks(
    cand: dict[str, JsonValue],
    api_path: str,
    framework: str,
    kwarg_values: typing.Mapping[str, JsonValue] | None = None,
    arg_dtypes: typing.Sequence[str] | typing.Mapping[str, str] | JsonValue = None,
    kwarg_dtypes: typing.Mapping[str, str] | JsonValue = None,
    arg_ranks: (
        typing.Sequence[int | str] | typing.Mapping[str, int | str] | JsonValue
    ) = None,
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
    dtype_errors: list[str] = []
    rank_errors: list[str] = []
    cand_params = _extract_cand_params(cand)

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
                passed_ranks[str(cand_params[idx]["name"])] = (
                    rk if isinstance(rk, int) else str(rk)
                )

    if kwarg_values:
        for k, val in kwarg_values.items():
            val_str = str(val).lower().replace("torch.", "")
            if (
                any(k_type in val_str for k_type in ("int", "float", "complex", "bool"))
                and k != "reduction"
            ):
                if k not in passed_dtypes:
                    passed_dtypes[k] = val_str

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

    for p in cand_params:
        p_name = str(p.get("name") or "")
        raw_p_dtypes = p.get("allowed_dtypes") or p.get("dtypes")
        p_dtypes = (
            [str(d) for d in raw_p_dtypes] if isinstance(raw_p_dtypes, list) else None
        )
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
                if not _matches_rank_constraint(passed_rk, p_rank):
                    rank_errors.append(
                        f"Rank {passed_rk} is not supported for parameter '{p_name}' of '{api_path}'. Expected rank: {p_rank}"
                    )

    if is_float_complex_api and passed_dtypes:
        for _p_k, dt in passed_dtypes.items():
            if any(
                bad in dt.lower()
                for bad in (
                    "int",
                    "bool",
                    "uint",
                    "qint",
                    "quint",
                    "string",
                    "object",
                )
            ):
                if not any(f"Dtype '{dt}'" in err for err in dtype_errors):
                    dtype_errors.append(
                        f"Dtype '{dt}' is not supported for '{api_path}'. Floating-point or complex dtype required."
                    )

    is_logical_bitwise = any(
        op in clean_api
        for op in (
            "bitwise_and",
            "bitwise_or",
            "bitwise_xor",
            "bitwise_not",
            "logical_and",
            "logical_or",
            "logical_xor",
            "logical_not",
        )
    )
    if is_logical_bitwise and passed_dtypes:
        for _p_k, dt in passed_dtypes.items():
            if any(bad in dt.lower() for bad in ("float", "complex")):
                if not any(f"Dtype '{dt}'" in err for err in dtype_errors):
                    dtype_errors.append(
                        f"Dtype '{dt}' is not supported for '{api_path}'. Logical/bitwise operations require integer or boolean types."
                    )

    is_quant_api = any(
        op in clean_api
        for op in ("quantize", "dequantize", "q_per_tensor", "q_per_channel")
    )
    if is_quant_api and "dtype" in passed_dtypes:
        q_dt = passed_dtypes["dtype"].lower()
        valid_q = [
            "int8",
            "uint8",
            "qint8",
            "quint8",
            "float8_e4m3fn",
            "float8_e5m2",
        ]
        if not any(vq in q_dt for vq in valid_q):
            dtype_errors.append(
                f"Dtype '{q_dt}' is not a valid quantization dtype for '{api_path}'. Expected one of {valid_q}."
            )

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

    is_matmul = (
        any(op in clean_api for op in ("matmul", "mm", "bmm")) or clean_api == "dot"
    )
    if is_matmul and len(passed_shapes) >= 2:
        from ml_ecosystem_snapshots.compliance import validate_matmul_shapes

        strict_2d = clean_api == "mm"
        sh_list = list(passed_shapes.values())
        compat, _, err = validate_matmul_shapes(
            sh_list[0], sh_list[1], strict_2d=strict_2d
        )
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
    min_pos = sum(
        1
        for p in cand_params
        if p.get("default") is None
        and p.get("kind") in ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD")
    )
    max_pos = sum(
        1
        for p in cand_params
        if p.get("kind") in ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD")
    )
    cand_has_varargs = bool(cand.get("has_varargs", False)) or any(
        p.get("kind") == "VAR_POSITIONAL" for p in cand_params
    )
    if args_count < min_pos:
        return f"Missing required positional arguments: expected at least {min_pos}, got {args_count}"
    if not cand_has_varargs and args_count > max_pos:
        return f"Too many positional arguments: expected at most {max_pos}, got {args_count}"
    return None


def check_hallucination(
    framework: str,
    api_path: str,
    kwargs: list[str] | None = None,
    args_count: int | None = None,
    strict_kwargs: bool = True,
    kwarg_values: typing.Mapping[str, JsonValue] | None = None,
    version: str | None = None,
    arg_dtypes: typing.Sequence[str] | typing.Mapping[str, str] | JsonValue = None,
    kwarg_dtypes: typing.Mapping[str, str] | JsonValue = None,
    arg_ranks: (
        typing.Sequence[int | str] | typing.Mapping[str, int | str] | JsonValue
    ) = None,
    kwarg_ranks: typing.Mapping[str, int | str] | JsonValue = None,
    arg_shapes: typing.Sequence[typing.Sequence[int | str]] | JsonValue = None,
    kwarg_shapes: typing.Mapping[str, typing.Sequence[int | str]] | JsonValue = None,
    strict_c_extensions: bool = False,
    repository: SnapshotRepository | None = None,
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
        version: Optional framework version string.
        arg_dtypes: Optional list or dictionary mapping positional/keyword parameters to passed tensor dtypes.
        kwarg_dtypes: Optional dictionary mapping keyword parameter names to passed tensor dtypes.
        arg_ranks: Optional list or dictionary mapping positional/keyword parameters to passed tensor ranks.
        kwarg_ranks: Optional dictionary mapping keyword parameter names to passed tensor ranks.
        arg_shapes: Optional list of tensor shape sequences for positional arguments.
        kwarg_shapes: Optional dictionary mapping keyword argument names to tensor shape sequences.
        strict_c_extensions: Whether to treat unverified kwargs on opaque C-extension APIs as hallucinations.
        repository: Optional `SnapshotRepository` instance for snapshot lookup.

    Returns:
        `HallucinationCheckResult` model containing validation details.
    """
    if kwargs is None and kwarg_values is not None:
        kwargs = list(kwarg_values.keys())

    sig = get_api_signature(framework, api_path, version=version, repository=repository)
    snap = resolve_snapshot(framework, version=version, repository=repository)
    raw_source = snap.get("_snapshot_source")
    snapshot_source = (
        str(raw_source)
        if raw_source is not None
        else ("not_found" if not snap.get("categories") else "unknown")
    )
    if not sig:
        if snapshot_source == "not_found":
            reason = (
                f"API '{api_path}' could not be verified: no local ground-truth snapshot for framework '{framework}' "
                f"(version '{version or 'any'}') was found in bundled packages or local cache. To ground offline, "
                f"install '{framework}' or place a versioned snapshot JSON in the cache directory."
            )
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
    overload_dicts: list[dict[str, JsonValue]] = (
        [dict(ov) for ov in raw_overloads if isinstance(ov, dict)]
        if isinstance(raw_overloads, list)
        else []
    )
    candidates = [sig] + overload_dicts

    best_result: HallucinationCheckResult | None = None
    fewest_errors = 999999
    clean_api = api_path.split(".")[-1].lower()

    for cand in candidates:
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
            strict_c_extensions=strict_c_extensions,
        )
        enum_errors = _validate_candidate_enums_and_literals(cand, kwarg_values)
        dtype_errors, rank_errors = _validate_candidate_dtypes_and_ranks(
            cand=cand,
            api_path=api_path,
            framework=framework,
            kwarg_values=kwarg_values,
            arg_dtypes=arg_dtypes,
            kwarg_dtypes=kwarg_dtypes,
            arg_ranks=arg_ranks,
            kwarg_ranks=kwarg_ranks,
        )
        shape_errors = _validate_candidate_shapes(
            cand=cand,
            api_path=api_path,
            clean_api=clean_api,
            arg_shapes=arg_shapes,
            kwarg_shapes=kwarg_shapes,
        )
        positional_error = _validate_candidate_positional_arity(cand, args_count)

        cand_hallucinated = (
            len(invalid) > 0
            or positional_error is not None
            or len(enum_errors) > 0
            or len(dtype_errors) > 0
            or len(rank_errors) > 0
            or len(shape_errors) > 0
        )
        reason = "Valid API call"
        if cand_hallucinated:
            reasons = []
            if invalid:
                reasons.append(
                    f"Invalid/hallucinated keyword arguments: {', '.join(invalid)}"
                )
            if enum_errors:
                reasons.extend(enum_errors)
            if dtype_errors:
                reasons.extend(dtype_errors)
            if rank_errors:
                reasons.extend(rank_errors)
            if shape_errors:
                reasons.extend(shape_errors)
            if positional_error:
                reasons.append(positional_error)
            reason = "; ".join(reasons)
        elif unrecognized:
            reason = f"Unconstrained keyword arguments passed to **kwargs: {', '.join(unrecognized)}"

        if opaque_warning and not cand_hallucinated and reason == "Valid API call":
            reason = f"Valid API call with warning: {opaque_warning}"

        res = HallucinationCheckResult(
            api_exists=True,
            is_hallucinated=cand_hallucinated,
            invalid_kwargs=invalid,
            canonical_params=sorted(list(cand_names)),
            has_unconstrained_kwargs=cand_has_var_kwargs,
            reason=reason,
            snapshot_source=snapshot_source,
            signature_completeness=str(
                cand.get(
                    "signature_completeness", "opaque" if cand_is_opaque else "exact"
                )
            ),
            is_c_extension=bool(cand.get("is_c_extension", False) or cand_is_opaque),
            dtype_errors=dtype_errors if dtype_errors else None,
            rank_errors=rank_errors if rank_errors else None,
            shape_errors=shape_errors if shape_errors else None,
            warning=opaque_warning,
        )

        if not cand_hallucinated:
            return res

        err_count = (
            len(invalid)
            + len(enum_errors)
            + len(dtype_errors)
            + len(rank_errors)
            + len(shape_errors)
            + (1 if positional_error else 0)
        )
        if err_count < fewest_errors:
            fewest_errors = err_count
            best_result = res

    return best_result or HallucinationCheckResult(
        api_exists=True,
        is_hallucinated=True,
        invalid_kwargs=kwargs or [],
        canonical_params=[],
        has_unconstrained_kwargs=False,
        snapshot_source=snapshot_source,
        reason="No matching overload signature found.",
    )


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
        canonical_params = [
            str(p["name"]) for p in _extract_cand_params(sig) if "name" in p
        ]

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
        imports: dict[str, str] = {}
        calls: list[tuple[int, str, list[str], int, dict[str, JsonValue]]] = []

        class CallVisitor(ast.NodeVisitor):
            """Visitor for tracking imports and collecting function calls."""

            def visit_Import(self, node: ast.Import) -> None:
                """Track 'import foo' and 'import foo as bar' statements.

                Args:
                    node: The AST Import node.
                """
                for alias in node.names:
                    name = alias.name
                    asname = alias.asname or name
                    imports[asname] = name
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
                    imports[asname] = full
                self.generic_visit(node)

            def visit_Call(self, node: ast.Call) -> None:
                """Inspect Call AST nodes.

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
                    resolved_base = imports.get(root_id, root_id)
                    if attr_parts:
                        full_path = f"{resolved_base}.{'.'.join(reversed(attr_parts))}"
                    else:
                        full_path = resolved_base

                    kw_names = [kw.arg for kw in node.keywords if kw.arg]
                    kw_vals: dict[str, JsonValue] = {}
                    for kw in node.keywords:
                        if kw.arg and isinstance(kw.value, ast.Constant):
                            val = kw.value.value
                            assert (
                                isinstance(val, (str, int, float, bool)) or val is None
                            )
                            kw_vals[kw.arg] = val
                    calls.append(
                        (node.lineno, full_path, kw_names, len(node.args), kw_vals)
                    )
                self.generic_visit(node)

        visitor = CallVisitor()
        visitor.visit(tree)

        if calls:
            is_python = True
            for lineno, api_name, kw_list, arg_cnt, kw_vals in calls:
                fw = framework
                if not fw:
                    if api_name.startswith("torch.") or api_name.startswith("torch"):
                        fw = "torch"
                    elif api_name.startswith("jax.") or api_name.startswith("jax"):
                        fw = "jax"
                    elif (
                        api_name.startswith("tf.")
                        or api_name.startswith("tensorflow.")
                        or api_name.startswith("tensorflow")
                    ):
                        fw = "tensorflow"
                    elif (
                        api_name.startswith("np.")
                        or api_name.startswith("numpy.")
                        or api_name.startswith("numpy")
                    ):
                        fw = "numpy"
                    elif api_name.startswith("nn.") or api_name.startswith("F."):
                        fw = "torch"
                        api_name = f"torch.{api_name}"
                    elif api_name.startswith("stablehlo."):
                        fw = "stablehlo"

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
    except SyntaxError:
        is_python = False

    if not is_python:
        lines = code.strip().splitlines()
        for lineno, raw_line in enumerate(lines, 1):
            line = raw_line.strip()
            if not line or line.startswith(("//", "#", ";")):
                continue

            mlir_match = re.search(
                r"(?:%[a-zA-Z0-9_]+\s*=\s*)?([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)", line
            )
            sass_match = re.match(
                r"^(?:@!?U?P\d+\s+)?([A-Z0-9_\.]+)(?:\s+(.*))?;?$", line
            )
            rdna_match = re.match(r"^([vsa]_[a-zA-Z0-9_\.]+)(?:\s+(.*))?$", line)

            if mlir_match and (
                framework in ("mlir", "stablehlo")
                or any(
                    d in line
                    for d in (
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
                )
            ):
                op_name = mlir_match.group(1)
                total_analyzed += 1
                mlir_res = check_mlir_op(op_name, repository=repository)
                if not mlir_res.get("is_valid"):
                    findings.append(
                        CodeBlockFinding(
                            line=lineno,
                            type="mlir_error",
                            target=op_name,
                            reason="; ".join(mlir_res.get("errors", [])),
                        )
                    )

            elif sass_match and (
                framework == "nvidia_sass"
                or any(s in line for s in ("R0", "UR0", "P0", "sm_"))
                or line.endswith(";")
            ):
                from ml_ecosystem_snapshots.frameworks.nvidia_sass import (
                    tokenize_sass_line,
                )

                _pred, mnem, mods, ops = tokenize_sass_line(line)
                total_analyzed += 1
                sass_res = check_sass_instruction(
                    mnem,
                    operands=ops,
                    modifiers=mods,
                    sm_arch=sm_arch,
                    repository=repository,
                )
                if not sass_res.get("is_valid"):
                    findings.append(
                        CodeBlockFinding(
                            line=lineno,
                            type="sass_error",
                            target=mnem,
                            reason="; ".join(sass_res.get("errors", [])),
                        )
                    )

            elif rdna_match:
                from ml_ecosystem_snapshots.frameworks.amd_rdna import (
                    tokenize_rdna_line,
                )

                base_mnem, ops, _enc_suf, mods = tokenize_rdna_line(line)
                total_analyzed += 1
                rdna_res = check_rdna_instruction(
                    base_mnem,
                    operands=ops,
                    modifiers=mods,
                    gfx_arch=gfx_arch,
                    repository=repository,
                )
                if not rdna_res.get("is_valid"):
                    findings.append(
                        CodeBlockFinding(
                            line=lineno,
                            type="rdna_error",
                            target=base_mnem,
                            reason="; ".join(rdna_res.get("errors", [])),
                        )
                    )

    return CodeBlockCheckResult(
        total_analyzed=total_analyzed,
        hallucinations_detected=len(findings),
        is_valid=len(findings) == 0,
        findings=findings,
    )


__all__ = [
    "DTYPE_CANONICAL_ALIASES",
    "KNOWN_ENUM_VALUES",
    "_matches_rank_constraint",
    "_validate_candidate_dtypes_and_ranks",
    "_validate_candidate_enums_and_literals",
    "_validate_candidate_kwargs",
    "_validate_candidate_positional_arity",
    "_validate_candidate_shapes",
    "check_code_block",
    "check_hallucination",
    "explain_anti_pattern",
    "normalize_dtype_name",
]
