"""Single canonical Python ML framework call, overload, kwarg, and arity grounding verifier."""

from __future__ import annotations

import typing

import pydantic
import typing_extensions

from ml_ecosystem_snapshots.grounding.dtypes import (
    CallConstraintOptions,
    CallOptionValue,
    _validate_candidate_dtypes_and_ranks,
    _validate_candidate_enums_and_literals,
    _validate_candidate_shapes,
)
from ml_ecosystem_snapshots.grounding.engine import GroundingEngine
from ml_ecosystem_snapshots.grounding.models import (
    DiagnosticSeverity,
    GroundingReport,
    HallucinationCheckResult,
)
from ml_ecosystem_snapshots.models import (
    ExtendedGhostRef,
    GhostRef,
    JsonValue,
    ensure_json_dict,
    ensure_json_list,
    ensure_present,
)


def _extract_cand_params(cand: dict[str, JsonValue]) -> list[dict[str, JsonValue]]:
    """Extract a list of parameter dictionaries from a candidate signature dictionary.

    Args:
        cand: Candidate signature dictionary.

    Returns:
        List of parameter dictionaries.
    """
    raw = ensure_json_list(cand.get("params") or [])
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
        if p.get("default") is None and p.get("kind") in ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD")
    )
    max_pos = sum(1 for p in cand_params if p.get("kind") in ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD"))
    cand_has_varargs = bool(cand.get("has_varargs", False)) or any(
        p.get("kind") == "VAR_POSITIONAL" for p in cand_params
    )
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
    options: CallConstraintOptions,
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
        options: Additional hallucination validation options (`CallConstraintOptions`).

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

    err_count = (
        len(invalid)
        + len(enum_errors)
        + len(dtype_errors)
        + len(rank_errors)
        + len(shape_errors)
        + (1 if positional_error else 0)
    )
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


class PythonCallValidationOptions(pydantic.BaseModel):
    """Optional tensor dtype, rank, and shape constraints for `validate_python_call`."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    arg_dtypes: list[str] | dict[str, str] | None = None
    kwarg_dtypes: dict[str, str] | None = None
    arg_ranks: list[int | str] | dict[str, int | str] | None = None
    kwarg_ranks: dict[str, int | str] | None = None
    arg_shapes: list[typing.Sequence[int | str]] | None = None
    kwarg_shapes: dict[str, typing.Sequence[int | str]] | None = None


_PythonCallOptionValue: typing_extensions.TypeAlias = "list[str] | dict[str, str] | list[int | str] | dict[str, int | str] | list[typing.Sequence[int | str]] | dict[str, typing.Sequence[int | str]] | None"


def _normalize_candidate_dict(
    ref_or_dict: GhostRef | ExtendedGhostRef | typing.Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    """Convert an ExtendedGhostRef or dictionary into a normalized candidate dictionary.

    Args:
        ref_or_dict: ExtendedGhostRef model instance or raw dictionary.

    Returns:
        Normalized dictionary with 'params', 'has_varargs', 'accepted_kwargs', etc.
    """
    if isinstance(ref_or_dict, GhostRef):
        dumped: dict[str, JsonValue] = ref_or_dict.model_dump()
        return dumped
    return dict(ref_or_dict)


def _check_candidate_overload(
    cand: dict[str, JsonValue],
    args_count: int | None,
    kwargs: typing.Sequence[str],
) -> tuple[list[str], str | None]:
    """Validate positional arity and keyword arguments against a single candidate overload.

    Args:
        cand: Normalized candidate signature dictionary.
        args_count: Optional number of positional arguments passed.
        kwargs: Sequence of keyword argument names passed.

    Returns:
        Tuple of (invalid_kwargs, positional_error).
    """
    invalid, _unrecognized, _has_var_kw, _is_opaque, _warn, _names = _validate_candidate_kwargs(
        cand=cand,
        kwargs=kwargs,
        strict_kwargs=False,
        strict_c_extensions=False,
    )
    pos_err = _validate_candidate_positional_arity(cand, args_count)
    return invalid, pos_err


def _collect_candidate_param_names(
    ref: ExtendedGhostRef,
    candidates: list[dict[str, JsonValue]],
) -> set[str]:
    """Gather all recognized parameter and accepted_kwargs names across candidate overloads.

    Args:
        ref: Primary `ExtendedGhostRef` specification.
        candidates: List of normalized candidate signature dictionaries.

    Returns:
        Set of recognized parameter names.
    """
    all_param_names: set[str] = set()
    for cand in candidates:
        cand_params = ensure_json_list(cand.get("params"))
        for p in cand_params:
            if isinstance(p, dict) and p.get("name"):
                all_param_names.add(str(p["name"]))
    if ref.accepted_kwargs:
        all_param_names.update(ref.accepted_kwargs)
    return all_param_names


def _suggest_python_kwarg_fix(
    clean_fw: str,
    kw: str,
    all_param_names: set[str],
) -> str | None:
    """Suggest a canonical keyword argument name for common cross-framework mixups.

    Args:
        clean_fw: Normalized lowercase framework name.
        kw: Unrecognized keyword argument name.
        all_param_names: Set of all valid parameter names across candidate signatures.

    Returns:
        Suggested canonical parameter name if a known mapping applies, or None.
    """
    if clean_fw == "torch" and kw == "axis" and "dim" in all_param_names:
        return "dim"
    if clean_fw in ("numpy", "jax", "tensorflow") and kw == "dim" and "axis" in all_param_names:
        return "axis"
    if clean_fw == "torch" and kw == "keepdims" and "keepdim" in all_param_names:
        return "keepdim"
    if clean_fw in ("numpy", "jax", "tensorflow") and kw == "keepdim" and "keepdims" in all_param_names:
        return "keepdims"
    return None


def _record_python_call_diagnostics(
    clean_fw: str,
    api_path: str,
    ref: ExtendedGhostRef,
    candidates: list[dict[str, JsonValue]],
    args_len: int,
    all_param_names: set[str],
    best_invalid_kws: list[str],
    report: GroundingReport,
) -> None:
    """Record keyword and positional arity diagnostics on `report` when no overload matched.

    Args:
        clean_fw: Normalized lowercase framework name.
        api_path: Fully qualified API symbol path.
        ref: Primary `ExtendedGhostRef` specification.
        candidates: List of normalized candidate signature dictionaries.
        args_len: Count of positional arguments passed to the call.
        all_param_names: Set of all valid parameter names across overloads.
        best_invalid_kws: Invalid keyword arguments from the closest candidate overload.
        report: Mutable `GroundingReport` to record diagnostics on.
    """
    for kw in best_invalid_kws:
        if kw not in all_param_names:
            suggested_kw = _suggest_python_kwarg_fix(clean_fw, kw, all_param_names)
            report.add_diagnostic(
                field=f"kwargs.{kw}",
                message=f"Unexpected or hallucinated keyword argument '{kw}' for '{api_path}'.",
                severity=DiagnosticSeverity.ERROR,
                suggested_fix=suggested_kw,
            )

    pos_params = [
        p
        for p in ref.params
        if "KEYWORD_ONLY" not in str(p.kind)
        and "VAR_POSITIONAL" not in str(p.kind)
        and "VAR_KEYWORD" not in str(p.kind)
    ]
    has_var_args = bool(ref.has_varargs) or any("VAR_POSITIONAL" in str(p.kind) for p in ref.params)

    if not has_var_args and args_len > len(pos_params) and pos_params:
        overload_accepts_len = any(
            _check_candidate_overload(cand=cand, args_count=args_len, kwargs=[])[1] is None for cand in candidates[1:]
        )
        if not overload_accepts_len:
            report.add_diagnostic(
                field="args",
                message=(
                    f"Too many positional arguments passed to '{api_path}'. Expected at most {len(pos_params)}, but received {args_len}."
                ),
                severity=DiagnosticSeverity.ERROR,
            )


def validate_python_call(
    framework: str,
    api_path: str,
    args: typing.Sequence[JsonValue],
    kwargs: typing.Mapping[str, JsonValue],
    engine: GroundingEngine | None = None,
    **options: _PythonCallOptionValue,
) -> GroundingReport:
    """Verify a Python framework API call against the ground-truth offline snapshot.

    Delegates core parameter, overload, enum, dtype, rank, and shape checks to a
    unified verification flow consistent with `check_hallucination`.

    Args:
        framework: Framework name (e.g., 'torch', 'jax', 'numpy', 'tensorflow').
        api_path: Fully qualified API symbol path (e.g., 'torch.sum', 'jax.numpy.mean').
        args: List of positional arguments provided to the call.
        kwargs: Dictionary of keyword arguments provided to the call.
        engine: Optional GroundingEngine instance.
        **options: Optional keyword arguments (`arg_dtypes`, `kwarg_dtypes`, `arg_ranks`,
            `kwarg_ranks`, `arg_shapes`, `kwarg_shapes`) specifying tensor metadata.

    Returns:
        GroundingReport containing validation status, diagnostics, and suggested fixes.
    """
    opts = PythonCallValidationOptions.model_validate(options)
    eng = engine or GroundingEngine()
    clean_fw = framework.lower().strip()

    report = GroundingReport(
        is_grounded=True,
        target=clean_fw,
        symbol=api_path,
    )

    ref = eng.get_symbol(clean_fw, api_path)
    if not ref:
        suggested = eng.suggest_closest_symbol(clean_fw, api_path)
        report.add_diagnostic(
            field="api_path",
            message=f"API '{api_path}' was not found in ground-truth snapshot for '{clean_fw}'.",
            severity=DiagnosticSeverity.ERROR,
            suggested_fix=suggested,
        )
        return report

    report.matched_ref = ref

    candidates: list[dict[str, JsonValue]] = [_normalize_candidate_dict(ref)]
    for ov in ref.overloads or ():
        candidates.append(_normalize_candidate_dict(ov))

    all_param_names = _collect_candidate_param_names(ref, candidates)
    kw_names = list(kwargs.keys())
    clean_api = api_path.split(".")[-1].lower()

    best_invalid_kws: list[str] = []
    best_extra_errors: list[str] = []
    fewest_errors = 999999

    for cand in candidates:
        invalid_kws, pos_err = _check_candidate_overload(
            cand=cand,
            args_count=len(args) if args else None,
            kwargs=kw_names,
        )
        enum_errs = _validate_candidate_enums_and_literals(cand, dict(kwargs))
        dtype_errs, rank_errs = _validate_candidate_dtypes_and_ranks(
            cand=cand,
            api_path=api_path,
            framework=clean_fw,
            kwarg_values=dict(kwargs),
            arg_dtypes=opts.arg_dtypes,
            kwarg_dtypes=opts.kwarg_dtypes,
            arg_ranks=opts.arg_ranks,
            kwarg_ranks=opts.kwarg_ranks,
        )
        shape_errs = _validate_candidate_shapes(
            cand=cand,
            api_path=api_path,
            clean_api=clean_api,
            arg_shapes=opts.arg_shapes,
            kwarg_shapes=opts.kwarg_shapes,
        )
        extra_errs = enum_errs + dtype_errs + rank_errs + shape_errs
        total_err_count = len(invalid_kws) + (1 if pos_err else 0) + len(extra_errs)
        if total_err_count == 0:
            return report
        if total_err_count < fewest_errors:
            fewest_errors = total_err_count
            best_invalid_kws = invalid_kws
            best_extra_errors = extra_errs

    _record_python_call_diagnostics(
        clean_fw=clean_fw,
        api_path=api_path,
        ref=ref,
        candidates=candidates,
        args_len=len(args),
        all_param_names=all_param_names,
        best_invalid_kws=best_invalid_kws,
        report=report,
    )

    for err_msg in best_extra_errors:
        report.add_diagnostic(
            field="arguments",
            message=err_msg,
            severity=DiagnosticSeverity.ERROR,
        )

    return report


def verify_python_call(
    framework: str,
    api_path: str,
    sig: typing.Mapping[str, JsonValue] | None,
    snapshot_source: str = "not_found",
    kwargs: list[str] | None = None,
    args_count: int | None = None,
    strict_kwargs: bool = True,
    kwarg_values: typing.Mapping[str, JsonValue] | None = None,
    **extra_constraints: CallOptionValue | str,
) -> HallucinationCheckResult:
    """Verify whether a Python API call uses valid parameters, arity, dtypes, ranks, and shapes.

    Args:
        framework: Target framework name (e.g. torch, jax).
        api_path: Canonical API path (e.g. torch.sum, jax.numpy.mean).
        sig: Resolved signature dictionary from snapshot or `None` if API does not exist.
        snapshot_source: Provenance label for the backing snapshot.
        kwargs: Optional list of keyword argument names passed by generated code.
        args_count: Optional number of positional arguments passed.
        strict_kwargs: If True, reject unrecognized kwargs even when `**kwargs` is present.
        kwarg_values: Optional mapping of passed keyword argument names to constant values.
        **extra_constraints: Optional inline constraint kwargs (`version`, `arg_dtypes`, `kwarg_dtypes`, etc.).

    Returns:
        Structured `HallucinationCheckResult` report.
    """
    if kwargs is None and kwarg_values is not None:
        kwargs = list(kwarg_values.keys())
    raw_ver = extra_constraints.get("version")
    ver_label = str(raw_ver) if isinstance(raw_ver, str) and raw_ver else "any"
    filtered_constraints = {k: v for k, v in extra_constraints.items() if k != "version"}
    opts = CallConstraintOptions.model_validate(filtered_constraints)
    if sig is None:
        if snapshot_source == "not_found":
            reason = (
                f"API '{api_path}' could not be verified: no local ground-truth snapshot for framework "
                f"'{framework}' (version '{ver_label}') was found in bundled packages or local cache. "
                f"To ground offline, install '{framework}' or place a versioned snapshot JSON in the cache directory."
            )
        else:
            reason = f"API '{api_path}' does not exist in ground-truth '{framework}' snapshot."
        return HallucinationCheckResult(
            api_exists=False,
            is_hallucinated=True,
            invalid_kwargs=kwargs or [],
            canonical_params=[],
            has_unconstrained_kwargs=False,
            reason=reason,
            snapshot_source=snapshot_source,
        )

    candidates: list[dict[str, JsonValue]] = [dict(sig)]
    raw_overloads = sig.get("overloads")
    if isinstance(raw_overloads, list) and raw_overloads:
        candidates.extend(ensure_json_dict(ov) for ov in raw_overloads)

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
        if not res.is_hallucinated:
            return res
        if err_count < fewest_errors:
            fewest_errors = err_count
            best_result = res

    return ensure_present(best_result, "overload candidate evaluation result")


__all__ = [
    "PythonCallValidationOptions",
    "validate_python_call",
    "verify_python_call",
]
