"""Python ML framework signature and argument grounding verifier."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ml_ecosystem_snapshots.grounding.engine import GroundingEngine
from ml_ecosystem_snapshots.grounding.models import (
    DiagnosticSeverity,
    GroundingReport,
)


def _normalize_candidate_dict(ref_or_dict: Any) -> Dict[str, Any]:
    """Convert an ExtendedGhostRef or dictionary into a normalized candidate dictionary.

    Args:
        ref_or_dict: ExtendedGhostRef model instance or raw dictionary.

    Returns:
        Normalized dictionary with 'params', 'has_varargs', 'accepted_kwargs', etc.
    """
    if isinstance(ref_or_dict, dict):
        return ref_or_dict
    dumped: Dict[str, Any] = ref_or_dict.model_dump()
    return dumped


def _check_candidate_overload(
    cand: Dict[str, Any],
    args_count: Optional[int],
    kwargs: Sequence[str],
) -> Tuple[List[str], Optional[str]]:
    """Validate positional arity and keyword arguments against a single candidate overload.

    Args:
        cand: Normalized candidate signature dictionary.
        args_count: Optional number of positional arguments passed.
        kwargs: Sequence of keyword argument names passed.

    Returns:
        Tuple of (invalid_kwargs, positional_error).
    """
    from ml_ecosystem_snapshots.mcp.tools.hallucination import (
        _validate_candidate_kwargs,
        _validate_candidate_positional_arity,
    )

    invalid, _unrecognized, _has_var_kw, _is_opaque, _warn, _names = (
        _validate_candidate_kwargs(
            cand=cand,
            kwargs=kwargs,
            strict_kwargs=False,
            strict_c_extensions=False,
        )
    )
    pos_err = _validate_candidate_positional_arity(cand, args_count)
    return invalid, pos_err


def validate_python_call(
    framework: str,
    api_path: str,
    args: List[Any],
    kwargs: Dict[str, Any],
    engine: Optional[GroundingEngine] = None,
    arg_dtypes: Optional[Union[List[str], Dict[str, str]]] = None,
    kwarg_dtypes: Optional[Dict[str, str]] = None,
    arg_ranks: Optional[
        Union[List[Union[int, str]], Dict[str, Union[int, str]]]
    ] = None,
    kwarg_ranks: Optional[Dict[str, Union[int, str]]] = None,
    arg_shapes: Optional[List[Sequence[Union[int, str]]]] = None,
    kwarg_shapes: Optional[Dict[str, Sequence[Union[int, str]]]] = None,
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
        arg_dtypes: Optional list or dictionary of positional argument tensor dtypes.
        kwarg_dtypes: Optional dictionary of keyword argument tensor dtypes.
        arg_ranks: Optional list or dictionary of positional argument tensor ranks.
        kwarg_ranks: Optional dictionary of keyword argument tensor ranks.
        arg_shapes: Optional list of positional argument tensor shapes.
        kwarg_shapes: Optional dictionary of keyword argument tensor shapes.

    Returns:
        GroundingReport containing validation status, diagnostics, and suggested fixes.
    """
    from ml_ecosystem_snapshots.mcp.tools.hallucination import (
        _validate_candidate_dtypes_and_ranks,
        _validate_candidate_enums_and_literals,
        _validate_candidate_shapes,
    )

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

    primary_cand = _normalize_candidate_dict(ref)
    candidates: List[Dict[str, Any]] = [primary_cand]
    for ov in getattr(ref, "overloads", []):
        candidates.append(_normalize_candidate_dict(ov))

    all_param_names = {
        str(p.get("name"))
        for cand in candidates
        for p in cand.get("params", [])
        if isinstance(p, dict) and p.get("name")
    }
    if ref.accepted_kwargs:
        all_param_names.update(ref.accepted_kwargs)

    any_has_var_kwargs = any(
        any(
            "VAR_KEYWORD" in str(p.get("kind", ""))
            for p in cand.get("params", [])
            if isinstance(p, dict)
        )
        and not (
            cand.get("accepted_kwargs")
            or (cand.get("domain_metadata") or {}).get("accepted_kwargs")
        )
        for cand in candidates
    )

    kw_names = list(kwargs.keys())
    clean_api = api_path.split(".")[-1].lower()

    best_extra_errors: List[str] = []
    fewest_errors = 999999

    for cand in candidates:
        invalid_kws, pos_err = _check_candidate_overload(
            cand=cand,
            args_count=len(args) if args else None,
            kwargs=kw_names,
        )
        enum_errs = _validate_candidate_enums_and_literals(cand, kwargs)
        dtype_errs, rank_errs = _validate_candidate_dtypes_and_ranks(
            cand=cand,
            api_path=api_path,
            framework=clean_fw,
            kwarg_values=kwargs,
            arg_dtypes=arg_dtypes,
            kwarg_dtypes=kwarg_dtypes,
            arg_ranks=arg_ranks,
            kwarg_ranks=kwarg_ranks,
        )
        shape_errs = _validate_candidate_shapes(
            cand=cand,
            api_path=api_path,
            clean_api=clean_api,
            arg_shapes=arg_shapes,
            kwarg_shapes=kwarg_shapes,
        )
        extra_errs = enum_errs + dtype_errs + rank_errs + shape_errs
        total_err_count = len(invalid_kws) + (1 if pos_err else 0) + len(extra_errs)
        if total_err_count == 0:
            return report
        if total_err_count < fewest_errors:
            fewest_errors = total_err_count
            best_extra_errors = extra_errs

    # Check for hallucinated kwargs across all overloads
    for kw in kwargs:
        if kw not in all_param_names and not any_has_var_kwargs:
            suggested_kw: Optional[str] = None
            if clean_fw == "torch" and kw == "axis" and "dim" in all_param_names:
                suggested_kw = "dim"
            elif (
                clean_fw in ("numpy", "jax", "tensorflow")
                and kw == "dim"
                and "axis" in all_param_names
            ):
                suggested_kw = "axis"
            elif (
                clean_fw == "torch"
                and kw == "keepdims"
                and "keepdim" in all_param_names
            ):
                suggested_kw = "keepdim"
            elif (
                clean_fw in ("numpy", "jax", "tensorflow")
                and kw == "keepdim"
                and "keepdims" in all_param_names
            ):
                suggested_kw = "keepdims"

            report.add_diagnostic(
                field=f"kwargs.{kw}",
                message=f"Unexpected or hallucinated keyword argument '{kw}' for '{api_path}'.",
                severity=DiagnosticSeverity.ERROR,
                suggested_fix=suggested_kw,
            )

    # Check positional argument count across all overloads
    pos_params = [
        p
        for p in ref.params
        if "KEYWORD_ONLY" not in str(getattr(p, "kind", ""))
        and "VAR_POSITIONAL" not in str(getattr(p, "kind", ""))
        and "VAR_KEYWORD" not in str(getattr(p, "kind", ""))
    ]
    has_var_args = getattr(ref, "has_varargs", False) or any(
        "VAR_POSITIONAL" in str(getattr(p, "kind", "")) for p in ref.params
    )

    if not has_var_args and len(args) > len(pos_params) and pos_params:
        overload_accepts_len = any(
            _check_candidate_overload(cand=cand, args_count=len(args), kwargs=[])[1]
            is None
            for cand in candidates[1:]
        )
        if not overload_accepts_len:
            report.add_diagnostic(
                field="args",
                message=(
                    f"Too many positional arguments passed to '{api_path}'. "
                    f"Expected at most {len(pos_params)}, but received {len(args)}."
                ),
                severity=DiagnosticSeverity.ERROR,
            )

    for err_msg in best_extra_errors:
        report.add_diagnostic(
            field="arguments",
            message=err_msg,
            severity=DiagnosticSeverity.ERROR,
        )

    return report


__all__ = [
    "_check_candidate_overload",
    "_normalize_candidate_dict",
    "validate_python_call",
]
