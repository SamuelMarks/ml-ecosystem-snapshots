"""SciPy API Snapshot Extractor.

Extracts mathematical special functions (Bessel, Gamma, Error, Hypergeometric),
linear algebra routines, signal processing kernels, n-dimensional filters,
and statistical distributions from the SciPy library.
"""

from __future__ import annotations

import importlib
import types

from ml_switcheroo_ir.schema.ghost import (
    GhostRef,
    SemanticTier,
)

from ml_ecosystem_snapshots.frameworks._base import (
    CanonicalSubmoduleOpSpec,
    CanonicalSubmoduleOpTuple,
    build_canonical_submodule_ops,
    collect_from_submodules_or_canonical,
)
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector

_SCIPY_OP_DEFS: list[CanonicalSubmoduleOpTuple] = [
    ("erf", "scipy.special.erf", "function", "special", [("z", "POSITIONAL_ONLY", None)], "Returns the error function of complex argument."),
    ("erfc", "scipy.special.erfc", "function", "special", [("x", "POSITIONAL_ONLY", None)], "Complementary error function, 1 - erf(x)."),
    ("gamma", "scipy.special.gamma", "function", "special", [("z", "POSITIONAL_ONLY", None)], "gamma function."),
    ("gammaln", "scipy.special.gammaln", "function", "special", [("z", "POSITIONAL_ONLY", None)], "Logarithm of the absolute value of the gamma function."),
    ("betaln", "scipy.special.betaln", "function", "special", [("a", "POSITIONAL_ONLY", None), ("b", "POSITIONAL_ONLY", None)], "Natural logarithm of absolute value of beta function."),
    ("digamma", "scipy.special.digamma", "function", "special", [("z", "POSITIONAL_ONLY", None)], "The digamma function, logarithmic derivative of the gamma function."),
    ("polygamma", "scipy.special.polygamma", "function", "special", [("n", "POSITIONAL_ONLY", None), ("x", "POSITIONAL_ONLY", None)], "Polygamma functions n-th derivative of digamma at x."),
    ("j0", "scipy.special.j0", "function", "special", [("x", "POSITIONAL_ONLY", None)], "Bessel function of the first kind of order 0."),
    ("j1", "scipy.special.j1", "function", "special", [("x", "POSITIONAL_ONLY", None)], "Bessel function of the first kind of order 1."),
    ("y0", "scipy.special.y0", "function", "special", [("x", "POSITIONAL_ONLY", None)], "Bessel function of the second kind of order 0."),
    ("y1", "scipy.special.y1", "function", "special", [("x", "POSITIONAL_ONLY", None)], "Bessel function of the second kind of order 1."),
    ("i0", "scipy.special.i0", "function", "special", [("x", "POSITIONAL_ONLY", None)], "Modified Bessel function of order 0."),
    ("i1", "scipy.special.i1", "function", "special", [("x", "POSITIONAL_ONLY", None)], "Modified Bessel function of order 1."),
    ("k0", "scipy.special.k0", "function", "special", [("x", "POSITIONAL_ONLY", None)], "Modified Bessel function of the second kind of order 0."),
    ("k1", "scipy.special.k1", "function", "special", [("x", "POSITIONAL_ONLY", None)], "Modified Bessel function of the second kind of order 1."),
    ("sinc", "scipy.special.sinc", "function", "special", [("x", "POSITIONAL_ONLY", None)], "Return the normalized sinc function."),
    ("logit", "scipy.special.logit", "function", "special", [("p", "POSITIONAL_ONLY", None)], "Logit function: log(p / (1 - p))."),
    ("expit", "scipy.special.expit", "function", "special", [("x", "POSITIONAL_ONLY", None)], "Expit (logistic sigmoid) function: 1 / (1 + exp(-x))."),
    ("softmax", "scipy.special.softmax", "function", "special", [("x", "POSITIONAL_ONLY", None), ("axis", "KEYWORD_ONLY", "None")], "Softmax function along given axis."),
    (
        "logsumexp",
        "scipy.special.logsumexp",
        "function",
        "special",
        [("a", "POSITIONAL_ONLY", None), ("axis", "KEYWORD_ONLY", "None"), ("b", "KEYWORD_ONLY", "None"), ("keepdims", "KEYWORD_ONLY", "False"), ("return_sign", "KEYWORD_ONLY", "False")],
        "Compute the log of the sum of exponentials of input elements.",
    ),
    ("inv", "scipy.linalg.inv", "function", "linalg", [("a", "POSITIONAL_ONLY", None), ("overwrite_a", "KEYWORD_ONLY", "False"), ("check_finite", "KEYWORD_ONLY", "True")], "Compute the inverse of a matrix."),
    ("pinv", "scipy.linalg.pinv", "function", "linalg", [("a", "POSITIONAL_ONLY", None), ("atol", "KEYWORD_ONLY", "None"), ("rtol", "KEYWORD_ONLY", "None"), ("return_rank", "KEYWORD_ONLY", "False"), ("check_finite", "KEYWORD_ONLY", "True")], "Compute the Moore-Penrose pseudo-inverse of a matrix."),
    ("det", "scipy.linalg.det", "function", "linalg", [("a", "POSITIONAL_ONLY", None), ("overwrite_a", "KEYWORD_ONLY", "False"), ("check_finite", "KEYWORD_ONLY", "True")], "Compute the determinant of a matrix."),
    ("norm", "scipy.linalg.norm", "function", "linalg", [("a", "POSITIONAL_ONLY", None), ("ord", "KEYWORD_ONLY", "None"), ("axis", "KEYWORD_ONLY", "None"), ("keepdims", "KEYWORD_ONLY", "False"), ("check_finite", "KEYWORD_ONLY", "True")], "Matrix or vector norm."),
    ("cholesky", "scipy.linalg.cholesky", "function", "linalg", [("a", "POSITIONAL_ONLY", None), ("lower", "KEYWORD_ONLY", "False"), ("overwrite_a", "KEYWORD_ONLY", "False"), ("check_finite", "KEYWORD_ONLY", "True")], "Compute the Cholesky decomposition of a matrix."),
    ("qr", "scipy.linalg.qr", "function", "linalg", [("a", "POSITIONAL_ONLY", None), ("overwrite_a", "KEYWORD_ONLY", "False"), ("mode", "KEYWORD_ONLY", "'full'"), ("pivoting", "KEYWORD_ONLY", "False"), ("check_finite", "KEYWORD_ONLY", "True")], "Compute QR decomposition of a matrix."),
    (
        "svd",
        "scipy.linalg.svd",
        "function",
        "linalg",
        [("a", "POSITIONAL_ONLY", None), ("full_matrices", "KEYWORD_ONLY", "True"), ("compute_uv", "KEYWORD_ONLY", "True"), ("overwrite_a", "KEYWORD_ONLY", "False"), ("check_finite", "KEYWORD_ONLY", "True"), ("lapack_driver", "KEYWORD_ONLY", "'gesdd'")],
        "Singular Value Decomposition.",
    ),
    (
        "solve",
        "scipy.linalg.solve",
        "function",
        "linalg",
        [("a", "POSITIONAL_ONLY", None), ("b", "POSITIONAL_ONLY", None), ("lower", "KEYWORD_ONLY", "False"), ("overwrite_a", "KEYWORD_ONLY", "False"), ("overwrite_b", "KEYWORD_ONLY", "False"), ("check_finite", "KEYWORD_ONLY", "True"), ("assume_a", "KEYWORD_ONLY", "'gen'")],
        "Solves the linear equation set a * x = b for the unknown x.",
    ),
    ("expm", "scipy.linalg.expm", "function", "linalg", [("a", "POSITIONAL_ONLY", None)], "Compute the matrix exponential using Pade approximation."),
    ("beta", "scipy.special.beta", "function", "special", [("a", "POSITIONAL_ONLY", None), ("b", "POSITIONAL_ONLY", None)], "Beta function."),
    ("betainc", "scipy.special.betainc", "function", "special", [("a", "POSITIONAL_ONLY", None), ("b", "POSITIONAL_ONLY", None), ("x", "POSITIONAL_ONLY", None)], "Incomplete beta integral."),
    ("dawsn", "scipy.special.dawsn", "function", "special", [("x", "POSITIONAL_ONLY", None)], "Dawson's integral."),
    ("expn", "scipy.special.expn", "function", "special", [("n", "POSITIONAL_ONLY", None), ("x", "POSITIONAL_ONLY", None)], "Generalized exponential integral."),
    ("fresnel", "scipy.special.fresnel", "function", "special", [("z", "POSITIONAL_ONLY", None)], "Fresnel integrals."),
    ("gammainc", "scipy.special.gammainc", "function", "special", [("a", "POSITIONAL_ONLY", None), ("x", "POSITIONAL_ONLY", None)], "Regularized lower incomplete gamma function."),
    ("gammaincc", "scipy.special.gammaincc", "function", "special", [("a", "POSITIONAL_ONLY", None), ("x", "POSITIONAL_ONLY", None)], "Regularized upper incomplete gamma function."),
    ("i0e", "scipy.special.i0e", "function", "special", [("x", "POSITIONAL_ONLY", None)], "Exponentially scaled modified Bessel function of order 0."),
    ("i1e", "scipy.special.i1e", "function", "special", [("x", "POSITIONAL_ONLY", None)], "Exponentially scaled modified Bessel function of order 1."),
    ("jv", "scipy.special.jv", "function", "special", [("v", "POSITIONAL_ONLY", None), ("z", "POSITIONAL_ONLY", None)], "Bessel function of the first kind of real order."),
    ("k0e", "scipy.special.k0e", "function", "special", [("x", "POSITIONAL_ONLY", None)], "Exponentially scaled modified Bessel function K of order 0."),
    ("k1e", "scipy.special.k1e", "function", "special", [("x", "POSITIONAL_ONLY", None)], "Exponentially scaled modified Bessel function K of order 1."),
    ("spence", "scipy.special.spence", "function", "special", [("z", "POSITIONAL_ONLY", None)], "Spence's function, also known as the dilogarithm."),
    ("zeta", "scipy.special.zeta", "function", "special", [("x", "POSITIONAL_ONLY", None), ("q", "POSITIONAL_OR_KEYWORD", "None")], "Riemann zeta function of two arguments."),
    ("cho_solve", "scipy.linalg.cho_solve", "function", "linalg", [("c_and_lower", "POSITIONAL_ONLY", None), ("b", "POSITIONAL_ONLY", None)], "Solve the linear equations Ax = b, given the Cholesky factorization of A."),
    ("eigh_tridiagonal", "scipy.linalg.eigh_tridiagonal", "function", "linalg", [("d", "POSITIONAL_ONLY", None), ("e", "POSITIONAL_ONLY", None)], "Solve eigenvalue problem for a real symmetric tridiagonal matrix."),
    ("hessenberg", "scipy.linalg.hessenberg", "function", "linalg", [("a", "POSITIONAL_ONLY", None), ("calc_q", "KEYWORD_ONLY", "False")], "Compute Hessenberg form of a matrix."),
    ("lu", "scipy.linalg.lu", "function", "linalg", [("a", "POSITIONAL_ONLY", None)], "Compute LU decomposition of a matrix."),
    ("lu_factor", "scipy.linalg.lu_factor", "function", "linalg", [("a", "POSITIONAL_ONLY", None)], "Compute pivoted LU decomposition of a matrix."),
    ("lu_solve", "scipy.linalg.lu_solve", "function", "linalg", [("lu_and_piv", "POSITIONAL_ONLY", None), ("b", "POSITIONAL_ONLY", None)], "Solve an equation system, a * x = b, given the LU factorization of a."),
    ("polar", "scipy.linalg.polar", "function", "linalg", [("a", "POSITIONAL_ONLY", None), ("side", "KEYWORD_ONLY", "'right'")], "Compute the polar decomposition."),
    ("schur", "scipy.linalg.schur", "function", "linalg", [("a", "POSITIONAL_ONLY", None)], "Compute Schur decomposition of a matrix."),
    ("solve_banded", "scipy.linalg.solve_banded", "function", "linalg", [("l_and_u", "POSITIONAL_ONLY", None), ("ab", "POSITIONAL_ONLY", None), ("b", "POSITIONAL_ONLY", None)], "Solve the equation a x = b for x, assuming a is banded matrix."),
    ("maximum_filter", "scipy.ndimage.maximum_filter", "function", "ndimage", [("input", "POSITIONAL_ONLY", None), ("size", "KEYWORD_ONLY", "None")], "Calculate a multidimensional maximum filter."),
    ("beta", "scipy.stats.beta", "function", "stats", [("a", "POSITIONAL_ONLY", None), ("b", "POSITIONAL_ONLY", None)], "A beta continuous random variable."),
    ("poisson", "scipy.stats.poisson", "function", "stats", [("mu", "POSITIONAL_ONLY", None)], "A Poisson discrete random variable."),
    ("convolve", "scipy.signal.convolve", "function", "signal", [("in1", "POSITIONAL_ONLY", None), ("in2", "POSITIONAL_ONLY", None), ("mode", "KEYWORD_ONLY", "'full'"), ("method", "KEYWORD_ONLY", "'auto'")], "Convolve two N-dimensional arrays."),
    ("correlate", "scipy.signal.correlate", "function", "signal", [("in1", "POSITIONAL_ONLY", None), ("in2", "POSITIONAL_ONLY", None), ("mode", "KEYWORD_ONLY", "'full'"), ("method", "KEYWORD_ONLY", "'auto'")], "Cross-correlate two N-dimensional arrays."),
    ("fftconvolve", "scipy.signal.fftconvolve", "function", "signal", [("in1", "POSITIONAL_ONLY", None), ("in2", "POSITIONAL_ONLY", None), ("mode", "KEYWORD_ONLY", "'full'"), ("axes", "KEYWORD_ONLY", "None")], "Convolve two N-dimensional arrays using FFT."),
    ("medfilt", "scipy.signal.medfilt", "function", "signal", [("volume", "POSITIONAL_ONLY", None), ("kernel_size", "KEYWORD_ONLY", "None")], "Perform a median filter on an N-dimensional array."),
]

CANONICAL_SCIPY_OPS: list[CanonicalSubmoduleOpSpec] = build_canonical_submodule_ops(_SCIPY_OP_DEFS)


def _get_scipy_submodule(submod: str) -> types.ModuleType | None:
    """Safely import a specific SciPy submodule.

    Args:
        submod: Name of the SciPy submodule (e.g. 'special', 'linalg').

    Returns:
        The submodule object or None if not available.
    """
    try:
        return importlib.import_module(f"scipy.{submod}")
    except (ImportError, ModuleNotFoundError, AttributeError, RuntimeError, TypeError, ValueError, OSError):
        return None


def collect_api(category: SemanticTier, include_nonpublic: bool = False) -> list[GhostRef]:
    """Collect SciPy mathematical, linear algebra, and signal processing APIs.

    Args:
        category: The SemanticTier category.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        List of GhostRef objects representing SciPy operations.
    """
    if category not in (
        SemanticTier.ARRAY_API,
        SemanticTier.UTIL,
    ):
        return []

    target_submodules = ["special", "linalg", "signal", "ndimage", "stats"]
    loaded_submods: dict[str, types.ModuleType] = {}
    for sub in target_submodules:
        mod = _get_scipy_submodule(sub)
        if mod is not None:
            loaded_submods[sub] = mod

    return collect_from_submodules_or_canonical(
        loaded_submods=loaded_submods,
        canonical_ops=CANONICAL_SCIPY_OPS,
        framework_tag="scipy",
        default_submodule="special",
        default_doc_prefix="SciPy",
        inspector_fn=GhostInspector.inspect,
    )
