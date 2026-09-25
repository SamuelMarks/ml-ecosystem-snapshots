"""SciPy API Snapshot Extractor.

Extracts mathematical special functions (Bessel, Gamma, Error, Hypergeometric),
linear algebra routines, signal processing kernels, n-dimensional filters,
and statistical distributions from the SciPy library.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

from ml_switcheroo_ir.schema.ghost import (
    GhostParam,
    GhostPythonRef,
    GhostRef,
    ParameterKind,
    SemanticTier,
)
from ..models import GhostInspector


CANONICAL_SCIPY_OPS: List[Dict[str, Any]] = [
    # Special functions
    {
        "name": "erf",
        "api_path": "scipy.special.erf",
        "submodule": "special",
        "params": [
            {"name": "z", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Returns the error function of complex argument.",
    },
    {
        "name": "erfc",
        "api_path": "scipy.special.erfc",
        "submodule": "special",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Complementary error function, 1 - erf(x).",
    },
    {
        "name": "gamma",
        "api_path": "scipy.special.gamma",
        "submodule": "special",
        "params": [
            {"name": "z", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "gamma function.",
    },
    {
        "name": "gammaln",
        "api_path": "scipy.special.gammaln",
        "submodule": "special",
        "params": [
            {"name": "z", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Logarithm of the absolute value of the gamma function.",
    },
    {
        "name": "betaln",
        "api_path": "scipy.special.betaln",
        "submodule": "special",
        "params": [
            {"name": "a", "kind": "POSITIONAL_ONLY"},
            {"name": "b", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Natural logarithm of absolute value of beta function.",
    },
    {
        "name": "digamma",
        "api_path": "scipy.special.digamma",
        "submodule": "special",
        "params": [
            {"name": "z", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "The digamma function, logarithmic derivative of the gamma function.",
    },
    {
        "name": "polygamma",
        "api_path": "scipy.special.polygamma",
        "submodule": "special",
        "params": [
            {"name": "n", "kind": "POSITIONAL_ONLY"},
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Polygamma functions n-th derivative of digamma at x.",
    },
    {
        "name": "j0",
        "api_path": "scipy.special.j0",
        "submodule": "special",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Bessel function of the first kind of order 0.",
    },
    {
        "name": "j1",
        "api_path": "scipy.special.j1",
        "submodule": "special",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Bessel function of the first kind of order 1.",
    },
    {
        "name": "y0",
        "api_path": "scipy.special.y0",
        "submodule": "special",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Bessel function of the second kind of order 0.",
    },
    {
        "name": "y1",
        "api_path": "scipy.special.y1",
        "submodule": "special",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Bessel function of the second kind of order 1.",
    },
    {
        "name": "i0",
        "api_path": "scipy.special.i0",
        "submodule": "special",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Modified Bessel function of order 0.",
    },
    {
        "name": "i1",
        "api_path": "scipy.special.i1",
        "submodule": "special",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Modified Bessel function of order 1.",
    },
    {
        "name": "k0",
        "api_path": "scipy.special.k0",
        "submodule": "special",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Modified Bessel function of the second kind of order 0.",
    },
    {
        "name": "k1",
        "api_path": "scipy.special.k1",
        "submodule": "special",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Modified Bessel function of the second kind of order 1.",
    },
    {
        "name": "sinc",
        "api_path": "scipy.special.sinc",
        "submodule": "special",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Return the normalized sinc function.",
    },
    {
        "name": "logit",
        "api_path": "scipy.special.logit",
        "submodule": "special",
        "params": [
            {"name": "p", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Logit function: log(p / (1 - p)).",
    },
    {
        "name": "expit",
        "api_path": "scipy.special.expit",
        "submodule": "special",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Expit (logistic sigmoid) function: 1 / (1 + exp(-x)).",
    },
    {
        "name": "softmax",
        "api_path": "scipy.special.softmax",
        "submodule": "special",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
            {"name": "axis", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Softmax function along given axis.",
    },
    {
        "name": "logsumexp",
        "api_path": "scipy.special.logsumexp",
        "submodule": "special",
        "params": [
            {"name": "a", "kind": "POSITIONAL_ONLY"},
            {"name": "axis", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "b", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "keepdims", "kind": "KEYWORD_ONLY", "default": "False"},
            {"name": "return_sign", "kind": "KEYWORD_ONLY", "default": "False"},
        ],
        "docstring": "Compute the log of the sum of exponentials of input elements.",
    },
    # Linear algebra functions
    {
        "name": "inv",
        "api_path": "scipy.linalg.inv",
        "submodule": "linalg",
        "params": [
            {"name": "a", "kind": "POSITIONAL_ONLY"},
            {"name": "overwrite_a", "kind": "KEYWORD_ONLY", "default": "False"},
            {"name": "check_finite", "kind": "KEYWORD_ONLY", "default": "True"},
        ],
        "docstring": "Compute the inverse of a matrix.",
    },
    {
        "name": "pinv",
        "api_path": "scipy.linalg.pinv",
        "submodule": "linalg",
        "params": [
            {"name": "a", "kind": "POSITIONAL_ONLY"},
            {"name": "atol", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "rtol", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "return_rank", "kind": "KEYWORD_ONLY", "default": "False"},
            {"name": "check_finite", "kind": "KEYWORD_ONLY", "default": "True"},
        ],
        "docstring": "Compute the Moore-Penrose pseudo-inverse of a matrix.",
    },
    {
        "name": "det",
        "api_path": "scipy.linalg.det",
        "submodule": "linalg",
        "params": [
            {"name": "a", "kind": "POSITIONAL_ONLY"},
            {"name": "overwrite_a", "kind": "KEYWORD_ONLY", "default": "False"},
            {"name": "check_finite", "kind": "KEYWORD_ONLY", "default": "True"},
        ],
        "docstring": "Compute the determinant of a matrix.",
    },
    {
        "name": "norm",
        "api_path": "scipy.linalg.norm",
        "submodule": "linalg",
        "params": [
            {"name": "a", "kind": "POSITIONAL_ONLY"},
            {"name": "ord", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "axis", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "keepdims", "kind": "KEYWORD_ONLY", "default": "False"},
            {"name": "check_finite", "kind": "KEYWORD_ONLY", "default": "True"},
        ],
        "docstring": "Matrix or vector norm.",
    },
    {
        "name": "cholesky",
        "api_path": "scipy.linalg.cholesky",
        "submodule": "linalg",
        "params": [
            {"name": "a", "kind": "POSITIONAL_ONLY"},
            {"name": "lower", "kind": "KEYWORD_ONLY", "default": "False"},
            {"name": "overwrite_a", "kind": "KEYWORD_ONLY", "default": "False"},
            {"name": "check_finite", "kind": "KEYWORD_ONLY", "default": "True"},
        ],
        "docstring": "Compute the Cholesky decomposition of a matrix.",
    },
    {
        "name": "qr",
        "api_path": "scipy.linalg.qr",
        "submodule": "linalg",
        "params": [
            {"name": "a", "kind": "POSITIONAL_ONLY"},
            {"name": "overwrite_a", "kind": "KEYWORD_ONLY", "default": "False"},
            {"name": "mode", "kind": "KEYWORD_ONLY", "default": "'full'"},
            {"name": "pivoting", "kind": "KEYWORD_ONLY", "default": "False"},
            {"name": "check_finite", "kind": "KEYWORD_ONLY", "default": "True"},
        ],
        "docstring": "Compute QR decomposition of a matrix.",
    },
    {
        "name": "svd",
        "api_path": "scipy.linalg.svd",
        "submodule": "linalg",
        "params": [
            {"name": "a", "kind": "POSITIONAL_ONLY"},
            {"name": "full_matrices", "kind": "KEYWORD_ONLY", "default": "True"},
            {"name": "compute_uv", "kind": "KEYWORD_ONLY", "default": "True"},
            {"name": "overwrite_a", "kind": "KEYWORD_ONLY", "default": "False"},
            {"name": "check_finite", "kind": "KEYWORD_ONLY", "default": "True"},
            {"name": "lapack_driver", "kind": "KEYWORD_ONLY", "default": "'gesdd'"},
        ],
        "docstring": "Singular Value Decomposition.",
    },
    {
        "name": "solve",
        "api_path": "scipy.linalg.solve",
        "submodule": "linalg",
        "params": [
            {"name": "a", "kind": "POSITIONAL_ONLY"},
            {"name": "b", "kind": "POSITIONAL_ONLY"},
            {"name": "lower", "kind": "KEYWORD_ONLY", "default": "False"},
            {"name": "overwrite_a", "kind": "KEYWORD_ONLY", "default": "False"},
            {"name": "overwrite_b", "kind": "KEYWORD_ONLY", "default": "False"},
            {"name": "check_finite", "kind": "KEYWORD_ONLY", "default": "True"},
            {"name": "assume_a", "kind": "KEYWORD_ONLY", "default": "'gen'"},
        ],
        "docstring": "Solves the linear equation set a * x = b for the unknown x.",
    },
    {
        "name": "expm",
        "api_path": "scipy.linalg.expm",
        "submodule": "linalg",
        "params": [
            {"name": "a", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Compute the matrix exponential using Pade approximation.",
    },
    # Signal processing
    {
        "name": "convolve",
        "api_path": "scipy.signal.convolve",
        "submodule": "signal",
        "params": [
            {"name": "in1", "kind": "POSITIONAL_ONLY"},
            {"name": "in2", "kind": "POSITIONAL_ONLY"},
            {"name": "mode", "kind": "KEYWORD_ONLY", "default": "'full'"},
            {"name": "method", "kind": "KEYWORD_ONLY", "default": "'auto'"},
        ],
        "docstring": "Convolve two N-dimensional arrays.",
    },
    {
        "name": "correlate",
        "api_path": "scipy.signal.correlate",
        "submodule": "signal",
        "params": [
            {"name": "in1", "kind": "POSITIONAL_ONLY"},
            {"name": "in2", "kind": "POSITIONAL_ONLY"},
            {"name": "mode", "kind": "KEYWORD_ONLY", "default": "'full'"},
            {"name": "method", "kind": "KEYWORD_ONLY", "default": "'auto'"},
        ],
        "docstring": "Cross-correlate two N-dimensional arrays.",
    },
    {
        "name": "fftconvolve",
        "api_path": "scipy.signal.fftconvolve",
        "submodule": "signal",
        "params": [
            {"name": "in1", "kind": "POSITIONAL_ONLY"},
            {"name": "in2", "kind": "POSITIONAL_ONLY"},
            {"name": "mode", "kind": "KEYWORD_ONLY", "default": "'full'"},
            {"name": "axes", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Convolve two N-dimensional arrays using FFT.",
    },
    {
        "name": "medfilt",
        "api_path": "scipy.signal.medfilt",
        "submodule": "signal",
        "params": [
            {"name": "volume", "kind": "POSITIONAL_ONLY"},
            {"name": "kernel_size", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Perform a median filter on an N-dimensional array.",
    },
]


def _get_scipy_submodule(submod: str) -> Any:
    """Safely import a specific SciPy submodule.

    Args:
        submod: Name of the SciPy submodule (e.g. 'special', 'linalg').

    Returns:
        The submodule object or None if not available.
    """
    try:
        return importlib.import_module(f"scipy.{submod}")
    except (ImportError, Exception):
        return None


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
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

    # Map categories to submodules
    target_submodules = ["special", "linalg", "signal"]
    loaded_submods: Dict[str, Any] = {}
    for sub in target_submodules:
        mod = _get_scipy_submodule(sub)
        if mod is not None:
            loaded_submods[sub] = mod

    if not loaded_submods:
        refs: List[GhostRef] = []
        for op in CANONICAL_SCIPY_OPS:
            params = [
                GhostParam(
                    name=p["name"],
                    kind=getattr(ParameterKind, p.get("kind", "POSITIONAL_OR_KEYWORD")),
                    default=p.get("default"),
                )
                for p in op.get("params", [])
            ]
            refs.append(
                GhostPythonRef(
                    name=op["name"],
                    api_path=op["api_path"],
                    kind="function",
                    params=params,
                    docstring=op.get("docstring", f"SciPy {op['name']} API."),
                    environment_tags=["scipy", op.get("submodule", "special")],
                    domain_metadata={"submodule": op.get("submodule", "special")},
                )
            )
        return refs

    results: List[GhostRef] = []
    seen: set[str] = set()

    for op in CANONICAL_SCIPY_OPS:
        sub = op.get("submodule", "special")
        if sub in loaded_submods:
            mod = loaded_submods[sub]
            name = op["name"]
            if hasattr(mod, name):
                obj = getattr(mod, name)
                try:
                    ref = GhostInspector.inspect(obj, op["api_path"], is_public=True)
                    ref.environment_tags = list(ref.environment_tags or ()) + [
                        "scipy",
                        sub,
                    ]
                    results.append(ref)
                    seen.add(op["api_path"])
                except Exception:
                    pass

    # Fill any missed ops from canonical definitions
    for op in CANONICAL_SCIPY_OPS:
        if op["api_path"] not in seen:
            params = [
                GhostParam(
                    name=p["name"],
                    kind=getattr(ParameterKind, p.get("kind", "POSITIONAL_OR_KEYWORD")),
                    default=p.get("default"),
                )
                for p in op.get("params", [])
            ]
            results.append(
                GhostPythonRef(
                    name=op["name"],
                    api_path=op["api_path"],
                    kind="function",
                    params=params,
                    docstring=op.get("docstring", f"SciPy {op['name']} API."),
                    environment_tags=["scipy", op.get("submodule", "special")],
                    domain_metadata={"submodule": op.get("submodule", "special")},
                )
            )

    return results
