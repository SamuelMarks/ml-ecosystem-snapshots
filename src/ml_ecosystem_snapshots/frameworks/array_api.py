"""Python Array API Standard Snapshot Extractor.

Extracts formal operator signatures, parameter constraints, positional/keyword distinctions,
and semantic specifications from the Python Array API Standard (v2023.12 / v2024.12).
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

_ARRAY_API_OP_DEFS: list[CanonicalSubmoduleOpTuple] = [
    ("add", "array_api.add", "function", "", [("x1", "POSITIONAL_ONLY", None), ("x2", "POSITIONAL_ONLY", None)], "Calculates the sum for each element of an array."),
    ("subtract", "array_api.subtract", "function", "", [("x1", "POSITIONAL_ONLY", None), ("x2", "POSITIONAL_ONLY", None)], "Calculates the difference for each element of an array."),
    ("multiply", "array_api.multiply", "function", "", [("x1", "POSITIONAL_ONLY", None), ("x2", "POSITIONAL_ONLY", None)], "Calculates the product for each element of an array."),
    ("divide", "array_api.divide", "function", "", [("x1", "POSITIONAL_ONLY", None), ("x2", "POSITIONAL_ONLY", None)], "Calculates the division for each element of an array."),
    ("floor_divide", "array_api.floor_divide", "function", "", [("x1", "POSITIONAL_ONLY", None), ("x2", "POSITIONAL_ONLY", None)], "Calculates the floor division for each element of an array."),
    ("pow", "array_api.pow", "function", "", [("x1", "POSITIONAL_ONLY", None), ("x2", "POSITIONAL_ONLY", None)], "Calculates x1 raised to the power of x2 for each element."),
    ("remainder", "array_api.remainder", "function", "", [("x1", "POSITIONAL_ONLY", None), ("x2", "POSITIONAL_ONLY", None)], "Calculates the remainder of division each element of an array."),
    ("abs", "array_api.abs", "function", "", [("x", "POSITIONAL_ONLY", None)], "Calculates the absolute value for each element of an array."),
    ("negative", "array_api.negative", "function", "", [("x", "POSITIONAL_ONLY", None)], "Calculates the numerical negative for each element of an array."),
    ("positive", "array_api.positive", "function", "", [("x", "POSITIONAL_ONLY", None)], "Calculates the numerical positive for each element of an array."),
    ("sign", "array_api.sign", "function", "", [("x", "POSITIONAL_ONLY", None)], "Returns an indication of the sign of a number for each element."),
    ("square", "array_api.square", "function", "", [("x", "POSITIONAL_ONLY", None)], "Squares each element of an array."),
    ("sqrt", "array_api.sqrt", "function", "", [("x", "POSITIONAL_ONLY", None)], "Calculates the square root for each element of an array."),
    ("exp", "array_api.exp", "function", "", [("x", "POSITIONAL_ONLY", None)], "Calculates the exponential for each element of an array."),
    ("expm1", "array_api.expm1", "function", "", [("x", "POSITIONAL_ONLY", None)], "Calculates exp(x) - 1 for each element of an array."),
    ("log", "array_api.log", "function", "", [("x", "POSITIONAL_ONLY", None)], "Calculates the natural logarithm of each element of an array."),
    ("log1p", "array_api.log1p", "function", "", [("x", "POSITIONAL_ONLY", None)], "Calculates log(1 + x) for each element of an array."),
    ("log2", "array_api.log2", "function", "", [("x", "POSITIONAL_ONLY", None)], "Calculates the base 2 logarithm of each element of an array."),
    ("log10", "array_api.log10", "function", "", [("x", "POSITIONAL_ONLY", None)], "Calculates the base 10 logarithm of each element of an array."),
    ("sin", "array_api.sin", "function", "", [("x", "POSITIONAL_ONLY", None)], "Calculates the sine of each element of an array."),
    ("cos", "array_api.cos", "function", "", [("x", "POSITIONAL_ONLY", None)], "Calculates the cosine of each element of an array."),
    ("tan", "array_api.tan", "function", "", [("x", "POSITIONAL_ONLY", None)], "Calculates the tangent of each element of an array."),
    ("asin", "array_api.asin", "function", "", [("x", "POSITIONAL_ONLY", None)], "Calculates the inverse sine of each element of an array."),
    ("acos", "array_api.acos", "function", "", [("x", "POSITIONAL_ONLY", None)], "Calculates the inverse cosine of each element of an array."),
    ("atan", "array_api.atan", "function", "", [("x", "POSITIONAL_ONLY", None)], "Calculates the inverse tangent of each element of an array."),
    ("atan2", "array_api.atan2", "function", "", [("x1", "POSITIONAL_ONLY", None), ("x2", "POSITIONAL_ONLY", None)], "Calculates the inverse tangent of x1/x2 for each element."),
    ("sinh", "array_api.sinh", "function", "", [("x", "POSITIONAL_ONLY", None)], "Calculates the hyperbolic sine of each element of an array."),
    ("cosh", "array_api.cosh", "function", "", [("x", "POSITIONAL_ONLY", None)], "Calculates the hyperbolic cosine of each element of an array."),
    ("tanh", "array_api.tanh", "function", "", [("x", "POSITIONAL_ONLY", None)], "Calculates the hyperbolic tangent of each element of an array."),
    ("asinh", "array_api.asinh", "function", "", [("x", "POSITIONAL_ONLY", None)], "Calculates the inverse hyperbolic sine of each element of an array."),
    ("acosh", "array_api.acosh", "function", "", [("x", "POSITIONAL_ONLY", None)], "Calculates the inverse hyperbolic cosine of each element of an array."),
    ("atanh", "array_api.atanh", "function", "", [("x", "POSITIONAL_ONLY", None)], "Calculates the inverse hyperbolic tangent of each element of an array."),
    ("ceil", "array_api.ceil", "function", "", [("x", "POSITIONAL_ONLY", None)], "Rounds each element of an array towards positive infinity."),
    ("floor", "array_api.floor", "function", "", [("x", "POSITIONAL_ONLY", None)], "Rounds each element of an array towards negative infinity."),
    ("round", "array_api.round", "function", "", [("x", "POSITIONAL_ONLY", None)], "Rounds each element of an array to the nearest integer."),
    ("trunc", "array_api.trunc", "function", "", [("x", "POSITIONAL_ONLY", None)], "Truncates each element of an array to an integral value."),
    ("clip", "array_api.clip", "function", "", [("x", "POSITIONAL_ONLY", None), ("min", "KEYWORD_ONLY", "None"), ("max", "KEYWORD_ONLY", "None")], "Clips array values between min and max bounds."),
    ("equal", "array_api.equal", "function", "", [("x1", "POSITIONAL_ONLY", None), ("x2", "POSITIONAL_ONLY", None)], "Calculates element-wise equality between x1 and x2."),
    ("not_equal", "array_api.not_equal", "function", "", [("x1", "POSITIONAL_ONLY", None), ("x2", "POSITIONAL_ONLY", None)], "Calculates element-wise inequality between x1 and x2."),
    ("greater", "array_api.greater", "function", "", [("x1", "POSITIONAL_ONLY", None), ("x2", "POSITIONAL_ONLY", None)], "Calculates element-wise greater than comparison."),
    ("greater_equal", "array_api.greater_equal", "function", "", [("x1", "POSITIONAL_ONLY", None), ("x2", "POSITIONAL_ONLY", None)], "Calculates element-wise greater than or equal comparison."),
    ("less", "array_api.less", "function", "", [("x1", "POSITIONAL_ONLY", None), ("x2", "POSITIONAL_ONLY", None)], "Calculates element-wise less than comparison."),
    ("less_equal", "array_api.less_equal", "function", "", [("x1", "POSITIONAL_ONLY", None), ("x2", "POSITIONAL_ONLY", None)], "Calculates element-wise less than or equal comparison."),
    ("maximum", "array_api.maximum", "function", "", [("x1", "POSITIONAL_ONLY", None), ("x2", "POSITIONAL_ONLY", None)], "Computes the maximum of elements along corresponding dimensions."),
    ("minimum", "array_api.minimum", "function", "", [("x1", "POSITIONAL_ONLY", None), ("x2", "POSITIONAL_ONLY", None)], "Computes the minimum of elements along corresponding dimensions."),
    ("bitwise_and", "array_api.bitwise_and", "function", "", [("x1", "POSITIONAL_ONLY", None), ("x2", "POSITIONAL_ONLY", None)], "Computes the bitwise AND of two arrays."),
    ("bitwise_or", "array_api.bitwise_or", "function", "", [("x1", "POSITIONAL_ONLY", None), ("x2", "POSITIONAL_ONLY", None)], "Computes the bitwise OR of two arrays."),
    ("bitwise_xor", "array_api.bitwise_xor", "function", "", [("x1", "POSITIONAL_ONLY", None), ("x2", "POSITIONAL_ONLY", None)], "Computes the bitwise XOR of two arrays."),
    ("bitwise_invert", "array_api.bitwise_invert", "function", "", [("x", "POSITIONAL_ONLY", None)], "Computes the bitwise inversion of an array."),
    ("bitwise_left_shift", "array_api.bitwise_left_shift", "function", "", [("x1", "POSITIONAL_ONLY", None), ("x2", "POSITIONAL_ONLY", None)], "Shifts the bits of x1 to the left by x2."),
    ("bitwise_right_shift", "array_api.bitwise_right_shift", "function", "", [("x1", "POSITIONAL_ONLY", None), ("x2", "POSITIONAL_ONLY", None)], "Shifts the bits of x1 to the right by x2."),
    ("logical_and", "array_api.logical_and", "function", "", [("x1", "POSITIONAL_ONLY", None), ("x2", "POSITIONAL_ONLY", None)], "Computes the logical AND of two arrays."),
    ("logical_or", "array_api.logical_or", "function", "", [("x1", "POSITIONAL_ONLY", None), ("x2", "POSITIONAL_ONLY", None)], "Computes the logical OR of two arrays."),
    ("logical_xor", "array_api.logical_xor", "function", "", [("x1", "POSITIONAL_ONLY", None), ("x2", "POSITIONAL_ONLY", None)], "Computes the logical XOR of two arrays."),
    ("logical_not", "array_api.logical_not", "function", "", [("x", "POSITIONAL_ONLY", None)], "Computes the logical NOT of an array."),
    ("zeros", "array_api.zeros", "function", "", [("shape", "POSITIONAL_OR_KEYWORD", None), ("dtype", "KEYWORD_ONLY", "None"), ("device", "KEYWORD_ONLY", "None")], "Returns a new array having a specified shape and filled with zeros."),
    ("ones", "array_api.ones", "function", "", [("shape", "POSITIONAL_OR_KEYWORD", None), ("dtype", "KEYWORD_ONLY", "None"), ("device", "KEYWORD_ONLY", "None")], "Returns a new array having a specified shape and filled with ones."),
    ("full", "array_api.full", "function", "", [("shape", "POSITIONAL_OR_KEYWORD", None), ("fill_value", "POSITIONAL_OR_KEYWORD", None), ("dtype", "KEYWORD_ONLY", "None"), ("device", "KEYWORD_ONLY", "None")], "Returns a new array having a specified shape and filled with fill_value."),
    ("empty", "array_api.empty", "function", "", [("shape", "POSITIONAL_OR_KEYWORD", None), ("dtype", "KEYWORD_ONLY", "None"), ("device", "KEYWORD_ONLY", "None")], "Returns an uninitialized array having a specified shape."),
    ("zeros_like", "array_api.zeros_like", "function", "", [("x", "POSITIONAL_ONLY", None), ("dtype", "KEYWORD_ONLY", "None"), ("device", "KEYWORD_ONLY", "None")], "Returns a new array with the same shape as x and filled with zeros."),
    ("ones_like", "array_api.ones_like", "function", "", [("x", "POSITIONAL_ONLY", None), ("dtype", "KEYWORD_ONLY", "None"), ("device", "KEYWORD_ONLY", "None")], "Returns a new array with the same shape as x and filled with ones."),
    ("full_like", "array_api.full_like", "function", "", [("x", "POSITIONAL_ONLY", None), ("fill_value", "POSITIONAL_OR_KEYWORD", None), ("dtype", "KEYWORD_ONLY", "None"), ("device", "KEYWORD_ONLY", "None")], "Returns a new array with the same shape as x and filled with fill_value."),
    ("empty_like", "array_api.empty_like", "function", "", [("x", "POSITIONAL_ONLY", None), ("dtype", "KEYWORD_ONLY", "None"), ("device", "KEYWORD_ONLY", "None")], "Returns an uninitialized array with the same shape as x."),
    (
        "arange",
        "array_api.arange",
        "function",
        "",
        [("start", "POSITIONAL_OR_KEYWORD", None), ("stop", "POSITIONAL_OR_KEYWORD", "None"), ("step", "POSITIONAL_OR_KEYWORD", "1"), ("dtype", "KEYWORD_ONLY", "None"), ("device", "KEYWORD_ONLY", "None")],
        "Returns evenly spaced values within a given interval.",
    ),
    (
        "linspace",
        "array_api.linspace",
        "function",
        "",
        [("start", "POSITIONAL_OR_KEYWORD", None), ("stop", "POSITIONAL_OR_KEYWORD", None), ("num", "POSITIONAL_OR_KEYWORD", None), ("dtype", "KEYWORD_ONLY", "None"), ("device", "KEYWORD_ONLY", "None"), ("endpoint", "KEYWORD_ONLY", "True")],
        "Returns evenly spaced numbers over a specified interval.",
    ),
    (
        "eye",
        "array_api.eye",
        "function",
        "",
        [("n_rows", "POSITIONAL_OR_KEYWORD", None), ("n_cols", "POSITIONAL_OR_KEYWORD", "None"), ("k", "KEYWORD_ONLY", "0"), ("dtype", "KEYWORD_ONLY", "None"), ("device", "KEYWORD_ONLY", "None")],
        "Returns a 2D array with ones on the diagonal and zeros elsewhere.",
    ),
    ("asarray", "array_api.asarray", "function", "", [("obj", "POSITIONAL_ONLY", None), ("dtype", "KEYWORD_ONLY", "None"), ("device", "KEYWORD_ONLY", "None"), ("copy", "KEYWORD_ONLY", "None")], "Converts the input to an array."),
    ("matmul", "array_api.matmul", "function", "", [("x1", "POSITIONAL_ONLY", None), ("x2", "POSITIONAL_ONLY", None)], "Computes the matrix product of two arrays."),
    ("matrix_transpose", "array_api.matrix_transpose", "function", "", [("x", "POSITIONAL_ONLY", None)], "Transposes a matrix (or a stack of matrices)."),
    ("tensordot", "array_api.tensordot", "function", "", [("x1", "POSITIONAL_ONLY", None), ("x2", "POSITIONAL_ONLY", None), ("axes", "KEYWORD_ONLY", "2")], "Computes tensor contraction along specified axes."),
    ("vecdot", "array_api.vecdot", "function", "", [("x1", "POSITIONAL_ONLY", None), ("x2", "POSITIONAL_ONLY", None), ("axis", "KEYWORD_ONLY", "-1")], "Computes vector dot product along the specified axis."),
    ("sum", "array_api.sum", "function", "", [("x", "POSITIONAL_ONLY", None), ("axis", "KEYWORD_ONLY", "None"), ("dtype", "KEYWORD_ONLY", "None"), ("keepdims", "KEYWORD_ONLY", "False")], "Calculates the sum of elements along specified axes."),
    ("prod", "array_api.prod", "function", "", [("x", "POSITIONAL_ONLY", None), ("axis", "KEYWORD_ONLY", "None"), ("dtype", "KEYWORD_ONLY", "None"), ("keepdims", "KEYWORD_ONLY", "False")], "Calculates the product of elements along specified axes."),
    ("mean", "array_api.mean", "function", "", [("x", "POSITIONAL_ONLY", None), ("axis", "KEYWORD_ONLY", "None"), ("keepdims", "KEYWORD_ONLY", "False")], "Calculates the arithmetic mean along specified axes."),
    ("std", "array_api.std", "function", "", [("x", "POSITIONAL_ONLY", None), ("axis", "KEYWORD_ONLY", "None"), ("correction", "KEYWORD_ONLY", "0.0"), ("keepdims", "KEYWORD_ONLY", "False")], "Calculates the standard deviation along specified axes."),
    ("var", "array_api.var", "function", "", [("x", "POSITIONAL_ONLY", None), ("axis", "KEYWORD_ONLY", "None"), ("correction", "KEYWORD_ONLY", "0.0"), ("keepdims", "KEYWORD_ONLY", "False")], "Calculates the variance along specified axes."),
    ("max", "array_api.max", "function", "", [("x", "POSITIONAL_ONLY", None), ("axis", "KEYWORD_ONLY", "None"), ("keepdims", "KEYWORD_ONLY", "False")], "Calculates the maximum value along specified axes."),
    ("min", "array_api.min", "function", "", [("x", "POSITIONAL_ONLY", None), ("axis", "KEYWORD_ONLY", "None"), ("keepdims", "KEYWORD_ONLY", "False")], "Calculates the minimum value along specified axes."),
    ("cumulative_sum", "array_api.cumulative_sum", "function", "", [("x", "POSITIONAL_ONLY", None), ("axis", "KEYWORD_ONLY", "None"), ("dtype", "KEYWORD_ONLY", "None"), ("include_initial", "KEYWORD_ONLY", "False")], "Calculates the cumulative sum of elements along a specified axis."),
    ("argmax", "array_api.argmax", "function", "", [("x", "POSITIONAL_ONLY", None), ("axis", "KEYWORD_ONLY", "None"), ("keepdims", "KEYWORD_ONLY", "False")], "Returns indices of maximum values along a specified axis."),
    ("argmin", "array_api.argmin", "function", "", [("x", "POSITIONAL_ONLY", None), ("axis", "KEYWORD_ONLY", "None"), ("keepdims", "KEYWORD_ONLY", "False")], "Returns indices of minimum values along a specified axis."),
    ("nonzero", "array_api.nonzero", "function", "", [("x", "POSITIONAL_ONLY", None)], "Returns indices of elements that are non-zero."),
    ("where", "array_api.where", "function", "", [("condition", "POSITIONAL_ONLY", None), ("x1", "POSITIONAL_ONLY", None), ("x2", "POSITIONAL_ONLY", None)], "Selects elements from x1 or x2 depending on condition."),
    ("sort", "array_api.sort", "function", "", [("x", "POSITIONAL_ONLY", None), ("axis", "KEYWORD_ONLY", "-1"), ("descending", "KEYWORD_ONLY", "False"), ("stable", "KEYWORD_ONLY", "True")], "Returns a sorted copy of an array."),
    ("argsort", "array_api.argsort", "function", "", [("x", "POSITIONAL_ONLY", None), ("axis", "KEYWORD_ONLY", "-1"), ("descending", "KEYWORD_ONLY", "False"), ("stable", "KEYWORD_ONLY", "True")], "Returns the indices that would sort an array."),
    ("concat", "array_api.concat", "function", "", [("arrays", "POSITIONAL_ONLY", None), ("axis", "KEYWORD_ONLY", "0")], "Joins a sequence of arrays along an existing axis."),
    ("stack", "array_api.stack", "function", "", [("arrays", "POSITIONAL_ONLY", None), ("axis", "KEYWORD_ONLY", "0")], "Joins a sequence of arrays along a new axis."),
    ("reshape", "array_api.reshape", "function", "", [("x", "POSITIONAL_ONLY", None), ("shape", "POSITIONAL_ONLY", None), ("copy", "KEYWORD_ONLY", "None")], "Gives a new shape to an array without changing its data."),
    ("squeeze", "array_api.squeeze", "function", "", [("x", "POSITIONAL_ONLY", None), ("axis", "POSITIONAL_ONLY", None)], "Removes singleton dimensions (axes) from an array."),
    ("expand_dims", "array_api.expand_dims", "function", "", [("x", "POSITIONAL_ONLY", None), ("axis", "KEYWORD_ONLY", "0")], "Expands the shape of an array by inserting a new axis."),
    ("permute_dims", "array_api.permute_dims", "function", "", [("x", "POSITIONAL_ONLY", None), ("axes", "POSITIONAL_ONLY", None)], "Permutes the dimensions of an array."),
    ("flip", "array_api.flip", "function", "", [("x", "POSITIONAL_ONLY", None), ("axis", "KEYWORD_ONLY", "None")], "Reverses the order of elements in an array along the given axis."),
    ("roll", "array_api.roll", "function", "", [("x", "POSITIONAL_ONLY", None), ("shift", "POSITIONAL_ONLY", None), ("axis", "KEYWORD_ONLY", "None")], "Rolls array elements along a specified axis."),
    ("all", "array_api.all", "function", "", [("x", "POSITIONAL_ONLY", None), ("axis", "KEYWORD_ONLY", "None"), ("keepdims", "KEYWORD_ONLY", "False")], "Tests whether all input array elements evaluate to True."),
    ("any", "array_api.any", "function", "", [("x", "POSITIONAL_ONLY", None), ("axis", "KEYWORD_ONLY", "None"), ("keepdims", "KEYWORD_ONLY", "False")], "Tests whether any input array elements evaluate to True."),
    ("isfinite", "array_api.isfinite", "function", "", [("x", "POSITIONAL_ONLY", None)], "Tests each element for finiteness."),
    ("isinf", "array_api.isinf", "function", "", [("x", "POSITIONAL_ONLY", None)], "Tests each element for positive or negative infinity."),
    ("isnan", "array_api.isnan", "function", "", [("x", "POSITIONAL_ONLY", None)], "Tests each element for NaN."),
    ("logaddexp", "array_api.logaddexp", "function", "", [("x1", "POSITIONAL_ONLY", None), ("x2", "POSITIONAL_ONLY", None)], "Calculates the logarithm of the sum of exponentiations."),
    ("unique_all", "array_api.unique_all", "function", "", [("x", "POSITIONAL_ONLY", None)], "Returns unique elements, indices, inverse indices, and counts."),
    ("unique_counts", "array_api.unique_counts", "function", "", [("x", "POSITIONAL_ONLY", None)], "Returns unique elements and their counts."),
    ("unique_inverse", "array_api.unique_inverse", "function", "", [("x", "POSITIONAL_ONLY", None)], "Returns unique elements and inverse indices."),
    ("unique_values", "array_api.unique_values", "function", "", [("x", "POSITIONAL_ONLY", None)], "Returns the minimum value for a given floating-point data type."),
]

CANONICAL_ARRAY_API_OPS: list[CanonicalSubmoduleOpSpec] = build_canonical_submodule_ops(_ARRAY_API_OP_DEFS)


def _get_array_api_module() -> types.ModuleType | None:
    """Retrieve an installed Array API implementation if available.

    Returns:
        The array_api_compat module or numpy.array_api if present, else None.
    """
    for mod_name in ("array_api_compat.numpy", "numpy.array_api", "array_api_compat"):
        try:
            return importlib.import_module(mod_name)
        except Exception as exc:
            del exc
    return None


def collect_api(category: SemanticTier, include_nonpublic: bool = False) -> list[GhostRef]:
    """Collect Python Array API standard function and operator definitions.

    Args:
        category: The SemanticTier category.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        List of GhostRef objects representing Array API Standard operations.
    """
    if category not in (
        SemanticTier.ARRAY_API,
        SemanticTier.UTIL,
    ):
        return []

    mod = _get_array_api_module()
    loaded_submods = {"standard": mod} if mod is not None else {}
    return collect_from_submodules_or_canonical(
        loaded_submods=loaded_submods,
        canonical_ops=CANONICAL_ARRAY_API_OPS,
        framework_tag="array_api",
        default_submodule="standard",
        default_doc_prefix="Array API Standard",
        inspector_fn=GhostInspector.inspect,
    )
