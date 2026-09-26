"""Python Array API Standard Snapshot Extractor.

Extracts formal operator signatures, parameter constraints, positional/keyword distinctions,
and semantic specifications from the Python Array API Standard (v2023.12 / v2024.12).
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


CANONICAL_ARRAY_API_OPS: List[Dict[str, Any]] = [
    # Elementwise binary ops
    {
        "name": "add",
        "api_path": "array_api.add",
        "params": [
            {"name": "x1", "kind": "POSITIONAL_ONLY"},
            {"name": "x2", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates the sum for each element of an array.",
    },
    {
        "name": "subtract",
        "api_path": "array_api.subtract",
        "params": [
            {"name": "x1", "kind": "POSITIONAL_ONLY"},
            {"name": "x2", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates the difference for each element of an array.",
    },
    {
        "name": "multiply",
        "api_path": "array_api.multiply",
        "params": [
            {"name": "x1", "kind": "POSITIONAL_ONLY"},
            {"name": "x2", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates the product for each element of an array.",
    },
    {
        "name": "divide",
        "api_path": "array_api.divide",
        "params": [
            {"name": "x1", "kind": "POSITIONAL_ONLY"},
            {"name": "x2", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates the division for each element of an array.",
    },
    {
        "name": "floor_divide",
        "api_path": "array_api.floor_divide",
        "params": [
            {"name": "x1", "kind": "POSITIONAL_ONLY"},
            {"name": "x2", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates the floor division for each element of an array.",
    },
    {
        "name": "pow",
        "api_path": "array_api.pow",
        "params": [
            {"name": "x1", "kind": "POSITIONAL_ONLY"},
            {"name": "x2", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates x1 raised to the power of x2 for each element.",
    },
    {
        "name": "remainder",
        "api_path": "array_api.remainder",
        "params": [
            {"name": "x1", "kind": "POSITIONAL_ONLY"},
            {"name": "x2", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates the remainder of division each element of an array.",
    },
    # Elementwise unary ops
    {
        "name": "abs",
        "api_path": "array_api.abs",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates the absolute value for each element of an array.",
    },
    {
        "name": "negative",
        "api_path": "array_api.negative",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates the numerical negative for each element of an array.",
    },
    {
        "name": "positive",
        "api_path": "array_api.positive",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates the numerical positive for each element of an array.",
    },
    {
        "name": "sign",
        "api_path": "array_api.sign",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Returns an indication of the sign of a number for each element.",
    },
    {
        "name": "square",
        "api_path": "array_api.square",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Squares each element of an array.",
    },
    {
        "name": "sqrt",
        "api_path": "array_api.sqrt",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates the square root for each element of an array.",
    },
    {
        "name": "exp",
        "api_path": "array_api.exp",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates the exponential for each element of an array.",
    },
    {
        "name": "expm1",
        "api_path": "array_api.expm1",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates exp(x) - 1 for each element of an array.",
    },
    {
        "name": "log",
        "api_path": "array_api.log",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates the natural logarithm of each element of an array.",
    },
    {
        "name": "log1p",
        "api_path": "array_api.log1p",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates log(1 + x) for each element of an array.",
    },
    {
        "name": "log2",
        "api_path": "array_api.log2",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates the base 2 logarithm of each element of an array.",
    },
    {
        "name": "log10",
        "api_path": "array_api.log10",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates the base 10 logarithm of each element of an array.",
    },
    {
        "name": "sin",
        "api_path": "array_api.sin",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates the sine of each element of an array.",
    },
    {
        "name": "cos",
        "api_path": "array_api.cos",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates the cosine of each element of an array.",
    },
    {
        "name": "tan",
        "api_path": "array_api.tan",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates the tangent of each element of an array.",
    },
    {
        "name": "asin",
        "api_path": "array_api.asin",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates the inverse sine of each element of an array.",
    },
    {
        "name": "acos",
        "api_path": "array_api.acos",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates the inverse cosine of each element of an array.",
    },
    {
        "name": "atan",
        "api_path": "array_api.atan",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates the inverse tangent of each element of an array.",
    },
    {
        "name": "atan2",
        "api_path": "array_api.atan2",
        "params": [
            {"name": "x1", "kind": "POSITIONAL_ONLY"},
            {"name": "x2", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates the inverse tangent of x1/x2 for each element.",
    },
    {
        "name": "sinh",
        "api_path": "array_api.sinh",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates the hyperbolic sine of each element of an array.",
    },
    {
        "name": "cosh",
        "api_path": "array_api.cosh",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates the hyperbolic cosine of each element of an array.",
    },
    {
        "name": "tanh",
        "api_path": "array_api.tanh",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates the hyperbolic tangent of each element of an array.",
    },
    {
        "name": "asinh",
        "api_path": "array_api.asinh",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates the inverse hyperbolic sine of each element of an array.",
    },
    {
        "name": "acosh",
        "api_path": "array_api.acosh",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates the inverse hyperbolic cosine of each element of an array.",
    },
    {
        "name": "atanh",
        "api_path": "array_api.atanh",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates the inverse hyperbolic tangent of each element of an array.",
    },
    # Rounding & Clipping
    {
        "name": "ceil",
        "api_path": "array_api.ceil",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Rounds each element of an array towards positive infinity.",
    },
    {
        "name": "floor",
        "api_path": "array_api.floor",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Rounds each element of an array towards negative infinity.",
    },
    {
        "name": "round",
        "api_path": "array_api.round",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Rounds each element of an array to the nearest integer.",
    },
    {
        "name": "trunc",
        "api_path": "array_api.trunc",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Truncates each element of an array to an integral value.",
    },
    {
        "name": "clip",
        "api_path": "array_api.clip",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
            {"name": "min", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "max", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Clips array values between min and max bounds.",
    },
    # Comparison ops
    {
        "name": "equal",
        "api_path": "array_api.equal",
        "params": [
            {"name": "x1", "kind": "POSITIONAL_ONLY"},
            {"name": "x2", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates element-wise equality between x1 and x2.",
    },
    {
        "name": "not_equal",
        "api_path": "array_api.not_equal",
        "params": [
            {"name": "x1", "kind": "POSITIONAL_ONLY"},
            {"name": "x2", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates element-wise inequality between x1 and x2.",
    },
    {
        "name": "greater",
        "api_path": "array_api.greater",
        "params": [
            {"name": "x1", "kind": "POSITIONAL_ONLY"},
            {"name": "x2", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates element-wise greater than comparison.",
    },
    {
        "name": "greater_equal",
        "api_path": "array_api.greater_equal",
        "params": [
            {"name": "x1", "kind": "POSITIONAL_ONLY"},
            {"name": "x2", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates element-wise greater than or equal comparison.",
    },
    {
        "name": "less",
        "api_path": "array_api.less",
        "params": [
            {"name": "x1", "kind": "POSITIONAL_ONLY"},
            {"name": "x2", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates element-wise less than comparison.",
    },
    {
        "name": "less_equal",
        "api_path": "array_api.less_equal",
        "params": [
            {"name": "x1", "kind": "POSITIONAL_ONLY"},
            {"name": "x2", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates element-wise less than or equal comparison.",
    },
    {
        "name": "maximum",
        "api_path": "array_api.maximum",
        "params": [
            {"name": "x1", "kind": "POSITIONAL_ONLY"},
            {"name": "x2", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Computes the maximum of elements along corresponding dimensions.",
    },
    {
        "name": "minimum",
        "api_path": "array_api.minimum",
        "params": [
            {"name": "x1", "kind": "POSITIONAL_ONLY"},
            {"name": "x2", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Computes the minimum of elements along corresponding dimensions.",
    },
    # Logic & Bitwise
    {
        "name": "bitwise_and",
        "api_path": "array_api.bitwise_and",
        "params": [
            {"name": "x1", "kind": "POSITIONAL_ONLY"},
            {"name": "x2", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Computes the bitwise AND of two arrays.",
    },
    {
        "name": "bitwise_or",
        "api_path": "array_api.bitwise_or",
        "params": [
            {"name": "x1", "kind": "POSITIONAL_ONLY"},
            {"name": "x2", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Computes the bitwise OR of two arrays.",
    },
    {
        "name": "bitwise_xor",
        "api_path": "array_api.bitwise_xor",
        "params": [
            {"name": "x1", "kind": "POSITIONAL_ONLY"},
            {"name": "x2", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Computes the bitwise XOR of two arrays.",
    },
    {
        "name": "bitwise_invert",
        "api_path": "array_api.bitwise_invert",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Computes the bitwise inversion of an array.",
    },
    {
        "name": "bitwise_left_shift",
        "api_path": "array_api.bitwise_left_shift",
        "params": [
            {"name": "x1", "kind": "POSITIONAL_ONLY"},
            {"name": "x2", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Shifts the bits of x1 to the left by x2.",
    },
    {
        "name": "bitwise_right_shift",
        "api_path": "array_api.bitwise_right_shift",
        "params": [
            {"name": "x1", "kind": "POSITIONAL_ONLY"},
            {"name": "x2", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Shifts the bits of x1 to the right by x2.",
    },
    {
        "name": "logical_and",
        "api_path": "array_api.logical_and",
        "params": [
            {"name": "x1", "kind": "POSITIONAL_ONLY"},
            {"name": "x2", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Computes the logical AND of two arrays.",
    },
    {
        "name": "logical_or",
        "api_path": "array_api.logical_or",
        "params": [
            {"name": "x1", "kind": "POSITIONAL_ONLY"},
            {"name": "x2", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Computes the logical OR of two arrays.",
    },
    {
        "name": "logical_xor",
        "api_path": "array_api.logical_xor",
        "params": [
            {"name": "x1", "kind": "POSITIONAL_ONLY"},
            {"name": "x2", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Computes the logical XOR of two arrays.",
    },
    {
        "name": "logical_not",
        "api_path": "array_api.logical_not",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Computes the logical NOT of an array.",
    },
    # Array Creation
    {
        "name": "zeros",
        "api_path": "array_api.zeros",
        "params": [
            {"name": "shape", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "dtype", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "device", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Returns a new array having a specified shape and filled with zeros.",
    },
    {
        "name": "ones",
        "api_path": "array_api.ones",
        "params": [
            {"name": "shape", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "dtype", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "device", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Returns a new array having a specified shape and filled with ones.",
    },
    {
        "name": "full",
        "api_path": "array_api.full",
        "params": [
            {"name": "shape", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "fill_value", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "dtype", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "device", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Returns a new array having a specified shape and filled with fill_value.",
    },
    {
        "name": "empty",
        "api_path": "array_api.empty",
        "params": [
            {"name": "shape", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "dtype", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "device", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Returns an uninitialized array having a specified shape.",
    },
    {
        "name": "zeros_like",
        "api_path": "array_api.zeros_like",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
            {"name": "dtype", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "device", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Returns a new array with the same shape as x and filled with zeros.",
    },
    {
        "name": "ones_like",
        "api_path": "array_api.ones_like",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
            {"name": "dtype", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "device", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Returns a new array with the same shape as x and filled with ones.",
    },
    {
        "name": "full_like",
        "api_path": "array_api.full_like",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
            {"name": "fill_value", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "dtype", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "device", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Returns a new array with the same shape as x and filled with fill_value.",
    },
    {
        "name": "empty_like",
        "api_path": "array_api.empty_like",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
            {"name": "dtype", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "device", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Returns an uninitialized array with the same shape as x.",
    },
    {
        "name": "arange",
        "api_path": "array_api.arange",
        "params": [
            {"name": "start", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "stop", "kind": "POSITIONAL_OR_KEYWORD", "default": "None"},
            {"name": "step", "kind": "POSITIONAL_OR_KEYWORD", "default": "1"},
            {"name": "dtype", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "device", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Returns evenly spaced values within a given interval.",
    },
    {
        "name": "linspace",
        "api_path": "array_api.linspace",
        "params": [
            {"name": "start", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "stop", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "num", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "dtype", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "device", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "endpoint", "kind": "KEYWORD_ONLY", "default": "True"},
        ],
        "docstring": "Returns evenly spaced numbers over a specified interval.",
    },
    {
        "name": "eye",
        "api_path": "array_api.eye",
        "params": [
            {"name": "n_rows", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "n_cols", "kind": "POSITIONAL_OR_KEYWORD", "default": "None"},
            {"name": "k", "kind": "KEYWORD_ONLY", "default": "0"},
            {"name": "dtype", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "device", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Returns a 2D array with ones on the diagonal and zeros elsewhere.",
    },
    {
        "name": "asarray",
        "api_path": "array_api.asarray",
        "params": [
            {"name": "obj", "kind": "POSITIONAL_ONLY"},
            {"name": "dtype", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "device", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "copy", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Converts the input to an array.",
    },
    # Linear Algebra / Matrices
    {
        "name": "matmul",
        "api_path": "array_api.matmul",
        "params": [
            {"name": "x1", "kind": "POSITIONAL_ONLY"},
            {"name": "x2", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Computes the matrix product of two arrays.",
    },
    {
        "name": "matrix_transpose",
        "api_path": "array_api.matrix_transpose",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Transposes a matrix (or a stack of matrices).",
    },
    {
        "name": "tensordot",
        "api_path": "array_api.tensordot",
        "params": [
            {"name": "x1", "kind": "POSITIONAL_ONLY"},
            {"name": "x2", "kind": "POSITIONAL_ONLY"},
            {"name": "axes", "kind": "KEYWORD_ONLY", "default": "2"},
        ],
        "docstring": "Computes tensor contraction along specified axes.",
    },
    {
        "name": "vecdot",
        "api_path": "array_api.vecdot",
        "params": [
            {"name": "x1", "kind": "POSITIONAL_ONLY"},
            {"name": "x2", "kind": "POSITIONAL_ONLY"},
            {"name": "axis", "kind": "KEYWORD_ONLY", "default": "-1"},
        ],
        "docstring": "Computes vector dot product along the specified axis.",
    },
    # Reductions & Stats
    {
        "name": "sum",
        "api_path": "array_api.sum",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
            {"name": "axis", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "dtype", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "keepdims", "kind": "KEYWORD_ONLY", "default": "False"},
        ],
        "docstring": "Calculates the sum of elements along specified axes.",
    },
    {
        "name": "prod",
        "api_path": "array_api.prod",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
            {"name": "axis", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "dtype", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "keepdims", "kind": "KEYWORD_ONLY", "default": "False"},
        ],
        "docstring": "Calculates the product of elements along specified axes.",
    },
    {
        "name": "mean",
        "api_path": "array_api.mean",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
            {"name": "axis", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "keepdims", "kind": "KEYWORD_ONLY", "default": "False"},
        ],
        "docstring": "Calculates the arithmetic mean along specified axes.",
    },
    {
        "name": "std",
        "api_path": "array_api.std",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
            {"name": "axis", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "correction", "kind": "KEYWORD_ONLY", "default": "0.0"},
            {"name": "keepdims", "kind": "KEYWORD_ONLY", "default": "False"},
        ],
        "docstring": "Calculates the standard deviation along specified axes.",
    },
    {
        "name": "var",
        "api_path": "array_api.var",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
            {"name": "axis", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "correction", "kind": "KEYWORD_ONLY", "default": "0.0"},
            {"name": "keepdims", "kind": "KEYWORD_ONLY", "default": "False"},
        ],
        "docstring": "Calculates the variance along specified axes.",
    },
    {
        "name": "max",
        "api_path": "array_api.max",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
            {"name": "axis", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "keepdims", "kind": "KEYWORD_ONLY", "default": "False"},
        ],
        "docstring": "Calculates the maximum value along specified axes.",
    },
    {
        "name": "min",
        "api_path": "array_api.min",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
            {"name": "axis", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "keepdims", "kind": "KEYWORD_ONLY", "default": "False"},
        ],
        "docstring": "Calculates the minimum value along specified axes.",
    },
    {
        "name": "cumulative_sum",
        "api_path": "array_api.cumulative_sum",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
            {"name": "axis", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "dtype", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "include_initial", "kind": "KEYWORD_ONLY", "default": "False"},
        ],
        "docstring": "Calculates the cumulative sum of elements along a specified axis.",
    },
    # Search, Set & Sort
    {
        "name": "argmax",
        "api_path": "array_api.argmax",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
            {"name": "axis", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "keepdims", "kind": "KEYWORD_ONLY", "default": "False"},
        ],
        "docstring": "Returns indices of maximum values along a specified axis.",
    },
    {
        "name": "argmin",
        "api_path": "array_api.argmin",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
            {"name": "axis", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "keepdims", "kind": "KEYWORD_ONLY", "default": "False"},
        ],
        "docstring": "Returns indices of minimum values along a specified axis.",
    },
    {
        "name": "nonzero",
        "api_path": "array_api.nonzero",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Returns indices of elements that are non-zero.",
    },
    {
        "name": "where",
        "api_path": "array_api.where",
        "params": [
            {"name": "condition", "kind": "POSITIONAL_ONLY"},
            {"name": "x1", "kind": "POSITIONAL_ONLY"},
            {"name": "x2", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Selects elements from x1 or x2 depending on condition.",
    },
    {
        "name": "sort",
        "api_path": "array_api.sort",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
            {"name": "axis", "kind": "KEYWORD_ONLY", "default": "-1"},
            {"name": "descending", "kind": "KEYWORD_ONLY", "default": "False"},
            {"name": "stable", "kind": "KEYWORD_ONLY", "default": "True"},
        ],
        "docstring": "Returns a sorted copy of an array.",
    },
    {
        "name": "argsort",
        "api_path": "array_api.argsort",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
            {"name": "axis", "kind": "KEYWORD_ONLY", "default": "-1"},
            {"name": "descending", "kind": "KEYWORD_ONLY", "default": "False"},
            {"name": "stable", "kind": "KEYWORD_ONLY", "default": "True"},
        ],
        "docstring": "Returns the indices that would sort an array.",
    },
    # Manipulation
    {
        "name": "concat",
        "api_path": "array_api.concat",
        "params": [
            {"name": "arrays", "kind": "POSITIONAL_ONLY"},
            {"name": "axis", "kind": "KEYWORD_ONLY", "default": "0"},
        ],
        "docstring": "Joins a sequence of arrays along an existing axis.",
    },
    {
        "name": "stack",
        "api_path": "array_api.stack",
        "params": [
            {"name": "arrays", "kind": "POSITIONAL_ONLY"},
            {"name": "axis", "kind": "KEYWORD_ONLY", "default": "0"},
        ],
        "docstring": "Joins a sequence of arrays along a new axis.",
    },
    {
        "name": "reshape",
        "api_path": "array_api.reshape",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
            {"name": "shape", "kind": "POSITIONAL_ONLY"},
            {"name": "copy", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Gives a new shape to an array without changing its data.",
    },
    {
        "name": "squeeze",
        "api_path": "array_api.squeeze",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
            {"name": "axis", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Removes singleton dimensions (axes) from an array.",
    },
    {
        "name": "expand_dims",
        "api_path": "array_api.expand_dims",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
            {"name": "axis", "kind": "KEYWORD_ONLY", "default": "0"},
        ],
        "docstring": "Expands the shape of an array by inserting a new axis.",
    },
    {
        "name": "permute_dims",
        "api_path": "array_api.permute_dims",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
            {"name": "axes", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Permutes the dimensions of an array.",
    },
    {
        "name": "flip",
        "api_path": "array_api.flip",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
            {"name": "axis", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Reverses the order of elements in an array along the given axis.",
    },
    {
        "name": "roll",
        "api_path": "array_api.roll",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
            {"name": "shift", "kind": "POSITIONAL_ONLY"},
            {"name": "axis", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Rolls array elements along a specified axis.",
    },
    # Logic Utilities
    {
        "name": "all",
        "api_path": "array_api.all",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
            {"name": "axis", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "keepdims", "kind": "KEYWORD_ONLY", "default": "False"},
        ],
        "docstring": "Tests whether all input array elements evaluate to True.",
    },
    {
        "name": "any",
        "api_path": "array_api.any",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
            {"name": "axis", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "keepdims", "kind": "KEYWORD_ONLY", "default": "False"},
        ],
        "docstring": "Tests whether any input array elements evaluate to True.",
    },
    # Elementwise floating-point checks & math
    {
        "name": "isfinite",
        "api_path": "array_api.isfinite",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Tests each element for finiteness.",
    },
    {
        "name": "isinf",
        "api_path": "array_api.isinf",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Tests each element for positive or negative infinity.",
    },
    {
        "name": "isnan",
        "api_path": "array_api.isnan",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Tests each element for NaN.",
    },
    {
        "name": "logaddexp",
        "api_path": "array_api.logaddexp",
        "params": [
            {"name": "x1", "kind": "POSITIONAL_ONLY"},
            {"name": "x2", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Calculates the logarithm of the sum of exponentiations.",
    },
    # Set & unique operations
    {
        "name": "unique_all",
        "api_path": "array_api.unique_all",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Returns unique elements, indices, inverse indices, and counts.",
    },
    {
        "name": "unique_counts",
        "api_path": "array_api.unique_counts",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Returns unique elements and their counts.",
    },
    {
        "name": "unique_inverse",
        "api_path": "array_api.unique_inverse",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Returns unique elements and inverse indices.",
    },
    {
        "name": "unique_values",
        "api_path": "array_api.unique_values",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Returns unique elements of an input array.",
    },
]


def _get_array_api_module() -> Any:
    """Retrieve an installed Array API implementation if available.

    Returns:
        The array_api_compat module or numpy.array_api if present, else None.
    """
    for mod_name in ("array_api_compat.numpy", "numpy.array_api", "array_api_compat"):
        try:
            return importlib.import_module(mod_name)
        except (ImportError, Exception):
            pass
    return None


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
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
    if mod is None:
        refs: List[GhostRef] = []
        for op in CANONICAL_ARRAY_API_OPS:
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
                    docstring=op.get(
                        "docstring", f"Array API Standard {op['name']} operation."
                    ),
                    environment_tags=["array_api", "standard"],
                    domain_metadata={"standard": "2024.12"},
                )
            )
        return refs

    results: List[GhostRef] = []
    seen: set[str] = set()

    for op in CANONICAL_ARRAY_API_OPS:
        name = op["name"]
        if hasattr(mod, name):
            obj = getattr(mod, name)
            try:
                ref = GhostInspector.inspect(obj, f"array_api.{name}", is_public=True)
                ref.environment_tags = list(ref.environment_tags or ()) + [
                    "array_api",
                    "standard",
                ]
                results.append(ref)
                seen.add(name)
            except Exception:
                pass

    # Fill any missed ops from canonical definitions
    for op in CANONICAL_ARRAY_API_OPS:
        if op["name"] not in seen:
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
                    docstring=op.get(
                        "docstring", f"Array API Standard {op['name']} operation."
                    ),
                    environment_tags=["array_api", "standard"],
                    domain_metadata={"standard": "2024.12"},
                )
            )

    return results
