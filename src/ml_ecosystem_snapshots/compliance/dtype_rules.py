"""Broadcasting shape rules, matmul shape validators, and dtype bitwidth/transcendental rules."""

from __future__ import annotations

import math
import typing

DTYPE_BITWIDTHS: dict[str, int] = {
    "float64": 64,
    "int64": 64,
    "uint64": 64,
    "complex128": 128,
    "float32": 32,
    "int32": 32,
    "uint32": 32,
    "complex64": 64,
    "float16": 16,
    "bfloat16": 16,
    "int16": 16,
    "uint16": 16,
    "float8_e4m3fn": 8,
    "float8_e4m3b11fnuz": 8,
    "float8_e5m2": 8,
    "fp8_e4m3fn": 8,
    "fp8_e4m3fnuz": 8,
    "fp8_e5m2": 8,
    "fp8_e5m2fnuz": 8,
    "int8": 8,
    "uint8": 8,
    "qint8": 8,
    "quint8": 8,
    "bool": 8,
    "int4": 4,
    "uint4": 4,
    "qint4": 4,
    "int2": 2,
    "string": 64,
    "object": 64,
}


def validate_broadcast_shapes(
    shape_a: typing.Sequence[int | str], shape_b: typing.Sequence[int | str]
) -> tuple[bool, list[int] | None, str | None]:
    """Validate NumPy/PyTorch broadcasting compatibility between two tensor shapes.

    Rules:
        - Trailing dimensions are aligned.
        - Two dimensions are compatible if they are equal, or one of them is 1.
        - Wildcard/symbolic dimensions ('?', -1) are treated as dynamically compatible.

    Args:
        shape_a: Shape sequence of the first tensor operand.
        shape_b: Shape sequence of the second tensor operand.

    Returns:
        A tuple of (is_compatible, broadcast_resulting_shape, error_message).
    """
    reversed_a = list(reversed(shape_a))
    reversed_b = list(reversed(shape_b))
    max_len = max(len(reversed_a), len(reversed_b))

    result_shape: list[int] = []
    for i in range(max_len):
        dim_a = reversed_a[i] if i < len(reversed_a) else 1
        dim_b = reversed_b[i] if i < len(reversed_b) else 1

        if str(dim_a) in ("?", "-1", "None") or str(dim_b) in ("?", "-1", "None"):
            result_shape.append(-1)
            continue

        try:
            int_a = int(dim_a)
            int_b = int(dim_b)
        except (ValueError, TypeError):
            result_shape.append(-1)
            continue

        if int_a == int_b:
            result_shape.append(int_a)
        elif int_a == 1:
            result_shape.append(int_b)
        elif int_b == 1:
            result_shape.append(int_a)
        else:
            return (
                False,
                None,
                f"Shape mismatch: dimension at reverse index {i} cannot broadcast between {dim_a} and {dim_b} (shapes: {list(shape_a)} vs {list(shape_b)}).",
            )

    return True, list(reversed(result_shape)), None


def _dims_conflict(dim_lhs: int | str, dim_rhs: int | str) -> bool:
    """Check whether two contracting dimensions are concrete and unequal.

    Args:
        dim_lhs: Left-hand contracting dimension value or symbol.
        dim_rhs: Right-hand contracting dimension value or symbol.

    Returns:
        True if both dimensions are concrete integers that do not match.
    """
    return str(dim_lhs) not in ("?", "-1") and str(dim_rhs) not in ("?", "-1") and int(dim_lhs) != int(dim_rhs)


def _parse_dim_or_dynamic(dim: int | str) -> int:
    """Convert a dimension value to an integer or `-1` if symbolic.

    Args:
        dim: Dimension value or symbolic marker.

    Returns:
        Parsed non-negative integer dimension or `-1` for symbolic dimensions.
    """
    return int(dim) if str(dim).isdigit() else -1


def _validate_low_rank_matmul(
    shape_a: typing.Sequence[int | str],
    shape_b: typing.Sequence[int | str],
) -> tuple[bool, list[int] | None, str | None]:
    """Validate matrix multiplication when at least one operand has rank below 2.

    Args:
        shape_a: Shape sequence of the LHS tensor operand.
        shape_b: Shape sequence of the RHS tensor operand.

    Returns:
        A tuple of `(is_compatible, output_shape, error_message)`.
    """
    if len(shape_a) == 1 and len(shape_b) == 1:
        return (
            (
                False,
                None,
                f"1D vector dot product mismatch: {shape_a[0]} != {shape_b[0]}.",
            )
            if _dims_conflict(shape_a[0], shape_b[0])
            else (True, [], None)
        )

    if len(shape_a) == 1 and len(shape_b) >= 2:
        k_lhs = shape_a[0]
        k_rhs = shape_b[-2]
        if _dims_conflict(k_lhs, k_rhs):
            return (
                False,
                None,
                f"Vector-matrix multiplication contracting dimension mismatch: inner dimension {k_lhs} != {k_rhs} (shapes: {list(shape_a)} vs {list(shape_b)}).",
            )
        batch_res = [_parse_dim_or_dynamic(d) for d in shape_b[:-2]]
        return True, batch_res + [_parse_dim_or_dynamic(shape_b[-1])], None

    if len(shape_a) >= 2 and len(shape_b) == 1:
        k_lhs = shape_a[-1]
        k_rhs = shape_b[0]
        if _dims_conflict(k_lhs, k_rhs):
            return (
                False,
                None,
                f"Matrix-vector multiplication contracting dimension mismatch: inner dimension {k_lhs} != {k_rhs} (shapes: {list(shape_a)} vs {list(shape_b)}).",
            )
        batch_res = [_parse_dim_or_dynamic(d) for d in shape_a[:-2]]
        return True, batch_res + [_parse_dim_or_dynamic(shape_a[-2])], None

    return (
        False,
        None,
        f"Matrix multiplication requires at least 2D operands (got ranks {len(shape_a)} and {len(shape_b)}).",
    )


def validate_matmul_shapes(
    shape_a: typing.Sequence[int | str],
    shape_b: typing.Sequence[int | str],
    strict_2d: bool = False,
) -> tuple[bool, list[int] | None, str | None]:
    """Validate matrix multiplication shape and contracting inner-dimension constraints.

    Rules:
        - If strict_2d is True (e.g. torch.mm), both shapes must be strictly 2D.
        - For 2D matrices (M, K) x (K, N), inner dimension K must match.
        - For batch matrices (...B, M, K) x (...B, K, N), batch dimensions must broadcast.
        - For 1D vectors (K,) x (K,), dot product requires matching dimension.

    Args:
        shape_a: Shape sequence of LHS matrix tensor.
        shape_b: Shape sequence of RHS matrix tensor.
        strict_2d: Flag enforcing strictly rank-2 matrix operands (torch.mm).

    Returns:
        A tuple of (is_compatible, output_shape, error_message).
    """
    if strict_2d and (len(shape_a) != 2 or len(shape_b) != 2):
        return (
            False,
            None,
            f"Strict 2D matmul error: both operands must have rank 2, got ranks {len(shape_a)} and {len(shape_b)}.",
        )

    if len(shape_a) < 2 or len(shape_b) < 2:
        return _validate_low_rank_matmul(shape_a, shape_b)

    k_lhs = shape_a[-1]
    k_rhs = shape_b[-2]
    if _dims_conflict(k_lhs, k_rhs):
        return (
            False,
            None,
            f"Matrix multiplication contracting dimension mismatch: inner dimension {k_lhs} != {k_rhs} (shapes: {list(shape_a)} vs {list(shape_b)}).",
        )

    core_shape = [
        _parse_dim_or_dynamic(shape_a[-2]),
        _parse_dim_or_dynamic(shape_b[-1]),
    ]
    batch_a = shape_a[:-2]
    batch_b = shape_b[:-2]
    if batch_a or batch_b:
        compat, b_shape, err = validate_broadcast_shapes(batch_a, batch_b)
        if not compat:
            return False, None, f"Batch dimension broadcasting error in matmul: {err}"
        return True, (b_shape or []) + core_shape, None

    return True, core_shape, None


def get_dtype_bitwidth(dtype_name: str) -> int:
    """Return bitwidth for canonical, sub-byte, and quantized data types.

    Args:
        dtype_name: Canonical or normalized dtype name.

    Returns:
        Bitwidth in bits (e.g. 8 for qint8, 4 for qint4/int4, 2 for int2, 32 for float32).
    """
    from ml_ecosystem_snapshots.grounding.dtypes import normalize_dtype_name

    clean = normalize_dtype_name(dtype_name)
    return DTYPE_BITWIDTHS.get(clean, 32)


def estimate_tensor_memory_bytes(shape: typing.Sequence[int], dtype_name: str) -> float:
    """Estimate memory footprint in bytes supporting sub-byte and quantized bitwidths.

    Args:
        shape: Tensor dimensions sequence.
        dtype_name: Canonical or normalized data type name.

    Returns:
        Total estimated memory footprint in bytes.
    """
    bits = get_dtype_bitwidth(dtype_name)
    num_elements = math.prod(shape) if shape else 1
    total_bits = num_elements * bits
    return total_bits / 8.0


def validate_dtype_for_op(op_name: str, dtype_name: str) -> tuple[bool, str | None]:
    """Validate that a data type is legally applicable to a mathematical operator.

    Rejects quantized, integer, boolean, and non-numeric types for transcendental
    operations (e.g., 'sin', 'exp', 'log', 'cos', 'sqrt', 'cholesky', 'linalg_inv').

    Args:
        op_name: Operation or function name.
        dtype_name: Data type name to validate.

    Returns:
        A tuple of (is_valid, error_message_or_None).
    """
    from ml_ecosystem_snapshots.grounding.dtypes import normalize_dtype_name

    clean_op = op_name.split(".")[-1].lower()
    clean_dt = normalize_dtype_name(dtype_name)

    transcendental_ops = {
        "sin",
        "cos",
        "tan",
        "exp",
        "log",
        "sqrt",
        "rsqrt",
        "sigmoid",
        "tanh",
        "cholesky",
        "linalg_inv",
        "inv",
    }

    if clean_op in transcendental_ops:
        non_float_prefixes = (
            "int",
            "uint",
            "qint",
            "quint",
            "bool",
            "string",
            "object",
        )
        if any(clean_dt.startswith(p) for p in non_float_prefixes):
            return False, (
                f"Data type '{dtype_name}' is not supported for transcendental operation '{op_name}'. Floating-point or complex dtype required."
            )

    return True, None


__all__ = [
    "DTYPE_BITWIDTHS",
    "estimate_tensor_memory_bytes",
    "get_dtype_bitwidth",
    "validate_broadcast_shapes",
    "validate_dtype_for_op",
    "validate_matmul_shapes",
]
