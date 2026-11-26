"""Total broadcasting shape rules, matmul validators, and dtype bitwidth rules."""

from __future__ import annotations

import math
import typing

from ml_switcheroo_ir.identifiers import NonBoolInt, PositiveInt
from ml_switcheroo_ir.typeclasses.outcome import Err, Ok, Result
from ml_switcheroo_ir.types import normalize_dtype as _ir_normalize_dtype

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
    shape_a: typing.Sequence[int | str],
    shape_b: typing.Sequence[int | str],
) -> Result[str, tuple[int, ...]]:
    """Validate NumPy/PyTorch broadcasting compatibility totally into `Result[str, tuple[int, ...]]`.

    Rules:
        - Trailing dimensions are aligned.
        - Two dimensions are compatible if they are equal, or one of them is 1.
        - Wildcard/symbolic dimensions (`'?'`, `-1`, `'None'`) broadcast dynamically as `-1`.

    Args:
        shape_a: Shape sequence of the first tensor operand.
        shape_b: Shape sequence of the second tensor operand.

    Returns:
        `Ok(broadcast_shape_tuple)` if compatible, else `Err(error_message)`.
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
            return Err(
                f"Shape mismatch: dimension at reverse index {i} cannot broadcast between {dim_a} and {dim_b} (shapes: {list(shape_a)} vs {list(shape_b)})."
            )

    return Ok(tuple(reversed(result_shape)))


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
) -> Result[str, tuple[int, ...]]:
    """Validate matrix multiplication when at least one operand has rank below 2.

    Args:
        shape_a: Shape sequence of the LHS tensor operand.
        shape_b: Shape sequence of the RHS tensor operand.

    Returns:
        `Ok(output_shape_tuple)` if valid, else `Err(error_message)`.
    """
    if len(shape_a) == 1 and len(shape_b) == 1:
        return (
            Err(f"1D vector dot product mismatch: {shape_a[0]} != {shape_b[0]}.")
            if _dims_conflict(shape_a[0], shape_b[0])
            else Ok(())
        )

    if len(shape_a) == 1 and len(shape_b) >= 2:
        k_lhs = shape_a[0]
        k_rhs = shape_b[-2]
        if _dims_conflict(k_lhs, k_rhs):
            return Err(
                f"Vector-matrix multiplication contracting dimension mismatch: inner dimension {k_lhs} != {k_rhs} (shapes: {list(shape_a)} vs {list(shape_b)})."
            )
        batch_res = tuple(_parse_dim_or_dynamic(d) for d in shape_b[:-2])
        return Ok((*batch_res, _parse_dim_or_dynamic(shape_b[-1])))

    if len(shape_a) >= 2 and len(shape_b) == 1:
        k_lhs = shape_a[-1]
        k_rhs = shape_b[0]
        if _dims_conflict(k_lhs, k_rhs):
            return Err(
                f"Matrix-vector multiplication contracting dimension mismatch: inner dimension {k_lhs} != {k_rhs} (shapes: {list(shape_a)} vs {list(shape_b)})."
            )
        batch_res = tuple(_parse_dim_or_dynamic(d) for d in shape_a[:-2])
        return Ok((*batch_res, _parse_dim_or_dynamic(shape_a[-2])))

    return Err(f"Matrix multiplication requires at least 2D operands (got ranks {len(shape_a)} and {len(shape_b)}).")


def validate_matmul_shapes(
    shape_a: typing.Sequence[int | str],
    shape_b: typing.Sequence[int | str],
    strict_2d: bool = False,
) -> Result[str, tuple[int, ...]]:
    """Validate matrix multiplication shape and contracting inner-dimension constraints totally.

    Args:
        shape_a: Shape sequence of LHS matrix tensor.
        shape_b: Shape sequence of RHS matrix tensor.
        strict_2d: Flag enforcing strictly rank-2 matrix operands (`torch.mm`).

    Returns:
        `Ok(output_shape_tuple)` if compatible, else `Err(error_message)`.
    """
    if strict_2d and (len(shape_a) != 2 or len(shape_b) != 2):
        return Err(
            f"Strict 2D matmul error: both operands must have rank 2, got ranks {len(shape_a)} and {len(shape_b)}."
        )

    if len(shape_a) < 2 or len(shape_b) < 2:
        return _validate_low_rank_matmul(shape_a, shape_b)

    k_lhs = shape_a[-1]
    k_rhs = shape_b[-2]
    if _dims_conflict(k_lhs, k_rhs):
        return Err(
            f"Matrix multiplication contracting dimension mismatch: inner dimension {k_lhs} != {k_rhs} (shapes: {list(shape_a)} vs {list(shape_b)})."
        )

    core_shape = (
        _parse_dim_or_dynamic(shape_a[-2]),
        _parse_dim_or_dynamic(shape_b[-1]),
    )
    batch_a = shape_a[:-2]
    batch_b = shape_b[:-2]
    if batch_a or batch_b:
        b_res = validate_broadcast_shapes(batch_a, batch_b)
        if isinstance(b_res, Err):
            return Err(f"Batch dimension broadcasting error in matmul: {b_res.error}")
        return Ok((*b_res.value, *core_shape))

    return Ok(core_shape)


def get_dtype_bitwidth(dtype_name: str) -> Result[str, PositiveInt]:
    """Resolve bitwidth totally for canonical, sub-byte, and quantized data types.

    Args:
        dtype_name: Canonical or normalized dtype name.

    Returns:
        `Ok(PositiveInt(bits))` when `dtype_name` is non-empty, else `Err(reason)`.
    """
    from ml_ecosystem_snapshots.grounding.dtypes import normalize_dtype_name

    if not dtype_name.strip():
        return Err("Data type name must be non-empty.")
    clean = normalize_dtype_name(dtype_name)
    bits = DTYPE_BITWIDTHS.get(clean, 32)
    return Ok(PositiveInt(NonBoolInt(bits)))


def estimate_tensor_memory_bytes(
    shape: typing.Sequence[int],
    dtype_name: str,
) -> Result[str, float]:
    """Estimate memory footprint in bytes totally supporting sub-byte and quantized bitwidths.

    Args:
        shape: Tensor dimensions sequence (all dimensions must be non-negative).
        dtype_name: Canonical or normalized data type name.

    Returns:
        `Ok(estimated_bytes)` on valid shape and dtype, else `Err(reason)`.
    """
    if any(dim < 0 for dim in shape):
        return Err(f"Tensor shape dimensions must be non-negative, got {list(shape)}.")
    bits_res = get_dtype_bitwidth(dtype_name)
    if isinstance(bits_res, Err):
        return bits_res
    num_elements = math.prod(shape) if shape else 1
    total_bits = num_elements * int(bits_res.value)
    return Ok(total_bits / 8.0)


def canonicalize_ir_dtype_name(dtype_name: str) -> str:
    """Normalize `dtype_name` via Tier-0 `ml_switcheroo_ir.types.normalize_dtype` with alias fallback.

    Args:
        dtype_name: Raw framework or dialect data type string.

    Returns:
        Canonical lowercase dtype string.
    """
    from ml_ecosystem_snapshots.grounding.dtypes import normalize_dtype_name

    fallback = normalize_dtype_name(dtype_name)
    try:
        ir_dt = _ir_normalize_dtype(fallback)
        return str(ir_dt.value).lower()
    except ValueError:
        return fallback


def validate_dtype_for_op(op_name: str, dtype_name: str) -> Result[str, str]:
    """Validate totally that a data type is legally applicable to a mathematical operator.

    Args:
        op_name: Operation or function name.
        dtype_name: Data type name to validate.

    Returns:
        `Ok(normalized_dtype)` if compatible, else `Err(error_message)`.
    """
    clean_op = op_name.split(".")[-1].lower()
    clean_dt = canonicalize_ir_dtype_name(dtype_name)

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
            return Err(
                f"Data type '{dtype_name}' is not supported for transcendental operation '{op_name}'. Floating-point or complex dtype required."
            )

    return Ok(clean_dt)


__all__ = [
    "DTYPE_BITWIDTHS",
    "canonicalize_ir_dtype_name",
    "estimate_tensor_memory_bytes",
    "get_dtype_bitwidth",
    "validate_broadcast_shapes",
    "validate_dtype_for_op",
    "validate_matmul_shapes",
]
