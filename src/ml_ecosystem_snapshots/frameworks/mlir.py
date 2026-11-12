"""MLIR Dialect API Snapshot Extractor.

Extracts MLIR dialects and operations (e.g., arith, math, tensor, linalg, scf, func,
memref, gpu, vector, llvm, nvvm, rocdl) by parsing an exhaustive JSON dump preserving SSA types,
attributes, results, and regions.
"""

from __future__ import annotations

import json
import os
import re
import typing
from enum import Enum

from ml_switcheroo_ir.schema.ghost import (
    GhostParam,
    GhostRef,
    ParameterKind,
    SemanticTier,
)

from ml_ecosystem_snapshots.frameworks._base import CollectorDomain, DomainCollectorAdapter

from ..models import (
    ExtendedGhostParam,
    GhostMlirRef,
    GhostResult,
    IRParameterRole,
    JsonValue,
    OperandDirection,
)


class MLIRDialectTrait(str, Enum):
    """Core verification traits for MLIR operations."""

    SAME_OPERANDS_AND_RESULT_TYPE = "SameOperandsAndResultType"
    SAME_TYPE_OPERANDS = "SameTypeOperands"
    COMMUTATIVE = "Commutative"
    ELEMENTWISE = "Elementwise"
    SCALARIZABLE = "Scalarizable"
    ISOLATED_FROM_ABOVE = "IsolatedFromAbove"
    ATTR_SIZED_OPERAND_SEGMENTS = "AttrSizedOperandSegments"
    ATTR_SIZED_RESULT_SEGMENTS = "AttrSizedResultSegments"


class MLIRTypeCategory(str, Enum):
    """High-level categories of MLIR types."""

    TENSOR = "tensor"
    VECTOR = "vector"
    MEMREF = "memref"
    INTEGER = "integer"
    FLOAT = "float"
    INDEX = "index"
    COMPLEX = "complex"
    CUSTOM = "custom"


_MLIR_SUBSTRING_TYPES: tuple[tuple[str, MLIRTypeCategory], ...] = (
    ("tensor", MLIRTypeCategory.TENSOR),
    ("vector", MLIRTypeCategory.VECTOR),
    ("memref", MLIRTypeCategory.MEMREF),
)


def classify_mlir_type(type_str: str) -> str:
    """Classify an MLIR type string into a recognized type category.

    Args:
        type_str: Raw type string (e.g. 'tensor<4x?xf32>', 'vector<[4]xi32>', 'i32', 'memref<10xf32>').

    Returns:
        The MLIRTypeCategory value string.
    """
    cleaned = type_str.strip().lower()
    for token, category in _MLIR_SUBSTRING_TYPES:
        if token in cleaned:
            return category.value
    if cleaned == "index":
        return MLIRTypeCategory.INDEX.value
    if re.match(r"^(s|u)?i\d+$", cleaned) or "integer" in cleaned:
        return MLIRTypeCategory.INTEGER.value
    if re.match(r"^(f16|f32|f64|bf16|f8e\d+m\d+.*)$", cleaned) or "float" in cleaned:
        return MLIRTypeCategory.FLOAT.value
    if cleaned.startswith("complex<"):
        return MLIRTypeCategory.COMPLEX.value
    return MLIRTypeCategory.CUSTOM.value


_CONTAINER_CATEGORIES: frozenset[str] = frozenset(
    {
        MLIRTypeCategory.TENSOR.value,
        MLIRTypeCategory.VECTOR.value,
        MLIRTypeCategory.MEMREF.value,
    }
)


def _validate_mlir_type_syntax(type_str: str, cleaned: str, cat: str) -> list[str]:
    """Validate angle-bracket container syntax or integer regex syntax for an MLIR type.

    Args:
        type_str: Original unstripped type string for error messages.
        cleaned: Whitespace-stripped type string.
        cat: Classified `MLIRTypeCategory` string value.

    Returns:
        List of syntax validation error messages.
    """
    if cat in _CONTAINER_CATEGORIES:
        if not (cleaned.startswith(f"{cat}<") and cleaned.endswith(">")):
            return [f"Malformed MLIR {cat} type syntax: '{type_str}'"]
    elif cat == MLIRTypeCategory.INTEGER.value and not re.match(r"^(s|u)?i\d+$", cleaned.lower()):
        return [f"Malformed MLIR integer type: '{type_str}'"]
    return []


def _validate_mlir_element_constraint(
    type_str: str,
    cleaned: str,
    cat: str,
    constraint: str,
    allowed_cats: tuple[str, ...],
    kind_label: str,
) -> list[str]:
    """Validate scalar or container element type against allowed scalar type categories.

    Args:
        type_str: Original type string for error reporting.
        cleaned: Whitespace-stripped type string.
        cat: Classified category of `cleaned`.
        constraint: Original TableGen constraint string.
        allowed_cats: Tuple of allowed `MLIRTypeCategory` values.
        kind_label: Label ('integer' or 'float') for diagnostic formatting.

    Returns:
        List of element type constraint mismatch messages.
    """
    if cat in _CONTAINER_CATEGORIES:
        inner = cleaned.rstrip(">").split("x")[-1].strip()
        if classify_mlir_type(inner) not in allowed_cats:
            return [
                f"Type '{type_str}' does not satisfy {kind_label} constraint '{constraint}': element type '{inner}' is not {kind_label}."
            ]
        return []
    if cat not in allowed_cats:
        return [
            f"Type '{type_str}' does not satisfy {kind_label} constraint '{constraint}': expected {kind_label} type."
        ]
    return []


def _validate_mlir_shaped_constraint(
    type_str: str,
    cleaned: str,
    cat: str,
    constraint: str,
    c_low: str,
) -> list[str]:
    """Validate shaped MLIR container constraints such as RankedTensorOf, UnrankedTensorOf, VectorOf, or MemRefOf.

    Args:
        type_str: Original type string for error reporting.
        cleaned: Whitespace-stripped type string.
        cat: Classified `MLIRTypeCategory` string value.
        constraint: Original TableGen ODS constraint string.
        c_low: Lowercase normalized constraint string.

    Returns:
        List of container constraint mismatch messages.
    """
    if "unrankedtensorof" in c_low:
        return (
            []
            if (cat == MLIRTypeCategory.TENSOR.value and "*" in cleaned)
            else [f"Type '{type_str}' does not satisfy UnrankedTensorOf constraint."]
        )
    if "rankedtensorof" in c_low:
        return (
            []
            if (cat == MLIRTypeCategory.TENSOR.value and "*" not in cleaned)
            else [f"Type '{type_str}' does not satisfy RankedTensorOf constraint '{constraint}'."]
        )
    if "vectorof" in c_low:
        return (
            []
            if cat == MLIRTypeCategory.VECTOR.value
            else [f"Type '{type_str}' does not satisfy VectorOf constraint '{constraint}'."]
        )
    return (
        []
        if cat == MLIRTypeCategory.MEMREF.value
        else [f"Type '{type_str}' does not satisfy MemRefOf constraint '{constraint}'."]
    )


def _validate_mlir_type_constraint(type_str: str, cleaned: str, cat: str, constraint: str) -> list[str]:
    """Validate an MLIR type against a TableGen ODS type constraint string.

    Args:
        type_str: Original type string for error messages.
        cleaned: Whitespace-stripped type string.
        cat: Classified `MLIRTypeCategory` value.
        constraint: TableGen ODS constraint string.

    Returns:
        List of constraint validation error messages.
    """
    c_low = constraint.lower()
    if any(tok in c_low for tok in ("unrankedtensorof", "rankedtensorof", "vectorof", "memrefof")):
        return _validate_mlir_shaped_constraint(type_str, cleaned, cat, constraint, c_low)
    if any(tok in c_low for tok in ("anyinteger", "anysignlessinteger", "integerlike")):
        return _validate_mlir_element_constraint(
            type_str,
            cleaned,
            cat,
            constraint,
            (MLIRTypeCategory.INTEGER.value, MLIRTypeCategory.INDEX.value),
            "integer",
        )
    if "anyfloat" in c_low or "floatlike" in c_low:
        return _validate_mlir_element_constraint(
            type_str,
            cleaned,
            cat,
            constraint,
            (MLIRTypeCategory.FLOAT.value,),
            "float",
        )
    errors: list[str] = []
    if "anytensor" in c_low or "staticshapetensor" in c_low:
        if cat != MLIRTypeCategory.TENSOR.value:
            errors.append(f"Type '{type_str}' does not satisfy tensor constraint '{constraint}': expected tensor type.")
        elif "staticshapetensor" in c_low and ("?" in cleaned or "*" in cleaned):
            errors.append(
                f"Type '{type_str}' does not satisfy static shape constraint '{constraint}': dynamic or unranked dimensions not permitted."
            )
    elif c_low == "index" and cat != MLIRTypeCategory.INDEX.value:
        errors.append(f"Type '{type_str}' does not satisfy index constraint '{constraint}': expected 'index' type.")
    return errors


def validate_mlir_type(type_str: str, constraint: str | None = None) -> list[str]:
    """Validate MLIR SSA type syntax and check compliance against TableGen type constraints.

    Validates:
        - Tensor types: RankedTensorOf<...>, UnrankedTensorOf (tensor<*xf32>), dynamic dims (tensor<?x?xf32>)
        - Vector types: VectorOf<...>, scalable vectors (vector<[4]xf32>)
        - Memory references: MemRefOf<...>, strided layouts, address spaces
        - Integer types: Signless (i1, i8, i16, i32, i64), signed (si32), unsigned (ui32), boolean (i1), index
        - Floating-point types: f16, f32, f64, bf16, etc.

    Args:
        type_str: Actual SSA type string to validate.
        constraint: Optional TableGen ODS constraint string (e.g. 'RankedTensorOf<[F16, F32]>', 'AnyInteger').

    Returns:
        List of type validation errors.
    """
    cleaned = type_str.strip()
    cat = classify_mlir_type(cleaned)
    errors = _validate_mlir_type_syntax(type_str, cleaned, cat)
    if constraint:
        errors.extend(_validate_mlir_type_constraint(type_str, cleaned, cat, constraint))
    return errors


def _validate_segment_traits(
    op_name: str,
    t_set: set[str],
    operand_types: list[str] | None,
    attributes: list[str] | None,
    structured_attributes: typing.Mapping[str, JsonValue] | None,
) -> list[str]:
    """Validate `AttrSizedOperandSegments` and `AttrSizedResultSegments` traits.

    Args:
        op_name: Qualified MLIR operation name.
        t_set: Set of trait strings attached to the operation.
        operand_types: Optional list of SSA operand type strings.
        attributes: Optional list of attribute names present on the operation.
        structured_attributes: Optional mapping of structured attribute values.

    Returns:
        List of segment trait validation error messages.
    """
    errors: list[str] = []
    attrs_provided = attributes is not None or structured_attributes is not None

    if MLIRDialectTrait.ATTR_SIZED_OPERAND_SEGMENTS.value in t_set:
        has_op_seg = bool(
            (attributes and "operand_segment_sizes" in attributes)
            or (structured_attributes and "operand_segment_sizes" in structured_attributes)
        )
        if structured_attributes and "operand_segment_sizes" in structured_attributes:
            seg_val = structured_attributes["operand_segment_sizes"]
            if isinstance(seg_val, list) and operand_types is not None:
                seg_ints = [x for x in seg_val if isinstance(x, int)]
                if sum(seg_ints) != len(operand_types):
                    errors.append(
                        f"AttrSizedOperandSegments mismatch for '{op_name}': sum of operand_segment_sizes ({sum(seg_ints)}) does not match operand count ({len(operand_types)})."
                    )
        if not has_op_seg and attrs_provided:
            errors.append(
                f"Missing required attribute 'operand_segment_sizes' for op '{op_name}' with AttrSizedOperandSegments trait."
            )

    if MLIRDialectTrait.ATTR_SIZED_RESULT_SEGMENTS.value in t_set:
        has_res_seg = bool(
            (attributes and "result_segment_sizes" in attributes)
            or (structured_attributes and "result_segment_sizes" in structured_attributes)
        )
        if not has_res_seg and attrs_provided:
            errors.append(
                f"Missing required attribute 'result_segment_sizes' for op '{op_name}' with AttrSizedResultSegments trait."
            )

    return errors


def validate_mlir_traits(
    op_name: str,
    traits: list[str],
    operand_types: list[str] | None = None,
    result_types: list[str] | None = None,
    attributes: list[str] | None = None,
    structured_attributes: typing.Mapping[str, JsonValue] | None = None,
) -> list[str]:
    """Validate dialect verification traits for an MLIR operation invocation.

    Checks:
        - SameOperandsAndResultType: all operand and result types must be identical
        - SameTypeOperands: all operand types must match
        - AttrSizedOperandSegments: requires operand_segment_sizes attribute
        - AttrSizedResultSegments: requires result_segment_sizes attribute
        - Elementwise / Scalarizable shape compatibility

    Args:
        op_name: Qualified operation name (e.g. 'arith.addf').
        traits: List of trait names associated with the operation.
        operand_types: Optional list of SSA operand type strings.
        result_types: Optional list of SSA result type strings.
        attributes: Optional list of attribute names present on the op.
        structured_attributes: Optional dictionary of structured attribute values.

    Returns:
        List of trait verification error messages.
    """
    errors: list[str] = []
    t_set = set(traits)

    if MLIRDialectTrait.SAME_OPERANDS_AND_RESULT_TYPE.value in t_set:
        all_types = list(operand_types or []) + list(result_types or [])
        if len(set(all_types)) > 1:
            errors.append(
                f"Trait violation for '{op_name}': SameOperandsAndResultType requires all operand and result types to be identical (got operands={operand_types}, results={result_types})."
            )

    if MLIRDialectTrait.SAME_TYPE_OPERANDS.value in t_set and operand_types and len(set(operand_types)) > 1:
        errors.append(
            f"Trait violation for '{op_name}': SameTypeOperands requires all operand types to match (got {operand_types})."
        )

    errors.extend(_validate_segment_traits(op_name, t_set, operand_types, attributes, structured_attributes))

    if MLIRDialectTrait.COMMUTATIVE.value in t_set and operand_types is not None and len(operand_types) < 2:
        errors.append(
            f"Trait violation for '{op_name}': Commutative trait requires at least 2 operands, got {len(operand_types)}."
        )

    if MLIRDialectTrait.ELEMENTWISE.value in t_set and operand_types and len(operand_types) >= 2:
        elem_types = [t.split("<")[-1].rstrip(">") for t in operand_types if "<" in t]
        if len(set(elem_types)) > 1:
            errors.append(
                f"Trait violation for '{op_name}': Elementwise trait requires operands to have matching element types (got {elem_types})."
            )

    if (
        MLIRDialectTrait.ISOLATED_FROM_ABOVE.value in t_set
        and structured_attributes
        and structured_attributes.get("has_external_captures")
    ):
        errors.append(
            f"Trait violation for '{op_name}': IsolatedFromAbove forbids implicit captures of SSA values defined above the region."
        )

    return errors


def validate_mlir_region(
    op_name: str,
    region_name: str,
    block_args: list[str],
    yield_types: list[str],
    isolated_from_above: bool = False,
) -> list[str]:
    """Validate MLIR region body block arguments, yield terminator types, and capture isolation.

    Args:
        op_name: Qualified operation name (e.g. 'scf.for', 'func.func', 'gpu.launch').
        region_name: Identifier name of the region (e.g. 'body', 'region').
        block_args: List of SSA block argument type strings.
        yield_types: List of yield/return terminator type strings.
        isolated_from_above: Whether the region has the IsolatedFromAbove trait.

    Returns:
        List of region verification error messages.
    """
    errors: list[str] = []
    clean_op = op_name.strip().lower()

    if clean_op in ("scf.for", "for"):
        if region_name == "body":
            if not block_args or block_args[0] != "index":
                errors.append(
                    f"scf.for body region first block argument (induction variable) must be 'index' type (got {block_args[0] if block_args else 'none'})."
                )
            if len(block_args) > 1:
                iter_args = block_args[1:]
                if yield_types and yield_types != iter_args:
                    errors.append(
                        f"scf.for body region yield types ({yield_types}) must match loop-carried iter_args ({iter_args})."
                    )

    elif clean_op in ("scf.while", "while"):
        if region_name == "before":
            if not yield_types or not any("i1" in yt for yt in yield_types):
                errors.append(f"scf.while before region must yield a condition boolean i1 (got {yield_types}).")

    return errors


def validate_mlir_successors(
    op_name: str,
    successors: list[str],
    expected_count: int | None = None,
) -> list[str]:
    """Validate successor block targets for MLIR control flow operations.

    Args:
        op_name: Qualified operation name (e.g. 'cf.br', 'cf.cond_br').
        successors: List of successor block identifier labels (e.g. ['^bb1', '^bb2']).
        expected_count: Optional expected number of branch successors.

    Returns:
        List of successor validation error messages.
    """
    errors: list[str] = []
    clean_op = op_name.strip().lower()

    for succ in successors:
        s_clean = succ.strip()
        if not (s_clean.startswith("^") or s_clean.startswith("bb") or s_clean.startswith("block")):
            errors.append(f"Malformed successor block label '{succ}' for '{op_name}': expected prefix '^' or 'bb'.")

    if expected_count is not None and len(successors) != expected_count:
        errors.append(f"Successor count mismatch for '{op_name}': expected {expected_count}, got {len(successors)}.")

    if clean_op == "cf.br" and len(successors) != 1:
        errors.append(f"Unconditional branch 'cf.br' requires exactly 1 successor block (got {len(successors)}).")
    elif clean_op == "cf.cond_br" and len(successors) != 2:
        errors.append(
            f"Conditional branch 'cf.cond_br' requires exactly 2 successor blocks (true_dest, false_dest, got {len(successors)})."
        )

    return errors


def _get_canonical_mlir_records() -> list[dict[str, JsonValue]]:
    """Return canonical baseline MLIR operations for offline operation when unbuilt.

    Returns:
        List of MLIR operation dictionaries.
    """
    return [
        {
            "class_name": "AddFOp",
            "api_path": "arith.addf",
            "dialect": "arith",
            "operands": [
                {"name": "lhs", "type": "AnyFloat"},
                {"name": "rhs", "type": "AnyFloat"},
            ],
            "results": [{"name": "result", "type": "AnyFloat"}],
            "attributes": [{"name": "fastmath", "type": "FastMathFlagsAttr"}],
            "traits": ["SameOperandsAndResultType", "Commutative"],
            "summary": "Floating-point addition operation",
        },
        {
            "class_name": "SubFOp",
            "api_path": "arith.subf",
            "dialect": "arith",
            "operands": [
                {"name": "lhs", "type": "AnyFloat"},
                {"name": "rhs", "type": "AnyFloat"},
            ],
            "results": [{"name": "result", "type": "AnyFloat"}],
            "attributes": [{"name": "fastmath", "type": "FastMathFlagsAttr"}],
            "traits": ["SameOperandsAndResultType"],
            "summary": "Floating-point subtraction operation",
        },
        {
            "class_name": "MulFOp",
            "api_path": "arith.mulf",
            "dialect": "arith",
            "operands": [
                {"name": "lhs", "type": "AnyFloat"},
                {"name": "rhs", "type": "AnyFloat"},
            ],
            "results": [{"name": "result", "type": "AnyFloat"}],
            "attributes": [{"name": "fastmath", "type": "FastMathFlagsAttr"}],
            "traits": ["SameOperandsAndResultType", "Commutative"],
            "summary": "Floating-point multiplication operation",
        },
        {
            "class_name": "DivFOp",
            "api_path": "arith.divf",
            "dialect": "arith",
            "operands": [
                {"name": "lhs", "type": "AnyFloat"},
                {"name": "rhs", "type": "AnyFloat"},
            ],
            "results": [{"name": "result", "type": "AnyFloat"}],
            "attributes": [{"name": "fastmath", "type": "FastMathFlagsAttr"}],
            "traits": ["SameOperandsAndResultType"],
            "summary": "Floating-point division operation",
        },
        {
            "class_name": "AddIOp",
            "api_path": "arith.addi",
            "dialect": "arith",
            "operands": [
                {"name": "lhs", "type": "AnyInteger"},
                {"name": "rhs", "type": "AnyInteger"},
            ],
            "results": [{"name": "result", "type": "AnyInteger"}],
            "attributes": [{"name": "overflowFlags", "type": "IntegerOverflowFlagsAttr"}],
            "traits": ["SameOperandsAndResultType", "Commutative"],
            "summary": "Integer addition operation",
        },
        {
            "class_name": "SubIOp",
            "api_path": "arith.subi",
            "dialect": "arith",
            "operands": [
                {"name": "lhs", "type": "AnyInteger"},
                {"name": "rhs", "type": "AnyInteger"},
            ],
            "results": [{"name": "result", "type": "AnyInteger"}],
            "attributes": [{"name": "overflowFlags", "type": "IntegerOverflowFlagsAttr"}],
            "traits": ["SameOperandsAndResultType"],
            "summary": "Integer subtraction operation",
        },
        {
            "class_name": "MulIOp",
            "api_path": "arith.muli",
            "dialect": "arith",
            "operands": [
                {"name": "lhs", "type": "AnyInteger"},
                {"name": "rhs", "type": "AnyInteger"},
            ],
            "results": [{"name": "result", "type": "AnyInteger"}],
            "attributes": [{"name": "overflowFlags", "type": "IntegerOverflowFlagsAttr"}],
            "traits": ["SameOperandsAndResultType", "Commutative"],
            "summary": "Integer multiplication operation",
        },
        {
            "class_name": "ExpOp",
            "api_path": "math.exp",
            "dialect": "math",
            "operands": [{"name": "operand", "type": "AnyFloat"}],
            "results": [{"name": "result", "type": "AnyFloat"}],
            "attributes": [],
            "traits": ["SameOperandsAndResultType"],
            "summary": "Exponential operation",
        },
        {
            "class_name": "FuncOp",
            "api_path": "func.func",
            "dialect": "func",
            "operands": [],
            "results": [],
            "attributes": [
                {"name": "sym_name", "type": "StringAttr"},
                {"name": "function_type", "type": "TypeAttr"},
            ],
            "traits": ["IsolatedFromAbove"],
            "summary": "Function definition",
        },
        {
            "class_name": "ReturnOp",
            "api_path": "func.return",
            "dialect": "func",
            "operands": [{"name": "operands", "type": "Variadic<AnyType>"}],
            "results": [],
            "attributes": [],
            "traits": ["Terminator"],
            "summary": "Function return operation",
        },
        {
            "class_name": "ForOp",
            "api_path": "scf.for",
            "dialect": "scf",
            "operands": [
                {"name": "lowerBound", "type": "Index"},
                {"name": "upperBound", "type": "Index"},
                {"name": "step", "type": "Index"},
            ],
            "results": [{"name": "results", "type": "Variadic<AnyType>"}],
            "attributes": [],
            "traits": ["RecursiveMemoryEffects"],
            "summary": "Structured control flow for loop",
        },
        {
            "class_name": "YieldOp",
            "api_path": "scf.yield",
            "dialect": "scf",
            "operands": [{"name": "results", "type": "Variadic<AnyType>"}],
            "results": [],
            "attributes": [],
            "traits": ["Terminator"],
            "summary": "SCF loop yield",
        },
        {
            "class_name": "EmptyOp",
            "api_path": "tensor.empty",
            "dialect": "tensor",
            "operands": [{"name": "dynamicSizes", "type": "Variadic<Index>"}],
            "results": [{"name": "result", "type": "AnyTensor"}],
            "attributes": [{"name": "staticSizes", "type": "DenseI64ArrayAttr"}],
            "traits": [],
            "summary": "Create empty uninitialized tensor",
        },
        {
            "class_name": "ExtractOp",
            "api_path": "tensor.extract",
            "dialect": "tensor",
            "operands": [
                {"name": "tensor", "type": "AnyTensor"},
                {"name": "indices", "type": "Variadic<Index>"},
            ],
            "results": [{"name": "result", "type": "AnyType"}],
            "attributes": [],
            "traits": ["Pure"],
            "summary": "Element extraction operation",
        },
        {
            "class_name": "MatmulOp",
            "api_path": "linalg.matmul",
            "dialect": "linalg",
            "operands": [
                {"name": "inputs", "type": "Variadic<AnyShaped>"},
                {"name": "outputs", "type": "Variadic<AnyShaped>"},
            ],
            "results": [{"name": "results", "type": "Variadic<AnyRankedTensor>"}],
            "attributes": [],
            "traits": ["StructuredOpTrait"],
            "summary": "Linalg matrix multiplication",
        },
    ]


def _read_mlir_op_records() -> list[dict[str, JsonValue]]:
    """Load raw MLIR operation dictionary records from disk or canonical fallback.

    Returns:
        List of MLIR operation dictionaries, or empty list on error / disabled fallback.
    """
    json_path = os.path.join(os.path.dirname(__file__), "mlir_exhaustive.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, encoding="utf-8") as f:
                data: JsonValue = json.load(f)
            if isinstance(data, dict):
                if "categories" in data and isinstance(data["categories"], dict):
                    all_ops: list[dict[str, JsonValue]] = []
                    for cat_ops in data["categories"].values():
                        if isinstance(cat_ops, list):
                            all_ops.extend(item for item in cat_ops if isinstance(item, dict))
                    return all_ops
                raw_cand = data.get("operations") or data.get("instructions") or data.get("items") or []
                return [item for item in (raw_cand if isinstance(raw_cand, list) else []) if isinstance(item, dict)]
            return [item for item in (data if isinstance(data, list) else []) if isinstance(item, dict)]
        except (json.JSONDecodeError, OSError):
            return []
    if os.environ.get("MLIR_DISABLE_FALLBACK") == "1":
        return []
    return _get_canonical_mlir_records()


def _build_mlir_op_params(op: dict[str, JsonValue]) -> list[ExtendedGhostParam]:
    """Construct SSA operand, attribute, and region parameters for an MLIR operation record.

    Args:
        op: Raw MLIR operation dictionary from the snapshot catalog.

    Returns:
        List of `ExtendedGhostParam` entries.
    """
    params: list[ExtendedGhostParam] = []

    raw_operands = op.get("operands", [])
    for operand in raw_operands if isinstance(raw_operands, list) else []:
        op_name = str(operand["name"]) if isinstance(operand, dict) else str(operand)
        op_type = str(operand.get("type", "Value")) if isinstance(operand, dict) else "Value"
        params.append(
            ExtendedGhostParam(
                name=op_name,
                kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                annotation=op_type,
                standardized_name="operand",
                description=f"SSA operand {op_name} of type {op_type}",
                direction=OperandDirection.READ,
                role=IRParameterRole.OPERAND,
            )
        )

    raw_attributes = op.get("attributes", [])
    for attribute in raw_attributes if isinstance(raw_attributes, list) else []:
        attr_name = str(attribute["name"]) if isinstance(attribute, dict) else str(attribute)
        attr_type = str(attribute.get("type", "Attribute")) if isinstance(attribute, dict) else "Attribute"
        params.append(
            ExtendedGhostParam(
                name=attr_name,
                kind=ParameterKind.KEYWORD_ONLY,
                annotation=attr_type,
                standardized_name="attribute",
                description=f"Buildable attribute {attr_name} of type {attr_type}",
                role=IRParameterRole.ATTRIBUTE,
            )
        )

    raw_regions = op.get("regions", [])
    for region in raw_regions if isinstance(raw_regions, list) else []:
        reg_name = str(region["name"]) if isinstance(region, dict) else str(region)
        params.append(
            ExtendedGhostParam(
                name=reg_name,
                kind=ParameterKind.KEYWORD_ONLY,
                annotation="Region",
                standardized_name="region",
                description=f"Op region {reg_name}",
                role=IRParameterRole.REGION,
            )
        )

    return params


def _load_mlir_exhaustive() -> list[GhostRef]:
    """Load the MLIR exhaustive JSON dump or canonical fallback and map it to GhostRefs.

    Returns:
        A list of GhostRef items representing MLIR operations.
    """
    ops_list = _read_mlir_op_records()
    refs: list[GhostRef] = []
    for op in ops_list:
        params = _build_mlir_op_params(op)

        returns_type: str | None = None
        ghost_results: list[GhostResult] = []
        results = op.get("results", [])
        if isinstance(results, list) and results:
            res_types: list[str] = []
            for r in results:
                r_name = str(r.get("name", "result")) if isinstance(r, dict) else "result"
                r_type = str(r.get("type", "Value")) if isinstance(r, dict) else str(r)
                res_types.append(r_type)
                ghost_results.append(GhostResult(name=r_name, type=r_type))
            returns_type = res_types[0] if len(res_types) == 1 else f"tuple[{', '.join(res_types)}]"

        docstring_parts = [str(op.get("description") or f"MLIR {op.get('class_name', 'Op')} operation.")]
        raw_traits = op.get("traits", [])
        traits: list[str] = [str(t) for t in (raw_traits if isinstance(raw_traits, list) else [])]
        if traits:
            docstring_parts.append(f"Traits: {', '.join(traits)}")

        domain_metadata: dict[str, JsonValue] = {
            "dialect": op.get("dialect"),
            "traits": list(traits),
            "regions": op.get("regions", []),
        }

        ssa_operands: list[GhostParam] = [p for p in params if p.role == IRParameterRole.OPERAND]
        params_list: list[GhostParam] = list(params)

        raw_succs = op.get("successors")
        successors: list[str] | None = [str(s) for s in raw_succs] if isinstance(raw_succs, list) else None
        raw_attrs_val = op.get("attributes")
        raw_regs_val = op.get("regions")

        refs.append(
            GhostMlirRef(
                name=str(op.get("class_name", "UnknownOp")),
                api_path=str(op.get("api_path", "")),
                kind="operation",
                params=params_list,
                operands=ssa_operands,
                docstring="\n".join(docstring_parts),
                returns_type=returns_type,
                returns=ghost_results,
                environment_tags=["cpu", "cuda", "rocm", "tpu"],
                domain_metadata=domain_metadata,
                traits=traits,
                attributes=raw_attrs_val if isinstance(raw_attrs_val, dict) else None,
                regions=raw_regs_val if isinstance(raw_regs_val, dict) else None,
                successors=successors,
            )
        )

    return refs


def collect_domain(include_nonpublic: bool = False) -> list[GhostRef]:
    """Collect MLIR dialect operation schemas without `SemanticTier` shoehorning.

    Args:
        include_nonpublic: Whether to include non-public operations.

    Returns:
        List of `GhostMlirRef` items.
    """
    del include_nonpublic
    return _load_mlir_exhaustive()


DOMAIN_COLLECTOR = DomainCollectorAdapter(CollectorDomain.COMPILER_IR, "operations", collect_domain)


def collect_api(category: SemanticTier, include_nonpublic: bool = False) -> list[GhostRef]:
    """Entrypoint to collect the MLIR API signature for a given category.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered for the requested category.
    """
    if category != SemanticTier.UTIL:
        return []

    return collect_domain(include_nonpublic)


get_canonical_mlir_records = _get_canonical_mlir_records
load_mlir_exhaustive = _load_mlir_exhaustive
