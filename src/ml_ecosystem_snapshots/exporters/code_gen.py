"""Code generators for Pydantic V2, Protocol Buffers, C++17 headers, and TypeScript interfaces."""

from __future__ import annotations

import ast
import typing

from ml_switcheroo_ir.schema.ghost import GhostParam, GhostRef


def _extract_ast_type_identifiers(annotation: str | None) -> list[str]:
    """Parse a type annotation via `ast.parse` and return lowercase type identifier tokens.

    Falls back to lowercase token splitting when `annotation` uses non-Python syntax
    (e.g. `tensor<4xf32>`).

    Args:
        annotation: Raw type annotation string.

    Returns:
        List of normalized lowercase identifier tokens in AST traversal order.
    """
    if not annotation:
        return []
    cleaned = str(annotation).strip()
    try:
        tree = ast.parse(cleaned, mode="eval")
        tokens = [
            node.id.lower() if isinstance(node, ast.Name) else node.attr.lower()
            for node in ast.walk(tree)
            if isinstance(node, (ast.Name, ast.Attribute))
        ]
        return tokens if tokens else [cleaned.lower()]
    except SyntaxError:
        return [cleaned.lower()]


def _render_pydantic_param_field(param: GhostParam) -> str:
    """Render a single Pydantic model field assignment line from a `GhostParam`.

    Args:
        param: Parameter definition to format as a Pydantic field.

    Returns:
        Formatted 4-space indented field declaration line.
    """
    raw_typ = param.annotation if param.annotation else "JsonValue"
    typ = "JsonValue" if raw_typ == "Any" else raw_typ
    desc = param.description.replace('"', "'") if param.description else ""
    default_val = param.default

    if param.kind == "VAR_POSITIONAL":
        typ = f"tuple[{typ}, ...]"
        field_def = f'pydantic.Field(default=(), description="{desc}")' if desc else "pydantic.Field(default=())"
        return f"    {param.name}: {typ} = {field_def}"
    if param.kind == "VAR_KEYWORD":
        typ = f"dict[str, {typ}]"
        field_def = (
            f'pydantic.Field(default_factory=dict, description="{desc}")'
            if desc
            else "pydantic.Field(default_factory=dict)"
        )
        return f"    {param.name}: {typ} = {field_def}"
    if default_val is None:
        field_def = f'pydantic.Field(..., description="{desc}")' if desc else "pydantic.Field(...)"
        return f"    {param.name}: {typ} = {field_def}"
    field_def = f'pydantic.Field(default={default_val}, description="{desc}")' if desc else f"{default_val}"
    return f"    {param.name}: {typ} = {field_def}"


def _render_single_pydantic_model(
    name: str,
    doc: str | None,
    params: typing.Sequence[GhostParam],
    validate_varargs: bool,
) -> list[str]:
    """Render Python source code lines for a single Pydantic `BaseModel` class.

    Args:
        name: Class name for the Pydantic model.
        doc: Optional docstring for the class.
        params: Sequence of `GhostParam` specifications.
        validate_varargs: Whether to include `*args` and `**kwargs` as validated fields.

    Returns:
        List of Python source code lines for the class.
    """
    m_lines = [
        f"class {name}(pydantic.BaseModel):",
        f'    """{doc}"""' if doc else '    """Generated Pydantic model."""',
        "",
        '    model_config = pydantic.ConfigDict(extra="forbid")',
    ]
    valid_params = [p for p in params if validate_varargs or p.kind not in ("VAR_POSITIONAL", "VAR_KEYWORD")]
    for param in valid_params:
        m_lines.append(_render_pydantic_param_field(param))
    return m_lines


def to_pydantic(ref: GhostRef, validate_varargs: bool = False) -> str:
    """Convert a GhostRef into a Pydantic V2 class definition string.

    Supports overloaded signatures by generating a PEP 604 union of Pydantic models,
    variable positional arguments as validated tuple fields, and parameter descriptions
    in pydantic.Field(description=...) for OpenAPI/schema generation.

    Args:
        ref: The GhostRef to convert.
        validate_varargs: Whether to represent *args as validated tuple fields instead of skipping.

    Returns:
        A string containing the Python source for the Pydantic model.
    """
    header_lines = [
        "import typing",
        "import pydantic",
        "",
        "",
    ]
    all_models: list[str] = []
    if ref.overloads:
        all_variants = [ref] + [ov for ov in ref.overloads if isinstance(ov, GhostRef)]
        variant_names: list[str] = []
        for i, variant in enumerate(all_variants):
            var_name = f"{ref.name}Variant{i}"
            variant_names.append(var_name)
            all_models.extend(
                _render_single_pydantic_model(
                    var_name,
                    variant.docstring or ref.docstring,
                    variant.params,
                    validate_varargs,
                )
            )
            all_models.append("")

        all_models.append(f"{ref.name} = {' | '.join(variant_names)}")
    else:
        all_models.extend(_render_single_pydantic_model(ref.name, ref.docstring, ref.params, validate_varargs))

    if any("JsonValue" in line for line in all_models):
        header_lines.insert(2, "from ml_ecosystem_snapshots.models import JsonValue")

    return "\n".join(header_lines + all_models) + "\n"


PROTO_TENSOR_DEFINITION = """message TensorProto {
  repeated int64 shape = 1;
  string dtype = 2;
  bytes raw_data = 3;
}"""

PROTO_ENUM_DEFINITIONS: dict[str, str] = {
    "reduction": """enum ReductionType {
  REDUCTION_UNSPECIFIED = 0;
  REDUCTION_NONE = 1;
  REDUCTION_MEAN = 2;
  REDUCTION_SUM = 3;
}""",
    "padding": """enum PaddingMode {
  PADDING_UNSPECIFIED = 0;
  PADDING_VALID = 1;
  PADDING_SAME = 2;
  PADDING_ZEROS = 3;
  PADDING_REFLECT = 4;
  PADDING_REPLICATE = 5;
  PADDING_CIRCULAR = 6;
}""",
    "layout": """enum LayoutMode {
  LAYOUT_UNSPECIFIED = 0;
  LAYOUT_NCHW = 1;
  LAYOUT_NHWC = 2;
  LAYOUT_NCDHW = 3;
  LAYOUT_NDHWC = 4;
}""",
    "mode": """enum InterpolationMode {
  INTERPOLATION_UNSPECIFIED = 0;
  INTERPOLATION_NEAREST = 1;
  INTERPOLATION_LINEAR = 2;
  INTERPOLATION_BILINEAR = 3;
  INTERPOLATION_BICUBIC = 4;
  INTERPOLATION_TRILINEAR = 5;
  INTERPOLATION_AREA = 6;
}""",
}


PROTO_DEFINITIONS_BY_TYPE: dict[str, str] = {
    "TensorProto": PROTO_TENSOR_DEFINITION,
    "ReductionType": PROTO_ENUM_DEFINITIONS["reduction"],
    "PaddingMode": PROTO_ENUM_DEFINITIONS["padding"],
    "LayoutMode": PROTO_ENUM_DEFINITIONS["layout"],
    "InterpolationMode": PROTO_ENUM_DEFINITIONS["mode"],
}


_PARAM_NAME_TO_PROTO_ENUM: dict[str, str] = {
    "reduction": "ReductionType",
    "reduction_type": "ReductionType",
    "padding": "PaddingMode",
    "padding_mode": "PaddingMode",
    "layout": "LayoutMode",
    "layout_mode": "LayoutMode",
}

_PROTO_SUBSTRING_TYPES: tuple[tuple[str, str], ...] = (
    ("list", "repeated string"),
    ("tuple", "repeated string"),
    ("dict", "map<string, string>"),
    ("int", "int64"),
    ("float", "double"),
    ("bool", "bool"),
    ("str", "string"),
)


def py_type_to_proto(typ: str | None, param_name: str = "") -> str:
    """Map Python type to Protobuf type using AST token extraction and enum mapping.

    Args:
        typ: Python type string.
        param_name: Parameter name for shape and enum heuristic mapping.

    Returns:
        Protobuf type string.
    """
    if not typ:
        return "string"

    clean_t = str(typ).lower().strip()
    p_norm = param_name.lower().strip()
    ast_tokens = _extract_ast_type_identifiers(clean_t)

    if clean_t in ("tensor", "torch.tensor", "tensorproto", "ndarray", "array") or clean_t.startswith("tensor<"):
        return "TensorProto"

    if p_norm in ("shape", "size", "dims", "dimensions") or clean_t in (
        "shape",
        "dimensions",
        "tuple[int, ...]",
        "sequence[int]",
    ):
        return "repeated int64"

    if p_norm in _PARAM_NAME_TO_PROTO_ENUM or (p_norm == "mode" and "interp" in clean_t):
        return _PARAM_NAME_TO_PROTO_ENUM.get(p_norm, "InterpolationMode")

    for substring, proto_type in _PROTO_SUBSTRING_TYPES:
        if any(substring in tok for tok in ast_tokens) or substring in clean_t:
            return proto_type
    return "string"


def to_protobuf(ref: GhostRef, package: str = "ml_framework") -> str:
    """Convert a GhostRef into a Protocol Buffers (.proto) message definition.

    Args:
        ref: The GhostRef to convert.
        package: The protobuf package name.

    Returns:
        A string containing the .proto definition.
    """
    preamble_blocks: list[str] = []
    seen_blocks: set[str] = set()

    field_lines: list[str] = []
    field_num = 1
    for param in ref.params:
        if param.kind in ("VAR_POSITIONAL", "VAR_KEYWORD"):
            continue

        proto_type = _py_type_to_proto(param.annotation, param_name=param.name)
        def_block = PROTO_DEFINITIONS_BY_TYPE.get(proto_type)
        if def_block and def_block not in seen_blocks:
            seen_blocks.add(def_block)
            preamble_blocks.append(def_block)

        if param.default is not None and not proto_type.startswith("repeated") and not proto_type.startswith("map"):
            proto_type = f"optional {proto_type}"

        desc = f" // {param.description}" if param.description else ""
        field_lines.append(f"  {proto_type} {param.name} = {field_num};{desc}")
        field_num += 1

    lines = [
        'syntax = "proto3";',
        "",
        f"package {package};",
        "",
    ]
    if preamble_blocks:
        lines.extend(preamble_blocks)
        lines.append("")

    lines.append(f"// {ref.docstring}" if ref.docstring else f"// Generated message for {ref.api_path}")
    lines.append(f"message {ref.name} {{")
    lines.extend(field_lines)
    lines.append("}")
    return "\n".join(lines) + "\n"


def _map_annotation_to_cpp_type(annotation: str | None) -> str:
    """Map a Python type annotation string to a C++17 type via AST token inspection.

    Args:
        annotation: Optional Python type annotation string.

    Returns:
        C++17 type string.
    """
    tokens = _extract_ast_type_identifiers(annotation)
    if any(any(k in t for k in ("tensor", "array", "list")) for t in tokens):
        return "std::vector<float>"
    if any("int" in t for t in tokens):
        return "int64_t"
    if any("bool" in t for t in tokens):
        return "bool"
    if any("str" in t for t in tokens):
        return "std::string"
    return "float"


def to_cpp_header(ref: GhostRef, namespace: str = "ml_ecosystem") -> str:
    """Generate modern C++17 struct definitions and function bindings for a GhostRef.

    Args:
        ref: The GhostRef specification to convert.
        namespace: C++ enclosing namespace name.

    Returns:
        Formatted C++ header string with struct definition and docstrings.
    """
    clean_name = ref.name.replace(".", "_")
    lines: list[str] = [
        "// Generated by ml-ecosystem-snapshots",
        "#pragma once",
        "",
        "#include <string>",
        "#include <vector>",
        "#include <optional>",
        "#include <cstdint>",
        "",
        f"namespace {namespace} {{",
        "",
    ]
    if ref.docstring:
        lines.append(f"// {ref.docstring.strip().splitlines()[0]}")
    lines.append(f"struct {clean_name} {{")
    for p in ref.params:
        cpp_type = _map_annotation_to_cpp_type(p.annotation)
        lines.append(f"    {cpp_type} {p.name};")
    lines.append("};")
    lines.append("")
    lines.append(f"}} // namespace {namespace}")
    return "\n".join(lines) + "\n"


def _map_annotation_to_ts_type(annotation: str | None) -> str:
    """Map a Python type annotation string to a TypeScript type via AST token inspection.

    Args:
        annotation: Optional Python type annotation string.

    Returns:
        TypeScript type string.
    """
    tokens = _extract_ast_type_identifiers(annotation)
    if any(any(k in t for k in ("list", "tensor", "array")) for t in tokens):
        return "unknown[]"
    if any(any(k in t for k in ("int", "float", "number")) for t in tokens):
        return "number"
    if any("bool" in t for t in tokens):
        return "boolean"
    if any("str" in t for t in tokens):
        return "string"
    return "unknown"


def to_typescript_interface(ref: GhostRef) -> str:
    """Generate TypeScript interface definitions for a GhostRef.

    Args:
        ref: The GhostRef specification to convert.

    Returns:
        Formatted TypeScript interface definition string.
    """
    clean_name = ref.name.replace(".", "_")
    lines: list[str] = []
    if ref.docstring:
        lines.append(f"/** {ref.docstring.strip().splitlines()[0]} */")
    lines.append(f"export interface {clean_name} {{")
    for p in ref.params:
        ts_type = _map_annotation_to_ts_type(p.annotation)
        opt = "?" if p.default is not None else ""
        lines.append(f"  {p.name}{opt}: {ts_type};")
    lines.append("}")
    return "\n".join(lines)


_py_type_to_proto = py_type_to_proto
to_pydantic_code = to_pydantic
to_typescript = to_typescript_interface


__all__ = [
    "PROTO_DEFINITIONS_BY_TYPE",
    "PROTO_ENUM_DEFINITIONS",
    "PROTO_TENSOR_DEFINITION",
    "py_type_to_proto",
    "to_cpp_header",
    "to_protobuf",
    "to_pydantic",
    "to_pydantic_code",
    "to_typescript",
    "to_typescript_interface",
]
