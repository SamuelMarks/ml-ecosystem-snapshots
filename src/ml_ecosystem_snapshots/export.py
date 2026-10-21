"""Export module for generating JSON Schema, OpenAPI specifications, Pydantic classes, and Protobuf definitions."""

from __future__ import annotations

from collections import OrderedDict
import typing

from ml_switcheroo_ir.schema.ghost import GhostParam, GhostRef
import pydantic

from ml_ecosystem_snapshots.models import JsonValue

if typing.TYPE_CHECKING:
    from ml_ecosystem_snapshots.mcp.cache import SnapshotRepository


class CddIrModel(pydantic.BaseModel):
    """Typed cdd-python Intermediate Representation (IR) model."""

    model_config = pydantic.ConfigDict(extra="forbid")

    name: str
    type: str = "static"
    doc: str = ""
    params: dict[str, dict[str, str]] = pydantic.Field(default_factory=dict)
    returns: dict[str, dict[str, str]] | None = None

    @typing.overload
    def __getitem__(self, key: typing.Literal["name", "type", "doc"]) -> str: ...

    @typing.overload
    def __getitem__(
        self, key: typing.Literal["params"]
    ) -> dict[str, dict[str, str]]: ...

    @typing.overload
    def __getitem__(
        self, key: typing.Literal["returns"]
    ) -> dict[str, dict[str, str]] | None: ...

    @typing.overload
    def __getitem__(self, key: str) -> str | dict[str, dict[str, str]] | JsonValue: ...

    def __getitem__(self, key: str) -> str | dict[str, dict[str, str]] | JsonValue:
        """Retrieve a CDD IR field by dictionary key.

        Args:
            key: Field name ('name', 'type', 'doc', 'params', or 'returns').

        Returns:
            Field value.
        """
        val: str | dict[str, dict[str, str]] | JsonValue = getattr(self, key)
        return val

    def to_cdd_dict(self) -> dict[str, JsonValue]:
        """Convert to a mutable dictionary with `OrderedDict` params/returns for `cdd-python`.

        Returns:
            Dictionary suitable for `cdd-python` emitters.
        """
        params_od: dict[str, JsonValue] = OrderedDict(
            (k, dict(v)) for k, v in self.params.items()
        )
        returns_od: dict[str, JsonValue] | None = (
            OrderedDict((k, dict(v)) for k, v in self.returns.items())
            if self.returns is not None
            else None
        )
        return {
            "name": self.name,
            "type": self.type,
            "doc": self.doc,
            "params": params_od,
            "returns": returns_od,
        }


def _ghost_to_cdd_ir(ref: GhostRef) -> CddIrModel:
    """Convert a GhostRef to cdd-python Intermediate Representation (IR).

    Args:
        ref: The `GhostRef` instance to convert into a `CddIrModel`.

    Returns:
        The `CddIrModel` representation.
    """
    params_map: dict[str, dict[str, str]] = {}

    for param in ref.params:
        p_dict: dict[str, str] = {}
        if param.annotation:
            p_dict["typ"] = param.annotation
        if param.description:
            p_dict["doc"] = param.description
        if param.default is not None:
            p_dict["default"] = str(param.default)
        params_map[param.name] = p_dict

    returns_map: dict[str, dict[str, str]] | None = None
    if ref.returns_type or ref.returns_description:
        ret_dict: dict[str, str] = {}
        if ref.returns_type:
            ret_dict["typ"] = ref.returns_type
        if ref.returns_description:
            ret_dict["doc"] = ref.returns_description
        returns_map = {"return_type": ret_dict}

    return CddIrModel(
        name=ref.name,
        type="class" if ref.kind == "class" else "static",
        doc=ref.docstring or "",
        params=params_map,
        returns=returns_map,
    )


def to_json_schema(ref: GhostRef) -> dict[str, JsonValue]:
    """Convert a GhostRef into a JSON Schema.

    Args:
        ref: The GhostRef to convert.

    Returns:
        A dictionary representing the JSON Schema.
    """
    import cdd.json_schema.emit

    ir = _ghost_to_cdd_ir(ref)
    schema: dict[str, JsonValue] = cdd.json_schema.emit.json_schema(
        ir.to_cdd_dict(), identifier=ref.api_path
    )
    return schema


def to_openapi(refs: typing.Sequence[GhostRef]) -> dict[str, JsonValue]:
    """Convert a sequence of GhostRefs into an OpenAPI specification.

    Args:
        refs: A sequence of GhostRefs.

    Returns:
        A dictionary representing the OpenAPI specification.
    """
    import cdd.compound.openapi.emit
    from cdd.compound.openapi.utils.emit_openapi_utils import NameModelRouteIdCrud

    nm_cruds = []
    for ref in refs:
        ir = _ghost_to_cdd_ir(ref)
        route = f"/{ref.api_path.replace('.', '/')}"

        nm_cruds.append(
            NameModelRouteIdCrud(
                name=ref.name,
                model=ir.to_cdd_dict(),
                route=route,
                id=None,
                crud="CRU",
            )
        )

    spec: dict[str, JsonValue] = cdd.compound.openapi.emit.openapi(nm_cruds)
    return spec


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

    def _render_single_model(
        name: str,
        doc: str | None,
        params: typing.Sequence[GhostParam],
    ) -> list[str]:
        """Render Python source code lines for a single Pydantic model.

        Args:
            name: Class name for the Pydantic model.
            doc: Optional docstring for the class.
            params: Sequence of GhostParam specifications.

        Returns:
            List of Python source code lines for the class.
        """
        m_lines = [f"class {name}(pydantic.BaseModel):"]
        if doc:
            m_lines.append(f'    """{doc}"""')
        else:
            m_lines.append('    """Generated Pydantic model."""')
        m_lines.append("")
        m_lines.append('    model_config = pydantic.ConfigDict(extra="forbid")')

        valid_params = []
        for p in params:
            if p.kind in ("VAR_POSITIONAL", "VAR_KEYWORD") and not validate_varargs:
                continue
            valid_params.append(p)

        if not valid_params:
            return m_lines

        for param in valid_params:
            raw_typ = param.annotation if param.annotation else "object"
            typ = "object" if raw_typ == "Any" else raw_typ
            desc = param.description.replace('"', "'") if param.description else ""
            default_val = param.default

            if param.kind == "VAR_POSITIONAL":
                typ = f"tuple[{typ}, ...]"
                field_def = (
                    f'pydantic.Field(default=(), description="{desc}")'
                    if desc
                    else "pydantic.Field(default=())"
                )
                m_lines.append(f"    {param.name}: {typ} = {field_def}")
                continue
            if param.kind == "VAR_KEYWORD":
                typ = f"dict[str, {typ}]"
                field_def = (
                    f'pydantic.Field(default_factory=dict, description="{desc}")'
                    if desc
                    else "pydantic.Field(default_factory=dict)"
                )
                m_lines.append(f"    {param.name}: {typ} = {field_def}")
                continue

            if default_val is None:
                field_def = (
                    f'pydantic.Field(..., description="{desc}")'
                    if desc
                    else "pydantic.Field(...)"
                )
                m_lines.append(f"    {param.name}: {typ} = {field_def}")
            else:
                field_def = (
                    f'pydantic.Field(default={default_val}, description="{desc}")'
                    if desc
                    else f"{default_val}"
                )
                m_lines.append(f"    {param.name}: {typ} = {field_def}")

        return m_lines

    all_models: list[str] = []
    if ref.overloads:
        all_variants = [ref] + [ov for ov in ref.overloads if isinstance(ov, GhostRef)]
        variant_names = []
        for i, variant in enumerate(all_variants):
            var_name = f"{ref.name}Variant{i}"
            variant_names.append(var_name)
            all_models.extend(
                _render_single_model(
                    var_name, variant.docstring or ref.docstring, variant.params
                )
            )
            all_models.append("")

        all_models.append(f"{ref.name} = {' | '.join(variant_names)}")
    else:
        all_models.extend(_render_single_model(ref.name, ref.docstring, ref.params))

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


def _py_type_to_proto(typ: str | None, param_name: str = "") -> str:
    """Map Python type to Protobuf type with structured message and enum mapping.

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

    if clean_t in (
        "tensor",
        "torch.tensor",
        "tensorproto",
        "ndarray",
        "array",
    ) or clean_t.startswith("tensor<"):
        return "TensorProto"

    if p_norm in ("shape", "size", "dims", "dimensions") or clean_t in (
        "shape",
        "dimensions",
        "tuple[int, ...]",
        "sequence[int]",
    ):
        return "repeated int64"

    if p_norm in ("reduction", "reduction_type"):
        return "ReductionType"
    if p_norm in ("padding", "padding_mode"):
        return "PaddingMode"
    if p_norm in ("layout", "layout_mode"):
        return "LayoutMode"
    if p_norm == "mode" and "interp" in clean_t:
        return "InterpolationMode"

    if "list" in clean_t or "tuple" in clean_t:
        return "repeated string"
    elif "dict" in clean_t:
        return "map<string, string>"
    elif "int" in clean_t:
        return "int64"
    elif "float" in clean_t:
        return "double"
    elif "bool" in clean_t:
        return "bool"
    elif "str" in clean_t:
        return "string"
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

        if (
            param.default is not None
            and not proto_type.startswith("repeated")
            and not proto_type.startswith("map")
        ):
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

    lines.append(
        f"// {ref.docstring}"
        if ref.docstring
        else f"// Generated message for {ref.api_path}"
    )
    lines.append(f"message {ref.name} {{")
    lines.extend(field_lines)
    lines.append("}")
    return "\n".join(lines) + "\n"


def export_llm_prompt_context(refs: typing.Sequence[GhostRef]) -> str:
    """Export GhostRef sequence into a compact, typed format optimized for LLM prompting.

    Formats functions, operands, attributes, and constraints into concise markdown
    specifications designed to prevent LLM hallucinations during code generation.

    Args:
        refs: Sequence of GhostRef items to format.

    Returns:
        A Markdown-formatted string with compact signatures and parameter constraints.
    """
    blocks: list[str] = []

    for ref in refs:
        params_str_list: list[str] = []
        param_details: list[str] = []

        for p in ref.params:
            type_annot = p.annotation or "Any"
            default_part = f" = {p.default}" if p.default is not None else ""
            params_str_list.append(f"{p.name}: {type_annot}{default_part}")

            desc = f" - {p.description}" if p.description else ""

            constraints: list[str] = []
            p_dtypes = getattr(p, "allowed_dtypes", None)
            if p_dtypes:
                constraints.append(f"dtypes: {p_dtypes}")
            p_rank = getattr(p, "rank_constraint", None)
            if p_rank is not None:
                constraints.append(f"rank: {p_rank}")
            constr_str = f" [{', '.join(constraints)}]" if constraints else ""

            param_details.append(
                f"  - `{p.name}` ({p.kind}, type `{type_annot}`{default_part}){constr_str}{desc}"
            )

        sig = f"{ref.api_path}({', '.join(params_str_list)})"
        if ref.returns_type:
            sig += f" -> {ref.returns_type}"

        lines = [
            f"### `{ref.api_path}`",
            f"- **Kind**: `{ref.kind}`",
            f"- **Signature**: `{sig}`",
        ]

        if param_details:
            lines.append("- **Parameters**:")
            lines.extend(param_details)

        if ref.raises:
            lines.append(f"- **Raises**: {', '.join(ref.raises)}")

        if ref.environment_tags:
            lines.append(f"- **Environments**: {', '.join(ref.environment_tags)}")

        if ref.docstring:
            first_line = ref.docstring.strip().splitlines()[0]
            lines.append(f"- **Summary**: {first_line}")

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks) + "\n"


def export_sass_prompt_context(refs: typing.Sequence[GhostRef]) -> str:
    """Export SASS instruction specifications as canonical assembly prompt templates.

    Formats GPU instructions into real SASS assembly syntax templates rather than
    synthetic Python signatures, specifying valid register files, modifiers, and SM architectures.

    Args:
        refs: Sequence of SASS GhostRef items to format.

    Returns:
        Markdown string formatted with assembly syntax templates and register constraints.
    """
    blocks: list[str] = []

    for ref in refs:
        meta = getattr(ref, "domain_metadata", None) or {}
        modifiers = meta.get("modifiers") or []
        archs = meta.get("valid_architectures") or ref.environment_tags or []
        op_sigs = meta.get("operand_signatures") or []

        mod_suffix = (
            f"{{.{' | .'.join(m.lstrip('.') for m in modifiers[:4])}}}"
            if modifiers
            else ""
        )
        if op_sigs:
            sample_sig = op_sigs[0]
            op_terms = []
            for i, op_t in enumerate(sample_sig):
                if i == 0:
                    op_terms.append(f"R{i} (dst)")
                else:
                    op_terms.append(f"R{i} ({op_t})")
            syntax_str = f"[@P0] {ref.name}{mod_suffix} {', '.join(op_terms)};"
        else:
            syntax_str = f"[@P0] {ref.name}{mod_suffix};"

        lines = [
            f"### `{ref.name}`",
            f"- **Syntax**: `{syntax_str}`",
        ]

        if archs:
            clean_archs = [a for a in archs if a != "cuda"]
            if clean_archs:
                lines.append(f"- **Architectures**: {', '.join(clean_archs)}")

        if modifiers:
            lines.append(f"- **Modifiers**: {', '.join(modifiers)}")

        if ref.params:
            lines.append("- **Operands**:")
            for p in ref.params:
                dir_val = getattr(p, "direction", None)
                dir_str = f" [{dir_val}]" if dir_val else ""
                lines.append(
                    f"  - `{p.name}` ({p.standardized_name or 'operand'}, type `{p.annotation or 'Register'}`){dir_str}"
                )

        if ref.docstring:
            first_line = ref.docstring.strip().splitlines()[0]
            lines.append(f"- **Description**: {first_line}")

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks) + "\n"


def export_mlir_prompt_context(refs: typing.Sequence[GhostRef]) -> str:
    """Export MLIR / StableHLO operations as canonical SSA IR prompt templates.

    Formats compiler operations into true MLIR syntax templates, distinguishing
    runtime SSA operands from compile-time attributes, regions, and multi-result signatures.

    Args:
        refs: Sequence of MLIR or StableHLO GhostRef items to format.

    Returns:
        Markdown string formatted with canonical MLIR operation specifications.
    """
    blocks: list[str] = []

    for ref in refs:
        meta = getattr(ref, "domain_metadata", None) or {}
        traits = meta.get("traits") or []
        regions = meta.get("regions") or []

        operands: list[str] = []
        attributes: list[str] = []

        for p in ref.params:
            role = getattr(p, "role", None)
            if str(role) == "IRParameterRole.REGION" or p.annotation == "Region":
                continue
            if str(role) == "IRParameterRole.ATTRIBUTE" or p.kind == "KEYWORD_ONLY":
                attributes.append(f"{p.name} = ...")
            else:
                operands.append(f"%{p.name}")

        op_str = ", ".join(operands)
        attr_str = f" {{{', '.join(attributes)}}}" if attributes else ""
        ret_str = ref.returns_type or "none"
        syntax_str = f"%res = {ref.api_path}({op_str}){attr_str} : {ret_str}"

        lines = [
            f"### `{ref.api_path}`",
            f"- **Syntax**: `{syntax_str}`",
        ]

        if operands:
            lines.append("- **SSA Operands**:")
            for p in ref.params:
                if (
                    str(getattr(p, "role", None)) != "IRParameterRole.ATTRIBUTE"
                    and p.annotation != "Region"
                ):
                    lines.append(f"  - `%{p.name}` (type `{p.annotation or 'Value'}`)")

        if attributes:
            lines.append("- **Attributes**:")
            for p in ref.params:
                if (
                    str(getattr(p, "role", None)) == "IRParameterRole.ATTRIBUTE"
                    or p.kind == "KEYWORD_ONLY"
                ):
                    lines.append(
                        f"  - `{p.name}` (type `{p.annotation or 'Attribute'}`)"
                    )

        results = getattr(ref, "returns", None)
        if results:
            lines.append("- **Results**:")
            for r in results:
                lines.append(f"  - `%{r.name}` (type `{r.type}`)")

        if traits:
            lines.append(f"- **Traits**: {', '.join(str(t) for t in traits)}")

        if regions:
            lines.append(f"- **Regions**: {', '.join(str(r) for r in regions)}")

        if ref.docstring:
            first_line = ref.docstring.strip().splitlines()[0]
            lines.append(f"- **Description**: {first_line}")

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks) + "\n"


COMMON_HALLUCINATION_GUARDS: dict[str, list[str]] = {
    "torch": [
        "Do NOT pass 'axis' to 'torch.sum' or reduction operations; use 'dim'.",
        "Do NOT pass 'keepdims' to PyTorch reduction ops; use 'keepdim'.",
        "Do NOT pass integer dtypes (e.g. 'int32') to linear algebra ops like 'torch.linalg.inv' or 'torch.cholesky'; floating-point or complex required.",
    ],
    "nvidia_sass": [
        "Uniform registers (UR0-UR63) require Volta+ (sm_70+).",
        "Align 64-bit register pairs to even indices (e.g. R0:R1, not R1:R2).",
        "Align 128-bit register quads to modulo-4 starting indices (e.g. R0:R3).",
    ],
    "amd_rdna": [
        "Dual-issue instructions (v_dual_*) are strictly restricted to GFX11/RDNA3 and GFX12/RDNA4.",
        "Matrix accumulator instructions (v_mfma_*) are strictly restricted to GFX9/CDNA.",
        "Align 64-bit vector registers to even indices (v[0:1], s[0:1]).",
    ],
    "mlir": [
        "Do NOT pass floating-point types to signless integer ops like 'arith.addi'; use 'arith.addf'.",
        "Ensure SSA operand segment sizes match AttrSizedOperandSegments attributes.",
    ],
    "stablehlo": [
        "Verify DotDimensionNumbers contracting dimensions against input operand ranks.",
        "Broadcast dimensions length in 'stablehlo.broadcast_in_dim' must equal input rank.",
    ],
}


def export_scoped_prompt_context(
    framework: str,
    module_prefix: str | None = None,
    max_symbols: int = 50,
    repository: SnapshotRepository | None = None,
) -> str:
    """Export a scoped prompt context with a hierarchical summary index.

    Prevents context window explosion by providing an indexed summary of available
    operations and detailed parameter constraints for up to max_symbols matching APIs.

    Args:
        framework: Name of the framework (e.g. 'torch', 'stablehlo', 'nvidia_sass').
        module_prefix: Optional namespace filter (e.g. 'torch.nn.functional', 'arith').
        max_symbols: Maximum number of detailed symbols to include (default: 50).
        repository: Optional `SnapshotRepository` instance for snapshot lookup.

    Returns:
        Structured Markdown string containing an index summary followed by detailed signatures.
    """
    from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
    from ml_ecosystem_snapshots.mcp.cache import resolve_snapshot

    snap = resolve_snapshot(framework, repository=repository)
    raw_categories = snap.get("categories", {})

    all_raw_items: list[dict[str, JsonValue]] = []
    assert isinstance(raw_categories, dict)
    for _cat, items in sorted(raw_categories.items()):
        assert isinstance(items, list)
        for raw_it in items:
            if isinstance(raw_it, dict):
                all_raw_items.append(dict(raw_it))

    matching_items: list[dict[str, JsonValue]] = []
    for item in all_raw_items:
        path = str(
            item.get("api_path") or item.get("name") or item.get("mnemonic") or ""
        )
        if module_prefix:
            if path.startswith(module_prefix) or path.startswith(f"{module_prefix}."):
                matching_items.append(item)
        else:
            matching_items.append(item)

    total_matching = len(matching_items)
    selected_items = matching_items[:max_symbols]

    hydrated_refs: list[GhostRef] = []
    for item in selected_items:
        try:
            hydrated_refs.append(GhostInspector.hydrate(item))
        except Exception:
            pass

    index_lines: list[str] = [
        f"# Framework Grounding Context: `{framework}`",
    ]

    guards = COMMON_HALLUCINATION_GUARDS.get(framework.lower())
    if guards:
        index_lines.append("\n## Common Hallucination Guards & Anti-Patterns")
        for g in guards:
            index_lines.append(f"- {g}")

    index_lines.append(f"\n## Index of Available Operations ({total_matching} total)")

    grouped: dict[str, list[str]] = {}
    for item in matching_items:
        path = str(
            item.get("api_path") or item.get("name") or item.get("mnemonic") or ""
        )
        parts = path.split(".")
        group_key = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
        if group_key not in grouped:
            grouped[group_key] = []
        grouped[group_key].append(path)

    for group_name, members in sorted(grouped.items()):
        sample = ", ".join(f"`{m.split('.')[-1]}`" for m in members[:8])
        if len(members) > 8:
            sample += f", ... (+{len(members) - 8} more)"
        index_lines.append(f"- **{group_name}** ({len(members)} ops): {sample}")

    index_lines.append("\n## Detailed Operation Signatures")

    if framework in ("nvidia_sass", "amd_rdna"):
        details = export_sass_prompt_context(hydrated_refs)
    elif framework in ("mlir", "stablehlo"):
        details = export_mlir_prompt_context(hydrated_refs)
    else:
        details = export_llm_prompt_context(hydrated_refs)

    result_parts = ["\n".join(index_lines), details]

    if total_matching > max_symbols:
        result_parts.append(
            f"*(Showing {max_symbols} of {total_matching} operations to prevent context explosion. "
            f"Query specific submodules or APIs using 'search_apis' or 'get_api_signature' for complete constraints.)*\n"
        )

    return "\n\n".join(result_parts).strip() + "\n"


def export_rdna_prompt_context(refs: typing.Sequence[GhostRef]) -> str:
    """Export AMD RDNA / CDNA instruction specifications as canonical assembly prompt templates.

    Args:
        refs: Sequence of AMD RDNA GhostRef items to format.

    Returns:
        Markdown string formatted with RDNA assembly syntax and architecture constraints.
    """
    blocks: list[str] = []
    for ref in refs:
        encoding = "VOP"
        archs: list[str] = []
        for tag in ref.environment_tags or []:
            if any(tag.startswith(pfx) for pfx in ("VOP", "SOP", "SMEM", "FLAT", "DS")):
                encoding = tag
            else:
                archs.append(tag)
        lines = [
            f"### `{ref.name}`",
            f"- **Encoding**: `{encoding}`",
        ]
        if archs:
            lines.append(f"- **Architectures**: {', '.join(archs)}")
        if ref.params:
            lines.append("- **Operands**:")
            for p in ref.params:
                lines.append(f"  - `{p.name}`: `{p.annotation or 'vgpr'}`")
        if ref.docstring:
            lines.append(f"- **Description**: {ref.docstring.strip().splitlines()[0]}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def export_ptx_prompt_context(refs: typing.Sequence[GhostRef]) -> str:
    """Export NVIDIA PTX instruction specifications as canonical assembly prompt templates.

    Args:
        refs: Sequence of PTX GhostRef items to format.

    Returns:
        Markdown string formatted with PTX assembly syntax and SM requirements.
    """
    blocks: list[str] = []
    for ref in refs:
        min_sm = "sm_50"
        supp_types: list[str] = []
        for tag in ref.environment_tags or []:
            if tag.startswith("sm_"):
                min_sm = tag
            elif tag.startswith("."):
                supp_types.append(tag)
        lines = [
            f"### `{ref.name}`",
            f"- **Minimum SM**: `{min_sm}`",
        ]
        if supp_types:
            lines.append(f"- **Supported Types**: `{', '.join(supp_types)}`")
        if ref.params:
            lines.append("- **Operands**:")
            for p in ref.params:
                lines.append(f"  - `{p.name}`: `{p.annotation or 'reg'}`")
        if ref.docstring:
            lines.append(f"- **Description**: {ref.docstring.strip().splitlines()[0]}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def export_stablehlo_prompt_context(refs: typing.Sequence[GhostRef]) -> str:
    """Export StableHLO operations as canonical MLIR dialect SSA prompt templates.

    Args:
        refs: Sequence of StableHLO GhostRef items to format.

    Returns:
        Markdown string formatted with canonical StableHLO operation templates.
    """
    return export_mlir_prompt_context(refs)


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
        cpp_type = "float"
        if p.annotation:
            ann = p.annotation.lower()
            if "tensor" in ann or "array" in ann or "list" in ann:
                cpp_type = "std::vector<float>"
            elif "int" in ann:
                cpp_type = "int64_t"
            elif "bool" in ann:
                cpp_type = "bool"
            elif "str" in ann:
                cpp_type = "std::string"
        lines.append(f"    {cpp_type} {p.name};")
    lines.append("};")
    lines.append("")
    lines.append(f"}} // namespace {namespace}")
    return "\n".join(lines) + "\n"


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
        ts_type = "unknown"
        if p.annotation:
            ann = p.annotation.lower()
            if "list" in ann or "tensor" in ann or "array" in ann:
                ts_type = "unknown[]"
            elif "int" in ann or "float" in ann or "number" in ann:
                ts_type = "number"
            elif "bool" in ann:
                ts_type = "boolean"
            elif "str" in ann:
                ts_type = "string"
        opt = "?" if p.default is not None else ""
        lines.append(f"  {p.name}{opt}: {ts_type};")
    lines.append("}")
    return "\n".join(lines)


to_pydantic_code = to_pydantic
