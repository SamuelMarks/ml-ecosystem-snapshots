"""LLM grounding prompt context exporters for Python, SASS, RDNA, PTX, MLIR, and StableHLO."""

from __future__ import annotations

import typing

from ml_switcheroo_ir.schema.ghost import (
    ExtendedGhostParam,
    ExtendedGhostRef,
    GhostParam,
    GhostRef,
)

from ml_ecosystem_snapshots.models import JsonValue

if typing.TYPE_CHECKING:
    from ml_ecosystem_snapshots.mcp.cache import SnapshotRepository


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
            p_dtypes = p.allowed_dtypes if isinstance(p, ExtendedGhostParam) else None
            if p_dtypes:
                constraints.append(f"dtypes: {p_dtypes}")
            p_rank = p.rank_constraint if isinstance(p, ExtendedGhostParam) else None
            if p_rank is not None:
                constraints.append(f"rank: {p_rank}")
            constr_str = f" [{', '.join(constraints)}]" if constraints else ""

            param_details.append(f"  - `{p.name}` ({p.kind}, type `{type_annot}`{default_part}){constr_str}{desc}")

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


def _build_sass_syntax_str(
    name: str,
    modifiers: list[str],
    op_sigs: list[list[str]],
) -> str:
    """Construct a canonical SASS assembly syntax template string for an instruction.

    Args:
        name: SASS instruction mnemonic.
        modifiers: List of supported instruction modifier strings.
        op_sigs: List of operand signature lists.

    Returns:
        Formatted SASS instruction template string.
    """
    mod_suffix = f"{{.{' | .'.join(m.lstrip('.') for m in modifiers[:4])}}}" if modifiers else ""
    if not op_sigs:
        return f"[@P0] {name}{mod_suffix};"
    op_terms = [f"R{i} (dst)" if i == 0 else f"R{i} ({op_t})" for i, op_t in enumerate(op_sigs[0])]
    return f"[@P0] {name}{mod_suffix} {', '.join(op_terms)};"


def _format_single_sass_ref(ref: GhostRef) -> str:
    """Format a single SASS `GhostRef` into a Markdown assembly specification block.

    Args:
        ref: SASS `GhostRef` item to format.

    Returns:
        Markdown block string for the instruction.
    """
    meta_raw = ref.domain_metadata if isinstance(ref, ExtendedGhostRef) else None
    meta = meta_raw if isinstance(meta_raw, dict) else {}
    raw_modifiers = meta.get("modifiers")
    modifiers: list[str] = [str(m) for m in raw_modifiers] if isinstance(raw_modifiers, list) else []
    raw_archs = meta.get("valid_architectures")
    archs: list[str] = [str(a) for a in raw_archs] if isinstance(raw_archs, list) and raw_archs else list(ref.environment_tags or [])
    raw_op_sigs = meta.get("operand_signatures")
    op_sigs: list[list[str]] = [[str(item) for item in sig] for sig in raw_op_sigs if isinstance(sig, list)] if isinstance(raw_op_sigs, list) else []

    syntax_str = _build_sass_syntax_str(ref.name, modifiers, op_sigs)
    lines = [
        f"### `{ref.name}`",
        f"- **Syntax**: `{syntax_str}`",
    ]

    clean_archs = [a for a in archs if a != "cuda"]
    if clean_archs:
        lines.append(f"- **Architectures**: {', '.join(clean_archs)}")

    if modifiers:
        lines.append(f"- **Modifiers**: {', '.join(modifiers)}")

    if ref.params:
        lines.append("- **Operands**:")
        for p in ref.params:
            dir_val = p.direction if isinstance(p, ExtendedGhostParam) else None
            dir_str = f" [{dir_val}]" if dir_val else ""
            lines.append(f"  - `{p.name}` ({p.standardized_name or 'operand'}, type `{p.annotation or 'Register'}`){dir_str}")

    if ref.docstring:
        lines.append(f"- **Description**: {ref.docstring.strip().splitlines()[0]}")

    return "\n".join(lines)


def export_sass_prompt_context(refs: typing.Sequence[GhostRef]) -> str:
    """Export SASS instruction specifications as canonical assembly prompt templates.

    Formats GPU instructions into real SASS assembly syntax templates rather than
    synthetic Python signatures, specifying valid register files, modifiers, and SM architectures.

    Args:
        refs: Sequence of SASS GhostRef items to format.

    Returns:
        Markdown string formatted with assembly syntax templates and register constraints.
    """
    blocks = [_format_single_sass_ref(ref) for ref in refs]
    return "\n\n".join(blocks) + "\n"


def _classify_mlir_params(
    params: typing.Sequence[GhostParam],
) -> tuple[list[GhostParam], list[GhostParam]]:
    """Partition MLIR operation parameters into SSA operands and compile-time attributes.

    Args:
        params: Sequence of `GhostParam` definitions on an MLIR operation.

    Returns:
        Tuple of `(ssa_operand_params, attribute_params)` excluding regions.
    """
    ssa_params: list[GhostParam] = []
    attr_params: list[GhostParam] = []
    for p in params:
        role = p.role if isinstance(p, ExtendedGhostParam) else None
        if str(role) == "IRParameterRole.REGION" or p.annotation == "Region":
            continue
        if str(role) == "IRParameterRole.ATTRIBUTE" or p.kind == "KEYWORD_ONLY":
            attr_params.append(p)
        else:
            ssa_params.append(p)
    return ssa_params, attr_params


def _format_single_mlir_ref(ref: GhostRef) -> str:
    """Format a single MLIR or StableHLO `GhostRef` into a Markdown SSA template block.

    Args:
        ref: MLIR or StableHLO `GhostRef` item to format.

    Returns:
        Markdown block string for the operation.
    """
    meta = (ref.domain_metadata or {}) if isinstance(ref, ExtendedGhostRef) else {}
    raw_traits = meta.get("traits")
    traits = [str(t) for t in raw_traits] if isinstance(raw_traits, list) else []
    raw_regions = meta.get("regions")
    regions = [str(r) for r in raw_regions] if isinstance(raw_regions, list) else []

    ssa_params, attr_params = _classify_mlir_params(ref.params)
    operands = [f"%{p.name}" for p in ssa_params]
    attributes = [f"{p.name} = ..." for p in attr_params]

    op_str = ", ".join(operands)
    attr_str = f" {{{', '.join(attributes)}}}" if attributes else ""
    ret_str = ref.returns_type or "none"
    syntax_str = f"%res = {ref.api_path}({op_str}){attr_str} : {ret_str}"

    lines = [
        f"### `{ref.api_path}`",
        f"- **Syntax**: `{syntax_str}`",
    ]

    if ssa_params:
        lines.append("- **SSA Operands**:")
        for p in ssa_params:
            lines.append(f"  - `%{p.name}` (type `{p.annotation or 'Value'}`)")

    if attr_params:
        lines.append("- **Attributes**:")
        for p in attr_params:
            lines.append(f"  - `{p.name}` (type `{p.annotation or 'Attribute'}`)")

    results = ref.returns if isinstance(ref, ExtendedGhostRef) else None
    if results:
        lines.append("- **Results**:")
        for r in results:
            lines.append(f"  - `%{r.name}` (type `{r.type}`)")

    if traits:
        lines.append(f"- **Traits**: {', '.join(traits)}")

    if regions:
        lines.append(f"- **Regions**: {', '.join(regions)}")

    if ref.docstring:
        lines.append(f"- **Description**: {ref.docstring.strip().splitlines()[0]}")

    return "\n".join(lines)


def export_mlir_prompt_context(refs: typing.Sequence[GhostRef]) -> str:
    """Export MLIR / StableHLO operations as canonical SSA IR prompt templates.

    Formats compiler operations into true MLIR syntax templates, distinguishing
    runtime SSA operands from compile-time attributes, regions, and multi-result signatures.

    Args:
        refs: Sequence of MLIR or StableHLO GhostRef items to format.

    Returns:
        Markdown string formatted with canonical MLIR operation specifications.
    """
    blocks = [_format_single_mlir_ref(ref) for ref in refs]
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
    "chex": [
        "Pass 'custom_message' (keyword-only) rather than 'msg' or 'message' to Chex assertions.",
        "Pass 'expected_shapes' to 'chex.assert_shape' and 'expected_ranks' to 'chex.assert_rank'.",
    ],
    "grain": [
        "Pass 'data_source', 'sampler', and 'worker_count' as keyword arguments to 'grain.DataLoader' rather than PyTorch's 'dataset' or 'num_workers'.",
    ],
    "orbax": [
        "Use 'max_to_keep' in 'orbax.checkpoint.CheckpointManagerOptions' rather than 'max_checkpoints' or 'keep_last_n'.",
        "Wrap saved/restored PyTrees in 'orbax.checkpoint.args.StandardSave' / 'StandardRestore' or 'PyTreeSave' / 'PyTreeRestore'.",
    ],
    "optax": [
        "Pass 'learning_rate' rather than 'lr' to Optax optimizers ('optax.adam', 'optax.adamw', 'optax.sgd').",
        "Pass separate 'b1' and 'b2' scalars rather than 'betas=(b1, b2)' to Optax momentum optimizers.",
    ],
}


def _collect_matching_scoped_items(
    raw_categories: JsonValue,
    module_prefix: str | None,
) -> list[dict[str, JsonValue]]:
    """Extract and filter snapshot symbol items matching an optional `module_prefix`.

    Args:
        raw_categories: Raw `categories` dictionary from a resolved snapshot.
        module_prefix: Optional namespace prefix filter.

    Returns:
        Filtered list of raw symbol dictionaries.
    """
    assert isinstance(raw_categories, dict)
    matching_items: list[dict[str, JsonValue]] = []
    for _cat, items in sorted(raw_categories.items()):
        assert isinstance(items, list)
        for raw_it in items:
            if not isinstance(raw_it, dict):
                continue
            item = dict(raw_it)
            path = str(item.get("api_path") or item.get("name") or item.get("mnemonic") or "")
            if not module_prefix or path.startswith(module_prefix) or path.startswith(f"{module_prefix}."):
                matching_items.append(item)
    return matching_items


def _build_scoped_index_lines(
    framework: str,
    matching_items: list[dict[str, JsonValue]],
) -> list[str]:
    """Build the Markdown header, hallucination guard section, and grouped index summary lines.

    Args:
        framework: Target framework identifier.
        matching_items: Filtered list of matching raw symbol dictionaries.

    Returns:
        List of Markdown lines for the index section.
    """
    index_lines: list[str] = [f"# Framework Grounding Context: `{framework}`"]

    guards = COMMON_HALLUCINATION_GUARDS.get(framework.lower())
    if guards:
        index_lines.append("\n## Common Hallucination Guards & Anti-Patterns")
        for g in guards:
            index_lines.append(f"- {g}")

    index_lines.append(f"\n## Index of Available Operations ({len(matching_items)} total)")

    grouped: dict[str, list[str]] = {}
    for item in matching_items:
        path = str(item.get("api_path") or item.get("name") or item.get("mnemonic") or "")
        parts = path.split(".")
        group_key = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
        grouped.setdefault(group_key, []).append(path)

    for group_name, members in sorted(grouped.items()):
        sample = ", ".join(f"`{m.split('.')[-1]}`" for m in members[:8])
        if len(members) > 8:
            sample += f", ... (+{len(members) - 8} more)"
        index_lines.append(f"- **{group_name}** ({len(members)} ops): {sample}")

    index_lines.append("\n## Detailed Operation Signatures")
    return index_lines


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
    matching_items = _collect_matching_scoped_items(snap.get("categories", {}), module_prefix)
    total_matching = len(matching_items)

    hydrated_refs: list[GhostRef] = []
    for item in matching_items[:max_symbols]:
        try:
            hydrated_refs.append(GhostInspector.hydrate(item))
        except Exception as exc:
            del exc

    index_lines = _build_scoped_index_lines(framework, matching_items)

    if framework in ("nvidia_sass", "amd_rdna"):
        details = export_sass_prompt_context(hydrated_refs)
    elif framework in ("mlir", "stablehlo"):
        details = export_mlir_prompt_context(hydrated_refs)
    else:
        details = export_llm_prompt_context(hydrated_refs)

    result_parts = ["\n".join(index_lines), details]

    if total_matching > max_symbols:
        result_parts.append(f"*(Showing {max_symbols} of {total_matching} operations to prevent context explosion. Query specific submodules or APIs using 'search_apis' or 'get_api_signature' for complete constraints.)*\n")

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


to_llm_prompt = export_llm_prompt_context
to_scoped_llm_prompt = export_scoped_prompt_context
to_sass_prompt = export_sass_prompt_context
to_mlir_prompt = export_mlir_prompt_context
to_rdna_prompt = export_rdna_prompt_context
to_ptx_prompt = export_ptx_prompt_context
to_stablehlo_prompt = export_stablehlo_prompt_context


__all__ = [
    "COMMON_HALLUCINATION_GUARDS",
    "export_llm_prompt_context",
    "export_mlir_prompt_context",
    "export_ptx_prompt_context",
    "export_rdna_prompt_context",
    "export_sass_prompt_context",
    "export_scoped_prompt_context",
    "export_stablehlo_prompt_context",
    "to_llm_prompt",
    "to_mlir_prompt",
    "to_ptx_prompt",
    "to_rdna_prompt",
    "to_sass_prompt",
    "to_scoped_llm_prompt",
    "to_stablehlo_prompt",
]
