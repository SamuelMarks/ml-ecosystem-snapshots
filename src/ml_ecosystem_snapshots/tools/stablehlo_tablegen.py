"""StableHLO TableGen `.td` and JSON dump extraction utilities."""

from __future__ import annotations

import json
import re
import typing

from ml_ecosystem_snapshots.models import JsonValue, TableGenRecordModel, ensure_json_dict
from ml_ecosystem_snapshots.tools.mlir_tablegen import (
    TableGenASTParser,
    _extract_brace_block,
    _extract_tblgen_json_traits,
    strip_tablegen_comments,
)
from ml_ecosystem_snapshots.utils import extract_tablegen_traits


def is_attribute(type_str: str) -> bool:
    """Determine if a spec type string represents an attribute rather than an operand.

    Args:
        type_str: The type string from the specification.

    Returns:
        True if the type represents an attribute, False otherwise.
    """
    t = type_str.lower()
    if any(k in t for k in ("tensor", "value", "anytype", "anyshaped")):
        return False
    return (
        "constant" in t
        or "enum" in t
        or "function" in t
        or "region" in t
        or "attribute" in t
        or "floattype" in t
        or "si64" == t.strip("`")
        or "string" in t
        or "aliasing parts" in t
        or "attr" in t
        or "dimensionnumbers" in t
        or "precision" in t
        or "algorithm" in t
        or "config" in t
        or "direction" in t
        or "distribution" in t
        or "comparison" in t
        or "transpose" in t
        or "mode" in t
        or "fft" in t
        or "dense" in t
        or "array" in t
    )


def build_verification_and_region_metadata(
    op_name: str,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    """Construct verification rules and region signature metadata for a StableHLO op.

    Args:
        op_name: Unqualified StableHLO operation mnemonic (e.g. 'dot_general', 'reduce').

    Returns:
        Tuple of `(verification_rules, region_signatures)` dictionaries.
    """
    verification_rules: dict[str, JsonValue] = {}
    if op_name in ("dot_general", "dot"):
        verification_rules["dimension_numbers"] = "DotDimensionNumbersAttr"
    elif op_name in ("convolution", "conv"):
        verification_rules["dimension_numbers"] = "ConvDimensionNumbersAttr"
    elif op_name in ("scatter",):
        verification_rules["dimension_numbers"] = "ScatterDimensionNumbersAttr"
    elif op_name in ("gather",):
        verification_rules["dimension_numbers"] = "GatherDimensionNumbersAttr"
    elif op_name in ("broadcast_in_dim",):
        verification_rules["broadcast"] = "broadcast_dimensions"

    region_signatures: dict[str, JsonValue] = {}
    if op_name == "reduce":
        region_signatures["body"] = {
            "block_arguments": ["tensor<T>", "tensor<T>"],
            "yield_types": ["tensor<T>"],
            "terminator": "stablehlo.return",
        }
    elif op_name == "while":
        region_signatures["cond"] = {
            "block_arguments": ["(args...)"],
            "yield_types": ["tensor<i1>"],
            "terminator": "stablehlo.return",
        }
        region_signatures["body"] = {
            "block_arguments": ["(args...)"],
            "yield_types": ["(results...)"],
            "terminator": "stablehlo.return",
        }
    elif op_name == "sort":
        region_signatures["comparator"] = {
            "block_arguments": ["tensor<T>", "tensor<T>"],
            "yield_types": ["tensor<i1>"],
            "terminator": "stablehlo.return",
        }

    return verification_rules, region_signatures


def _extract_stablehlo_base_classes(
    clean_content: str,
) -> dict[str, dict[str, str | list[str] | None]]:
    """Parse StableHLO TableGen `class` definitions into a lookup registry.

    Args:
        clean_content: Macro-expanded TableGen source string.

    Returns:
        Mapping from class name to dictionary of `args`, `results`, and `supers`.
    """
    base_classes: dict[str, dict[str, str | list[str] | None]] = {}
    for m in re.finditer(
        r"\bclass\s+([A-Za-z0-9_]+)(?:<[\s\S]*?>)?\s*(?::\s*([^{;]+))?\s*(\{|;)",
        clean_content,
    ):
        cls_name = m.group(1)
        supers_raw = m.group(2) or ""
        term = m.group(3)
        cls_body, _ = _extract_brace_block(clean_content, m.end()) if term == "{" else ("", m.end())

        args_m = re.search(r"let\s+arguments\s*=\s*\(\s*ins(.*?)\);", cls_body, re.DOTALL) or re.search(
            r"Arguments<\s*\(\s*ins(.*?)\)>", supers_raw, re.DOTALL
        )
        res_m = re.search(r"let\s+results\s*=\s*\(\s*outs(.*?)\);", cls_body, re.DOTALL) or re.search(
            r"Results<\s*\(\s*outs(.*?)\)>", supers_raw, re.DOTALL
        )
        supers = [s.split("<")[0].strip() for s in re.split(r",", supers_raw) if s.strip()]
        base_classes[cls_name] = {
            "args": args_m.group(1).strip() if args_m else None,
            "results": res_m.group(1).strip() if res_m else None,
            "supers": supers,
        }
    return base_classes


def _resolve_stablehlo_inherited_field(
    base_classes: typing.Mapping[str, typing.Mapping[str, str | list[str] | None]],
    cls_name: str,
    field_name: str,
) -> str | None:
    """Recursively resolve an inherited TableGen field from `base_classes`.

    Args:
        base_classes: Parsed StableHLO TableGen class registry.
        cls_name: Name of the class to resolve.
        field_name: Field key to lookup (`'args'` or `'results'`).

    Returns:
        Resolved field string value or None.
    """
    if cls_name not in base_classes:
        return None
    val = base_classes[cls_name].get(field_name)
    if val is not None:
        return str(val)
    raw_supers = base_classes[cls_name].get("supers", [])
    for s in raw_supers if isinstance(raw_supers, list) else []:
        res = _resolve_stablehlo_inherited_field(base_classes, s, field_name)
        if res is not None:
            return str(res)
    return None


def _scan_def_header_and_body(clean_content: str, start: int) -> tuple[str, str]:
    """Scan balanced template arguments and body for a TableGen `def` block.

    Args:
        clean_content: Macro-expanded TableGen source string.
        start: Character index immediately after the base class name.

    Returns:
        Tuple of `(targs_raw, body)`.
    """
    depth_angle = 0
    depth_paren = 0
    depth_bracket = 0
    p = start
    term: str | None = None
    while p < len(clean_content):
        c = clean_content[p]
        if c == "<":
            depth_angle += 1
        elif c == ">":
            depth_angle -= 1
        elif c == "(":
            depth_paren += 1
        elif c == ")":
            depth_paren -= 1
        elif c == "[":
            depth_bracket += 1
        elif c == "]":
            depth_bracket -= 1
        elif depth_angle == 0 and depth_paren == 0 and depth_bracket == 0 and c in ("{", ";"):
            term = c
            break
        p += 1

    targs_raw = clean_content[start:p].strip()
    body, _ = _extract_brace_block(clean_content, p + 1) if term == "{" else ("", p)
    return targs_raw, body


def _camel_to_snake_mnemonic(class_name: str) -> str:
    """Convert a StableHLO camelCase operation class name into a snake_case mnemonic.

    Args:
        class_name: Operation class name (e.g. `'StableHLO_BroadcastInDimOp'`).

    Returns:
        Lowercase snake_case mnemonic string.
    """
    raw_name = class_name.replace("StableHLO_", "").replace("Op", "")
    return re.sub(r"(?<!^)(?=[A-Z])", "_", raw_name).lower()


def _resolve_stablehlo_op_name(class_name: str, targs_raw: str, body: str) -> str:
    """Resolve the StableHLO operation mnemonic from body, template arguments, or class name.

    Args:
        class_name: Operation `def` identifier.
        targs_raw: Raw template arguments string from the `def` header.
        body: Inner body of the `def` block.

    Returns:
        Unqualified StableHLO operation mnemonic.
    """
    mnem = re.search(r'let\s+mnemonic\s*=\s*"([a-z0-9_]+)"', body)
    if mnem:
        return mnem.group(1)
    m_targ = re.search(r'"([a-z0-9_]+)"', targs_raw)
    if m_targ:
        return m_targ.group(1)
    return _camel_to_snake_mnemonic(class_name)


def _split_angle_comma(raw_str: str) -> list[str]:
    """Split a TableGen argument or result list on commas outside `<...>` angle brackets.

    Args:
        raw_str: Raw comma-separated TableGen declaration string.

    Returns:
        List of trimmed declaration segments.
    """
    depth_angle = 0
    parts_list: list[str] = []
    curr_chars: list[str] = []
    for c in raw_str:
        if c == "<":
            depth_angle += 1
        elif c == ">":
            depth_angle -= 1
        elif c == "," and depth_angle == 0:
            parts_list.append("".join(curr_chars).strip())
            curr_chars = []
            continue
        curr_chars.append(c)
    if curr_chars:
        parts_list.append("".join(curr_chars).strip())
    return parts_list


def _append_stablehlo_arg(
    arg_name: str,
    arg_type: str,
    params: list[JsonValue],
    operands: list[JsonValue],
    attributes: list[JsonValue],
) -> None:
    """Append a parsed StableHLO argument into `params` and either `attributes` or `operands`.

    Args:
        arg_name: Argument identifier string.
        arg_type: TableGen type annotation string.
        params: Mutable list of GhostRef parameter dictionaries.
        operands: Mutable list of SSA operand dictionaries.
        attributes: Mutable list of attribute dictionaries.
    """
    is_attr = is_attribute(arg_type)
    kind = "KEYWORD_ONLY" if is_attr else "POSITIONAL_OR_KEYWORD"
    params.append(
        {
            "name": arg_name,
            "annotation": arg_type,
            "default": None,
            "kind": kind,
            "description": f"{'Attribute' if is_attr else 'Operand'} of type {arg_type}",
            "standardized_name": "attribute" if is_attr else "operand",
        }
    )
    target = attributes if is_attr else operands
    target.append({"name": arg_name, "type": arg_type})


def _append_stablehlo_region(
    reg_name: str,
    regions: list[JsonValue],
    params: list[JsonValue],
) -> None:
    """Append a parsed region identifier into `regions` and `params`.

    Args:
        reg_name: Region block identifier string.
        regions: Mutable list of region name strings.
        params: Mutable list of GhostRef parameter dictionaries.
    """
    regions.append(reg_name)
    params.append(
        {
            "name": reg_name,
            "annotation": "Region",
            "default": None,
            "kind": "KEYWORD_ONLY",
            "description": f"Op region block {reg_name}",
            "standardized_name": "region",
        }
    )


def _format_returns_type(res_types: list[str]) -> str | None:
    """Format a list of result types into a single return type annotation or tuple string.

    Args:
        res_types: List of result type annotation strings.

    Returns:
        Formatted return type string or None if `res_types` is empty.
    """
    if len(res_types) == 1:
        return res_types[0]
    if len(res_types) > 1:
        return f"tuple[{', '.join(res_types)}]"
    return None


def _parse_stablehlo_td_args(
    args_str: str | None,
) -> tuple[list[JsonValue], list[JsonValue], list[JsonValue]]:
    """Parse a TableGen `(ins ...)` string into `params`, `operands`, and `attributes`.

    Args:
        args_str: Inner `(ins ...)` declaration string or None.

    Returns:
        Three-element tuple of `(params, operands, attributes)`.
    """
    params: list[JsonValue] = []
    operands: list[JsonValue] = []
    attributes: list[JsonValue] = []
    if not args_str:
        return params, operands, attributes

    for arg_item in _split_angle_comma(args_str):
        if not arg_item:
            continue
        parts = arg_item.split(":$")
        if len(parts) == 2:
            arg_type, arg_name = parts[0].strip(), parts[1].strip()
        else:
            arg_parts = arg_item.split(":")
            arg_type = arg_parts[0].strip()
            arg_name = arg_parts[1].lstrip("$").strip() if len(arg_parts) > 1 else f"arg{len(params)}"
        _append_stablehlo_arg(arg_name, arg_type, params, operands, attributes)

    return params, operands, attributes


def _parse_stablehlo_td_regions(body: str, params: list[JsonValue]) -> list[JsonValue]:
    """Parse `let regions = (region ...);` from a TableGen `def` body.

    Args:
        body: Inner body of the TableGen `def` block.
        params: Mutable list of GhostRef parameter dictionaries.

    Returns:
        List of extracted region names.
    """
    regions: list[JsonValue] = []
    regions_match = re.search(r"let\s+regions\s*=\s*\(\s*region(.*?)\);", body, re.DOTALL)
    if regions_match:
        for reg_item in regions_match.group(1).strip().split(","):
            reg_item = reg_item.strip()
            if not reg_item:
                continue
            parts = reg_item.split(":$")
            reg_name = parts[1].strip() if len(parts) == 2 else reg_item
            _append_stablehlo_region(reg_name, regions, params)
    return regions


def _parse_stablehlo_td_results(
    res_str: str | None,
) -> tuple[list[JsonValue], str | None]:
    """Parse a TableGen `(outs ...)` string into result records and a return type string.

    Args:
        res_str: Inner `(outs ...)` declaration string or None.

    Returns:
        Tuple of `(results, returns_type)`.
    """
    results: list[JsonValue] = []
    res_types: list[str] = []
    if res_str:
        for res_item in _split_angle_comma(res_str):
            if not res_item:
                continue
            parts = res_item.split(":$")
            res_type = parts[0].strip()
            res_name = parts[1].strip() if len(parts) == 2 else "result"
            results.append({"name": res_name, "type": res_type})
            res_types.append(res_type)
    return results, _format_returns_type(res_types)


def _resolve_stablehlo_args_str(
    class_name: str,
    base_cls: str,
    targs_raw: str,
    body: str,
    base_classes: typing.Mapping[str, typing.Mapping[str, str | list[str] | None]],
) -> str | None:
    """Resolve the `(ins ...)` argument declaration string for a StableHLO operation.

    Args:
        class_name: Operation `def` identifier.
        base_cls: Immediate base class name.
        targs_raw: Raw template arguments string.
        body: Inner body of the `def` block.
        base_classes: Parsed StableHLO TableGen class registry.

    Returns:
        Resolved `(ins ...)` inner string or None.
    """
    args_match = re.search(r"let\s+arguments\s*=\s*\(\s*ins(.*?)\);", body, re.DOTALL) or re.search(
        r"Arguments<\s*\(\s*ins(.*?)\)>", targs_raw, re.DOTALL
    )
    args_str = (
        args_match.group(1).strip()
        if args_match
        else _resolve_stablehlo_inherited_field(base_classes, base_cls, "args")
    )
    if not args_str:
        combined = f"{base_cls} {class_name}"
        if "Binary" in combined:
            return "HLO_Tensor:$lhs, HLO_Tensor:$rhs"
        if "Unary" in combined or "Cast" in combined:
            return "HLO_Tensor:$operand"
    return args_str


def _resolve_stablehlo_results_str(
    base_cls: str,
    targs_raw: str,
    body: str,
    base_classes: typing.Mapping[str, typing.Mapping[str, str | list[str] | None]],
) -> str | None:
    """Resolve the `(outs ...)` result declaration string for a StableHLO operation.

    Args:
        base_cls: Immediate base class name.
        targs_raw: Raw template arguments string.
        body: Inner body of the `def` block.
        base_classes: Parsed StableHLO TableGen class registry.

    Returns:
        Resolved `(outs ...)` inner string or None.
    """
    results_match = re.search(r"let\s+results\s*=\s*\(\s*outs(.*?)\);", body, re.DOTALL) or re.search(
        r"Results<\s*\(\s*outs(.*?)\)>", targs_raw, re.DOTALL
    )
    res_str = (
        results_match.group(1).strip()
        if results_match
        else _resolve_stablehlo_inherited_field(base_classes, base_cls, "results")
    )
    if not res_str and any(k in base_cls for k in ("Binary", "Unary", "Cast")):
        return "HLO_Tensor:$result"
    return res_str


def _build_stablehlo_record(
    op_name: str,
    clean_class_name: str,
    parsed_args: tuple[list[JsonValue], list[JsonValue], list[JsonValue]],
    parsed_results: tuple[list[JsonValue], str | None],
    docstring: str | None,
    traits: list[JsonValue],
    regions: list[JsonValue],
) -> TableGenRecordModel:
    """Assemble a complete `TableGenRecordModel` for a StableHLO operation.

    Args:
        op_name: Unqualified StableHLO operation mnemonic.
        clean_class_name: Normalized C++ operation class name.
        parsed_args: Tuple of `(params, operands, attributes)` lists.
        parsed_results: Tuple of `(results, returns_type)`.
        docstring: Summary or description string or None.
        traits: List of operation trait strings.
        regions: List of operation region names.

    Returns:
        Populated `TableGenRecordModel` instance.
    """
    params, operands, attributes = parsed_args
    results, returns_type = parsed_results
    verification_rules, region_signatures = build_verification_and_region_metadata(op_name)
    return TableGenRecordModel(
        api_path=f"stablehlo.{op_name}",
        name=op_name,
        class_name=clean_class_name,
        kind="function",
        is_public=True,
        has_varargs=False,
        environment_tags=["cpu", "cuda", "rocm", "tpu"],
        aliases=[],
        overloads=[],
        params=[ensure_json_dict(p) for p in params],
        operands=[ensure_json_dict(o) for o in operands],
        attributes=[ensure_json_dict(a) for a in attributes],
        results=[ensure_json_dict(r) for r in results],
        returns_type=returns_type,
        returns_description=None,
        raises=[],
        docstring=docstring,
        traits=[str(t) for t in traits],
        regions=[ensure_json_dict(rg) if isinstance(rg, dict) else str(rg) for rg in regions],
        verification_rules=dict(verification_rules),
        region_signatures=dict(region_signatures),
    )


def parse_stablehlo_tablegen(content: str) -> list[TableGenRecordModel]:
    """Parse StableHLO TableGen / ODS definitions into structured operations.

    Args:
        content: Raw TableGen file content.

    Returns:
        List of structured GhostRef operation dictionaries.
    """
    clean_content = TableGenASTParser(content).expand()
    ghost_refs: list[TableGenRecordModel] = []
    base_classes = _extract_stablehlo_base_classes(clean_content)

    for m in re.finditer(r"\bdef\s+([A-Za-z0-9_]+Op)\s*:\s*([A-Za-z0-9_]+)", clean_content):
        class_name = m.group(1)
        base_cls = m.group(2)
        targs_raw, body = _scan_def_header_and_body(clean_content, m.end())

        op_name = _resolve_stablehlo_op_name(class_name, targs_raw, body)
        clean_class_name = class_name.replace("StableHLO_", "") if class_name.startswith("StableHLO_") else class_name

        traits: list[JsonValue] = []
        if "[" in targs_raw:
            traits.extend(extract_tablegen_traits(targs_raw))
        traits_in_body = re.search(r"let\s+traits\s*=\s*\[(.*?)\];", body, re.DOTALL)
        if traits_in_body:
            traits.extend(extract_tablegen_traits(traits_in_body.group(0)))

        args_str = _resolve_stablehlo_args_str(class_name, base_cls, targs_raw, body, base_classes)
        params, operands, attributes = _parse_stablehlo_td_args(args_str)
        regions = _parse_stablehlo_td_regions(body, params)

        res_str = _resolve_stablehlo_results_str(base_cls, targs_raw, body, base_classes)
        results, returns_type = _parse_stablehlo_td_results(res_str)

        summary_match = re.search(r'let\s+summary\s*=\s*"([^"]*)";', body)
        docstring = summary_match.group(1) if summary_match else None

        ghost_refs.append(
            _build_stablehlo_record(
                op_name,
                clean_class_name,
                (params, operands, attributes),
                (results, returns_type),
                docstring,
                traits,
                regions,
            )
        )

    return ghost_refs


def _parse_stablehlo_json_args(
    raw_args: JsonValue,
) -> tuple[list[JsonValue], list[JsonValue], list[JsonValue]]:
    """Extract `params`, `operands`, and `attributes` from a StableHLO TableGen JSON `arguments` field.

    Args:
        raw_args: Raw `arguments` field from `llvm-tblgen -dump-json`.

    Returns:
        Three-element tuple of `(params, operands, attributes)`.
    """
    params: list[JsonValue] = []
    operands: list[JsonValue] = []
    attributes: list[JsonValue] = []

    if isinstance(raw_args, dict) and raw_args.get("kind") == "dag":
        dag_args = raw_args.get("args", [])
        for arg in dag_args if isinstance(dag_args, list) else []:
            if isinstance(arg, (list, tuple)) and len(arg) >= 2:
                arg_type_obj, arg_name = arg[0], str(arg[1])
                arg_type = str(arg_type_obj.get("def", "Type")) if isinstance(arg_type_obj, dict) else str(arg_type_obj)
                _append_stablehlo_arg(arg_name, arg_type, params, operands, attributes)
    elif isinstance(raw_args, list):
        for arg in raw_args:
            aname = str(arg.get("name", "arg")) if isinstance(arg, dict) else str(arg)
            atype = str(arg.get("type", "Type")) if isinstance(arg, dict) else "Type"
            _append_stablehlo_arg(aname, atype, params, operands, attributes)

    return params, operands, attributes


def _parse_stablehlo_json_regions(
    raw_regions: JsonValue,
    params: list[JsonValue],
) -> list[JsonValue]:
    """Extract region names from a StableHLO TableGen JSON `regions` field.

    Args:
        raw_regions: Raw `regions` field from `llvm-tblgen -dump-json`.
        params: Mutable list of GhostRef parameter dictionaries.

    Returns:
        List of extracted region name strings.
    """
    regions: list[JsonValue] = []
    if isinstance(raw_regions, dict) and raw_regions.get("kind") == "dag":
        dag_regs = raw_regions.get("args", [])
        for reg in dag_regs if isinstance(dag_regs, list) else []:
            if isinstance(reg, (list, tuple)) and len(reg) >= 2:
                _append_stablehlo_region(str(reg[1]), regions, params)
    elif isinstance(raw_regions, list):
        for reg in raw_regions:
            _append_stablehlo_region(str(reg), regions, params)
    return regions


def _parse_stablehlo_json_results(
    raw_results: JsonValue,
) -> tuple[list[JsonValue], str | None]:
    """Extract result records and return type string from a StableHLO TableGen JSON `results` field.

    Args:
        raw_results: Raw `results` field from `llvm-tblgen -dump-json`.

    Returns:
        Tuple of `(results, returns_type)`.
    """
    results: list[JsonValue] = []
    res_types: list[str] = []
    if isinstance(raw_results, dict) and raw_results.get("kind") == "dag":
        dag_res = raw_results.get("args", [])
        for res in dag_res if isinstance(dag_res, list) else []:
            if isinstance(res, (list, tuple)) and len(res) >= 2:
                res_type_obj, res_name = res[0], str(res[1])
                res_type = str(res_type_obj.get("def", "Type")) if isinstance(res_type_obj, dict) else str(res_type_obj)
                results.append({"name": res_name or "result", "type": res_type})
                res_types.append(res_type)
    elif isinstance(raw_results, list):
        for res in raw_results:
            rname = str(res.get("name", "result")) if isinstance(res, dict) else "result"
            rtype = str(res.get("type", "Type")) if isinstance(res, dict) else "Type"
            results.append({"name": rname, "type": rtype})
            res_types.append(rtype)

    return results, _format_returns_type(res_types)


def parse_stablehlo_llvm_tblgen_json(
    json_data: str | typing.Mapping[str, JsonValue],
) -> list[TableGenRecordModel]:
    """Parse JSON dump from llvm-tblgen -dump-json for StableHLO into structured GhostRef operations.

    Args:
        json_data: Raw JSON string or dictionary loaded from llvm-tblgen -dump-json.

    Returns:
        List of structured GhostRef operation dictionaries.
    """
    data: typing.Mapping[str, JsonValue] = json.loads(json_data) if isinstance(json_data, str) else json_data

    ghost_refs: list[TableGenRecordModel] = []
    raw_instance_of = data.get("!instanceof", {})
    raw_op_defs = raw_instance_of.get("Op", []) if isinstance(raw_instance_of, dict) else []
    op_def_names = {str(x) for x in (raw_op_defs if isinstance(raw_op_defs, list) else [])}

    for rec_name, rec in data.items():
        if rec_name.startswith("!") or not isinstance(rec, dict):
            continue

        raw_supers = rec.get("!superclasses", [])
        superclasses = [str(s) for s in (raw_supers if isinstance(raw_supers, list) else [])]
        is_op = (
            rec_name in op_def_names
            or "StableHLO_Op" in superclasses
            or any("StableHLO_" in s for s in superclasses)
            or rec_name.startswith("StableHLO_")
            or rec_name.endswith("Op")
        )
        if not is_op:
            continue

        clean_class_name = rec_name.replace("StableHLO_", "")
        if not clean_class_name.endswith("Op"):
            clean_class_name = f"{clean_class_name}Op"

        mnemonic = rec.get("mnemonic") or _camel_to_snake_mnemonic(clean_class_name)
        op_name = str(mnemonic)

        params, operands, attributes = _parse_stablehlo_json_args(rec.get("arguments"))
        regions = _parse_stablehlo_json_regions(rec.get("regions"), params)
        results, returns_type = _parse_stablehlo_json_results(rec.get("results"))
        traits = _extract_tblgen_json_traits(rec.get("traits", []))
        docstring = rec.get("summary") or rec.get("description")

        ghost_refs.append(
            _build_stablehlo_record(
                op_name,
                clean_class_name,
                (params, operands, attributes),
                (results, returns_type),
                str(docstring) if docstring is not None else None,
                traits,
                regions,
            )
        )

    return ghost_refs


__all__ = [
    "TableGenASTParser",
    "build_verification_and_region_metadata",
    "is_attribute",
    "parse_stablehlo_llvm_tblgen_json",
    "parse_stablehlo_tablegen",
    "strip_tablegen_comments",
]
