"""TableGen `.td` lexer, AST parser, and ODS operation extraction utilities for MLIR."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import typing

from ml_ecosystem_snapshots.exceptions import SnapshotError
from ml_ecosystem_snapshots.models import JsonValue, TableGenRecordModel, ensure_json_dict
from ml_ecosystem_snapshots.utils import extract_tablegen_traits


def strip_tablegen_comments(text: str) -> str:
    """Strip block and line comments from TableGen source cleanly.

    Args:
        text: Raw TableGen source code.

    Returns:
        Source code with comments removed.
    """
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*", "", text)
    return text


class TableGenASTParser:
    """TableGen AST parser resolving multiclasses, defm instances, foreach loops, and concatenations."""

    def __init__(self, content: str) -> None:
        """Initialize TableGen AST parser with source content.

        Args:
            content: Raw TableGen source string.
        """
        self.raw_content = content
        self.content = strip_tablegen_comments(content)
        self.multiclasses: dict[str, dict[str, list[str] | str]] = {}

    def _extract_multiclasses(self, text: str) -> str:
        """Extract multiclass definitions and replace them with whitespace.

        Args:
            text: TableGen source code.

        Returns:
            TableGen source with multiclasses extracted into internal registry.
        """
        pattern = re.compile(r"\bmulticlass\s+([A-Za-z0-9_]+)(?:<([^>]*)>)?\s*\{", re.DOTALL)
        pos = 0
        chunks: list[str] = []
        while True:
            m = pattern.search(text, pos)
            if not m:
                chunks.append(text[pos:])
                break
            chunks.append(text[pos : m.start()])
            name = m.group(1)
            raw_params = m.group(2) or ""
            params = [p.strip() for p in raw_params.split(",") if p.strip()]

            start_idx = m.end()
            depth = 1
            i = start_idx
            while i < len(text) and depth > 0:
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                i += 1
            body = text[start_idx : i - 1]
            self.multiclasses[name] = {"params": params, "body": body}
            pos = i
        return "".join(chunks)

    def _expand_foreach(self, text: str) -> str:
        """Expand foreach loops into unrolled blocks.

        Args:
            text: TableGen source code.

        Returns:
            TableGen source with foreach loops unrolled.
        """
        nl = chr(10)
        pattern = re.compile(r"\bforeach\s+([A-Za-z0-9_]+)\s*=\s*\[(.*?)\]\s*in\s*\{", re.DOTALL)
        pos = 0
        chunks: list[str] = []
        while True:
            m = pattern.search(text, pos)
            if not m:
                chunks.append(text[pos:])
                break
            chunks.append(text[pos : m.start()])
            var_name = m.group(1)
            raw_items = m.group(2)
            items = [item.strip().strip('"').strip("'") for item in raw_items.split(",") if item.strip()]

            start_idx = m.end()
            depth = 1
            i = start_idx
            while i < len(text) and depth > 0:
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                i += 1
            body = text[start_idx : i - 1]

            unrolled_parts: list[str] = []
            for item in items:
                sub_body = re.sub(rf"\b{var_name}\b", item, body)
                unrolled_parts.append(sub_body)
            chunks.append(nl.join(unrolled_parts))
            pos = i
        return "".join(chunks)

    def _expand_defm(self, text: str) -> str:
        """Expand defm declarations into instantiated def blocks.

        Args:
            text: TableGen source code.

        Returns:
            TableGen source with defm declarations instantiated.
        """
        pattern = re.compile(
            r'\bdefm\s+([A-Za-z0-9_"]*)\s*:\s*([A-Za-z0-9_]+)(?:<([^>]*)>)?\s*;',
            re.DOTALL,
        )
        pos = 0
        chunks: list[str] = []
        while True:
            m = pattern.search(text, pos)
            if not m:
                chunks.append(text[pos:])
                break
            chunks.append(text[pos : m.start()])
            inst_name = (m.group(1) or "").strip().strip('"')
            mc_name = m.group(2)
            raw_args = m.group(3) or ""
            args = [a.strip().strip('"').strip("'") for a in raw_args.split(",") if a.strip()]

            if mc_name in self.multiclasses:
                mc_info = self.multiclasses[mc_name]
                mc_body = str(mc_info["body"])
                raw_mc_params = mc_info["params"]
                mc_params = raw_mc_params if isinstance(raw_mc_params, list) else []

                inst_body = mc_body
                for p_idx, p_name in enumerate(mc_params):
                    clean_p = p_name.split()[-1]
                    val = args[p_idx] if p_idx < len(args) else ""
                    inst_body = re.sub(rf"\b{clean_p}\b", val, inst_body)

                inst_body = re.sub(r"\bNAME\b", inst_name, inst_body)
                chunks.append(inst_body)
            else:
                chunks.append(m.group(0))
            pos = m.end()
        return "".join(chunks)

    def _resolve_concatenations(self, text: str) -> str:
        """Resolve '#' string concatenation operators.

        Args:
            text: TableGen source code.

        Returns:
            TableGen source with string concatenations resolved.
        """
        prev = None
        curr = text
        while prev != curr:
            prev = curr
            curr = re.sub(r'"([^"]*)"\s*#\s*"([^"]*)"', r'"\1\2"', curr)
            curr = re.sub(
                r'([A-Za-z0-9_]+)\s*#\s*"([^"]*)"',
                lambda m: str(m.group(1)) + str(m.group(2)),
                curr,
            )
            curr = re.sub(
                r'"([^"]*)"\s*#\s*([A-Za-z0-9_]+)',
                lambda m: str(m.group(1)) + str(m.group(2)),
                curr,
            )
            curr = re.sub(
                r"([A-Za-z0-9_]+)\s*#\s*([A-Za-z0-9_]+)",
                lambda m: str(m.group(1)) + str(m.group(2)),
                curr,
            )
        return curr

    def expand(self) -> str:
        """Expand all multiclasses, defm instances, foreach loops, and concatenations.

        Returns:
            Fully expanded TableGen source code.
        """
        expanded = self._extract_multiclasses(self.content)
        expanded = self._expand_foreach(expanded)
        expanded = self._resolve_concatenations(expanded)
        expanded = self._expand_defm(expanded)
        expanded = self._resolve_concatenations(expanded)
        expanded = re.sub(r'\bdef\s+"([^"]+)"', r"def \1", expanded)
        return expanded


_MLIR_ATTR_KEYWORDS: tuple[str, ...] = (
    "attr",
    "enum",
    "constant",
    "dictionary",
    "string",
    "bool",
    "i64",
    "i32",
    "f32",
    "dimension",
)


def _split_top_level_comma(raw_str: str) -> list[str]:
    """Split a TableGen declaration string on top-level commas outside `<>` and `()`.

    Args:
        raw_str: Inner TableGen declaration string from `(ins ...)` or `(outs ...)`.

    Returns:
        List of trimmed comma-separated declaration segments.
    """
    depth = 0
    parts: list[str] = []
    curr: list[str] = []
    for c in raw_str:
        if c in "<(":
            depth += 1
        elif c in ">)":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append("".join(curr).strip())
            curr = []
            continue
        curr.append(c)
    if curr:
        parts.append("".join(curr).strip())
    return parts


def _parse_type_and_name(item: str, default_name: str) -> tuple[str, str]:
    """Extract the TableGen type and identifier from a single declaration segment.

    Args:
        item: Trimmed segment such as `Type:$lhs`, `Type:lhs`, or `Type`.
        default_name: Fallback name used when no identifier is present or empty.

    Returns:
        Tuple of `(type_str, name_str)`.
    """
    if ":$" in item:
        raw_type, raw_name = item.split(":$", 1)
    elif ":" in item:
        raw_type, raw_name = item.split(":", 1)
        raw_name = raw_name.lstrip("$").strip()
    else:
        raw_type, raw_name = item, default_name
    clean_name = raw_name.strip() or default_name
    return raw_type.strip(), clean_name


def _is_mlir_attribute_type(type_str: str) -> bool:
    """Return whether a TableGen argument type represents an MLIR attribute.

    Args:
        type_str: TableGen type annotation string.

    Returns:
        True if the type matches known MLIR attribute keywords, False otherwise.
    """
    lowered = type_str.lower()
    return any(k in lowered for k in _MLIR_ATTR_KEYWORDS)


def parse_tablegen_args(
    args_str: str | None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Parse TableGen (ins ...) argument declarations into operands and attributes.

    Args:
        args_str: The inner argument string from (ins ...).

    Returns:
        Tuple of (operands list, attributes list).
    """
    operands: list[dict[str, str]] = []
    attributes: list[dict[str, str]] = []
    if not args_str:
        return operands, attributes

    for part in _split_top_level_comma(args_str):
        if not part:
            continue
        default_name = f"arg{len(operands) + len(attributes)}"
        atype, aname = _parse_type_and_name(part, default_name)
        if _is_mlir_attribute_type(atype):
            attributes.append({"name": aname, "type": atype})
        else:
            operands.append({"name": aname, "type": atype})

    return operands, attributes


def parse_tablegen_results(res_str: str | None) -> list[dict[str, str]]:
    """Parse TableGen (outs ...) result declarations into structured results.

    Args:
        res_str: The inner result string from (outs ...).

    Returns:
        List of result dictionaries containing name and type.
    """
    results: list[dict[str, str]] = []
    if not res_str:
        return results

    for part in _split_top_level_comma(res_str):
        if not part:
            continue
        rtype, rname = _parse_type_and_name(part, "result")
        results.append({"name": rname, "type": rtype})

    return results


def parse_tablegen_regions(reg_str: str | None) -> list[str]:
    """Parse TableGen (region ...) declarations into region names.

    Args:
        reg_str: The inner region string from (region ...).

    Returns:
        List of region name strings.
    """
    regions: list[str] = []
    if not reg_str:
        return regions

    for p in reg_str.split(","):
        p = p.strip()
        if not p:
            continue
        if ":$" in p:
            rname = p.split(":$")[1].strip()
        elif ":" in p:
            rname = p.split(":")[1].lstrip("$").strip()
        else:
            rname = p
        regions.append(rname)

    return regions


def _extract_brace_block(text: str, start_idx: int) -> tuple[str, int]:
    """Extract the balanced `{ ... }` block starting after an opening brace.

    Args:
        text: Full source text being scanned.
        start_idx: Index immediately following the opening `{`.

    Returns:
        Tuple of `(block_body, end_index_after_closing_brace)`.
    """
    depth = 1
    pos = start_idx
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1
    return text[start_idx : pos - 1], pos


def _extract_tablegen_classes(
    clean_content: str,
) -> dict[str, dict[str, str | list[str] | None]]:
    """Parse TableGen `class` declarations into a lookup table of fields and superclasses.

    Args:
        clean_content: Macro-expanded TableGen source text.

    Returns:
        Mapping from class name to dictionary of `args`, `res`, `reg`, and `supers`.
    """
    classes: dict[str, dict[str, str | list[str] | None]] = {}
    for m in re.finditer(
        r"\bclass\s+([A-Za-z0-9_]+)(?:<.*?>)?\s*(?::\s*([^{;]+))?\s*(\{|;)",
        clean_content,
    ):
        cname = m.group(1)
        supers_raw = m.group(2) or ""
        term = m.group(3)
        body, _ = _extract_brace_block(clean_content, m.end()) if term == "{" else ("", m.end())

        args_m = re.search(r"let\s+arguments\s*=\s*\(ins(.*?)\);", body, re.DOTALL) or re.search(
            r"Arguments<\s*\(ins(.*?)\)>", supers_raw, re.DOTALL
        )
        res_m = re.search(r"let\s+results\s*=\s*\(outs(.*?)\);", body, re.DOTALL) or re.search(
            r"Results<\s*\(outs(.*?)\)>", supers_raw, re.DOTALL
        )
        reg_m = re.search(r"let\s+regions\s*=\s*\(region(.*?)\);", body, re.DOTALL)

        supers = [s.split("<")[0].strip() for s in re.split(r",", supers_raw) if s.strip()]
        classes[cname] = {
            "args": args_m.group(1).strip() if args_m else None,
            "res": res_m.group(1).strip() if res_m else None,
            "reg": reg_m.group(1).strip() if reg_m else None,
            "supers": supers,
        }
    return classes


def _resolve_inherited_class_field(
    classes: typing.Mapping[str, typing.Mapping[str, str | list[str] | None]],
    cls_name: str,
    field_name: str,
) -> str | None:
    """Recursively resolve an inherited TableGen field from base classes.

    Args:
        classes: Parsed TableGen class registry.
        cls_name: Name of the class to inspect.
        field_name: Field key to lookup (`'args'`, `'res'`, or `'reg'`).

    Returns:
        Resolved field string value or None if not defined in the hierarchy.
    """
    if cls_name not in classes:
        return None
    val = classes[cls_name].get(field_name)
    if val is not None:
        return str(val)
    raw_supers = classes[cls_name]["supers"]
    for s in raw_supers if isinstance(raw_supers, list) else []:
        res = _resolve_inherited_class_field(classes, s, field_name)
        if res is not None:
            return str(res)
    return None


def _resolve_mlir_op_mnemonic(clean_class: str, targs: str, body: str) -> str:
    """Resolve the MLIR operation mnemonic from body, template arguments, or class name.

    Args:
        clean_class: Unprefixed operation class name.
        targs: Raw template arguments string from the `def` header.
        body: Inner body of the `def` block.

    Returns:
        Resolved lowercase operation mnemonic.
    """
    mnem_m = re.search(r'let\s+mnemonic\s*=\s*"([a-z0-9_.]+)"', body)
    if mnem_m:
        return mnem_m.group(1)
    first_targ = targs.split(",")[0].strip().strip('"') if targs else clean_class.lower()
    if first_targ and not first_targ.startswith("["):
        return first_targ
    return clean_class.lower()


def _resolve_mlir_args_str(
    raw_class: str,
    base_cls: str,
    extra_supers: str,
    body: str,
    classes: typing.Mapping[str, typing.Mapping[str, str | list[str] | None]],
) -> str | None:
    """Resolve the `(ins ...)` argument declaration string for an MLIR operation.

    Args:
        raw_class: Raw TableGen `def` identifier.
        base_cls: Immediate base class name.
        extra_supers: Additional superclass clause string.
        body: Inner body of the `def` block.
        classes: Parsed TableGen class registry.

    Returns:
        Resolved `(ins ...)` inner string or None.
    """
    args_match = re.search(r"let\s+arguments\s*=\s*\(ins(.*?)\);", body, re.DOTALL) or re.search(
        r"Arguments<\s*\(ins(.*?)\)>", extra_supers, re.DOTALL
    )
    args_str = args_match.group(1).strip() if args_match else _resolve_inherited_class_field(classes, base_cls, "args")
    if not args_str:
        combined = f"{base_cls} {raw_class}"
        if "Binary" in combined:
            return "Type:$lhs, Type:$rhs"
        if "Unary" in combined or "Cast" in combined:
            return "Type:$in"
    return args_str


def _resolve_mlir_results_str(
    base_cls: str,
    extra_supers: str,
    body: str,
    classes: typing.Mapping[str, typing.Mapping[str, str | list[str] | None]],
) -> str | None:
    """Resolve the `(outs ...)` result declaration string for an MLIR operation.

    Args:
        base_cls: Immediate base class name.
        extra_supers: Additional superclass clause string.
        body: Inner body of the `def` block.
        classes: Parsed TableGen class registry.

    Returns:
        Resolved `(outs ...)` inner string or None.
    """
    results_match = re.search(r"let\s+results\s*=\s*\(outs(.*?)\);", body, re.DOTALL) or re.search(
        r"Results<\s*\(outs(.*?)\)>", extra_supers, re.DOTALL
    )
    res_str = (
        results_match.group(1).strip() if results_match else _resolve_inherited_class_field(classes, base_cls, "res")
    )
    if not res_str and any(k in base_cls for k in ("Binary", "Unary", "Cast")):
        return "Type:$result"
    return res_str


def _apply_segment_trait(
    trait_name: str,
    attr_name: str,
    traits: list[str],
    attributes: list[dict[str, str]],
    base_cls: str,
    extra_supers: str,
) -> None:
    """Synthesize implicit segment size attributes and traits when variadic traits are present.

    Args:
        trait_name: Variadic segment trait identifier (e.g. `'AttrSizedOperandSegments'`).
        attr_name: Corresponding dense i32 array attribute name.
        traits: Mutable list of operation traits.
        attributes: Mutable list of operation attribute dictionaries.
        base_cls: Immediate base class name.
        extra_supers: Additional superclass clause string.
    """
    has_trait = any(trait_name in t for t in traits) or trait_name in base_cls or trait_name in extra_supers
    if has_trait:
        if not any(a.get("name") == attr_name for a in attributes):
            attributes.append({"name": attr_name, "type": "DenseI32ArrayAttr"})
        if trait_name not in traits:
            traits.append(trait_name)


def parse_mlir_tablegen(content: str, dialect_name: str) -> list[TableGenRecordModel]:
    """Parse MLIR TableGen / ODS definitions into structured operations.

    Args:
        content: Raw TableGen file content.
        dialect_name: Name of the dialect (e.g. 'arith', 'linalg').

    Returns:
        List of structured operation dictionaries.
    """
    clean_content = TableGenASTParser(content).expand()
    ops: list[TableGenRecordModel] = []
    classes = _extract_tablegen_classes(clean_content)

    pattern = re.compile(
        r"\bdef\s+([A-Za-z0-9_]+)\s*:\s*([A-Za-z0-9_]+)(?:<([^>]*)>)?(?:\s*,\s*([^{;]+))?\s*(\{|;)",
        re.DOTALL,
    )

    pos = 0
    while True:
        match = pattern.search(clean_content, pos)
        if not match:
            break
        raw_class = match.group(1)
        base_cls = match.group(2)
        targs = match.group(3) or ""
        extra_supers = match.group(4) or ""
        term = match.group(5)

        body, pos = _extract_brace_block(clean_content, match.end()) if term == "{" else ("", match.end())

        clean_class = raw_class.split("_")[-1]
        class_name = f"{clean_class}Op" if not clean_class.endswith("Op") else clean_class
        op_name = _resolve_mlir_op_mnemonic(clean_class, targs, body)

        traits: list[str] = []
        if "[" in targs:
            traits.extend(extract_tablegen_traits(targs))
        body_traits = re.search(r"let\s+traits\s*=\s*\[(.*?)\];", body, re.DOTALL)
        if body_traits:
            traits.extend(extract_tablegen_traits(body_traits.group(0)))

        args_str = _resolve_mlir_args_str(raw_class, base_cls, extra_supers, body, classes)
        operands, attributes = parse_tablegen_args(args_str)

        res_str = _resolve_mlir_results_str(base_cls, extra_supers, body, classes)
        results = parse_tablegen_results(res_str)

        regions_match = re.search(r"let\s+regions\s*=\s*\(region(.*?)\);", body, re.DOTALL)
        reg_str = (
            regions_match.group(1).strip()
            if regions_match
            else _resolve_inherited_class_field(classes, base_cls, "reg")
        )
        regions = parse_tablegen_regions(reg_str)

        summary_match = re.search(r'let\s+summary\s*=\s*"([^"]*)";', body)
        description = summary_match.group(1) if summary_match else None

        _apply_segment_trait(
            "AttrSizedOperandSegments", "operand_segment_sizes", traits, attributes, base_cls, extra_supers
        )
        _apply_segment_trait(
            "AttrSizedResultSegments", "result_segment_sizes", traits, attributes, base_cls, extra_supers
        )

        ops.append(
            TableGenRecordModel(
                api_path=f"{dialect_name}.{op_name}",
                dialect=dialect_name,
                class_name=class_name,
                operands=[dict(o) for o in operands],
                attributes=[dict(a) for a in attributes],
                results=[dict(r) for r in results],
                regions=list(regions),
                traits=list(traits),
                description=description,
            )
        )

    return ops


def _extract_tblgen_json_args(
    raw_args: JsonValue,
) -> tuple[list[JsonValue], list[JsonValue]]:
    """Extract operands and attributes from an `llvm-tblgen -dump-json` `arguments` field.

    Args:
        raw_args: Raw `arguments` field value from a TableGen JSON record.

    Returns:
        Tuple of `(operands, attributes)` lists.
    """
    operands: list[JsonValue] = []
    attributes: list[JsonValue] = []

    if isinstance(raw_args, dict) and raw_args.get("kind") == "dag":
        dag_args = raw_args.get("args", [])
        for arg in dag_args if isinstance(dag_args, list) else []:
            if isinstance(arg, (list, tuple)) and len(arg) >= 2:
                arg_type_obj, arg_name = arg[0], str(arg[1])
                arg_type = str(arg_type_obj.get("def", "Type")) if isinstance(arg_type_obj, dict) else str(arg_type_obj)
                target = attributes if _is_mlir_attribute_type(arg_type) else operands
                target.append({"name": arg_name, "type": arg_type})
    elif isinstance(raw_args, list):
        for arg in raw_args:
            aname = str(arg.get("name", "arg")) if isinstance(arg, dict) else str(arg)
            atype = str(arg.get("type", "Type")) if isinstance(arg, dict) else "Type"
            target = attributes if _is_mlir_attribute_type(atype) else operands
            target.append({"name": aname, "type": atype})

    return operands, attributes


def _extract_tblgen_json_results(raw_results: JsonValue) -> list[JsonValue]:
    """Extract result records from an `llvm-tblgen -dump-json` `results` field.

    Args:
        raw_results: Raw `results` field value from a TableGen JSON record.

    Returns:
        List of structured result dictionaries.
    """
    results: list[JsonValue] = []
    if isinstance(raw_results, dict) and raw_results.get("kind") == "dag":
        dag_res = raw_results.get("args", [])
        for res in dag_res if isinstance(dag_res, list) else []:
            if isinstance(res, (list, tuple)) and len(res) >= 2:
                res_type_obj, res_name = res[0], str(res[1])
                res_type = str(res_type_obj.get("def", "Type")) if isinstance(res_type_obj, dict) else str(res_type_obj)
                results.append({"name": res_name or "result", "type": res_type})
    elif isinstance(raw_results, list):
        for res in raw_results:
            rname = str(res.get("name", "result")) if isinstance(res, dict) else "result"
            rtype = str(res.get("type", "Type")) if isinstance(res, dict) else "Type"
            results.append({"name": rname, "type": rtype})
    return results


def _extract_tblgen_json_regions(raw_regions: JsonValue) -> list[JsonValue]:
    """Extract region names from an `llvm-tblgen -dump-json` `regions` field.

    Args:
        raw_regions: Raw `regions` field value from a TableGen JSON record.

    Returns:
        List of region name strings.
    """
    regions: list[JsonValue] = []
    if isinstance(raw_regions, dict) and raw_regions.get("kind") == "dag":
        dag_regs = raw_regions.get("args", [])
        for reg in dag_regs if isinstance(dag_regs, list) else []:
            if isinstance(reg, (list, tuple)) and len(reg) >= 2:
                regions.append(str(reg[1]))
    elif isinstance(raw_regions, list):
        regions.extend(str(r) for r in raw_regions)
    return regions


def _extract_tblgen_json_traits(raw_traits: JsonValue) -> list[JsonValue]:
    """Extract trait identifiers from an `llvm-tblgen -dump-json` `traits` field.

    Args:
        raw_traits: Raw `traits` field value from a TableGen JSON record.

    Returns:
        List of non-empty trait names.
    """
    traits: list[JsonValue] = []
    if isinstance(raw_traits, list):
        for t in raw_traits:
            t_name = str(t.get("def", "")) if isinstance(t, dict) else str(t)
            if t_name:
                traits.append(t_name)
    return traits


def parse_llvm_tblgen_json(
    json_data: str | typing.Mapping[str, JsonValue], dialect_name: str
) -> list[TableGenRecordModel]:
    """Parse JSON dump from llvm-tblgen -dump-json into structured MLIR operations.

    Args:
        json_data: Raw JSON string or dictionary loaded from llvm-tblgen -dump-json.
        dialect_name: Target dialect identifier (e.g. 'arith', 'math').

    Returns:
        List of structured operation dictionaries conforming to snapshot schema.
    """
    data: typing.Mapping[str, JsonValue] = json.loads(json_data) if isinstance(json_data, str) else json_data

    ops: list[TableGenRecordModel] = []
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
            or "Op" in superclasses
            or any(s.endswith("_Op") or s == "Op" for s in superclasses)
            or rec_name.endswith("Op")
            or "mnemonic" in rec
        )
        if not is_op:
            continue

        clean_class = rec_name.split("_")[-1]
        class_name = clean_class if clean_class.endswith("Op") else f"{clean_class}Op"
        mnemonic = rec.get("mnemonic") or (
            clean_class[:-2].lower() if clean_class.endswith("Op") else clean_class.lower()
        )

        operands, attributes = _extract_tblgen_json_args(rec.get("arguments"))
        results = _extract_tblgen_json_results(rec.get("results"))
        regions = _extract_tblgen_json_regions(rec.get("regions"))
        traits = _extract_tblgen_json_traits(rec.get("traits", []))

        description = rec.get("description") or rec.get("summary")

        ops.append(
            TableGenRecordModel(
                api_path=f"{dialect_name}.{mnemonic}",
                dialect=dialect_name,
                class_name=class_name,
                operands=[ensure_json_dict(o) for o in operands],
                attributes=[ensure_json_dict(a) for a in attributes],
                results=[ensure_json_dict(r) for r in results],
                regions=[ensure_json_dict(rg) if isinstance(rg, dict) else str(rg) for rg in regions],
                traits=[str(t) for t in traits],
                description=str(description) if description is not None else None,
            )
        )

    return ops


def dump_ast_via_llvm_tooling(
    td_path: str, dialect_name: str, tool_binary: str | None = None
) -> list[TableGenRecordModel] | None:
    """Dump AST using host llvm-tblgen or mlir-tblgen binary when available.

    Args:
        td_path: Path to the TableGen .td file.
        dialect_name: Dialect identifier (e.g. 'arith', 'math').
        tool_binary: Optional explicit path to TableGen executable.

    Returns:
        List of parsed operations if successful, or None if tooling unavailable or fails.
    """
    binary = tool_binary or shutil.which("llvm-tblgen") or shutil.which("mlir-tblgen")
    if not binary:
        return None

    try:
        proc = subprocess.run(
            [binary, "--dump-json", td_path],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return parse_llvm_tblgen_json(proc.stdout, dialect_name)
    except (
        SnapshotError,
        ImportError,
        AttributeError,
        TypeError,
        ValueError,
        LookupError,
        OSError,
        RuntimeError,
        SyntaxError,
        subprocess.SubprocessError,
    ) as e:
        print(f"Error running TableGen binary {binary}: {e}")
    return None


def validate_dialect_ops_ods(ops: typing.Sequence[typing.Mapping[str, JsonValue]], dialect_name: str) -> list[str]:
    """Validate extracted dialect operations against official ODS structural rules.

    Args:
        ops: List of parsed operation dictionaries.
        dialect_name: Dialect identifier (e.g. 'arith', 'math').

    Returns:
        List of validation error descriptions.
    """
    errors: list[str] = []
    prefix = f"{dialect_name}."
    for op in ops:
        path = str(op.get("api_path", ""))
        if not path.startswith(prefix):
            errors.append(f"Operation '{path}' does not match dialect prefix '{prefix}'.")
        if not op.get("class_name"):
            errors.append(f"Operation '{path}' missing class_name.")
        if "operands" not in op or not isinstance(op["operands"], list):
            errors.append(f"Operation '{path}' missing valid operands list.")
        if "attributes" not in op or not isinstance(op["attributes"], list):
            errors.append(f"Operation '{path}' missing valid attributes list.")
        if "results" not in op or not isinstance(op["results"], list):
            errors.append(f"Operation '{path}' missing valid results list.")
    return errors


__all__ = [
    "TableGenASTParser",
    "dump_ast_via_llvm_tooling",
    "parse_llvm_tblgen_json",
    "parse_mlir_tablegen",
    "parse_tablegen_args",
    "parse_tablegen_regions",
    "parse_tablegen_results",
    "strip_tablegen_comments",
    "validate_dialect_ops_ods",
]
