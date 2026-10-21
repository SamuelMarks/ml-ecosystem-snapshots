"""TableGen `.td` lexer, AST parser, and ODS operation extraction utilities for MLIR."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import typing

from ml_ecosystem_snapshots.models import JsonValue, TableGenRecordModel
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
        pattern = re.compile(
            r"\bmulticlass\s+([A-Za-z0-9_]+)(?:<([^>]*)>)?\s*\{", re.DOTALL
        )
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
        pattern = re.compile(
            r"\bforeach\s+([A-Za-z0-9_]+)\s*=\s*\[(.*?)\]\s*in\s*\{", re.DOTALL
        )
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
            items = [
                item.strip().strip('"').strip("'")
                for item in raw_items.split(",")
                if item.strip()
            ]

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
            args = [
                a.strip().strip('"').strip("'")
                for a in raw_args.split(",")
                if a.strip()
            ]

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

    depth = 0
    parts: list[str] = []
    curr: list[str] = []
    for c in args_str:
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

    for p in parts:
        p = p.strip()
        if not p:
            continue
        if ":$" in p:
            atype, aname = p.split(":$", 1)
        elif ":" in p:
            atype, aname = p.split(":", 1)
            aname = aname.lstrip("$").strip()
        else:
            atype, aname = p, f"arg{len(operands) + len(attributes)}"

        atype = atype.strip()
        aname = aname.strip()
        if not aname:
            aname = f"arg{len(operands) + len(attributes)}"

        is_attr = any(
            k in atype.lower()
            for k in (
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
        )
        if is_attr:
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

    depth = 0
    parts: list[str] = []
    curr: list[str] = []
    for c in res_str:
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

    for p in parts:
        p = p.strip()
        if not p:
            continue
        if ":$" in p:
            rtype, rname = p.split(":$", 1)
        elif ":" in p:
            rtype, rname = p.split(":", 1)
            rname = rname.lstrip("$").strip()
        else:
            rtype, rname = p, "result"
        results.append({"name": rname.strip() or "result", "type": rtype.strip()})

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

    classes: dict[str, dict[str, str | list[str] | None]] = {}
    for m in re.finditer(
        r"\bclass\s+([A-Za-z0-9_]+)(?:<.*?>)?\s*(?::\s*([^{;]+))?\s*(\{|;)",
        clean_content,
    ):
        cname = m.group(1)
        supers_raw = m.group(2) or ""
        term = m.group(3)
        body = ""
        if term == "{":
            start = m.end()
            depth = 1
            pos = start
            while pos < len(clean_content) and depth > 0:
                if clean_content[pos] == "{":
                    depth += 1
                elif clean_content[pos] == "}":
                    depth -= 1
                pos += 1
            body = clean_content[start : pos - 1]

        args_m = re.search(
            r"let\s+arguments\s*=\s*\(ins(.*?)\);", body, re.DOTALL
        ) or re.search(r"Arguments<\s*\(ins(.*?)\)>", supers_raw, re.DOTALL)
        res_m = re.search(
            r"let\s+results\s*=\s*\(outs(.*?)\);", body, re.DOTALL
        ) or re.search(r"Results<\s*\(outs(.*?)\)>", supers_raw, re.DOTALL)
        reg_m = re.search(r"let\s+regions\s*=\s*\(region(.*?)\);", body, re.DOTALL)

        supers = [
            s.split("<")[0].strip() for s in re.split(r",", supers_raw) if s.strip()
        ]
        classes[cname] = {
            "args": args_m.group(1).strip() if args_m else None,
            "res": res_m.group(1).strip() if res_m else None,
            "reg": reg_m.group(1).strip() if reg_m else None,
            "supers": supers,
        }

    def resolve_field(cls_name: str, field_name: str) -> str | None:
        """Recursively resolve inherited TableGen fields from base classes.

        Args:
            cls_name: Name of the class to resolve.
            field_name: Field key to lookup ('args', 'res', 'reg').

        Returns:
            Resolved field string value or None.
        """
        if cls_name not in classes:
            return None
        val = classes[cls_name].get(field_name)
        if val is not None:
            return str(val)
        raw_supers = classes[cls_name]["supers"]
        for s in raw_supers if isinstance(raw_supers, list) else []:
            res = resolve_field(s, field_name)
            if res is not None:
                return str(res)
        return None

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

        body = ""
        if term == "{":
            start_idx = match.end()
            depth = 1
            i = start_idx
            while i < len(clean_content) and depth > 0:
                if clean_content[i] == "{":
                    depth += 1
                elif clean_content[i] == "}":
                    depth -= 1
                i += 1
            body = clean_content[start_idx : i - 1]
            pos = i
        else:
            pos = match.end()

        clean_class = raw_class.split("_")[-1]
        class_name = (
            f"{clean_class}Op" if not clean_class.endswith("Op") else clean_class
        )

        mnem_m = re.search(r'let\s+mnemonic\s*=\s*"([a-z0-9_.]+)"', body)
        if mnem_m:
            op_name = mnem_m.group(1)
        else:
            first_targ = (
                targs.split(",")[0].strip().strip('"') if targs else clean_class.lower()
            )
            if first_targ and not first_targ.startswith("["):
                op_name = first_targ
            else:
                op_name = clean_class.lower()

        traits: list[str] = []
        if "[" in targs:
            traits.extend(extract_tablegen_traits(targs))
        body_traits = re.search(r"let\s+traits\s*=\s*\[(.*?)\];", body, re.DOTALL)
        if body_traits:
            traits.extend(extract_tablegen_traits(body_traits.group(0)))

        args_match = re.search(
            r"let\s+arguments\s*=\s*\(ins(.*?)\);", body, re.DOTALL
        ) or re.search(r"Arguments<\s*\(ins(.*?)\)>", extra_supers, re.DOTALL)
        args_str = args_match.group(1).strip() if args_match else None
        if not args_str:
            args_str = resolve_field(base_cls, "args")

        if not args_str and ("Binary" in base_cls or "Binary" in raw_class):
            args_str = "Type:$lhs, Type:$rhs"
        elif not args_str and ("Unary" in base_cls or "Unary" in raw_class):
            args_str = "Type:$in"
        elif not args_str and ("Cast" in base_cls or "Cast" in raw_class):
            args_str = "Type:$in"

        operands, attributes = parse_tablegen_args(args_str)

        results_match = re.search(
            r"let\s+results\s*=\s*\(outs(.*?)\);", body, re.DOTALL
        ) or re.search(r"Results<\s*\(outs(.*?)\)>", extra_supers, re.DOTALL)
        res_str = results_match.group(1).strip() if results_match else None
        if not res_str:
            res_str = resolve_field(base_cls, "res")

        if not res_str and (
            "Binary" in base_cls or "Unary" in base_cls or "Cast" in base_cls
        ):
            res_str = "Type:$result"

        results = parse_tablegen_results(res_str)

        regions_match = re.search(
            r"let\s+regions\s*=\s*\(region(.*?)\);", body, re.DOTALL
        )
        reg_str = regions_match.group(1).strip() if regions_match else None
        if not reg_str:
            reg_str = resolve_field(base_cls, "reg")

        regions = parse_tablegen_regions(reg_str)

        summary_match = re.search(r'let\s+summary\s*=\s*"([^"]*)";', body)
        description = summary_match.group(1) if summary_match else None

        if (
            any("AttrSizedOperandSegments" in t for t in traits)
            or "AttrSizedOperandSegments" in base_cls
            or "AttrSizedOperandSegments" in extra_supers
        ):
            if not any(a.get("name") == "operand_segment_sizes" for a in attributes):
                attributes.append(
                    {"name": "operand_segment_sizes", "type": "DenseI32ArrayAttr"}
                )
            if "AttrSizedOperandSegments" not in traits:
                traits.append("AttrSizedOperandSegments")

        if (
            any("AttrSizedResultSegments" in t for t in traits)
            or "AttrSizedResultSegments" in base_cls
            or "AttrSizedResultSegments" in extra_supers
        ):
            if not any(a.get("name") == "result_segment_sizes" for a in attributes):
                attributes.append(
                    {"name": "result_segment_sizes", "type": "DenseI32ArrayAttr"}
                )
            if "AttrSizedResultSegments" not in traits:
                traits.append("AttrSizedResultSegments")

        ops.append(
            TableGenRecordModel(
                {
                    "api_path": f"{dialect_name}.{op_name}",
                    "dialect": dialect_name,
                    "class_name": class_name,
                    "operands": [dict(o) for o in operands],
                    "attributes": [dict(a) for a in attributes],
                    "results": [dict(r) for r in results],
                    "regions": list(regions),
                    "traits": list(traits),
                    "description": description,
                }
            )
        )

    return ops


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
    if isinstance(json_data, str):
        data: typing.Mapping[str, JsonValue] = json.loads(json_data)
    else:
        data = json_data

    ops: list[TableGenRecordModel] = []
    raw_instance_of = data.get("!instanceof", {})
    raw_op_defs = (
        raw_instance_of.get("Op", []) if isinstance(raw_instance_of, dict) else []
    )
    op_def_names = {
        str(x) for x in (raw_op_defs if isinstance(raw_op_defs, list) else [])
    }

    for rec_name, rec in data.items():
        if rec_name.startswith("!") or not isinstance(rec, dict):
            continue

        raw_supers = rec.get("!superclasses", [])
        superclasses = [
            str(s) for s in (raw_supers if isinstance(raw_supers, list) else [])
        ]
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

        mnemonic = rec.get("mnemonic")
        if not mnemonic:
            mnemonic = (
                clean_class[:-2].lower()
                if clean_class.endswith("Op")
                else clean_class.lower()
            )

        operands: list[JsonValue] = []
        attributes: list[JsonValue] = []

        raw_args = rec.get("arguments")
        if isinstance(raw_args, dict) and raw_args.get("kind") == "dag":
            dag_args = raw_args.get("args", [])
            for arg in dag_args if isinstance(dag_args, list) else []:
                if isinstance(arg, (list, tuple)) and len(arg) >= 2:
                    arg_type_obj, arg_name = arg[0], str(arg[1])
                    arg_type = (
                        str(arg_type_obj.get("def", "Type"))
                        if isinstance(arg_type_obj, dict)
                        else str(arg_type_obj)
                    )
                    is_attr = any(
                        k in arg_type.lower()
                        for k in (
                            "attr",
                            "enum",
                            "constant",
                            "string",
                            "bool",
                            "i64",
                            "i32",
                            "f32",
                            "dimension",
                        )
                    )
                    if is_attr:
                        attributes.append({"name": arg_name, "type": arg_type})
                    else:
                        operands.append({"name": arg_name, "type": arg_type})
        elif isinstance(raw_args, list):
            for arg in raw_args:
                aname = (
                    str(arg.get("name", "arg")) if isinstance(arg, dict) else str(arg)
                )
                atype = (
                    str(arg.get("type", "Type")) if isinstance(arg, dict) else "Type"
                )
                is_attr = any(
                    k in atype.lower()
                    for k in (
                        "attr",
                        "enum",
                        "constant",
                        "string",
                        "bool",
                        "i64",
                        "i32",
                        "f32",
                        "dimension",
                    )
                )
                if is_attr:
                    attributes.append({"name": aname, "type": atype})
                else:
                    operands.append({"name": aname, "type": atype})

        results: list[JsonValue] = []
        raw_results = rec.get("results")
        if isinstance(raw_results, dict) and raw_results.get("kind") == "dag":
            dag_res = raw_results.get("args", [])
            for res in dag_res if isinstance(dag_res, list) else []:
                if isinstance(res, (list, tuple)) and len(res) >= 2:
                    res_type_obj, res_name = res[0], str(res[1])
                    res_type = (
                        str(res_type_obj.get("def", "Type"))
                        if isinstance(res_type_obj, dict)
                        else str(res_type_obj)
                    )
                    results.append({"name": res_name or "result", "type": res_type})
        elif isinstance(raw_results, list):
            for res in raw_results:
                rname = (
                    str(res.get("name", "result"))
                    if isinstance(res, dict)
                    else "result"
                )
                rtype = (
                    str(res.get("type", "Type")) if isinstance(res, dict) else "Type"
                )
                results.append({"name": rname, "type": rtype})

        regions: list[JsonValue] = []
        raw_regions = rec.get("regions")
        if isinstance(raw_regions, dict) and raw_regions.get("kind") == "dag":
            dag_regs = raw_regions.get("args", [])
            for reg in dag_regs if isinstance(dag_regs, list) else []:
                if isinstance(reg, (list, tuple)) and len(reg) >= 2:
                    regions.append(str(reg[1]))
        elif isinstance(raw_regions, list):
            regions.extend(str(r) for r in raw_regions)

        traits: list[JsonValue] = []
        raw_traits = rec.get("traits", [])
        if isinstance(raw_traits, list):
            for t in raw_traits:
                t_name = str(t.get("def", "")) if isinstance(t, dict) else str(t)
                if t_name:
                    traits.append(t_name)

        summary = rec.get("summary")
        description = rec.get("description") or summary

        ops.append(
            TableGenRecordModel(
                {
                    "api_path": f"{dialect_name}.{mnemonic}",
                    "dialect": dialect_name,
                    "class_name": class_name,
                    "operands": operands,
                    "attributes": attributes,
                    "results": results,
                    "regions": regions,
                    "traits": traits,
                    "description": str(description)
                    if description is not None
                    else None,
                }
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
    except Exception as e:
        print(f"Error running TableGen binary {binary}: {e}")
    return None


def validate_dialect_ops_ods(
    ops: typing.Sequence[typing.Mapping[str, JsonValue]], dialect_name: str
) -> list[str]:
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
            errors.append(
                f"Operation '{path}' does not match dialect prefix '{prefix}'."
            )
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
