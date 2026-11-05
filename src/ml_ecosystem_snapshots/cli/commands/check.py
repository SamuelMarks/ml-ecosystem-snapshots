"""CLI command handlers for compliance, ISA, IR, and specification verification."""

from __future__ import annotations

import argparse
import collections
import inspect
import json
import os
import sys
import typing

import typing_extensions
from ml_switcheroo_ir.schema.ghost import GhostRef

from ml_ecosystem_snapshots.mcp.models import DomainOpCheckResult
from ml_ecosystem_snapshots.models import JsonValue


def _read_namespace_attr(args: argparse.Namespace, attr_name: str, default: JsonValue = None) -> JsonValue:
    """Read an optional attribute from an `argparse.Namespace` via member lookup.

    Args:
        args: Parsed CLI `Namespace` instance.
        attr_name: Attribute name to look up.
        default: Default value when absent.

    Returns:
        Attribute value or `default`.
    """
    raw = dict(inspect.getmembers(args)).get(attr_name, default)
    assert isinstance(raw, (str, int, float, bool, list, dict)) or raw is None
    return raw


class _DomainOpNameCheckProto(typing.Protocol):
    """Protocol for MCP domain op check functions keyed by `op_name`."""

    def __call__(
        self,
        op_name: str,
        inputs_count: int | None = None,
        attributes: list[str] | None = None,
    ) -> DomainOpCheckResult | typing.Mapping[str, JsonValue]:
        """Validate a domain operation by `op_name`.

        Args:
            op_name: Operation name.
            inputs_count: Optional expected input count.
            attributes: Optional list of attribute names.

        Returns:
            Check result model or dictionary.
        """
        ...


class _DomainMnemonicCheckProto(typing.Protocol):
    """Protocol for MCP domain op check functions keyed by `mnemonic`."""

    def __call__(
        self,
        mnemonic: str,
        operands_count: int | None = None,
        attributes: list[str] | None = None,
    ) -> DomainOpCheckResult | typing.Mapping[str, JsonValue]:
        """Validate a domain instruction by `mnemonic`.

        Args:
            mnemonic: Instruction mnemonic.
            operands_count: Optional expected operand count.
            attributes: Optional list of attribute names.

        Returns:
            Check result model or dictionary.
        """
        ...


class _DomainOnnxCheckProto(typing.Protocol):
    """Protocol for ONNX domain op check function."""

    def __call__(
        self,
        op_name: str,
        domain: str = "",
        inputs_count: int | None = None,
        attributes: list[str] | None = None,
    ) -> DomainOpCheckResult | typing.Mapping[str, JsonValue]:
        """Validate an ONNX operation.

        Args:
            op_name: Operation name.
            domain: Operator domain.
            inputs_count: Optional expected input count.
            attributes: Optional list of attribute names.

        Returns:
            Check result model or dictionary.
        """
        ...


class _DomainMetalCheckProto(typing.Protocol):
    """Protocol for Metal domain op check function."""

    def __call__(
        self,
        op_name: str,
        address_space: str | None = None,
        inputs_count: int | None = None,
        attributes: list[str] | None = None,
    ) -> DomainOpCheckResult | typing.Mapping[str, JsonValue]:
        """Validate a Metal operation.

        Args:
            op_name: Operation name.
            address_space: Optional address space.
            inputs_count: Optional expected input count.
            attributes: Optional list of attribute names.

        Returns:
            Check result model or dictionary.
        """
        ...


DomainCheckCallable: typing_extensions.TypeAlias = "_DomainOpNameCheckProto | _DomainMnemonicCheckProto | _DomainOnnxCheckProto | _DomainMetalCheckProto"


def _is_onnx_check_fn(
    check_fn: DomainCheckCallable,
    extra_kwargs: dict[str, JsonValue] | None,
) -> typing_extensions.TypeGuard[_DomainOnnxCheckProto]:
    """Narrow `check_fn` to `_DomainOnnxCheckProto` when `domain` extra kwargs are supplied.

    Args:
        check_fn: Domain check callable.
        extra_kwargs: Optional extra keyword arguments dictionary.

    Returns:
        True if `extra_kwargs` contains a `'domain'` key.
    """
    return bool(extra_kwargs is not None and "domain" in extra_kwargs)


def _is_metal_check_fn(
    check_fn: DomainCheckCallable,
    extra_kwargs: dict[str, JsonValue] | None,
) -> typing_extensions.TypeGuard[_DomainMetalCheckProto]:
    """Narrow `check_fn` to `_DomainMetalCheckProto` when `address_space` extra kwargs are supplied.

    Args:
        check_fn: Domain check callable.
        extra_kwargs: Optional extra keyword arguments dictionary.

    Returns:
        True if `extra_kwargs` contains an `'address_space'` key.
    """
    return bool(extra_kwargs is not None and "address_space" in extra_kwargs)


def _is_mnemonic_check_fn(
    check_fn: DomainCheckCallable,
    op_attr: str,
) -> typing_extensions.TypeGuard[_DomainMnemonicCheckProto]:
    """Narrow `check_fn` to `_DomainMnemonicCheckProto` when `op_attr` is `'mnemonic'`.

    Args:
        check_fn: Domain check callable.
        op_attr: Target operation attribute name.

    Returns:
        True if `op_attr == 'mnemonic'`.
    """
    return op_attr == "mnemonic"


def _is_op_name_check_fn(
    check_fn: DomainCheckCallable,
    op_attr: str,
) -> typing_extensions.TypeGuard[_DomainOpNameCheckProto]:
    """Narrow `check_fn` to `_DomainOpNameCheckProto` for standard domain operations.

    Args:
        check_fn: Domain check callable.
        op_attr: Target operation attribute name.

    Returns:
        True if `op_attr != 'mnemonic'`.
    """
    return op_attr != "mnemonic"


def _run_domain_op_check_cli(
    args: argparse.Namespace,
    check_fn: DomainCheckCallable,
    label: str,
    op_attr: str = "op_name",
    count_attr: str = "inputs_count",
    count_param: str = "inputs_count",
    invalid_format: str = "Invalid:",
    extra_kwargs: dict[str, JsonValue] | None = None,
) -> None:
    """Execute a domain op verification check and format CLI stdout/exit codes.

    Args:
        args: Parsed CLI `Namespace` object.
        check_fn: Underlying MCP verification callable.
        label: Human-readable domain operation prefix (e.g. 'ONNX Operation', 'WASM Instruction').
        op_attr: Attribute name on `args` holding the operation or mnemonic string.
        count_attr: Attribute name on `args` holding the expected operand/input count.
        count_param: Keyword parameter name accepted by `check_fn` for operand/input count.
        invalid_format: Suffix string when reporting an invalid operation ('Invalid:' or 'is invalid:').
        extra_kwargs: Optional additional keyword arguments to forward to `check_fn`.
    """
    raw_attrs = _read_namespace_attr(args, "attributes")
    attrs: list[str] | None = [a.strip() for a in str(raw_attrs).split(",") if a.strip()] if raw_attrs else None
    target = str(_read_namespace_attr(args, op_attr, ""))
    raw_count = _read_namespace_attr(args, count_attr)
    count_val: int | None = int(raw_count) if isinstance(raw_count, int) else None

    res: DomainOpCheckResult | typing.Mapping[str, JsonValue]
    if extra_kwargs and _is_onnx_check_fn(check_fn, extra_kwargs):
        res = check_fn(
            op_name=target,
            domain=str(extra_kwargs["domain"] or ""),
            inputs_count=count_val,
            attributes=attrs,
        )
    elif extra_kwargs and _is_metal_check_fn(check_fn, extra_kwargs):
        addr_val = extra_kwargs["address_space"]
        res = check_fn(
            op_name=target,
            address_space=str(addr_val) if addr_val is not None else None,
            inputs_count=count_val,
            attributes=attrs,
        )
    elif _is_mnemonic_check_fn(check_fn, op_attr):
        res = check_fn(
            mnemonic=target,
            operands_count=count_val,
            attributes=attrs,
        )
    else:
        assert _is_op_name_check_fn(check_fn, op_attr)
        res = check_fn(
            op_name=target,
            inputs_count=count_val,
            attributes=attrs,
        )
    is_valid = res.is_valid if isinstance(res, DomainOpCheckResult) else bool(res.get("is_valid"))
    if not is_valid:
        print(f"{label} '{target}' {invalid_format}")
        errs = res.errors if isinstance(res, DomainOpCheckResult) else res.get("errors", [])
        assert isinstance(errs, list)
        for err in errs:
            print(f"  - {err}")
        sys.exit(1)

    exp_inputs = res.expected_inputs if isinstance(res, DomainOpCheckResult) else res.get("expected_inputs", [])
    exp_attrs = res.expected_attributes if isinstance(res, DomainOpCheckResult) else res.get("expected_attributes", [])
    inp_len = len(exp_inputs) if isinstance(exp_inputs, list) else 0
    attr_strs = [str(a) for a in exp_attrs] if isinstance(exp_attrs, list) else []
    print(f"{label} '{target}' is valid.")
    print(f"  Inputs:     {inp_len}")
    print(f"  Attributes: {', '.join(attr_strs)}")


def _check_assembly_or_mlir_target(
    target_path: str,
    target_ext: str,
    reference_snapshot: typing.Mapping[str, JsonValue],
) -> bool:
    """Run assembly or MLIR snippet compliance verification when the target file extension matches.

    Args:
        target_path: Path to the target source file.
        target_ext: Lowercase file extension of `target_path`.
        reference_snapshot: Loaded reference snapshot dictionary.

    Returns:
        True if `target_ext` was handled as an assembly or MLIR file, False otherwise.
    """
    if target_ext in (".sass", ".s"):
        from ml_ecosystem_snapshots.compliance import (
            check_rdna_assembly_compliance,
            check_sass_assembly_compliance,
        )

        with open(target_path, encoding="utf-8") as f:
            content = f.read()
        target = str(reference_snapshot.get("target", "")).lower()
        if "rdna" in target or "amd" in target:
            res = check_rdna_assembly_compliance(content)
            print(f"AMD RDNA snippet compliance: {res.get('verified_instructions')}/{res.get('total_instructions')} instructions valid.")
        else:
            res = check_sass_assembly_compliance(content)
            print(f"NVIDIA SASS snippet compliance: {res.get('verified_instructions')}/{res.get('total_instructions')} instructions valid.")
        if not res.get("is_compliant"):
            for err in res.get("errors", []):
                print(f"  - {err}")
            sys.exit(1)
        return True

    if target_ext == ".mlir":
        from ml_ecosystem_snapshots.compliance import check_mlir_text_compliance

        with open(target_path, encoding="utf-8") as f:
            content = f.read()
        res = check_mlir_text_compliance(content)
        print(f"MLIR snippet compliance: {res.get('verified_ops')}/{res.get('total_ops')} operations valid.")
        if not res.get("is_compliant"):
            for err in res.get("errors", []):
                print(f"  - {err}")
            sys.exit(1)
        return True

    return False


def _print_module_breakdown(matched: list[str], missing: list[str]) -> None:
    """Print per-module compliance percentage breakdown.

    Args:
        matched: List of matched API paths.
        missing: List of missing API paths.
    """
    print("Breakdown by Module:")
    matched_by_mod: dict[str, list[str]] = collections.defaultdict(list)
    missing_by_mod: dict[str, list[str]] = collections.defaultdict(list)

    for api_path in matched:
        mod = ".".join(api_path.split(".")[:-1])
        matched_by_mod[mod].append(api_path)

    for api_path in missing:
        mod = ".".join(api_path.split(".")[:-1])
        missing_by_mod[mod].append(api_path)

    all_mods = set(list(matched_by_mod.keys()) + list(missing_by_mod.keys()))
    for mod in sorted(all_mods):
        m = len(matched_by_mod[mod])
        total = m + len(missing_by_mod[mod])
        mod_score = (m / total) * 100 if total > 0 else 0
        print(f"  - {mod}: {mod_score:.1f}% ({m}/{total})")


def _format_missing_ref_row(fqn: str, ref: GhostRef) -> str:
    """Format a Markdown table row describing a missing API endpoint.

    Args:
        fqn: Fully qualified API path of the missing symbol.
        ref: Reference `GhostRef` definition for the missing symbol.

    Returns:
        Formatted Markdown table row string.
    """
    parts = fqn.split(".")
    framework = parts[0]
    symbol = parts[-1]
    namespace = ".".join(parts[:-1])

    if ref.kind == "MODULE":
        sig = "module"
    else:
        sig_parts = []
        for p in ref.params:
            p_str = p.name
            if p.annotation:
                p_str += f": {p.annotation}"
            if p.default:
                p_str += f"={p.default}"
            sig_parts.append(p_str)
        sig = f"({', '.join(sig_parts)})"
        if ref.returns_type:
            sig += f" -> {ref.returns_type}"

    doc = ref.docstring or ""
    if doc:
        doc = doc.strip().split("\n\n")[0].replace("\n", " ").replace("|", "\\|")
        if len(doc) > 100:
            doc = doc[:97] + "..."
    else:
        doc = "No docstring available."

    return f"| [ ] | {framework} | {namespace} | {symbol} | {fqn} | `{sig}` | {doc} |"


def _print_missing_apis_table(
    missing: list[str],
    reference_snapshot: typing.Mapping[str, JsonValue],
) -> None:
    """Print the Markdown table of missing APIs from a compliance check.

    Args:
        missing: List of missing fully qualified API paths.
        reference_snapshot: Reference snapshot dictionary containing categories.
    """
    from ml_switcheroo_ir.schema.ghost import GhostRef

    print(f"\nMissing APIs ({len(missing)}):\n")

    ref_map: dict[str, GhostRef] = {}
    cats = reference_snapshot.get("categories", {})
    assert isinstance(cats, dict)
    for _cat, items in cats.items():
        assert isinstance(items, list)
        for item in items:
            ref_item = GhostRef.model_validate(item)
            ref_map[ref_item.api_path] = ref_item
            for alias in ref_item.aliases or []:
                ref_map[alias] = ref_item

    print("|   | Framework | Namespace | Symbol | FQN | Signature | Docstring |")
    print("|---|---|---|---|---|---|---|")

    for fqn in sorted(missing):
        ref_match = ref_map.get(fqn)
        if not ref_match:
            continue
        print(_format_missing_ref_row(fqn, ref_match))


def cmd_check(args: argparse.Namespace) -> None:
    """Handle the check command for compliance checking.

    Args:
        args: Parsed command line arguments containing snapshot_json, target_path,
              target_prefix, and reference_prefix.
    """
    from ml_ecosystem_snapshots.cli.resolver import resolve_snapshot_path
    from ml_ecosystem_snapshots.compliance import extract_target_refs, score_compliance

    path = resolve_snapshot_path(args.snapshot_json)
    with open(path, encoding="utf-8") as f:
        reference_snapshot = json.load(f)

    target_path = str(_read_namespace_attr(args, "target_path", "") or "")
    target_ext = os.path.splitext(target_path)[1].lower() if target_path else ""

    if _check_assembly_or_mlir_target(target_path, target_ext, reference_snapshot):
        return

    print(f"Extracting target APIs from {args.target_path}...")
    target_refs = extract_target_refs(args.target_path, args.target_prefix, args.reference_prefix)

    print("Scoring compliance...")
    results = score_compliance(reference_snapshot, target_refs)

    score = results.get("score_percentage", 0.0)
    print("\n--- Compliance Report ---")
    print(f"Overall Compliance: {score}%\n")

    matched = results.get("matched", [])
    missing = results.get("missing", [])
    _print_module_breakdown(matched, missing)

    if missing:
        _print_missing_apis_table(missing, reference_snapshot)

    mismatched = results.get("mismatched", [])
    if mismatched:
        print(f"\nMismatched APIs ({len(mismatched)}):")
        for item in sorted(mismatched, key=lambda x: x["api_path"])[:20]:
            print(f"  ~ {item['api_path']}\n    Expected: {item['expected']}\n    Actual:   {item['actual']}")
        if len(mismatched) > 20:
            print(f"  ~ ... and {len(mismatched) - 20} more")


def cmd_check_sass(args: argparse.Namespace) -> None:
    """Validate NVIDIA SASS assembly file or instruction mnemonic from the CLI.

    Args:
        args: Parsed command line arguments containing mnemonic, operands, modifiers, sm_arch, or file.
    """
    from ml_ecosystem_snapshots.compliance import check_sass_assembly_compliance
    from ml_ecosystem_snapshots.mcp_server import check_sass_instruction

    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: SASS assembly file not found: {args.file}")
            sys.exit(1)
        with open(args.file, encoding="utf-8") as f:
            content = f.read()
        res = check_sass_assembly_compliance(content, sm_arch=args.sm_arch)
        if not res.get("is_compliant"):
            print(f"SASS Compliance Check Failed ({len(res.get('errors', []))} errors):")
            for err in res.get("errors", []):
                print(f"  - {err}")
            sys.exit(1)
        print(f"SASS snippet verified compliant: {res.get('verified_instructions')}/{res.get('total_instructions')} instructions valid.")
        return

    if not args.mnemonic:
        print("Error: Either a mnemonic or --file must be specified.")
        sys.exit(1)

    operands = [op.strip() for op in args.operands.split(",") if op.strip()] if args.operands else None
    modifiers = [mod.strip() for mod in args.modifiers.split(",") if mod.strip()] if args.modifiers else None

    inst_res = check_sass_instruction(
        mnemonic=args.mnemonic,
        operands=operands,
        modifiers=modifiers,
        sm_arch=args.sm_arch,
    )

    if not inst_res.get("is_valid"):
        print(f"SASS Instruction '{args.mnemonic}' Invalid:")
        for err in inst_res.get("errors", []):
            print(f"  - {err}")
        sys.exit(1)
    else:
        print(f"SASS Instruction '{args.mnemonic}' is valid.")
        print(f"  Supported Architectures: {', '.join(inst_res.get('supported_architectures', []))}")
        if inst_res.get("valid_modifiers"):
            print(f"  Recognized Modifiers:    {', '.join(inst_res.get('valid_modifiers', []))}")


def cmd_check_rdna(args: argparse.Namespace) -> None:
    """Validate AMD RDNA assembly file or instruction mnemonic from the CLI.

    Args:
        args: Parsed command line arguments containing mnemonic, operands, encoding, gfx_arch, modifiers, wave_size, or file.
    """
    from ml_ecosystem_snapshots.compliance import check_rdna_assembly_compliance
    from ml_ecosystem_snapshots.mcp_server import check_rdna_instruction

    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: RDNA assembly file not found: {args.file}")
            sys.exit(1)
        with open(args.file, encoding="utf-8") as f:
            content = f.read()
        res = check_rdna_assembly_compliance(content, gfx_arch=args.gfx_arch)
        if not res.get("is_compliant"):
            print(f"RDNA Compliance Check Failed ({len(res.get('errors', []))} errors):")
            for err in res.get("errors", []):
                print(f"  - {err}")
            sys.exit(1)
        print(f"RDNA snippet verified compliant: {res.get('verified_instructions')}/{res.get('total_instructions')} instructions valid.")
        return

    if not args.mnemonic:
        print("Error: Either a mnemonic or --file must be specified.")
        sys.exit(1)

    operands = [op.strip() for op in args.operands.split(",") if op.strip()] if args.operands else None
    modifiers = [mod.strip() for mod in args.modifiers.split(",") if mod.strip()] if args.modifiers else None

    inst_res = check_rdna_instruction(
        mnemonic=args.mnemonic,
        operands=operands,
        encoding=args.encoding,
        gfx_arch=args.gfx_arch,
        modifiers=modifiers,
        wave_size=args.wave_size,
    )

    if not inst_res.get("is_valid"):
        print(f"RDNA Instruction '{args.mnemonic}' Invalid:")
        for err in inst_res.get("errors", []):
            print(f"  - {err}")
        sys.exit(1)
    else:
        print(f"RDNA Instruction '{args.mnemonic}' is valid.")
        print(f"  Supported Architectures: {', '.join(inst_res.get('supported_architectures', []))}")
        if inst_res.get("encoding"):
            print(f"  Encoding:                {inst_res.get('encoding')}")


def cmd_check_ptx(args: argparse.Namespace) -> None:
    """Validate NVIDIA PTX instruction mnemonic from the CLI.

    Args:
        args: Parsed command line arguments containing mnemonic, types, operands, state_space, scope, vector_width, sm_arch.
    """
    from ml_ecosystem_snapshots.mcp_server import check_ptx_instruction

    mnemonic = _read_namespace_attr(args, "mnemonic")
    if not mnemonic:
        print("Error: Instruction mnemonic must be specified.")
        sys.exit(1)

    raw_types = _read_namespace_attr(args, "types")
    types = [t.strip() for t in str(raw_types).split(",") if t.strip()] if raw_types else None
    raw_operands = _read_namespace_attr(args, "operands")
    operands = [op.strip() for op in str(raw_operands).split(",") if op.strip()] if raw_operands else None

    state_space = _read_namespace_attr(args, "state_space")
    scope = _read_namespace_attr(args, "scope")
    vector_width = _read_namespace_attr(args, "vector_width")
    sm_arch = _read_namespace_attr(args, "sm_arch")

    res = check_ptx_instruction(
        mnemonic=str(mnemonic),
        types=types,
        operands=operands,
        state_space=str(state_space) if state_space is not None else None,
        scope=str(scope) if scope is not None else None,
        vector_width=str(vector_width) if vector_width is not None else None,
        sm_arch=str(sm_arch) if sm_arch is not None else None,
    )

    if not res.get("is_valid"):
        print(f"PTX Instruction '{mnemonic}' Invalid:")
        for err in res.get("errors", []):
            print(f"  - {err}")
        sys.exit(1)
    else:
        print(f"PTX Instruction '{mnemonic}' is valid.")
        if res.get("min_sm"):
            print(f"  Minimum SM Architecture: {res.get('min_sm')}")
        if res.get("supported_types"):
            print(f"  Supported Types:         {', '.join(res.get('supported_types', []))}")


def cmd_check_mlir(args: argparse.Namespace) -> None:
    """Validate MLIR text or operation from the CLI.

    Args:
        args: Parsed command line arguments containing op_name, operands_count, attributes, or file.
    """
    from ml_ecosystem_snapshots.compliance import check_mlir_text_compliance
    from ml_ecosystem_snapshots.mcp_server import check_mlir_op

    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: MLIR file not found: {args.file}")
            sys.exit(1)
        with open(args.file, encoding="utf-8") as f:
            content = f.read()
        res = check_mlir_text_compliance(content)
        if not res.get("is_compliant"):
            print(f"MLIR Compliance Check Failed ({len(res.get('errors', []))} errors):")
            for err in res.get("errors", []):
                print(f"  - {err}")
            sys.exit(1)
        print(f"MLIR snippet verified compliant: {res.get('verified_ops')}/{res.get('total_ops')} operations valid.")
        return

    if not args.op_name:
        print("Error: Either an op_name or --file must be specified.")
        sys.exit(1)

    attrs = [a.strip() for a in args.attributes.split(",") if a.strip()] if args.attributes else None

    op_res = check_mlir_op(
        op_name=args.op_name,
        operands_count=args.operands_count,
        attributes=attrs,
    )

    if not op_res.get("is_valid"):
        print(f"MLIR Operation '{args.op_name}' Invalid:")
        for err in op_res.get("errors", []):
            print(f"  - {err}")
        sys.exit(1)

    print(f"MLIR Operation '{args.op_name}' is valid.")
    print(f"  Operands:   {len(op_res.get('expected_operands', []))}")
    print(f"  Attributes: {', '.join(op_res.get('expected_attributes', []))}")


def cmd_check_stablehlo(args: argparse.Namespace) -> None:
    """Validate StableHLO text or operation from the CLI.

    Args:
        args: Parsed command line arguments containing op_name, operands_count, attributes, or file.
    """
    from ml_ecosystem_snapshots.compliance import check_mlir_text_compliance
    from ml_ecosystem_snapshots.mcp_server import check_stablehlo_op

    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: StableHLO file not found: {args.file}")
            sys.exit(1)
        with open(args.file, encoding="utf-8") as f:
            content = f.read()
        res = check_mlir_text_compliance(content)
        if not res.get("is_compliant"):
            print(f"StableHLO Compliance Check Failed ({len(res.get('errors', []))} errors):")
            for err in res.get("errors", []):
                print(f"  - {err}")
            sys.exit(1)
        print(f"StableHLO snippet verified compliant: {res.get('verified_ops')}/{res.get('total_ops')} operations valid.")
        return

    if not args.op_name:
        print("Error: Either an op_name or --file must be specified.")
        sys.exit(1)

    attrs = [a.strip() for a in args.attributes.split(",") if a.strip()] if args.attributes else None

    op_res = check_stablehlo_op(
        op_name=args.op_name,
        operands_count=args.operands_count,
        attributes=attrs,
    )

    if not op_res.get("is_valid"):
        print(f"StableHLO Operation '{args.op_name}' Invalid:")
        for err in op_res.get("errors", []):
            print(f"  - {err}")
        sys.exit(1)

    print(f"StableHLO Operation '{args.op_name}' is valid.")
    print(f"  Operands:   {len(op_res.get('expected_operands', []))}")
    print(f"  Attributes: {', '.join(op_res.get('expected_attributes', []))}")


def cmd_check_wgsl(args: argparse.Namespace) -> None:
    """Validate WebGPU WGSL shader code or individual operation from the CLI.

    Args:
        args: Parsed command line arguments containing op_name, inputs_count, attributes, or file.
    """
    from ml_ecosystem_snapshots.compliance import check_wgsl_shader_compliance
    from ml_ecosystem_snapshots.mcp_server import check_wgsl_op

    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: WGSL file not found: {args.file}")
            sys.exit(1)
        with open(args.file, encoding="utf-8") as f:
            content = f.read()
        res = check_wgsl_shader_compliance(content)
        if not res.get("is_compliant"):
            print(f"WGSL Compliance Check Failed ({len(res.get('errors', []))} errors):")
            for err in res.get("errors", []):
                print(f"  - {err}")
            sys.exit(1)
        print(f"WGSL shader verified compliant: {res.get('verified_ops', 0)} operations recognized.")
        return

    if not args.op_name:
        print("Error: Either an op_name or --file must be specified.")
        sys.exit(1)

    _run_domain_op_check_cli(args, check_wgsl_op, "WGSL Operation")


def cmd_check_onnx(args: argparse.Namespace) -> None:
    """Validate ONNX operation against specification definitions.

    Args:
        args: Parsed command line arguments.
    """
    from ml_ecosystem_snapshots.mcp_server import check_onnx_op

    _run_domain_op_check_cli(
        args,
        check_onnx_op,
        "ONNX Operation",
        extra_kwargs={"domain": str(_read_namespace_attr(args, "domain", "") or "")},
    )


def cmd_check_metal(args: argparse.Namespace) -> None:
    """Validate Apple Metal MSL compute operation against specification.

    Args:
        args: Parsed command line arguments.
    """
    from ml_ecosystem_snapshots.mcp_server import check_metal_op

    raw_addr = _read_namespace_attr(args, "address_space")
    _run_domain_op_check_cli(
        args,
        check_metal_op,
        "Metal Operation",
        extra_kwargs={"address_space": str(raw_addr) if raw_addr is not None else None},
    )


def cmd_check_wasm(args: argparse.Namespace) -> None:
    """Validate WebAssembly 2.0 / SIMD instruction against specification.

    Args:
        args: Parsed command line arguments.
    """
    from ml_ecosystem_snapshots.mcp_server import check_wasm_instruction

    _run_domain_op_check_cli(
        args,
        check_wasm_instruction,
        "WASM Instruction",
        op_attr="mnemonic",
        count_attr="operands_count",
        count_param="operands_count",
    )


def _resolve_simple_domain_check_fn(fn_name: str) -> _DomainOpNameCheckProto:
    """Resolve an MCP domain verifier callable by name without reflection.

    Args:
        fn_name: Name of the verifier function exported by `ml_ecosystem_snapshots.mcp_server`.

    Returns:
        Typed domain op check callable.
    """
    import ml_ecosystem_snapshots.mcp_server as _mcp_srv

    dispatch: dict[str, _DomainOpNameCheckProto] = {
        "check_webgl_op": _mcp_srv.check_webgl_op,
        "check_cpp_op": _mcp_srv.check_cpp_op,
        "check_array_api_op": _mcp_srv.check_array_api_op,
        "check_scipy_op": _mcp_srv.check_scipy_op,
        "check_torchvision_op": _mcp_srv.check_torchvision_op,
        "check_torchaudio_op": _mcp_srv.check_torchaudio_op,
        "check_safetensors_op": _mcp_srv.check_safetensors_op,
        "check_aten_op": _mcp_srv.check_aten_op,
        "check_nccl_op": _mcp_srv.check_nccl_op,
        "check_flash_attention_op": _mcp_srv.check_flash_attention_op,
        "check_chex_op": _mcp_srv.check_chex_op,
        "check_grain_op": _mcp_srv.check_grain_op,
        "check_orbax_op": _mcp_srv.check_orbax_op,
        "check_optax_op": _mcp_srv.check_optax_op,
    }
    return dispatch[fn_name]


def _make_simple_domain_check_cmd(
    fn_name: str,
    label: str,
    invalid_format: str = "is invalid:",
) -> typing.Callable[[argparse.Namespace], None]:
    """Create a CLI command handler for a standard MCP domain operation verifier.

    Args:
        fn_name: Name of the verifier function exported by `ml_ecosystem_snapshots.mcp_server`.
        label: Human-readable domain operation prefix.
        invalid_format: Suffix string when reporting an invalid operation.

    Returns:
        CLI handler callable accepting `argparse.Namespace`.
    """

    def _handler(args: argparse.Namespace) -> None:
        """Validate domain operation against specification.

        Args:
            args: Parsed command line arguments.
        """
        check_fn = _resolve_simple_domain_check_fn(fn_name)
        _run_domain_op_check_cli(
            args,
            check_fn,
            label,
            invalid_format=invalid_format,
        )

    return _handler


cmd_check_webgl = _make_simple_domain_check_cmd("check_webgl_op", "WebGL Operation", invalid_format="Invalid:")
cmd_check_cpp = _make_simple_domain_check_cmd("check_cpp_op", "C++ Operation", invalid_format="Invalid:")
cmd_check_array_api = _make_simple_domain_check_cmd("check_array_api_op", "Array API Operation")
cmd_check_scipy = _make_simple_domain_check_cmd("check_scipy_op", "SciPy Operation")
cmd_check_torchvision = _make_simple_domain_check_cmd("check_torchvision_op", "TorchVision Operation")
cmd_check_torchaudio = _make_simple_domain_check_cmd("check_torchaudio_op", "TorchAudio Operation")
cmd_check_safetensors = _make_simple_domain_check_cmd("check_safetensors_op", "SafeTensors Operation")
cmd_check_aten = _make_simple_domain_check_cmd("check_aten_op", "ATen Operation")
cmd_check_nccl = _make_simple_domain_check_cmd("check_nccl_op", "NCCL Operation")
cmd_check_flash_attention = _make_simple_domain_check_cmd("check_flash_attention_op", "Attention Operation")
cmd_check_chex = _make_simple_domain_check_cmd("check_chex_op", "Chex Operation")
cmd_check_grain = _make_simple_domain_check_cmd("check_grain_op", "Grain Operation")
cmd_check_orbax = _make_simple_domain_check_cmd("check_orbax_op", "Orbax Operation")
cmd_check_optax = _make_simple_domain_check_cmd("check_optax_op", "Optax Operation")


__all__ = [
    "_run_domain_op_check_cli",
    "cmd_check",
    "cmd_check_array_api",
    "cmd_check_aten",
    "cmd_check_chex",
    "cmd_check_cpp",
    "cmd_check_flash_attention",
    "cmd_check_grain",
    "cmd_check_metal",
    "cmd_check_mlir",
    "cmd_check_nccl",
    "cmd_check_onnx",
    "cmd_check_optax",
    "cmd_check_orbax",
    "cmd_check_ptx",
    "cmd_check_rdna",
    "cmd_check_safetensors",
    "cmd_check_sass",
    "cmd_check_scipy",
    "cmd_check_stablehlo",
    "cmd_check_torchaudio",
    "cmd_check_torchvision",
    "cmd_check_wasm",
    "cmd_check_webgl",
    "cmd_check_wgsl",
]
