"""Compliance checking module.

This module provides functionality to parse arbitrary Python files or packages
and compare them against a reference ML framework snapshot.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import os
from pathlib import Path
import re
import sys
import typing
import typing_extensions

import griffe
from ml_switcheroo_ir.schema.ghost import GhostParam, GhostRef
import pydantic

from ml_ecosystem_snapshots.models import JsonValue, SnapshotEnvelope
from ml_ecosystem_snapshots.utils import (
    get_framework_docstring_parser,
    resolve_griffe_parser,
)

ParamSignatureTuple: typing_extensions.TypeAlias = tuple[
    str, str, "str | None", "str | None"
]


class ComplianceMismatchEntry(pydantic.BaseModel):
    """Typed mismatch entry within `ComplianceScoreReport`."""

    model_config = pydantic.ConfigDict(extra="forbid")

    api_path: str
    expected: list[tuple[str, str, str | None, str | None]] = pydantic.Field(
        default_factory=list
    )
    actual: list[tuple[str, str, str | None, str | None]] = pydantic.Field(
        default_factory=list
    )
    reason: str | None = None

    @typing.overload
    def __getitem__(self, key: typing.Literal["api_path"]) -> str: ...

    @typing.overload
    def __getitem__(
        self, key: typing.Literal["expected", "actual"]
    ) -> list[ParamSignatureTuple]: ...

    @typing.overload
    def __getitem__(self, key: typing.Literal["reason"]) -> str | None: ...

    @typing.overload
    def __getitem__(self, key: str) -> str | list[ParamSignatureTuple] | JsonValue: ...

    def __getitem__(self, key: str) -> str | list[ParamSignatureTuple] | JsonValue:
        """Retrieve a mismatch entry field by dictionary key.

        Args:
            key: Field name ('api_path', 'expected', 'actual', or 'reason').

        Returns:
            Field value.
        """
        val: str | list[ParamSignatureTuple] | JsonValue = getattr(self, key)
        return val


class ComplianceScoreReport(pydantic.BaseModel):
    """Typed compliance report returned by `score_compliance`."""

    model_config = pydantic.ConfigDict(extra="forbid")

    score_percentage: float = 0.0
    total_endpoints: int = 0
    matched: list[str] = pydantic.Field(default_factory=list)
    missing: list[str] = pydantic.Field(default_factory=list)
    mismatched: list[ComplianceMismatchEntry] = pydantic.Field(default_factory=list)
    opaque_signatures: list[str] = pydantic.Field(default_factory=list)
    warnings: list[str] = pydantic.Field(default_factory=list)

    @typing.overload
    def __getitem__(self, key: typing.Literal["score_percentage"]) -> float: ...

    @typing.overload
    def __getitem__(self, key: typing.Literal["total_endpoints"]) -> int: ...

    @typing.overload
    def __getitem__(
        self,
        key: typing.Literal["matched", "missing", "opaque_signatures", "warnings"],
    ) -> list[str]: ...

    @typing.overload
    def __getitem__(
        self, key: typing.Literal["mismatched"]
    ) -> list[ComplianceMismatchEntry]: ...

    @typing.overload
    def __getitem__(
        self, key: str
    ) -> float | int | list[str] | list[ComplianceMismatchEntry] | JsonValue: ...

    def __getitem__(
        self, key: str
    ) -> float | int | list[str] | list[ComplianceMismatchEntry] | JsonValue:
        """Retrieve a compliance report field by dictionary key.

        Args:
            key: Field name.

        Returns:
            Field value.
        """
        val: float | int | list[str] | list[ComplianceMismatchEntry] | JsonValue = (
            getattr(self, key)
        )
        return val

    @typing.overload
    def get(
        self,
        key: typing.Literal["score_percentage"],
        default: float = 0.0,
    ) -> float: ...

    @typing.overload
    def get(
        self,
        key: typing.Literal["total_endpoints"],
        default: int = 0,
    ) -> int: ...

    @typing.overload
    def get(
        self,
        key: typing.Literal["matched", "missing", "opaque_signatures", "warnings"],
        default: list[str] | None = None,
    ) -> list[str]: ...

    @typing.overload
    def get(
        self,
        key: typing.Literal["mismatched"],
        default: list[ComplianceMismatchEntry] | None = None,
    ) -> list[ComplianceMismatchEntry]: ...

    @typing.overload
    def get(
        self,
        key: str,
        default: (
            float | int | list[str] | list[ComplianceMismatchEntry] | JsonValue
        ) = None,
    ) -> float | int | list[str] | list[ComplianceMismatchEntry] | JsonValue: ...

    def get(
        self,
        key: str,
        default: (
            float | int | list[str] | list[ComplianceMismatchEntry] | JsonValue
        ) = None,
    ) -> float | int | list[str] | list[ComplianceMismatchEntry] | JsonValue:
        """Retrieve a compliance report field by key with an optional default.

        Args:
            key: Field name.
            default: Default value when absent.

        Returns:
            Field value or `default`.
        """
        val: float | int | list[str] | list[ComplianceMismatchEntry] | JsonValue = (
            getattr(self, key, default)
        )
        return val


class SnippetComplianceReport(pydantic.BaseModel):
    """Typed compliance report returned by MLIR, WGSL, SASS, and RDNA snippet verifiers."""

    model_config = pydantic.ConfigDict(extra="forbid")

    is_compliant: bool = False
    total_ops: int = 0
    verified_ops: int = 0
    total_instructions: int = 0
    verified_instructions: int = 0
    errors: list[str] = pydantic.Field(default_factory=list)

    @typing.overload
    def __getitem__(self, key: typing.Literal["is_compliant"]) -> bool: ...

    @typing.overload
    def __getitem__(
        self,
        key: typing.Literal[
            "total_ops",
            "verified_ops",
            "total_instructions",
            "verified_instructions",
        ],
    ) -> int: ...

    @typing.overload
    def __getitem__(self, key: typing.Literal["errors"]) -> list[str]: ...

    @typing.overload
    def __getitem__(self, key: str) -> list[str] | JsonValue: ...

    def __getitem__(self, key: str) -> bool | int | list[str] | JsonValue:
        """Retrieve a snippet compliance field by dictionary key.

        Args:
            key: Field name.

        Returns:
            Field value.
        """
        val: bool | int | list[str] | JsonValue = getattr(self, key)
        return val

    @typing.overload
    def get(
        self,
        key: typing.Literal["is_compliant"],
        default: bool = False,
    ) -> bool: ...

    @typing.overload
    def get(
        self,
        key: typing.Literal[
            "total_ops",
            "verified_ops",
            "total_instructions",
            "verified_instructions",
        ],
        default: int = 0,
    ) -> int: ...

    @typing.overload
    def get(
        self,
        key: typing.Literal["errors"],
        default: list[str] | None = None,
    ) -> list[str]: ...

    @typing.overload
    def get(
        self,
        key: str,
        default: bool | int | list[str] | JsonValue = None,
    ) -> bool | int | list[str] | JsonValue: ...

    def get(
        self,
        key: str,
        default: bool | int | list[str] | JsonValue = None,
    ) -> bool | int | list[str] | JsonValue:
        """Retrieve a snippet compliance field by key with an optional default.

        Args:
            key: Field name.
            default: Default value when absent.

        Returns:
            Field value or `default`.
        """
        val: bool | int | list[str] | JsonValue = getattr(self, key, default)
        return val


_LOGGER = logging.getLogger(__name__)

DEFAULT_NAMESPACE_PREFIX_MAP: dict[str, str] = {
    "zero_jax": "jax",
    "zero_optax": "optax",
    "zero_chex": "chex",
    "zero_orbax": "orbax",
    "zero_grain": "grain",
}


@contextlib.contextmanager
def _temporary_sys_path(search_path: str) -> typing.Iterator[None]:
    """Temporarily prepend `search_path` to `sys.path` if not already present.

    Args:
        search_path: Directory path to add to `sys.path`.

    Yields:
        None while `search_path` is active on `sys.path`.
    """
    added = False
    if search_path not in sys.path:
        sys.path.insert(0, search_path)
        added = True
    try:
        yield
    finally:
        if added:
            sys.path.remove(search_path)


def get_module_info_from_path(
    file_path: str, target_prefix: str = ""
) -> tuple[str, str]:
    """Resolve a file path to a root search path and a module name.

    This function ascends the directory tree starting from the given path
    until it finds a directory that does not contain an `__init__.py` file.
    It returns the path to that root directory and the Python module name
    derived from the relative path.

    Args:
        file_path: The path to a Python file or directory.
        target_prefix: Optional module prefix to use as fallback.

    Returns:
        A tuple containing:
            - The root search path as a string.
            - The derived module name as a string.

    Raises:
        FileNotFoundError: If the provided path does not exist.
        ValueError: If the path does not appear to be a valid Python module.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {file_path}")

    is_dir = path.is_dir()
    current = path if is_dir else path.parent

    # Traverse upwards while an __init__.py is present
    while (
        Path(os.path.join(current, "__init__.py")).exists()
        and current.parent != current
    ):
        current = current.parent

    search_path = str(current)

    rel_path = path.relative_to(current)
    if is_dir:
        mod_name = ".".join(rel_path.parts)
    else:
        parts = list(rel_path.parts)
        if parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]
            if parts[-1] == "__init__":
                parts.pop()
        mod_name = ".".join(parts)

    if not mod_name:
        if target_prefix:
            mod_name = target_prefix.split(".")[0]
            if Path(os.path.join(Path(search_path), "src")).is_dir():
                search_path = str(Path(os.path.join(Path(search_path), "src")))
        else:
            raise ValueError(f"Could not derive module name from path: {file_path}")

    return search_path, mod_name


def extract_target_ast(
    file_path: str, target_prefix: str = ""
) -> griffe.Module | griffe.Object | None:
    """Extract the Griffe AST for an arbitrary target path.

    This function dynamically determines the module name and search path
    required to load the given file or directory using Griffe.

    Args:
        file_path: The path to a Python file or directory.
        target_prefix: The prefix of the target module.

    Returns:
        The extracted Griffe module AST.
    """
    search_path, mod_name = get_module_info_from_path(file_path, target_prefix)
    parser_choice = get_framework_docstring_parser(mod_name.split(".")[0])
    resolved_parser = resolve_griffe_parser(parser_choice)
    loaded = griffe.load(
        mod_name, search_paths=[search_path], docstring_parser=resolved_parser
    )
    assert isinstance(loaded, (griffe.Module, griffe.Object))
    return loaded


def align_namespace(
    api_path: str,
    target_prefix: str,
    reference_prefix: str,
    prefix_map: typing.Mapping[str, str] | None = None,
) -> str:
    """Align a target namespace path to a reference namespace path.

    Args:
        api_path: The API path to align.
        target_prefix: The prefix in the target namespace.
        reference_prefix: The prefix in the reference namespace.
        prefix_map: Optional mapping of custom source prefixes to reference prefixes.

    Returns:
        The aligned namespace string.
    """
    if api_path == target_prefix:
        return reference_prefix

    mapping = (
        dict(prefix_map)
        if prefix_map is not None
        else dict(DEFAULT_NAMESPACE_PREFIX_MAP)
    )

    for sub_target in ("jax", "optax"):
        candidate_prefix = f"{target_prefix}.{sub_target}."
        if api_path.startswith(candidate_prefix):
            return api_path.replace(candidate_prefix, f"{sub_target}.", 1)

    for z_pref, r_pref in mapping.items():
        if api_path.startswith(z_pref + "."):
            return api_path.replace(z_pref + ".", r_pref + ".", 1)
        if api_path == z_pref:
            return r_pref

    if api_path.startswith(f"{target_prefix}."):
        suffix = api_path[len(target_prefix) + 1 :]
        return f"{reference_prefix}.{suffix}"

    return api_path


def extract_target_refs_single(
    file_path: str, target_prefix: str, reference_prefix: str
) -> list[GhostRef]:
    """Extract a list of GhostRefs from a single target file path.

    This function parses the path using Griffe to find public definitions,
    dynamically imports them, and uses GhostInspector to create GhostRefs.
    It then aligns their namespaces to match the reference format.

    Args:
        file_path: The path to a Python file or directory.
        target_prefix: The prefix of the target module.
        reference_prefix: The prefix of the reference module.

    Returns:
        List of GhostRefs.
    """
    from ml_ecosystem_snapshots.introspection.inspector import GhostInspector

    search_path, mod_name = get_module_info_from_path(file_path, target_prefix)

    parser_choice = get_framework_docstring_parser(mod_name.split(".")[0])
    resolved_parser = resolve_griffe_parser(parser_choice)
    mod_ast = griffe.load(
        mod_name, search_paths=[search_path], docstring_parser=resolved_parser
    )

    refs: list[GhostRef] = []

    def walk(
        node: griffe.Module | griffe.Object | griffe.Alias, current_path: str
    ) -> None:
        """Recursively traverse a Griffe AST node and inspect public symbols.

        Args:
            node: Griffe AST node (`Module`, `Class`, `Function`, or `Alias`).
            current_path: Dotted Python module/symbol path of `node`.
        """
        if isinstance(node, (griffe.Function, griffe.Class, griffe.Alias)):
            try:
                parts = current_path.split(".")
                mod_p = parts[0]
                live_mod = importlib.import_module(mod_p)
                i = 1
                while i < len(parts):
                    try:
                        mod_p = f"{mod_p}.{parts[i]}"
                        live_mod = importlib.import_module(mod_p)
                        i += 1
                    except ImportError:
                        break

                obj: object = live_mod
                for p in parts[i:]:
                    obj = getattr(obj, p)

                node_path = getattr(node, "path", "")
                if node_path.startswith("zero_jax."):
                    current_path = (
                        node_path.replace("zero_jax.", f"{target_prefix}.jax.", 1)
                        .replace(".activation", "")
                        .replace(".nn.nn", ".nn")
                        .replace(".initializers.initializers", ".initializers")
                    )
                elif "zero_jax" in current_path:
                    current_path = current_path.replace(
                        "zero_jax.", f"{target_prefix}.jax.", 1
                    )
                aligned_path = align_namespace(
                    current_path, target_prefix, reference_prefix
                )
                refs.append(GhostInspector.inspect(obj, aligned_path))
            except Exception as e:
                _LOGGER.warning("Exception in walk for %s: %s", current_path, e)

        if isinstance(node, (griffe.Module, griffe.Class)):
            for name, member in node.members.items():
                if not name.startswith("_"):
                    walk(member, f"{current_path}.{name}")

    with _temporary_sys_path(search_path):
        assert isinstance(mod_ast, (griffe.Module, griffe.Object, griffe.Alias))
        walk(mod_ast, mod_name)
    return refs


def score_compliance(
    reference_snapshot: SnapshotEnvelope | typing.Mapping[str, JsonValue],
    target_refs: typing.Sequence[GhostRef],
    strict_c_extensions: bool = False,
) -> ComplianceScoreReport:
    """Score the compliance of target refs against a reference snapshot.

    Args:
        reference_snapshot: The reference snapshot dictionary or envelope containing categories.
        target_refs: The extracted and aligned GhostRefs from the target.
        strict_c_extensions: Whether to reject fallback matches with opaque C-extension signatures.

    Returns:
        A `ComplianceScoreReport` containing compliance metrics.
    """
    reference_map: dict[str, GhostRef] = {}
    raw_snap = (
        reference_snapshot.model_dump()
        if isinstance(reference_snapshot, SnapshotEnvelope)
        else reference_snapshot
    )
    cats_obj: JsonValue = raw_snap.get("categories", {})
    assert isinstance(cats_obj, dict)
    for _cat, items in cats_obj.items():
        assert isinstance(items, list)
        for item in items:
            ref = GhostRef.model_validate(item)
            reference_map[ref.api_path] = ref
            for alias in ref.aliases or []:
                reference_map[alias] = ref

    target_map = {ref.api_path: ref for ref in target_refs}

    total_reference_endpoints = len({ref.api_path for ref in reference_map.values()})
    if total_reference_endpoints == 0:
        return ComplianceScoreReport(
            score_percentage=0.0,
            total_endpoints=0,
            matched=[],
            missing=[],
            mismatched=[],
            opaque_signatures=[],
            warnings=[],
        )

    matched: list[str] = []
    missing: list[str] = []
    mismatched: list[ComplianceMismatchEntry] = []
    opaque_signatures: list[str] = []
    warnings: list[str] = []

    def sig_tuple(p: GhostParam) -> ParamSignatureTuple:
        """Build a normalized `(name, kind, default, annotation)` tuple for parameter comparison.

        Args:
            p: A `GhostParam` with `name`, `kind`, `default`, and `annotation`.

        Returns:
            Normalized 4-tuple `(name, kind, default, annotation)`.
        """
        name_val = p.name
        kind_val = str(p.kind)
        ann = p.annotation
        default: str | None = str(p.default) if p.default is not None else None

        if isinstance(ann, str) and ann.startswith('"') and ann.endswith('"'):
            ann = ann[1:-1]
        if isinstance(ann, str) and ann.startswith("'") and ann.endswith("'"):
            ann = ann[1:-1]
        if default in ("'```(None)```'", "```(None)```", "'___NONE___'"):
            default = "None"
        return (name_val, kind_val, default, ann)

    for api_path, ref_obj in reference_map.items():
        if api_path not in target_map:
            missing.append(api_path)
            continue

        target_obj = target_map[api_path]

        ref_sig = [sig_tuple(p) for p in ref_obj.params]
        tgt_sig = [sig_tuple(p) for p in target_obj.params]

        ref_is_opaque = (
            getattr(ref_obj, "signature_completeness", None) == "opaque"
            or "inexact_signature" in (ref_obj.environment_tags or [])
            or "opaque_c_extension" in (ref_obj.environment_tags or [])
        )
        tgt_is_opaque = (
            getattr(target_obj, "signature_completeness", None) == "opaque"
            or "inexact_signature" in (target_obj.environment_tags or [])
            or "opaque_c_extension" in (target_obj.environment_tags or [])
        )

        if ref_is_opaque or tgt_is_opaque:
            opaque_signatures.append(api_path)
            warnings.append(
                f"API '{api_path}' matched via opaque (*args, **kwargs) C-extension fallback."
            )
            if strict_c_extensions:
                mismatched.append(
                    ComplianceMismatchEntry(
                        api_path=api_path,
                        expected=ref_sig,
                        actual=tgt_sig,
                        reason="Opaque C-extension signature fallback rejected under strict_c_extensions",
                    )
                )
            else:
                matched.append(api_path)
            continue

        if ref_sig == tgt_sig:
            matched.append(api_path)
        else:
            has_varargs = any("VAR_POSITIONAL" in p.kind for p in target_obj.params)
            has_varkwargs = any("VAR_KEYWORD" in p.kind for p in target_obj.params)

            if has_varargs and has_varkwargs:
                matched.append(api_path)
            else:
                mismatched.append(
                    ComplianceMismatchEntry(
                        api_path=api_path, expected=ref_sig, actual=tgt_sig
                    )
                )

    score_percentage = (len(matched) / total_reference_endpoints) * 100.0

    return ComplianceScoreReport(
        score_percentage=round(score_percentage, 2),
        total_endpoints=total_reference_endpoints,
        matched=matched,
        missing=missing,
        mismatched=mismatched,
        opaque_signatures=opaque_signatures,
        warnings=warnings,
    )


def extract_target_refs(
    file_paths: typing.Sequence[str] | str,
    target_prefix: str,
    reference_prefix: str,
) -> list[GhostRef]:
    """Extract a list of GhostRefs from multiple target file paths.

    Args:
        file_paths: A list of file paths (or a single file path) to extract refs from.
        target_prefix: The prefix in the target namespace.
        reference_prefix: The prefix in the reference namespace.

    Returns:
        A list of GhostRef objects.
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    refs: list[GhostRef] = []
    for fp in file_paths:
        refs.extend(extract_target_refs_single(fp, target_prefix, reference_prefix))
    return refs


def check_mlir_text_compliance(mlir_text: str) -> SnippetComplianceReport:
    """Verify that an MLIR / StableHLO text snippet uses valid operations, attributes, and operands.

    Args:
        mlir_text: Text snippet of MLIR or StableHLO assembly code.

    Returns:
        `SnippetComplianceReport` with `is_compliant`, `total_ops`, `verified_ops`, and `errors`.
    """
    from ml_ecosystem_snapshots.mcp_server import check_mlir_op

    errors: list[str] = []
    total_ops = 0
    verified_ops = 0

    op_pattern = re.compile(
        r"(?:%\w+\s*=\s*)?([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)\s*(?:\(([^)]*)\))?"
    )

    for line in mlir_text.splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("//"):
            continue

        match = op_pattern.search(cleaned)
        if match:
            op_name = match.group(1)
            if op_name in ("builtin.module", "func.func"):
                continue

            total_ops += 1
            args_str = match.group(2)
            operands_count = (
                len([a for a in args_str.split(",") if a.strip()])
                if args_str is not None
                else None
            )

            res = check_mlir_op(op_name, operands_count=operands_count)
            if not res.get("op_exists"):
                errors.append(
                    f"Invalid MLIR operation '{op_name}' on line: '{cleaned}'"
                )
            elif not res.get("is_valid"):
                raw_errs = res.get("errors", [])
                assert isinstance(raw_errs, list)
                errors.extend([str(e) for e in raw_errs])
            else:
                verified_ops += 1

    return SnippetComplianceReport(
        is_compliant=len(errors) == 0,
        total_ops=total_ops,
        verified_ops=verified_ops,
        errors=errors,
    )


def check_wgsl_shader_compliance(shader_text: str) -> SnippetComplianceReport:
    """Verify that a WebGPU WGSL shader snippet uses valid builtin operations.

    Args:
        shader_text: Text snippet of WebGPU WGSL shader code.

    Returns:
        `SnippetComplianceReport` with `is_compliant`, `total_ops`, `verified_ops`, and `errors`.
    """
    from ml_ecosystem_snapshots.mcp_server import check_wgsl_op

    errors: list[str] = []
    total_ops = 0
    verified_ops = 0

    call_pattern = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)")

    ignored_keywords = {
        "fn",
        "if",
        "for",
        "while",
        "loop",
        "switch",
        "var",
        "let",
        "const",
        "return",
        "vec2",
        "vec3",
        "vec4",
        "mat2x2",
        "mat3x3",
        "mat4x4",
        "array",
        "ptr",
    }

    for line in shader_text.splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("//"):
            continue

        for match in call_pattern.finditer(cleaned):
            op_name = match.group(1)
            if op_name in ignored_keywords:
                continue

            res = check_wgsl_op(op_name)
            if res.get("op_exists"):
                total_ops += 1
                if res.get("is_valid"):
                    verified_ops += 1
                else:
                    errors.extend(res.get("errors", []))

    return SnippetComplianceReport(
        is_compliant=len(errors) == 0,
        total_ops=total_ops,
        verified_ops=verified_ops,
        errors=errors,
    )


def check_sass_assembly_compliance(
    assembly_text: str, sm_arch: str | None = None
) -> SnippetComplianceReport:
    """Verify that an NVIDIA SASS assembly snippet is valid for the target SM architecture.

    Args:
        assembly_text: NVIDIA SASS assembly text.
        sm_arch: Optional target SM architecture (e.g. 'sm_80', 'sm_90').

    Returns:
        `SnippetComplianceReport` with `is_compliant`, `total_instructions`, and `errors`.
    """
    from ml_ecosystem_snapshots.mcp_server import check_sass_instruction

    errors: list[str] = []
    total_insts = 0
    verified_insts = 0

    sass_line_pattern = re.compile(
        r"^(?:@!?P\d+\s+)?([A-Z0-9_]+)((?:\.[A-Z0-9_]+)*)(?:\s+(.*?))?;?$"
    )

    for line in assembly_text.splitlines():
        cleaned = re.sub(r"/\*.*?\*/", "", line).strip()
        if not cleaned or cleaned.startswith("//") or cleaned.startswith("#"):
            continue

        m = sass_line_pattern.match(cleaned)
        if m:
            base_mnemonic = m.group(1)
            mods_raw = m.group(2)
            operands_raw = m.group(3)

            modifiers = [mod for mod in mods_raw.split(".") if mod] if mods_raw else []
            operands = (
                [op.strip() for op in operands_raw.split(",") if op.strip()]
                if operands_raw
                else None
            )

            total_insts += 1
            res = check_sass_instruction(
                base_mnemonic,
                operands=operands,
                modifiers=modifiers,
                sm_arch=sm_arch,
            )
            if not res.get("is_valid"):
                raw_errs = res.get("errors", [])
                assert isinstance(raw_errs, list)
                errors.extend([str(e) for e in raw_errs])
            else:
                verified_insts += 1

    return SnippetComplianceReport(
        is_compliant=len(errors) == 0,
        total_instructions=total_insts,
        verified_instructions=verified_insts,
        errors=errors,
    )


def check_rdna_assembly_compliance(
    assembly_text: str, gfx_arch: str | None = None
) -> SnippetComplianceReport:
    """Verify that an AMD RDNA assembly snippet is valid for the target GFX architecture.

    Args:
        assembly_text: AMD RDNA assembly text.
        gfx_arch: Optional target GFX architecture (e.g. 'GFX11/RDNA3', 'GFX9/CDNA').

    Returns:
        `SnippetComplianceReport` with `is_compliant`, `total_instructions`, and `errors`.
    """
    from ml_ecosystem_snapshots.mcp_server import check_rdna_instruction

    errors: list[str] = []
    total_insts = 0
    verified_insts = 0

    rdna_line_pattern = re.compile(r"^([a-z0-9_]+)(?:\s+(.*?))?;?$")

    for line in assembly_text.splitlines():
        cleaned = re.sub(r"//.*", "", line).strip()
        if not cleaned or cleaned.startswith(";") or cleaned.startswith("#"):
            continue

        m = rdna_line_pattern.match(cleaned)
        if m:
            mnemonic = m.group(1)
            operands_raw = m.group(2)
            operands = (
                [op.strip() for op in operands_raw.split(",") if op.strip()]
                if operands_raw
                else None
            )

            total_insts += 1
            res = check_rdna_instruction(mnemonic, operands=operands, gfx_arch=gfx_arch)
            if not res.get("is_valid"):
                raw_errs = res.get("errors", [])
                assert isinstance(raw_errs, list)
                errors.extend([str(e) for e in raw_errs])
            else:
                verified_insts += 1

    return SnippetComplianceReport(
        is_compliant=len(errors) == 0,
        total_instructions=total_insts,
        verified_instructions=verified_insts,
        errors=errors,
    )


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
    if strict_2d:
        if len(shape_a) != 2 or len(shape_b) != 2:
            return (
                False,
                None,
                f"Strict 2D matmul error: both operands must have rank 2, got ranks {len(shape_a)} and {len(shape_b)}.",
            )

    if len(shape_a) == 1 and len(shape_b) == 1:
        if str(shape_a[0]) not in ("?", "-1") and str(shape_b[0]) not in ("?", "-1"):
            if int(shape_a[0]) != int(shape_b[0]):
                return (
                    False,
                    None,
                    f"1D vector dot product mismatch: {shape_a[0]} != {shape_b[0]}.",
                )
        return True, [], None

    if len(shape_a) == 1 and len(shape_b) >= 2:
        k_lhs = shape_a[0]
        k_rhs = shape_b[-2]
        if str(k_lhs) not in ("?", "-1") and str(k_rhs) not in ("?", "-1"):
            if int(k_lhs) != int(k_rhs):
                return (
                    False,
                    None,
                    f"Vector-matrix multiplication contracting dimension mismatch: inner dimension {k_lhs} != {k_rhs} (shapes: {list(shape_a)} vs {list(shape_b)}).",
                )
        batch_res = [int(d) if str(d).isdigit() else -1 for d in shape_b[:-2]]
        out_n = int(shape_b[-1]) if str(shape_b[-1]).isdigit() else -1
        return True, batch_res + [out_n], None

    if len(shape_a) >= 2 and len(shape_b) == 1:
        k_lhs = shape_a[-1]
        k_rhs = shape_b[0]
        if str(k_lhs) not in ("?", "-1") and str(k_rhs) not in ("?", "-1"):
            if int(k_lhs) != int(k_rhs):
                return (
                    False,
                    None,
                    f"Matrix-vector multiplication contracting dimension mismatch: inner dimension {k_lhs} != {k_rhs} (shapes: {list(shape_a)} vs {list(shape_b)}).",
                )
        batch_res = [int(d) if str(d).isdigit() else -1 for d in shape_a[:-2]]
        out_m = int(shape_a[-2]) if str(shape_a[-2]).isdigit() else -1
        return True, batch_res + [out_m], None

    if len(shape_a) < 2 or len(shape_b) < 2:
        return (
            False,
            None,
            f"Matrix multiplication requires at least 2D operands (got ranks {len(shape_a)} and {len(shape_b)}).",
        )

    k_lhs = shape_a[-1]
    k_rhs = shape_b[-2]
    if str(k_lhs) not in ("?", "-1") and str(k_rhs) not in ("?", "-1"):
        if int(k_lhs) != int(k_rhs):
            return (
                False,
                None,
                f"Matrix multiplication contracting dimension mismatch: inner dimension {k_lhs} != {k_rhs} (shapes: {list(shape_a)} vs {list(shape_b)}).",
            )

    batch_a = shape_a[:-2]
    batch_b = shape_b[:-2]
    if batch_a or batch_b:
        compat, b_shape, err = validate_broadcast_shapes(batch_a, batch_b)
        if not compat:
            return False, None, f"Batch dimension broadcasting error in matmul: {err}"
        out_shape = (b_shape or []) + [
            int(shape_a[-2]) if str(shape_a[-2]).isdigit() else -1,
            int(shape_b[-1]) if str(shape_b[-1]).isdigit() else -1,
        ]
    else:
        out_shape = [
            int(shape_a[-2]) if str(shape_a[-2]).isdigit() else -1,
            int(shape_b[-1]) if str(shape_b[-1]).isdigit() else -1,
        ]

    return True, out_shape, None


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


def get_dtype_bitwidth(dtype_name: str) -> int:
    """Return bitwidth for canonical, sub-byte, and quantized data types.

    Args:
        dtype_name: Canonical or normalized dtype name.

    Returns:
        Bitwidth in bits (e.g. 8 for qint8, 4 for qint4/int4, 2 for int2, 32 for float32).
    """
    from .mcp_server import normalize_dtype_name

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
    import math

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
    from .mcp_server import normalize_dtype_name

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
                f"Data type '{dtype_name}' is not supported for transcendental "
                f"operation '{op_name}'. Floating-point or complex dtype required."
            )

    return True, None
