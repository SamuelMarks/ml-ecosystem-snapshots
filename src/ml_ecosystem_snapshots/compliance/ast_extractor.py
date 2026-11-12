"""Static Python AST extraction and namespace alignment for compliance verification."""

from __future__ import annotations

import logging
import os
import typing
from pathlib import Path

import griffe
from ml_switcheroo_ir.schema.ghost import GhostRef

from ml_ecosystem_snapshots.compliance.models import NamespaceAlignmentRule
from ml_ecosystem_snapshots.exceptions import SnapshotError
from ml_ecosystem_snapshots.utils import (
    get_framework_docstring_parser,
    resolve_griffe_parser,
)

_LOGGER = logging.getLogger(__name__)


def get_module_info_from_path(file_path: str, target_prefix: str = "") -> tuple[str, str]:
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

    while Path(os.path.join(current, "__init__.py")).exists() and current.parent != current:
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
    file_path: str,
    target_prefix: str = "",
) -> griffe.Module | griffe.Object | griffe.Alias | None:
    """Extract the static Griffe AST for an arbitrary target path without code execution.

    Args:
        file_path: The path to a Python file or directory.
        target_prefix: The prefix of the target module.

    Returns:
        The extracted Griffe module AST.
    """
    search_path, mod_name = get_module_info_from_path(file_path, target_prefix)
    parser_choice = get_framework_docstring_parser(mod_name.split(".")[0])
    resolved_parser = resolve_griffe_parser(parser_choice)
    return griffe.load(
        mod_name,
        search_paths=[search_path],
        docstring_parser=resolved_parser,
        allow_inspection=False,
    )


def align_namespace(
    api_path: str,
    target_prefix: str,
    reference_prefix: str,
    alignment_rule: NamespaceAlignmentRule | None = None,
) -> str:
    """Align a target namespace path to a reference namespace path using an optional `NamespaceAlignmentRule`.

    Args:
        api_path: The API path to align.
        target_prefix: The prefix in the target namespace.
        reference_prefix: The prefix in the reference namespace.
        alignment_rule: Optional `NamespaceAlignmentRule` specifying prefix mappings and subpackage roots.

    Returns:
        The aligned namespace string.
    """
    if api_path == target_prefix:
        return reference_prefix

    if alignment_rule is not None:
        for sub_target in alignment_rule.subpackage_roots:
            candidate_prefix = f"{target_prefix}.{sub_target}."
            if api_path.startswith(candidate_prefix):
                return api_path.replace(candidate_prefix, f"{sub_target}.", 1)

        for src_pref, ref_pref in alignment_rule.prefix_map.items():
            if api_path.startswith(src_pref + "."):
                return api_path.replace(src_pref + ".", ref_pref + ".", 1)
            if api_path == src_pref:
                return ref_pref

    if api_path.startswith(f"{target_prefix}."):
        suffix = api_path[len(target_prefix) + 1 :]
        return f"{reference_prefix}.{suffix}"

    return api_path


def _apply_node_path_rule(
    node: griffe.Function | griffe.Class | griffe.Alias,
    current_path: str,
    target_prefix: str,
    alignment_rule: NamespaceAlignmentRule | None,
) -> str:
    """Apply optional `NamespaceAlignmentRule` path rewrites to a static Griffe node path.

    Args:
        node: Griffe AST node being inspected.
        current_path: Current dotted traversal path.
        target_prefix: Prefix of the target module.
        alignment_rule: Optional `NamespaceAlignmentRule`.

    Returns:
        Normalized dotted symbol path string.
    """
    if alignment_rule is None:
        return current_path

    node_path = str(node.path or "")
    for src_prefix, template in alignment_rule.node_prefix_rewrites.items():
        rendered = template.format(target_prefix=target_prefix)
        if node_path.startswith(src_prefix):
            rewritten = node_path.replace(src_prefix, rendered, 1)
            for old_seg, new_seg in alignment_rule.path_rewrites:
                rewritten = rewritten.replace(old_seg, new_seg)
            return rewritten
        if src_prefix in current_path:
            return current_path.replace(src_prefix, rendered, 1)

    return current_path


def extract_target_refs_single(
    file_path: str,
    target_prefix: str,
    reference_prefix: str,
    alignment_rule: NamespaceAlignmentRule | None = None,
) -> list[GhostRef]:
    """Statically extract `GhostRef` models from a target file or package without importing target code.

    Args:
        file_path: The path to a Python file or directory.
        target_prefix: The prefix of the target module.
        reference_prefix: The prefix of the reference module.
        alignment_rule: Optional `NamespaceAlignmentRule` for namespace normalization.

    Returns:
        List of statically extracted `GhostRef` instances.
    """
    from ml_ecosystem_snapshots.introspection.inspector import GhostInspector

    search_path, mod_name = get_module_info_from_path(file_path, target_prefix)

    parser_choice = get_framework_docstring_parser(mod_name.split(".")[0])
    resolved_parser = resolve_griffe_parser(parser_choice)
    mod_ast = griffe.load(
        mod_name,
        search_paths=[search_path],
        docstring_parser=resolved_parser,
        allow_inspection=False,
    )

    refs: list[GhostRef] = []

    def walk(node: griffe.Module | griffe.Object | griffe.Alias, current_path: str) -> None:
        """Recursively traverse a static Griffe AST node and inspect public symbols.

        Args:
            node: Griffe AST node (`Module`, `Class`, `Function`, or `Alias`).
            current_path: Dotted Python module/symbol path of `node`.
        """
        if isinstance(node, (griffe.Function, griffe.Class, griffe.Alias)):
            try:
                inspect_target: griffe.Function | griffe.Class | griffe.Object = (
                    node.final_target if isinstance(node, griffe.Alias) else node
                )
                normalized_path = _apply_node_path_rule(node, current_path, target_prefix, alignment_rule)
                aligned_path = align_namespace(
                    normalized_path,
                    target_prefix,
                    reference_prefix,
                    alignment_rule=alignment_rule,
                )
                refs.append(GhostInspector.inspect(inspect_target, aligned_path))
            except (
                SnapshotError,
                ImportError,
                AttributeError,
                TypeError,
                ValueError,
                OSError,
                RuntimeError,
                LookupError,
                SyntaxError,
                griffe.GriffeError,
            ) as e:
                _LOGGER.warning("Exception in walk for %s: %s", current_path, e)

        if isinstance(node, (griffe.Module, griffe.Class)):
            for name, member in node.members.items():
                if not name.startswith("_"):
                    walk(member, f"{current_path}.{name}")

    walk(mod_ast, mod_name)
    return refs


def extract_target_refs(
    file_paths: typing.Sequence[str] | str,
    target_prefix: str,
    reference_prefix: str,
    alignment_rule: NamespaceAlignmentRule | None = None,
) -> list[GhostRef]:
    """Extract a deduplicated list of `GhostRef`s statically from one or more target file paths.

    Args:
        file_paths: A single path string or sequence of paths to Python files or directories.
        target_prefix: The prefix of the target module.
        reference_prefix: The prefix of the reference module.
        alignment_rule: Optional `NamespaceAlignmentRule` for namespace normalization.

    Returns:
        Deduplicated list of statically extracted `GhostRef` instances.
    """
    paths = [file_paths] if isinstance(file_paths, str) else list(file_paths)
    seen_paths: set[str] = set()
    refs: list[GhostRef] = []
    for fpath in paths:
        for ref in extract_target_refs_single(
            fpath,
            target_prefix,
            reference_prefix,
            alignment_rule=alignment_rule,
        ):
            if ref.api_path not in seen_paths:
                seen_paths.add(ref.api_path)
                refs.append(ref)
    return refs


__all__ = [
    "NamespaceAlignmentRule",
    "align_namespace",
    "extract_target_ast",
    "extract_target_refs",
    "extract_target_refs_single",
    "get_module_info_from_path",
]
