"""Static Python AST extraction and pure namespace alignment for compliance verification."""

from __future__ import annotations

import logging
import os
import typing
from pathlib import Path

import griffe
from ml_switcheroo_ir.schema.ghost import GhostRef
from ml_switcheroo_ir.typeclasses.outcome import Err, Ok, Result

from ml_ecosystem_snapshots.compliance.models import NamespaceAlignmentRule
from ml_ecosystem_snapshots.exceptions import SnapshotError, SnapshotNotFoundError
from ml_ecosystem_snapshots.typeclasses import make_namespace_align_endo
from ml_ecosystem_snapshots.utils import (
    get_framework_docstring_parser,
    resolve_griffe_parser,
)

_LOGGER = logging.getLogger(__name__)


def get_module_info_from_path(
    file_path: str,
    target_prefix: str = "",
) -> Result[SnapshotError, tuple[str, str]]:
    """Resolve a file path totally into `(root_search_path, derived_module_name)`.

    Args:
        file_path: Path to a Python file or directory.
        target_prefix: Optional module prefix fallback.

    Returns:
        `Ok((search_path, module_name))` on success, `Err(SnapshotNotFoundError)` when
        `file_path` does not exist, or `Err(SnapshotError)` when module name cannot be derived.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        return Err(SnapshotNotFoundError(f"Path does not exist: {file_path}"))

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
            return Err(SnapshotError(f"Could not derive module name from path: {file_path}"))

    return Ok((search_path, mod_name))


def extract_target_ast(
    file_path: str,
    target_prefix: str = "",
) -> Result[SnapshotError, griffe.Module | griffe.Object | griffe.Alias]:
    """Extract the static Griffe AST totally for `file_path` without code execution.

    Args:
        file_path: Path to a Python file or directory.
        target_prefix: Prefix of the target module.

    Returns:
        `Ok(griffe_node)` on success, or `Err(SnapshotError)` on missing/unparseable target.
    """
    info_res = get_module_info_from_path(file_path, target_prefix)
    if isinstance(info_res, Err):
        return info_res
    search_path, mod_name = info_res.value
    parser_choice = get_framework_docstring_parser(mod_name.split(".")[0])
    resolved_parser = resolve_griffe_parser(parser_choice)
    try:
        loaded = griffe.load(
            mod_name,
            search_paths=[search_path],
            docstring_parser=resolved_parser,
            allow_inspection=False,
        )
        return Ok(loaded)
    except (OSError, ValueError, TypeError, SyntaxError, griffe.GriffeError) as exc:
        return Err(SnapshotError(f"Failed to load static AST for '{file_path}': {exc}"))


def align_namespace(
    api_path: str,
    target_prefix: str,
    reference_prefix: str,
    alignment_rule: NamespaceAlignmentRule | None = None,
) -> str:
    """Align a target namespace path to a reference namespace path via `make_namespace_align_endo`.

    Args:
        api_path: API path to align.
        target_prefix: Prefix in the target namespace.
        reference_prefix: Prefix in the reference namespace.
        alignment_rule: Optional `NamespaceAlignmentRule`.

    Returns:
        Aligned canonical API path string.
    """
    return make_namespace_align_endo(target_prefix, reference_prefix, alignment_rule).app(api_path)


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
    for src_prefix, template in alignment_rule.node_prefix_rewrites:
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
) -> Result[SnapshotError, tuple[GhostRef, ...]]:
    """Statically extract an immutable `GhostRef` tuple from `file_path` wrapped in `Result`.

    Args:
        file_path: Path to a Python file or directory.
        target_prefix: Prefix of the target module.
        reference_prefix: Prefix of the reference module.
        alignment_rule: Optional `NamespaceAlignmentRule` for namespace normalization.

    Returns:
        `Ok(tuple[GhostRef, ...])` on success, or `Err(SnapshotError)` on path/load failure.
    """
    from ml_ecosystem_snapshots.introspection.inspector import GhostInspector

    info_res = get_module_info_from_path(file_path, target_prefix)
    if isinstance(info_res, Err):
        return info_res
    search_path, mod_name = info_res.value

    ast_res = extract_target_ast(file_path, target_prefix)
    if isinstance(ast_res, Err):
        return ast_res
    del search_path
    mod_ast = ast_res.value

    refs: list[GhostRef] = []
    align_endo = make_namespace_align_endo(target_prefix, reference_prefix, alignment_rule)

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
                aligned_path = align_endo.app(normalized_path)
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
            ) as exc:
                _LOGGER.warning("Exception in walk for %s: %s", current_path, exc)

        if isinstance(node, (griffe.Module, griffe.Class)):
            for name, member in node.members.items():
                if not name.startswith("_"):
                    walk(member, f"{current_path}.{name}")

    walk(mod_ast, mod_name)
    return Ok(tuple(refs))


def extract_target_refs(
    file_paths: typing.Sequence[str] | str,
    target_prefix: str,
    reference_prefix: str,
    alignment_rule: NamespaceAlignmentRule | None = None,
) -> Result[SnapshotError, tuple[GhostRef, ...]]:
    """Extract a deduplicated `GhostRef` tuple statically from one or more target paths.

    Args:
        file_paths: Single path string or sequence of paths to Python files or directories.
        target_prefix: Prefix of the target module.
        reference_prefix: Prefix of the reference module.
        alignment_rule: Optional `NamespaceAlignmentRule` for namespace normalization.

    Returns:
        `Ok(tuple[GhostRef, ...])` on success, or `Err(SnapshotError)` if any path fails.
    """
    paths = (file_paths,) if isinstance(file_paths, str) else tuple(file_paths)
    seen_paths: set[str] = set()
    refs: list[GhostRef] = []
    for fpath in paths:
        single_res = extract_target_refs_single(
            fpath,
            target_prefix,
            reference_prefix,
            alignment_rule=alignment_rule,
        )
        if isinstance(single_res, Err):
            return single_res
        for ref in single_res.value:
            if ref.api_path not in seen_paths:
                seen_paths.add(ref.api_path)
                refs.append(ref)
    return Ok(tuple(refs))


__all__ = [
    "NamespaceAlignmentRule",
    "align_namespace",
    "extract_target_ast",
    "extract_target_refs",
    "extract_target_refs_single",
    "get_module_info_from_path",
]
