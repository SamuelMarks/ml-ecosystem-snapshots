"""Static Python AST extraction and namespace alignment for compliance verification."""

from __future__ import annotations

import contextlib
import importlib
import logging
import os
from pathlib import Path
import sys
import typing

import griffe
from ml_switcheroo_ir.schema.ghost import GhostRef

from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    InspectableTarget,
    read_runtime_attr,
)
from ml_ecosystem_snapshots.utils import (
    get_framework_docstring_parser,
    resolve_griffe_parser,
)

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

                obj: InspectableTarget = live_mod
                for p in parts[i:]:
                    obj = getattr(obj, p)

                node_path = str(read_runtime_attr(node, "path", "") or "")
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


__all__ = [
    "DEFAULT_NAMESPACE_PREFIX_MAP",
    "_temporary_sys_path",
    "align_namespace",
    "extract_target_ast",
    "extract_target_refs",
    "extract_target_refs_single",
    "get_module_info_from_path",
]
