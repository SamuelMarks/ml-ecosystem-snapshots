"""Script to generate backward-compatible ml_framework_snapshots shim package.

Creates thin forwarding modules in src/ml_framework_snapshots mirroring
src/ml_ecosystem_snapshots with 100% docstrings and full symbol re-exports.
"""

from __future__ import annotations

import os


def generate_shims(
    eco_dir: str = os.path.join("src", "ml_ecosystem_snapshots"),
    fw_dir: str = os.path.join("src", "ml_framework_snapshots"),
) -> None:
    """Generate all shim files from src/ml_ecosystem_snapshots to src/ml_framework_snapshots.

    Args:
        eco_dir: Source directory of canonical `ml_ecosystem_snapshots` package.
        fw_dir: Target directory for generated `ml_framework_snapshots` shims.
    """
    for root, _dirs, files in os.walk(eco_dir):
        rel = os.path.relpath(root, eco_dir)
        target_dir = os.path.join(fw_dir, rel) if rel != "." else fw_dir
        os.makedirs(target_dir, exist_ok=True)

        for f in files:
            if f.endswith(".py"):
                target_file = os.path.join(target_dir, f)
                mod_path: list[str] = []
                if rel != ".":
                    mod_path = rel.split(os.sep)
                if f != "__init__.py":
                    mod_path.append(f[:-3])

                submod = ".".join(mod_path)
                eco_target = (
                    f"ml_ecosystem_snapshots.{submod}"
                    if submod
                    else "ml_ecosystem_snapshots"
                )
                mod_name = (
                    f"ml_framework_snapshots.{submod}"
                    if submod
                    else "ml_framework_snapshots"
                )

                parts = [
                    '"""Backward-compatibility shim for ' + mod_name + ".",
                    "",
                    "Transparently re-exports all members from " + eco_target + ".",
                    '"""',
                    "",
                    "from __future__ import annotations",
                    "",
                    "from typing import Any, List",
                    "from ml_ecosystem_snapshots._alias import register_alias_finder as _register_alias_finder",
                    "import " + eco_target + " as _orig_mod",
                    "from " + eco_target + " import *  # noqa: F401, F403",
                    "",
                    "_register_alias_finder()",
                    "",
                    "__all__: List[str] = getattr(",
                    "    _orig_mod,",
                    '    "__all__",',
                    '    [k for k in dir(_orig_mod) if not k.startswith("_")],',
                    ")",
                    "",
                    "",
                    "def __getattr__(name: str) -> Any:",
                    '    """Delegate attribute access to the canonical module.',
                    "",
                    "    Args:",
                    "        name: Attribute name to look up.",
                    "",
                    "    Returns:",
                    "        The attribute value from the canonical module.",
                    '    """',
                    "    return getattr(_orig_mod, name)",
                    "",
                    "",
                    "def __dir__() -> List[str]:",
                    '    """Return the list of attributes exposed by the canonical module.',
                    "",
                    "    Returns:",
                    "        Sorted list of attribute names.",
                    '    """',
                    "    return dir(_orig_mod)",
                    "",
                ]
                with open(target_file, "w", encoding="utf-8") as fh:
                    fh.write(chr(10).join(parts))
            elif f == ".gitignore":
                target_file = os.path.join(target_dir, f)
                with open(os.path.join(root, f), encoding="utf-8") as sfh:
                    data = sfh.read()
                with open(target_file, "w", encoding="utf-8") as dfh:
                    dfh.write(data)


if __name__ == "__main__":
    generate_shims()
