"""Script to generate backward-compatible ml_ecosystem_snapshots shim package.

Creates thin forwarding modules in src/ml_ecosystem_snapshots mirroring
src/ml_ecosystem_snapshots with 100% docstrings and full symbol re-exports.
"""

from __future__ import annotations

import os


def generate_shims() -> None:
    """Generate all shim files from src/ml_ecosystem_snapshots to src/ml_framework_snapshots."""
    eco_dir = os.path.join("src", "ml_ecosystem_snapshots")
    fw_dir = os.path.join("src", "ml_framework_snapshots")
    nl = chr(10)

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
                    "import sys",
                    "from typing import TYPE_CHECKING",
                    "import " + eco_target + " as _orig_mod",
                    "",
                    "if TYPE_CHECKING:",
                    "    from " + eco_target + " import *  # noqa: F401, F403",
                    "",
                    "# Re-export all attributes including internal and dunder methods",
                    "for _k in dir(_orig_mod):",
                    "    globals()[_k] = getattr(_orig_mod, _k)",
                    "",
                    "__all__ = getattr(",
                    "    _orig_mod,",
                    '    "__all__",',
                    '    [k for k in dir(_orig_mod) if not k.startswith("_")],',
                    ")",
                    "",
                    "sys.modules[__name__] = _orig_mod",
                    "",
                ]
                with open(target_file, "w", encoding="utf-8") as fh:
                    fh.write(nl.join(parts))
            elif f == ".gitignore":
                target_file = os.path.join(target_dir, f)
                with open(os.path.join(root, f), encoding="utf-8") as sfh:
                    data = sfh.read()
                with open(target_file, "w", encoding="utf-8") as dfh:
                    dfh.write(data)


if __name__ == "__main__":
    generate_shims()
