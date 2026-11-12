"""TableGen source lexer and fetcher for NVIDIA PTX instruction definitions."""

from __future__ import annotations

import os
import re

from ml_ecosystem_snapshots.network_env import fetch_url

LLVM_NVPTX_BASE_URL = "https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/lib/Target/NVPTX"
NVPTX_TD_FILES = [
    "NVPTXInstrInfo.td",
    "NVPTXIntrinsics.td",
]

_PTX_DEF_PATTERN = re.compile(r'def\s+([A-Za-z0-9_]+)\s*:\s*[A-Za-z0-9_]*\s*<.*?"([a-z0-9_\.]+)"')


def fetch_nvptx_td_file(filename: str, local_dir: str | None = None) -> str:
    """Fetch an NVPTX TableGen file from a local directory or upstream LLVM GitHub.

    Args:
        filename: Name of the .td file (e.g. 'NVPTXInstrInfo.td').
        local_dir: Optional local path containing LLVM NVPTX TableGen files.

    Returns:
        File text content, or empty string if unavailable.
    """
    if local_dir:
        local_path = os.path.join(local_dir, filename)
        if os.path.isfile(local_path):
            with open(local_path, encoding="utf-8") as f:
                return f.read()

    url = f"{LLVM_NVPTX_BASE_URL}/{filename}"
    return fetch_url(url, timeout=15) or ""


def tokenize_nvptx_td_defs(content: str) -> list[tuple[str, str]]:
    """Strip comments and tokenize `(def_name, mnemonic)` pairs from NVPTX TableGen content.

    Args:
        content: Raw NVPTX TableGen text content.

    Returns:
        Ordered list of `(def_name, clean_mnemonic)` tuples.
    """
    stripped = re.sub(r"//.*", "", content)
    stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.DOTALL)
    tokens: list[tuple[str, str]] = []
    for match in _PTX_DEF_PATTERN.finditer(stripped):
        def_name = match.group(1)
        mnemonic = match.group(2)
        tokens.append((def_name, mnemonic))
    return tokens


__all__ = [
    "LLVM_NVPTX_BASE_URL",
    "NVPTX_TD_FILES",
    "fetch_nvptx_td_file",
    "tokenize_nvptx_td_defs",
]
