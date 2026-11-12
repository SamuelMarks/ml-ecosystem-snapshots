"""Modular NVIDIA PTX TableGen lexer, parser, and catalog scraper subpackage."""

from __future__ import annotations

from ml_ecosystem_snapshots.tools.ptx.lexer import (
    LLVM_NVPTX_BASE_URL,
    NVPTX_TD_FILES,
    fetch_nvptx_td_file,
    tokenize_nvptx_td_defs,
)
from ml_ecosystem_snapshots.tools.ptx.parser import parse_nvptx_td_content
from ml_ecosystem_snapshots.tools.ptx.scraper import (
    build_exhaustive_ptx_catalog,
    build_ptx_arg_parser,
    scrape_ptx,
)

__all__ = [
    "LLVM_NVPTX_BASE_URL",
    "NVPTX_TD_FILES",
    "build_exhaustive_ptx_catalog",
    "build_ptx_arg_parser",
    "fetch_nvptx_td_file",
    "parse_nvptx_td_content",
    "scrape_ptx",
    "tokenize_nvptx_td_defs",
]
