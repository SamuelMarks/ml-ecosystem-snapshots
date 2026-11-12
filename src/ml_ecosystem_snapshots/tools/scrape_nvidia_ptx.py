"""CLI entrypoint and public facade for NVIDIA PTX ISA TableGen scraping."""

from __future__ import annotations

from ml_ecosystem_snapshots.models import IsaInstructionRecordModel
from ml_ecosystem_snapshots.tools.ptx import (
    LLVM_NVPTX_BASE_URL,
    NVPTX_TD_FILES,
    build_exhaustive_ptx_catalog,
    build_ptx_arg_parser,
    fetch_nvptx_td_file,
    parse_nvptx_td_content,
)
from ml_ecosystem_snapshots.tools.ptx.scraper import scrape_ptx as _scrape_ptx_impl


def scrape_ptx(
    td_dir: str | None = None,
    output_path: str | None = None,
    isa_version: str = "8.5",
    cuda_version: str = "12.6.0",
) -> list[IsaInstructionRecordModel]:
    """Scrape NVPTX TableGen definitions and write the exhaustive PTX ISA dump.

    Args:
        td_dir: Optional directory containing local NVPTX TableGen files.
        output_path: Target path to write exhaustive JSON.
        isa_version: Target PTX ISA version string (e.g. 8.5).
        cuda_version: Target CUDA toolkit version string (e.g. 12.6.0).

    Returns:
        List of generated instruction metadata records.
    """
    return _scrape_ptx_impl(
        td_dir=td_dir,
        output_path=output_path,
        isa_version=isa_version,
        cuda_version=cuda_version,
        fetch_fn=lambda fname, ldir: fetch_nvptx_td_file(fname, local_dir=ldir),
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for NVIDIA PTX ISA scraping.

    Args:
        argv: Optional list of command-line arguments.
    """
    args, _ = build_ptx_arg_parser().parse_known_args(argv)
    records = scrape_ptx(
        td_dir=args.td_dir,
        output_path=args.output_path,
        isa_version=args.isa_version,
        cuda_version=args.cuda_version,
    )
    print(f"Scraped and generated {len(records)} PTX instructions.")


__all__ = [
    "LLVM_NVPTX_BASE_URL",
    "NVPTX_TD_FILES",
    "build_exhaustive_ptx_catalog",
    "fetch_nvptx_td_file",
    "main",
    "parse_nvptx_td_content",
    "scrape_ptx",
]

if __name__ == "__main__":
    main()
