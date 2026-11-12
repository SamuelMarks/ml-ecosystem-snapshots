"""NVIDIA PTX catalog builder and TableGen scraper orchestration."""

from __future__ import annotations

import argparse
import json
import os
import typing

from ml_ecosystem_snapshots.models import IsaInstructionRecordModel, JsonValue
from ml_ecosystem_snapshots.tools.ptx._memory_ops import PTX_MEMORY_OPS
from ml_ecosystem_snapshots.tools.ptx.lexer import NVPTX_TD_FILES, fetch_nvptx_td_file
from ml_ecosystem_snapshots.tools.ptx.parser import PTX_ARITH_OPS, parse_nvptx_td_content

PTX_SYNC_AND_TENSOR_OPS: list[dict[str, JsonValue]] = [
    # Barrier & Warp Synchronization
    {
        "mnemonic": "bar.sync",
        "description": "Synchronize threads in a CTA at a named barrier.",
        "category": "barrier",
        "supported_types": [".u32"],
        "vector_widths": [],
        "state_spaces": [],
        "operands": [
            {
                "name": "name",
                "role": "src0",
                "description": "Barrier index or count",
            }
        ],
        "min_sm": "sm_50",
    },
    {
        "mnemonic": "bar.arrive",
        "description": "Signal arrival at a barrier without blocking thread execution.",
        "category": "barrier",
        "supported_types": [".u32"],
        "vector_widths": [],
        "state_spaces": [],
        "operands": [{"name": "name", "role": "src0", "description": "Barrier index"}],
        "min_sm": "sm_50",
    },
    {
        "mnemonic": "bar.red",
        "description": "Perform reduction across threads arriving at a barrier.",
        "category": "barrier",
        "supported_types": [".pred"],
        "vector_widths": [],
        "state_spaces": [],
        "operands": [
            {"name": "p", "role": "dst", "description": "Result predicate"},
            {"name": "name", "role": "src0", "description": "Barrier index"},
            {
                "name": "pred_val",
                "role": "src1",
                "description": "Input predicate value",
            },
        ],
        "min_sm": "sm_50",
    },
    {
        "mnemonic": "barrier.cluster",
        "description": "Synchronize all CTAs within a Thread Block Cluster.",
        "category": "barrier",
        "supported_types": [],
        "vector_widths": [],
        "state_spaces": [],
        "operands": [],
        "min_sm": "sm_90",
    },
    {
        "mnemonic": "mbarrier.init",
        "description": "Initialize an asynchronous memory barrier object in shared memory.",
        "category": "barrier",
        "supported_types": [".b64"],
        "vector_widths": [],
        "state_spaces": [".shared"],
        "operands": [
            {
                "name": "addr",
                "role": "dst",
                "description": "Shared memory address of barrier",
            },
            {
                "name": "count",
                "role": "src0",
                "description": "Expected arrival count",
            },
        ],
        "min_sm": "sm_80",
    },
    {
        "mnemonic": "mbarrier.arrive",
        "description": "Signal arrival at an asynchronous memory barrier object.",
        "category": "barrier",
        "supported_types": [".b64"],
        "vector_widths": [],
        "state_spaces": [".shared"],
        "operands": [
            {
                "name": "state",
                "role": "dst",
                "description": "Output arrival token/state",
            },
            {
                "name": "addr",
                "role": "src0",
                "description": "Shared memory address of barrier",
            },
        ],
        "min_sm": "sm_80",
    },
    {
        "mnemonic": "mbarrier.test_wait",
        "description": "Test whether all expected threads have arrived at an mbarrier object.",
        "category": "barrier",
        "supported_types": [".b64"],
        "vector_widths": [],
        "state_spaces": [".shared"],
        "operands": [
            {
                "name": "p",
                "role": "dst",
                "description": "Result boolean predicate",
            },
            {
                "name": "addr",
                "role": "src0",
                "description": "Shared memory address of barrier",
            },
            {"name": "state", "role": "src1", "description": "Arrival token/state"},
        ],
        "min_sm": "sm_80",
    },
    # Tensor Core Matrix Operations
    {
        "mnemonic": "wmma.mma",
        "description": "Warp-level matrix multiply and accumulate on Tensor Cores (Volta / Turing / Ampere).",
        "category": "tensor",
        "supported_types": [
            ".f16",
            ".f32",
            ".s8",
            ".u8",
            ".s32",
            ".bf16",
            ".tf32",
        ],
        "vector_widths": [],
        "state_spaces": [],
        "operands": [
            {"name": "d", "role": "dst", "description": "Accumulator destination"},
            {"name": "a", "role": "src0", "description": "Matrix A fragment"},
            {"name": "b", "role": "src1", "description": "Matrix B fragment"},
            {"name": "c", "role": "src2", "description": "Matrix C accumulator"},
        ],
        "min_sm": "sm_70",
    },
    {
        "mnemonic": "mma.sync",
        "description": "Synchronous warp-level matrix multiply-accumulate across threads.",
        "category": "tensor",
        "supported_types": [
            ".f16",
            ".f32",
            ".s8",
            ".u8",
            ".s32",
            ".bf16",
            ".tf32",
            ".e4m3",
            ".e5m2",
        ],
        "vector_widths": [],
        "state_spaces": [],
        "operands": [
            {"name": "d", "role": "dst", "description": "Accumulator destination"},
            {"name": "a", "role": "src0", "description": "Matrix A fragment"},
            {"name": "b", "role": "src1", "description": "Matrix B fragment"},
            {"name": "c", "role": "src2", "description": "Matrix C accumulator"},
        ],
        "min_sm": "sm_80",
    },
    {
        "mnemonic": "wgmma.mma_async",
        "description": "Warpgroup-level asynchronous matrix multiply and accumulate (Hopper).",
        "category": "tensor",
        "supported_types": [
            ".f16",
            ".f32",
            ".bf16",
            ".tf32",
            ".e4m3",
            ".e5m2",
            ".s32",
        ],
        "vector_widths": [],
        "state_spaces": [],
        "operands": [
            {"name": "d", "role": "dst", "description": "Accumulator destination"},
            {
                "name": "a",
                "role": "src0",
                "description": "Matrix A shared memory descriptor or register fragment",
            },
            {
                "name": "b",
                "role": "src1",
                "description": "Matrix B shared memory descriptor",
            },
        ],
        "min_sm": "sm_90",
    },
    {
        "mnemonic": "wgmma.fence",
        "description": "Enforces ordering between register accesses and wgmma.mma_async instructions.",
        "category": "tensor",
        "supported_types": [],
        "vector_widths": [],
        "state_spaces": [],
        "operands": [],
        "min_sm": "sm_90",
    },
    {
        "mnemonic": "wgmma.commit_group",
        "description": "Commits all prior wgmma.mma_async instructions into a warpgroup batch.",
        "category": "tensor",
        "supported_types": [],
        "vector_widths": [],
        "state_spaces": [],
        "operands": [],
        "min_sm": "sm_90",
    },
    {
        "mnemonic": "wgmma.wait_group",
        "description": "Waits until at most N warpgroup batches remain outstanding.",
        "category": "tensor",
        "supported_types": [],
        "vector_widths": [],
        "state_spaces": [],
        "operands": [
            {
                "name": "n",
                "role": "src0",
                "description": "Number of remaining pending batches",
            }
        ],
        "min_sm": "sm_90",
    },
    {
        "mnemonic": "ldmatrix",
        "description": "Load matrix fragment collectively across warp from shared memory into registers.",
        "category": "tensor",
        "supported_types": [".b16"],
        "vector_widths": [".v2", ".v4"],
        "state_spaces": [".shared"],
        "operands": [
            {
                "name": "d",
                "role": "dst",
                "description": "Destination register tuple",
            },
            {
                "name": "a",
                "role": "src0",
                "description": "Shared memory base address",
            },
        ],
        "min_sm": "sm_75",
    },
    {
        "mnemonic": "stmatrix",
        "description": "Store matrix fragment collectively across warp from registers into shared memory.",
        "category": "tensor",
        "supported_types": [".b16"],
        "vector_widths": [".v2", ".v4"],
        "state_spaces": [".shared"],
        "operands": [
            {
                "name": "a",
                "role": "dst",
                "description": "Shared memory base address",
            },
            {"name": "b", "role": "src0", "description": "Source register tuple"},
        ],
        "min_sm": "sm_90",
    },
    # Control Flow Operations
    {
        "mnemonic": "bra",
        "description": "Branch to a target label conditionally or unconditionally.",
        "category": "control",
        "supported_types": [".uni"],
        "vector_widths": [],
        "state_spaces": [],
        "operands": [{"name": "tgt", "role": "src0", "description": "Target label"}],
        "min_sm": "sm_50",
    },
    {
        "mnemonic": "ret",
        "description": "Return execution from kernel or device function.",
        "category": "control",
        "supported_types": [".uni"],
        "vector_widths": [],
        "state_spaces": [],
        "operands": [],
        "min_sm": "sm_50",
    },
]


def build_exhaustive_ptx_catalog(
    base_instructions: typing.Sequence[IsaInstructionRecordModel | typing.Mapping[str, JsonValue]] | None = None,
) -> list[IsaInstructionRecordModel]:
    """Build the comprehensive NVIDIA PTX ISA 7.x / 8.x instruction catalog.

    Args:
        base_instructions: Optional base list of instructions from TableGen scraping.

    Returns:
        Exhaustive list of PTX instruction metadata records.
    """
    catalog: dict[str, IsaInstructionRecordModel] = {}
    if base_instructions:
        for inst in base_instructions:
            if isinstance(inst, IsaInstructionRecordModel):
                if inst.mnemonic and inst.mnemonic not in catalog:
                    catalog[inst.mnemonic] = inst
            else:
                m = inst.get("mnemonic")
                if isinstance(m, str) and m and m not in catalog:
                    catalog[m] = IsaInstructionRecordModel.model_validate(dict(inst))

    for op in PTX_ARITH_OPS + PTX_MEMORY_OPS + PTX_SYNC_AND_TENSOR_OPS:
        catalog[str(op["mnemonic"])] = IsaInstructionRecordModel.model_validate(op)

    return list(catalog.values())


def scrape_ptx(
    td_dir: str | None = None,
    output_path: str | None = None,
    isa_version: str = "8.5",
    cuda_version: str = "12.6.0",
    fetch_fn: typing.Callable[[str, str | None], str | None] | None = None,
) -> list[IsaInstructionRecordModel]:
    """Scrape NVPTX TableGen definitions and write the exhaustive PTX ISA dump.

    Args:
        td_dir: Optional directory containing local NVPTX TableGen files.
        output_path: Target path to write exhaustive JSON.
        isa_version: Target PTX ISA version string (e.g. 8.5).
        cuda_version: Target CUDA toolkit version string (e.g. 12.6.0).
        fetch_fn: Optional custom TableGen file fetcher callback.

    Returns:
        List of generated instruction metadata records.
    """
    fetcher = fetch_fn or fetch_nvptx_td_file
    base_instructions: list[IsaInstructionRecordModel] = []
    for td_file in NVPTX_TD_FILES:
        content = fetcher(td_file, td_dir)
        if content:
            base_instructions.extend(parse_nvptx_td_content(content))

    catalog = build_exhaustive_ptx_catalog(base_instructions)
    resolved_output = output_path or os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "frameworks",
        "nvidia_ptx_exhaustive.json",
    )

    exhaustive_data: dict[str, JsonValue] = {
        "schema_version": "2.0.0",
        "target": "nvidia_ptx",
        "version": isa_version,
        "upstream_version": f"cuda-{cuda_version}",
        "upstream_commit": f"cuda-{cuda_version}",
        "source_type": "tablegen",
        "supported_microarchitectures": [
            "sm_50",
            "sm_60",
            "sm_70",
            "sm_75",
            "sm_80",
            "sm_86",
            "sm_89",
            "sm_90",
            "sm_100",
        ],
        "instructions": [r.model_dump(exclude_none=True) for r in catalog],
    }

    nl = chr(10)
    with open(resolved_output, "w", encoding="utf-8") as f:
        json.dump(exhaustive_data, f, indent=2, sort_keys=True)
        f.write(nl)

    return catalog


def build_ptx_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for PTX TableGen scraping.

    Returns:
        Configured `argparse.ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser(description="Scrape NVIDIA PTX ISA definitions into JSON.")
    parser.add_argument("--td-dir", default=None, help="Optional local path to LLVM NVPTX TableGen files.")
    parser.add_argument(
        "--output-path",
        "--output",
        dest="output_path",
        default=None,
        help="Optional output path for nvidia_ptx_exhaustive.json.",
    )
    parser.add_argument("--isa-version", default="8.5", help="Target PTX ISA version string (default: 8.5).")
    parser.add_argument(
        "--cuda-version", default="12.6.0", help="Target CUDA toolkit version string (default: 12.6.0)."
    )
    return parser


__all__ = [
    "PTX_SYNC_AND_TENSOR_OPS",
    "build_exhaustive_ptx_catalog",
    "build_ptx_arg_parser",
    "scrape_ptx",
]
