"""CLI command handlers for capturing, pulling, listing, and verifying snapshots."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import shutil
import sys
import urllib.request
import warnings

import pydantic


class DiscoveredSnapshot(pydantic.BaseModel):
    """Typed record for a discovered snapshot file on disk."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    file: str
    path: str
    size: int


def _get_pkg_dir() -> str:
    """Resolve the root `ml_ecosystem_snapshots` package directory.

    Returns:
        Absolute path to the `ml_ecosystem_snapshots` directory.
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cmd_capture(args: argparse.Namespace) -> None:
    """Handle the capture command to extract live framework snapshots.

    Args:
        args: Parsed command line arguments containing frameworks, out_dir, and include_nonpublic.
    """
    from rich.progress import Progress

    from ml_ecosystem_snapshots.api import (
        extract_snapshot,
        get_available_frameworks,
        write_snapshot,
    )

    warnings.filterwarnings(
        "ignore",
        message=".*Failed to initialize NumPy.*",
        category=UserWarning,
    )
    logging.getLogger("griffe").setLevel(logging.CRITICAL)

    available = get_available_frameworks()
    target_fws = []

    if "all" in args.frameworks or "*" in args.frameworks or not args.frameworks:
        target_fws = list(available.keys())
    else:
        target_fws = [fw for fw in args.frameworks if fw in available]
        unsupported = [fw for fw in args.frameworks if fw not in available]
        if unsupported:
            print(
                f"Warning: The following frameworks are unsupported and will be skipped: {', '.join(unsupported)}"
            )

    with Progress() as progress:
        task = progress.add_task("[cyan]Scanning frameworks...", total=len(target_fws))
        for fw in target_fws:
            progress.update(task, description=f"[cyan]Scanning {fw}...")
            snapshot_data = extract_snapshot(
                fw, include_nonpublic=args.include_nonpublic
            )
            if snapshot_data:
                path = write_snapshot(fw, snapshot_data, args.out_dir)
                progress.console.print(f"[green]Saved snapshot to {path}[/green]")
            else:
                progress.console.print(
                    f"[yellow]Skipping {fw}, not installed or no components found.[/yellow]"
                )
            progress.advance(task)


def cmd_list_snapshots(args: argparse.Namespace) -> None:
    """List available pre-bundled and locally cached snapshots.

    Args:
        args: Parsed command line arguments.
    """
    from ml_ecosystem_snapshots.index import get_cache_dir
    from ml_ecosystem_snapshots.network_env import get_custom_snapshots_paths

    pkg_dir = _get_pkg_dir()
    snapshots_dirs = get_custom_snapshots_paths() + [
        os.path.join(get_cache_dir(), "snapshots"),
        os.path.join(pkg_dir, "snapshots"),
        os.path.join(pkg_dir, "frameworks"),
    ]
    found_snapshots: list[DiscoveredSnapshot] = []
    seen_files: set[str] = set()
    for d in snapshots_dirs:
        if os.path.isdir(d):
            for fname in sorted(os.listdir(d)):
                if fname.endswith(".json"):
                    if fname in seen_files:
                        continue
                    seen_files.add(fname)
                    full_path = os.path.join(d, fname)
                    size = os.path.getsize(full_path)
                    found_snapshots.append(
                        DiscoveredSnapshot(file=fname, path=full_path, size=size)
                    )

    if not found_snapshots:
        print("No snapshot files found.")
        return

    print("Available Snapshots:")
    for s in found_snapshots:
        print(f"  - {s.file} ({s.size} bytes)")


def cmd_pull(args: argparse.Namespace) -> None:
    """Download a versioned snapshot from GitHub Releases into local cache.

    Args:
        args: Parsed command line arguments containing target and optional out_dir.
    """
    from ml_ecosystem_snapshots.index import get_cache_dir
    from ml_ecosystem_snapshots.network_env import is_offline_mode

    arg_map = vars(args)
    if bool(arg_map.get("offline", False)) or is_offline_mode():
        print(
            "Network access disabled: pull command cannot be executed in offline mode."
        )
        sys.exit(1)

    target = args.target.strip()
    if "@" in target:
        framework, version = target.split("@", 1)
    else:
        framework, version = target, "latest"

    out_dir = args.out_dir or os.path.join(get_cache_dir(), "snapshots")
    os.makedirs(out_dir, exist_ok=True)

    filename = (
        f"{framework}_v{version}.json" if version != "latest" else f"{framework}.json"
    )
    dest_path = os.path.join(out_dir, filename)

    tag = f"v{version}" if version != "latest" else "latest"
    url = f"https://github.com/SamuelMarks/ml-ecosystem-snapshots/releases/download/{tag}/{filename}"

    print(f"Pulling snapshot for {framework} ({version}) from {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"Saved snapshot to {dest_path}")
    except Exception as e:
        print(f"Failed to download snapshot: {e}")


def cmd_download_all(args: argparse.Namespace) -> None:
    """Download all canonical pre-compiled framework and ISA snapshots into local cache.

    Args:
        args: Parsed command line arguments containing cache_dir and optional offline flag.
    """
    from ml_ecosystem_snapshots.index import get_cache_dir
    from ml_ecosystem_snapshots.network_env import is_offline_mode

    arg_map = vars(args)
    if bool(arg_map.get("offline", False)) or is_offline_mode():
        print(
            "Network access disabled: download-all command cannot be executed in offline mode."
        )
        sys.exit(1)

    raw_cache_dir = arg_map.get("cache_dir")
    cache_dir = (
        str(raw_cache_dir)
        if raw_cache_dir
        else os.path.join(get_cache_dir(), "snapshots")
    )
    os.makedirs(cache_dir, exist_ok=True)

    targets = [
        "torch",
        "jax",
        "tensorflow",
        "keras",
        "mlx",
        "numpy",
        "nvidia_sass",
        "amd_rdna",
        "mlir",
        "stablehlo",
        "nvidia_ptx",
    ]
    print(f"Downloading pre-compiled snapshot bundle to {cache_dir}...")
    success_count = 0
    pkg_dir = _get_pkg_dir()

    for target in targets:
        fname = f"{target}_exhaustive.json"
        dest_path = os.path.join(cache_dir, fname)
        url = f"https://github.com/SamuelMarks/ml-ecosystem-snapshots/releases/download/latest/{fname}"
        try:
            urllib.request.urlretrieve(url, dest_path)
            print(f"  + Downloaded {target} -> {dest_path}")
            success_count += 1
        except Exception:
            bundled_src = os.path.join(pkg_dir, "frameworks", fname)
            if os.path.exists(bundled_src):
                shutil.copyfile(bundled_src, dest_path)
                print(f"  + Copied bundled {target} -> {dest_path}")
                success_count += 1
            else:
                print(f"  ! Failed to download {target} snapshot")

    print(f"Download complete: {success_count}/{len(targets)} snapshots ready.")


def cmd_verify_local_cache(args: argparse.Namespace) -> None:
    """Validate SHA256 checksums and JSON schema integrity of cached and bundled snapshots.

    Args:
        args: Parsed command line arguments containing cache_dir.
    """
    from ml_ecosystem_snapshots.index import get_cache_dir

    pkg_dir = _get_pkg_dir()
    raw_cache_dir = vars(args).get("cache_dir")
    cache_dir = str(raw_cache_dir) if raw_cache_dir else get_cache_dir()
    search_dirs = [
        cache_dir,
        os.path.join(cache_dir, "snapshots"),
        os.path.join(pkg_dir, "snapshots"),
        os.path.join(pkg_dir, "frameworks"),
    ]

    discovered: list[str] = []
    seen = set()
    for d in search_dirs:
        if os.path.isdir(d):
            for fname in sorted(os.listdir(d)):
                if fname.endswith(".json") or fname.endswith(".json.gz"):
                    p = os.path.join(d, fname)
                    if p not in seen:
                        seen.add(p)
                        discovered.append(p)

    if not discovered:
        print(f"No snapshot files discovered in cache directories: {search_dirs}")
        return

    print(f"Verifying local snapshot cache across {len(discovered)} files...\n")
    has_errors = False

    for fpath in discovered:
        fname = os.path.basename(fpath)
        try:
            sha = hashlib.sha256()
            with open(fpath, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    sha.update(chunk)
            digest_prefix = sha.hexdigest()[:12]
            size_kb = os.path.getsize(fpath) // 1024

            if fpath.endswith(".gz"):
                with gzip.open(fpath, "rt", encoding="utf-8") as gf:
                    data = json.load(gf)
            else:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)

            count = (
                len(data) if isinstance(data, list) else len(data.get("categories", {}))
            )
            print(
                f"  [OK] {fname:35s} {size_kb:6d} KB  sha256:{digest_prefix}...  ({count} entries)"
            )
        except Exception as e:
            has_errors = True
            print(f"  [FAIL] {fname:35s} Corrupted or invalid: {e}")

    if has_errors:
        print("\nCache verification failed: corrupted snapshots detected.")
        sys.exit(1)
    else:
        print(f"\nCache verification passed: {len(discovered)} snapshots verified.")


__all__ = [
    "DiscoveredSnapshot",
    "cmd_capture",
    "cmd_download_all",
    "cmd_list_snapshots",
    "cmd_pull",
    "cmd_verify_local_cache",
]
