"""Grain (PyGrain) API Snapshot Extractor.

Extracts deterministic, reproducible data loading (`DataLoader`, `MapDataset`,
`IterDataset`), sampling (`IndexSampler`, `SequentialSampler`), sharding
(`ShardOptions`, `ShardByJaxProcess`), transformations (`MapTransform`,
`RandomMapTransform`, `FilterTransform`, `Batch`), random-access data sources
(`ArrayRecordDataSource`, `InMemoryDataSource`), and Orbax checkpoint handlers
(`PyGrainCheckpointHandler`) from Google's Grain (`grain` / `grain.python`) library.
"""

from __future__ import annotations

import importlib
import types
import typing

from ml_switcheroo_ir.schema.ghost import (
    GhostRef,
    SemanticTier,
)

from ml_ecosystem_snapshots.frameworks._base import (
    CanonicalParamSpec,
    CanonicalSubmoduleOpSpec,
    build_canonical_python_ref,
    collect_dir_callables,
    scan_griffe_module_members,
)
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.static_parsers import (
    GriffeCacheEntry,
    load_griffe_module_safely,
)
from ml_ecosystem_snapshots.network_env import is_offline_mode


_GRAIN_OP_DEFS: list[
    tuple[
        str,
        str,
        str,
        str,
        list[tuple[str, str, str | None]],
        str,
    ]
] = [
    # MapDataset, IterDataset & Dataset Iterators (dataset)
    (
        "MapDataset",
        "grain.MapDataset",
        "class",
        "dataset",
        [
            ("parents", "POSITIONAL_OR_KEYWORD", "()"),
        ],
        "Dataset supporting random-access indexing (`__getitem__` and `__len__`) and lazy functional transformations.",
    ),
    (
        "source",
        "grain.MapDataset.source",
        "function",
        "dataset",
        [
            ("source", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Create a `MapDataset` wrapping a random-access data source.",
    ),
    (
        "range",
        "grain.MapDataset.range",
        "function",
        "dataset",
        [
            ("start", "POSITIONAL_OR_KEYWORD", None),
            ("stop", "POSITIONAL_OR_KEYWORD", "None"),
            ("step", "POSITIONAL_OR_KEYWORD", "1"),
        ],
        "Create a `MapDataset` producing integers in `range(start, stop, step)`.",
    ),
    (
        "mix",
        "grain.MapDataset.mix",
        "function",
        "dataset",
        [
            ("datasets", "POSITIONAL_OR_KEYWORD", None),
            ("weights", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Create a `MapDataset` that interleaves elements from multiple `MapDataset` parents according to `weights`.",
    ),
    (
        "select_from_datasets",
        "grain.MapDataset.select_from_datasets",
        "function",
        "dataset",
        [
            ("datasets", "POSITIONAL_OR_KEYWORD", None),
            ("selection_map", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Create a `MapDataset` selecting elements across `datasets` using `selection_map`.",
    ),
    (
        "map",
        "grain.MapDataset.map",
        "function",
        "dataset",
        [
            ("transform", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Apply a deterministic mapping transformation or callable to each element in a `MapDataset`.",
    ),
    (
        "random_map",
        "grain.MapDataset.random_map",
        "function",
        "dataset",
        [
            ("transform", "POSITIONAL_OR_KEYWORD", None),
            ("seed", "KEYWORD_ONLY", "None"),
        ],
        "Apply a reproducible seeded stochastic mapping transformation to each element in a `MapDataset`.",
    ),
    (
        "filter",
        "grain.MapDataset.filter",
        "function",
        "dataset",
        [
            ("transform", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Filter elements of a `MapDataset` according to a boolean predicate or `FilterTransform`.",
    ),
    (
        "batch",
        "grain.MapDataset.batch",
        "function",
        "dataset",
        [
            ("batch_size", "POSITIONAL_OR_KEYWORD", None),
            ("drop_remainder", "KEYWORD_ONLY", "False"),
            ("batch_fn", "KEYWORD_ONLY", "None"),
        ],
        "Batch consecutive elements of a `MapDataset` into arrays of size `batch_size`.",
    ),
    (
        "slice",
        "grain.MapDataset.slice",
        "function",
        "dataset",
        [
            ("sl", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Subset a `MapDataset` using a `slice` object.",
    ),
    (
        "shuffle",
        "grain.MapDataset.shuffle",
        "function",
        "dataset",
        [
            ("seed", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Deterministically shuffle element indices of a `MapDataset` in O(1) memory using `seed`.",
    ),
    (
        "repeat",
        "grain.MapDataset.repeat",
        "function",
        "dataset",
        [
            ("num_epochs", "POSITIONAL_OR_KEYWORD", "None"),
            ("reassemble_every_epoch", "KEYWORD_ONLY", "True"),
        ],
        "Repeat a `MapDataset` for `num_epochs` epochs (or indefinitely when `None`).",
    ),
    (
        "to_iter_dataset",
        "grain.MapDataset.to_iter_dataset",
        "function",
        "dataset",
        [
            ("read_options", "POSITIONAL_OR_KEYWORD", "None"),
            ("allow_nones", "KEYWORD_ONLY", "False"),
        ],
        "Convert a `MapDataset` into a sequential `IterDataset` with threaded prefetching.",
    ),
    (
        "mp_prefetch",
        "grain.MapDataset.mp_prefetch",
        "function",
        "dataset",
        [
            ("options", "POSITIONAL_OR_KEYWORD", "None"),
            ("worker_init_fn", "KEYWORD_ONLY", "None"),
            ("sequential_slice", "KEYWORD_ONLY", "False"),
        ],
        "Prefetch dataset elements across multiple background worker processes.",
    ),
    (
        "IterDataset",
        "grain.IterDataset",
        "class",
        "dataset",
        [
            ("parents", "POSITIONAL_OR_KEYWORD", "()"),
        ],
        "Iterable dataset producing checkpointable `DatasetIterator` streams.",
    ),
    (
        "mix",
        "grain.IterDataset.mix",
        "function",
        "dataset",
        [
            ("datasets", "POSITIONAL_OR_KEYWORD", None),
            ("weights", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Interleave elements from multiple `IterDataset` streams according to `weights`.",
    ),
    (
        "map",
        "grain.IterDataset.map",
        "function",
        "dataset",
        [
            ("transform", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Apply a deterministic mapping transformation to each element in an `IterDataset`.",
    ),
    (
        "random_map",
        "grain.IterDataset.random_map",
        "function",
        "dataset",
        [
            ("transform", "POSITIONAL_OR_KEYWORD", None),
            ("seed", "KEYWORD_ONLY", "None"),
        ],
        "Apply a seeded random mapping transformation to each element in an `IterDataset`.",
    ),
    (
        "filter",
        "grain.IterDataset.filter",
        "function",
        "dataset",
        [
            ("transform", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Filter elements of an `IterDataset` using a predicate or `FilterTransform`.",
    ),
    (
        "batch",
        "grain.IterDataset.batch",
        "function",
        "dataset",
        [
            ("batch_size", "POSITIONAL_OR_KEYWORD", None),
            ("drop_remainder", "KEYWORD_ONLY", "False"),
            ("batch_fn", "KEYWORD_ONLY", "None"),
        ],
        "Batch consecutive elements of an `IterDataset` into arrays of size `batch_size`.",
    ),
    (
        "prefetch",
        "grain.IterDataset.prefetch",
        "function",
        "dataset",
        [
            ("multiprocessing_options", "POSITIONAL_OR_KEYWORD", None),
            ("worker_init_fn", "KEYWORD_ONLY", "None"),
        ],
        "Prefetch elements of an `IterDataset` using multiprocess workers.",
    ),
    (
        "mp_prefetch",
        "grain.IterDataset.mp_prefetch",
        "function",
        "dataset",
        [
            ("options", "POSITIONAL_OR_KEYWORD", "None"),
            ("worker_init_fn", "KEYWORD_ONLY", "None"),
            ("sequential_slice", "KEYWORD_ONLY", "False"),
        ],
        "Prefetch elements of an `IterDataset` using `MultiprocessingOptions`.",
    ),
    (
        "DatasetIterator",
        "grain.DatasetIterator",
        "class",
        "dataset",
        [
            ("parents", "POSITIONAL_OR_KEYWORD", "()"),
        ],
        "Stateful, checkpointable iterator over an `IterDataset` supporting `get_state()` and `set_state()`.",
    ),
    (
        "PyGrainDatasetIterator",
        "grain.PyGrainDatasetIterator",
        "class",
        "dataset",
        [],
        "Checkpointable iterator returned by `grain.DataLoader.__iter__()`.",
    ),
    (
        "SelectionWithReplacementMap",
        "grain.SelectionWithReplacementMap",
        "class",
        "dataset",
        [],
        "Index selection map for sampling with replacement across mixed `MapDataset` instances.",
    ),
    # DataLoader & Functional load (data_loader)
    (
        "DataLoader",
        "grain.DataLoader",
        "class",
        "data_loader",
        [
            ("data_source", "KEYWORD_ONLY", None),
            ("sampler", "KEYWORD_ONLY", None),
            ("operations", "KEYWORD_ONLY", "()"),
            ("worker_count", "KEYWORD_ONLY", "0"),
            ("worker_buffer_size", "KEYWORD_ONLY", "1"),
            ("shard_options", "KEYWORD_ONLY", "None"),
            ("read_options", "KEYWORD_ONLY", "None"),
            ("enable_profiling", "KEYWORD_ONLY", "False"),
        ],
        "Deterministic multiprocess data loader combining a random-access data source, sampler, and transformation operations.",
    ),
    (
        "load",
        "grain.load",
        "function",
        "data_loader",
        [
            ("data_source", "POSITIONAL_OR_KEYWORD", None),
            ("num_epochs", "KEYWORD_ONLY", "None"),
            ("shuffle", "KEYWORD_ONLY", "False"),
            ("seed", "KEYWORD_ONLY", "None"),
            ("shard_options", "KEYWORD_ONLY", "None"),
            ("transformations", "KEYWORD_ONLY", "()"),
            ("batch_size", "KEYWORD_ONLY", "None"),
            ("drop_remainder", "KEYWORD_ONLY", "False"),
            ("worker_count", "KEYWORD_ONLY", "0"),
            ("read_options", "KEYWORD_ONLY", "None"),
        ],
        "High-level convenience factory constructing a configured `grain.DataLoader`.",
    ),
    # Samplers (samplers)
    (
        "Sampler",
        "grain.Sampler",
        "class",
        "samplers",
        [],
        "Protocol/base class for deterministic record index samplers.",
    ),
    (
        "IndexSampler",
        "grain.IndexSampler",
        "class",
        "samplers",
        [
            ("num_records", "POSITIONAL_OR_KEYWORD", None),
            ("shard_options", "POSITIONAL_OR_KEYWORD", None),
            ("shuffle", "POSITIONAL_OR_KEYWORD", "False"),
            ("num_epochs", "POSITIONAL_OR_KEYWORD", "None"),
            ("seed", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Default index sampler supporting sharding, multi-epoch iteration, and seeded global shuffling.",
    ),
    (
        "SequentialSampler",
        "grain.SequentialSampler",
        "class",
        "samplers",
        [
            ("num_records", "POSITIONAL_OR_KEYWORD", None),
            ("shard_options", "POSITIONAL_OR_KEYWORD", None),
            ("num_epochs", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Sampler yielding record indices sequentially within each shard.",
    ),
    # Sharding & Options (sharding)
    (
        "ShardOptions",
        "grain.ShardOptions",
        "class",
        "sharding",
        [
            ("shard_index", "POSITIONAL_OR_KEYWORD", None),
            ("shard_count", "POSITIONAL_OR_KEYWORD", None),
            ("drop_remainder", "POSITIONAL_OR_KEYWORD", "False"),
        ],
        "Configuration specifying `shard_index`, `shard_count`, and `drop_remainder` for distributed data loading.",
    ),
    (
        "ShardByJaxProcess",
        "grain.ShardByJaxProcess",
        "class",
        "sharding",
        [
            ("drop_remainder", "POSITIONAL_OR_KEYWORD", "False"),
        ],
        "Shard options automatically partitioning data across `jax.process_index()` and `jax.process_count()`.",
    ),
    (
        "NoSharding",
        "grain.NoSharding",
        "class",
        "sharding",
        [],
        "Single-shard `ShardOptions(shard_index=0, shard_count=1)` configuration.",
    ),
    (
        "ReadOptions",
        "grain.ReadOptions",
        "class",
        "sharding",
        [
            ("num_threads", "POSITIONAL_OR_KEYWORD", "16"),
            ("prefetch_buffer_size", "POSITIONAL_OR_KEYWORD", "500"),
        ],
        "Options controlling I/O reader thread pool size and prefetch buffer capacity.",
    ),
    (
        "MultiprocessingOptions",
        "grain.MultiprocessingOptions",
        "class",
        "sharding",
        [
            ("num_workers", "POSITIONAL_OR_KEYWORD", "0"),
            ("per_worker_buffer_size", "POSITIONAL_OR_KEYWORD", "1"),
            ("enable_profiling", "POSITIONAL_OR_KEYWORD", "False"),
        ],
        "Options controlling worker process count and per-worker output buffer size.",
    ),
    # Transformations & Operations (transforms)
    (
        "Transformation",
        "grain.Transformation",
        "class",
        "transforms",
        [],
        "Base class for all Grain dataset transformations.",
    ),
    (
        "MapTransform",
        "grain.MapTransform",
        "class",
        "transforms",
        [],
        "Abstract 1-to-1 deterministic element transformation (`map(self, element)`).",
    ),
    (
        "RandomMapTransform",
        "grain.RandomMapTransform",
        "class",
        "transforms",
        [],
        "Abstract 1-to-1 stochastic element transformation (`random_map(self, element, rng)`).",
    ),
    (
        "FilterTransform",
        "grain.FilterTransform",
        "class",
        "transforms",
        [],
        "Abstract element filter transformation (`filter(self, element) -> bool`).",
    ),
    (
        "FlatMapTransform",
        "grain.FlatMapTransform",
        "class",
        "transforms",
        [],
        "Abstract 1-to-N element splitting transformation (`flat_map(self, element)`).",
    ),
    (
        "Batch",
        "grain.Batch",
        "class",
        "transforms",
        [
            ("batch_size", "POSITIONAL_OR_KEYWORD", None),
            ("drop_remainder", "POSITIONAL_OR_KEYWORD", "False"),
            ("batch_fn", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Transformation batching consecutive elements into PyTrees of arrays with leading dimension `batch_size`.",
    ),
    (
        "Operation",
        "grain.Operation",
        "class",
        "transforms",
        [],
        "Base class for legacy stream operations in `grain.DataLoader`.",
    ),
    (
        "MapOperation",
        "grain.MapOperation",
        "class",
        "transforms",
        [
            ("map_function", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Operation applying `map_function` to `Record` streams.",
    ),
    (
        "RandomMapOperation",
        "grain.RandomMapOperation",
        "class",
        "transforms",
        [
            ("map_function", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Operation applying seeded `map_function` to `Record` streams.",
    ),
    (
        "FilterOperation",
        "grain.FilterOperation",
        "class",
        "transforms",
        [
            ("condition_function", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Operation filtering `Record` streams using `condition_function`.",
    ),
    (
        "BatchOperation",
        "grain.BatchOperation",
        "class",
        "transforms",
        [
            ("batch_size", "POSITIONAL_OR_KEYWORD", None),
            ("drop_remainder", "POSITIONAL_OR_KEYWORD", "False"),
        ],
        "Operation batching `Record` streams into batches of `batch_size`.",
    ),
    # Data Sources & Records (data_sources)
    (
        "RandomAccessDataSource",
        "grain.RandomAccessDataSource",
        "class",
        "data_sources",
        [],
        "Protocol for random-access data sources implementing `__len__` and `__getitem__`.",
    ),
    (
        "ArrayRecordDataSource",
        "grain.ArrayRecordDataSource",
        "class",
        "data_sources",
        [
            ("paths", "POSITIONAL_OR_KEYWORD", None),
        ],
        "High-performance random-access data source backed by `.array_record` files.",
    ),
    (
        "InMemoryDataSource",
        "grain.InMemoryDataSource",
        "class",
        "data_sources",
        [
            ("elements", "POSITIONAL_OR_KEYWORD", "()"),
        ],
        "In-memory random-access data source wrapping a sequence of elements.",
    ),
    (
        "RangeDataSource",
        "grain.RangeDataSource",
        "class",
        "data_sources",
        [
            ("start", "POSITIONAL_OR_KEYWORD", None),
            ("stop", "POSITIONAL_OR_KEYWORD", None),
            ("step", "POSITIONAL_OR_KEYWORD", "1"),
        ],
        "Random-access data source yielding integers in `range(start, stop, step)`.",
    ),
    (
        "Record",
        "grain.Record",
        "class",
        "data_sources",
        [
            ("metadata", "POSITIONAL_OR_KEYWORD", None),
            ("data", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Container bundling record `data` with its `RecordMetadata`.",
    ),
    (
        "RecordMetadata",
        "grain.RecordMetadata",
        "class",
        "data_sources",
        [
            ("index", "POSITIONAL_OR_KEYWORD", None),
            ("rng", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Metadata tracking global record `index` and per-record PRNG state.",
    ),
    # Orbax Checkpoint Integration (checkpoint)
    (
        "PyGrainCheckpointHandler",
        "grain.PyGrainCheckpointHandler",
        "class",
        "checkpoint",
        [],
        "Orbax `CheckpointHandler` for saving and restoring `DatasetIterator` or `PyGrainDatasetIterator` state.",
    ),
    (
        "PyGrainCheckpointSave",
        "grain.PyGrainCheckpointSave",
        "class",
        "checkpoint",
        [
            ("item", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Orbax `CheckpointArgs` wrapper for saving a Grain dataset iterator.",
    ),
    (
        "PyGrainCheckpointRestore",
        "grain.PyGrainCheckpointRestore",
        "class",
        "checkpoint",
        [
            ("item", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Orbax `CheckpointArgs` wrapper for restoring a Grain dataset iterator in-place.",
    ),
    # Experimental Datasets (experimental)
    (
        "FirstFitPackIterDataset",
        "grain.experimental.FirstFitPackIterDataset",
        "class",
        "experimental",
        [
            ("parent", "POSITIONAL_OR_KEYWORD", None),
            ("length_struct", "KEYWORD_ONLY", None),
            ("num_packing_bins", "KEYWORD_ONLY", None),
            ("shuffle_bins", "KEYWORD_ONLY", "True"),
            ("padding_struct", "KEYWORD_ONLY", "None"),
            ("max_sequences_per_bin", "KEYWORD_ONLY", "None"),
        ],
        "Experimental `IterDataset` packing variable-length sequences into fixed-length bins using first-fit bin packing.",
    ),
    (
        "InterleaveIterDataset",
        "grain.experimental.InterleaveIterDataset",
        "class",
        "experimental",
        [
            ("datasets", "POSITIONAL_OR_KEYWORD", None),
            ("cycle_length", "KEYWORD_ONLY", None),
            ("num_make_iter_threads", "KEYWORD_ONLY", "1"),
            ("iter_buffer_size", "KEYWORD_ONLY", "1"),
        ],
        "Experimental `IterDataset` concurrently interleaving elements from a sequence of datasets.",
    ),
    (
        "ThreadPrefetchIterDataset",
        "grain.experimental.ThreadPrefetchIterDataset",
        "class",
        "experimental",
        [
            ("parent", "POSITIONAL_OR_KEYWORD", None),
            ("prefetch_buffer_size", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Experimental `IterDataset` prefetching elements on a background thread.",
    ),
    (
        "FlatMapIterDataset",
        "grain.experimental.FlatMapIterDataset",
        "class",
        "experimental",
        [
            ("parent", "POSITIONAL_OR_KEYWORD", None),
            ("transform", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Experimental `IterDataset` applying a `FlatMapTransform` to each parent element.",
    ),
]


def _build_grain_canonical_ops(
    defs: typing.Sequence[
        tuple[
            str,
            str,
            str,
            str,
            list[tuple[str, str, str | None]],
            str,
        ]
    ],
) -> list[CanonicalSubmoduleOpSpec]:
    """Convert compact Grain operation tuples into `CanonicalSubmoduleOpSpec` models.

    Args:
        defs: Sequence of `(name, api_path, kind, submodule, params, docstring)` tuples.

    Returns:
        List of validated `CanonicalSubmoduleOpSpec` models.
    """
    return [
        CanonicalSubmoduleOpSpec(
            name=name,
            api_path=api_path,
            kind=kind,
            submodule=submodule,
            params=[
                CanonicalParamSpec(name=pn, kind=pk, default=pd)
                for pn, pk, pd in params
            ],
            docstring=doc,
        )
        for name, api_path, kind, submodule, params, doc in defs
    ]


CANONICAL_GRAIN_OPS: list[CanonicalSubmoduleOpSpec] = _build_grain_canonical_ops(
    _GRAIN_OP_DEFS
)


def _get_grain_module() -> dict[str, types.ModuleType | None]:
    """Safely import `grain`, `grain.python`, and `grain.experimental` when not in offline mode.

    Returns:
        Dictionary mapping `'core'`, `'python'`, and `'experimental'` to imported modules or `None`.
    """
    mods: dict[str, types.ModuleType | None] = {
        "core": None,
        "python": None,
        "experimental": None,
    }
    if is_offline_mode():
        return mods
    for key, mod_path in (
        ("core", "grain"),
        ("python", "grain.python"),
        ("experimental", "grain.experimental"),
    ):
        try:
            mods[key] = importlib.import_module(mod_path)
        except Exception:
            pass
    return mods


def _get_grain_griffe_module() -> GriffeCacheEntry:
    """Statically load `grain` via the custom Griffe fork when not in offline mode.

    Returns:
        Loaded Griffe AST node for `grain` or `None` if unavailable or in offline mode.
    """
    if is_offline_mode():
        return None
    return load_griffe_module_safely("grain", framework_name="grain")


def _attach_grain_python_alias(ref: GhostRef) -> None:
    """Attach `grain.python.<suffix>` alias to a `grain.<suffix>` `GhostRef`.

    Args:
        ref: Target `GhostRef` to enrich with `grain.python.*` alias.
    """
    if ref.api_path.startswith("grain.") and not ref.api_path.startswith(
        "grain.python."
    ):
        suffix = ref.api_path[len("grain.") :]
        py_alias = f"grain.python.{suffix}"
        existing = list(ref.aliases or [])
        if py_alias not in existing:
            existing.append(py_alias)
            ref.aliases = existing


def collect_grain_api(include_nonpublic: bool = False) -> list[GhostRef]:
    """Collect Grain (`grain` / `grain.python`) dataset, sampler, and transformation APIs.

    Args:
        include_nonpublic: Whether to include underscore-prefixed private members.

    Returns:
        List of `GhostRef` items representing the Grain API surface.
    """
    env_tags = ("grain", "pygrain", "jax", "dataloader")
    results: list[GhostRef] = []
    seen_paths: set[str] = set()

    if not is_offline_mode():
        live_mods = _get_grain_module()
        for key, prefix in (
            ("core", "grain"),
            ("python", "grain"),
            ("experimental", "grain.experimental"),
        ):
            mod = live_mods.get(key)
            if mod is not None:
                for ref in collect_dir_callables(
                    mod,
                    api_prefix=prefix,
                    include_nonpublic=include_nonpublic,
                    inspector_fn=GhostInspector.inspect,
                ):
                    if ref.api_path not in seen_paths:
                        ref.environment_tags = list(ref.environment_tags or ()) + [
                            t for t in env_tags if t not in (ref.environment_tags or ())
                        ]
                        _attach_grain_python_alias(ref)
                        results.append(ref)
                        seen_paths.add(ref.api_path)

        if _get_grain_griffe_module() is not None:
            for ref in scan_griffe_module_members(
                module_path="grain",
                api_prefix="grain",
                include_nonpublic=include_nonpublic,
                environment_tags=env_tags,
                framework_name="grain",
                inspector_fn=GhostInspector.inspect,
            ):
                if ref.api_path not in seen_paths:
                    _attach_grain_python_alias(ref)
                    results.append(ref)
                    seen_paths.add(ref.api_path)

    for spec in CANONICAL_GRAIN_OPS:
        if spec.api_path not in seen_paths:
            ref = build_canonical_python_ref(
                spec,
                default_doc_prefix="Grain",
                environment_tags=env_tags,
                domain_metadata={"submodule": spec.submodule or "dataset"},
            )
            _attach_grain_python_alias(ref)
            results.append(ref)
            seen_paths.add(spec.api_path)

    return results


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> list[GhostRef]:
    """Collect Grain API definitions for the requested semantic tier.

    Args:
        category: Target `SemanticTier` category (`ARRAY_API`).
        include_nonpublic: Whether to include non-public symbols.

    Returns:
        List of `GhostRef` objects for `SemanticTier.ARRAY_API`, or `[]` otherwise.
    """
    if category == SemanticTier.ARRAY_API:
        return collect_grain_api(include_nonpublic=include_nonpublic)
    return []


__all__ = [
    "CANONICAL_GRAIN_OPS",
    "_GRAIN_OP_DEFS",
    "_attach_grain_python_alias",
    "_build_grain_canonical_ops",
    "_get_grain_griffe_module",
    "_get_grain_module",
    "collect_api",
    "collect_grain_api",
]
