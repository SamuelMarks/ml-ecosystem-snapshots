"""Orbax Checkpoint & Export API Snapshot Extractor.

Extracts checkpoint managers (`CheckpointManager`, `CheckpointManagerOptions`),
checkpointers (`Checkpointer`, `AsyncCheckpointer`, `PyTreeCheckpointer`,
`StandardCheckpointer`), handlers (`PyTreeCheckpointHandler`,
`StandardCheckpointHandler`, `JsonCheckpointHandler`, `ArrayCheckpointHandler`,
`ProtoCheckpointHandler`, `CompositeCheckpointHandler`), declarative save/restore
arguments (`orbax.checkpoint.args.*`), restoration utilities, Orbax v1 functional
APIs (`save_pytree`, `load_pytree`), and Orbax Export (`ExportManager`,
`JaxModule`, `ServingConfig`) using live introspection, the custom Griffe fork,
and canonical offline specifications.
"""

from __future__ import annotations

import importlib
import inspect
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
    scan_griffe_module_members,
)
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.static_parsers import (
    GriffeCacheEntry,
    load_griffe_module_safely,
)
from ml_ecosystem_snapshots.network_env import is_offline_mode
from ml_ecosystem_snapshots.utils import get_all_members

ocp: types.ModuleType | None = None
try:
    import orbax.checkpoint as _ocp_mod

    ocp = _ocp_mod
except Exception:
    pass


_ORBAX_OP_DEFS: list[
    tuple[
        str,
        str,
        str,
        str,
        list[tuple[str, str, str | None]],
        str,
    ]
] = [
    # Checkpointers & CheckpointManager (checkpoint)
    (
        "AbstractCheckpointer",
        "orbax.checkpoint.AbstractCheckpointer",
        "class",
        "checkpoint",
        [],
        "Abstract interface for synchronous and asynchronous Orbax checkpointers.",
    ),
    (
        "Checkpointer",
        "orbax.checkpoint.Checkpointer",
        "class",
        "checkpoint",
        [
            ("handler", "POSITIONAL_OR_KEYWORD", None),
            ("primary_host", "KEYWORD_ONLY", "0"),
            ("active_processes", "KEYWORD_ONLY", "None"),
            ("barrier_sync_fn", "KEYWORD_ONLY", "None"),
            ("path_permission_mode", "KEYWORD_ONLY", "None"),
            ("checkpoint_metadata_store", "KEYWORD_ONLY", "None"),
            ("temporary_path_cls", "KEYWORD_ONLY", "None"),
        ],
        "Synchronous checkpointer wrapping a `CheckpointHandler` for atomic step saves and restores.",
    ),
    (
        "AsyncCheckpointer",
        "orbax.checkpoint.AsyncCheckpointer",
        "class",
        "checkpoint",
        [
            ("handler", "POSITIONAL_OR_KEYWORD", None),
            ("timeout_secs", "POSITIONAL_OR_KEYWORD", "300"),
            ("primary_host", "KEYWORD_ONLY", "0"),
            ("active_processes", "KEYWORD_ONLY", "None"),
            ("barrier_sync_fn", "KEYWORD_ONLY", "None"),
            ("post_finalization_callback", "KEYWORD_ONLY", "None"),
            ("path_permission_mode", "KEYWORD_ONLY", "None"),
            ("async_options", "KEYWORD_ONLY", "None"),
            ("multiprocessing_options", "KEYWORD_ONLY", "None"),
            ("temporary_path_cls", "KEYWORD_ONLY", "None"),
        ],
        "Asynchronous checkpointer executing background device-to-host and disk I/O transfers.",
    ),
    (
        "PyTreeCheckpointer",
        "orbax.checkpoint.PyTreeCheckpointer",
        "class",
        "checkpoint",
        [
            ("primary_host", "POSITIONAL_OR_KEYWORD", "0"),
            ("use_ocdbt", "POSITIONAL_OR_KEYWORD", "True"),
            ("use_zarr3", "KEYWORD_ONLY", "False"),
        ],
        "Convenience checkpointer pre-configured with `PyTreeCheckpointHandler`.",
    ),
    (
        "StandardCheckpointer",
        "orbax.checkpoint.StandardCheckpointer",
        "class",
        "checkpoint",
        [
            ("primary_host", "POSITIONAL_OR_KEYWORD", "0"),
            ("multiprocessing_options", "KEYWORD_ONLY", "None"),
            ("async_options", "KEYWORD_ONLY", "None"),
        ],
        "Async checkpointer pre-configured with `StandardCheckpointHandler` for JAX PyTrees.",
    ),
    (
        "CheckpointManager",
        "orbax.checkpoint.CheckpointManager",
        "class",
        "checkpoint",
        [
            ("directory", "POSITIONAL_OR_KEYWORD", None),
            ("checkpointers", "POSITIONAL_OR_KEYWORD", "None"),
            ("options", "POSITIONAL_OR_KEYWORD", "None"),
            ("metadata", "POSITIONAL_OR_KEYWORD", "None"),
            ("item_names", "POSITIONAL_OR_KEYWORD", "None"),
            ("item_handlers", "POSITIONAL_OR_KEYWORD", "None"),
            ("logger", "POSITIONAL_OR_KEYWORD", "None"),
            ("handler_registry", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "High-level manager orchestrating step-numbered checkpoints, retention policies, metrics, and composite items.",
    ),
    (
        "CheckpointManagerOptions",
        "orbax.checkpoint.CheckpointManagerOptions",
        "class",
        "checkpoint",
        [
            ("save_interval_steps", "POSITIONAL_OR_KEYWORD", "1"),
            ("max_to_keep", "POSITIONAL_OR_KEYWORD", "None"),
            ("keep_time_interval", "POSITIONAL_OR_KEYWORD", "None"),
            ("keep_period", "POSITIONAL_OR_KEYWORD", "None"),
            ("should_save_fn", "POSITIONAL_OR_KEYWORD", "None"),
            ("best_fn", "POSITIONAL_OR_KEYWORD", "None"),
            ("best_mode", "POSITIONAL_OR_KEYWORD", "'max'"),
            ("keep_checkpoints_without_metrics", "POSITIONAL_OR_KEYWORD", "True"),
            ("step_prefix", "POSITIONAL_OR_KEYWORD", "None"),
            ("step_format_fixed_length", "POSITIONAL_OR_KEYWORD", "None"),
            ("step_name_format", "POSITIONAL_OR_KEYWORD", "None"),
            ("create", "POSITIONAL_OR_KEYWORD", "True"),
            ("cleanup_tmp_directories", "POSITIONAL_OR_KEYWORD", "False"),
            ("save_on_steps", "POSITIONAL_OR_KEYWORD", "None"),
            ("single_host_load_and_broadcast", "POSITIONAL_OR_KEYWORD", "False"),
            ("todelete_subdir", "POSITIONAL_OR_KEYWORD", "None"),
            ("enable_background_delete", "POSITIONAL_OR_KEYWORD", "False"),
            ("read_only", "POSITIONAL_OR_KEYWORD", "False"),
            ("enable_async_checkpointing", "POSITIONAL_OR_KEYWORD", "True"),
            ("async_options", "POSITIONAL_OR_KEYWORD", "None"),
            ("multiprocessing_options", "POSITIONAL_OR_KEYWORD", "None"),
            ("save_decision_policy", "POSITIONAL_OR_KEYWORD", "None"),
            ("preservation_policy", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Configuration options controlling `CheckpointManager` save frequency, retention, naming, and async execution.",
    ),
    (
        "AsyncOptions",
        "orbax.checkpoint.AsyncOptions",
        "class",
        "checkpoint",
        [
            ("timeout_secs", "POSITIONAL_OR_KEYWORD", "600"),
            ("barrier_sync_fn", "POSITIONAL_OR_KEYWORD", "None"),
            ("post_finalization_callback", "POSITIONAL_OR_KEYWORD", "None"),
            ("create_directories_asynchronously", "POSITIONAL_OR_KEYWORD", "True"),
            ("async_origin", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Options configuring asynchronous checkpointing timeouts and synchronization callbacks.",
    ),
    (
        "MultiprocessingOptions",
        "orbax.checkpoint.MultiprocessingOptions",
        "class",
        "checkpoint",
        [
            ("primary_host", "POSITIONAL_OR_KEYWORD", "0"),
            ("active_processes", "POSITIONAL_OR_KEYWORD", "None"),
            ("barrier_sync_key_prefix", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Options configuring multi-host JAX process coordination during checkpointing.",
    ),
    # Checkpoint Handlers (handlers)
    (
        "CheckpointHandler",
        "orbax.checkpoint.CheckpointHandler",
        "class",
        "handlers",
        [],
        "Base interface for synchronous serialization and deserialization of checkpointable items.",
    ),
    (
        "AsyncCheckpointHandler",
        "orbax.checkpoint.AsyncCheckpointHandler",
        "class",
        "handlers",
        [],
        "Base interface for asynchronous checkpoint handlers returning background commit futures.",
    ),
    (
        "PyTreeCheckpointHandler",
        "orbax.checkpoint.PyTreeCheckpointHandler",
        "class",
        "handlers",
        [
            ("aggregate_filename", "POSITIONAL_OR_KEYWORD", "None"),
            ("use_ocdbt", "POSITIONAL_OR_KEYWORD", "True"),
            ("use_zarr3", "POSITIONAL_OR_KEYWORD", "False"),
            ("primary_host", "POSITIONAL_OR_KEYWORD", "0"),
            ("type_handler_registry", "POSITIONAL_OR_KEYWORD", "None"),
            ("handler_impl", "POSITIONAL_OR_KEYWORD", "None"),
            ("save_concurrent_gb", "POSITIONAL_OR_KEYWORD", "None"),
            ("restore_concurrent_gb", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Handler for saving and restoring arbitrary JAX PyTrees using TensorStore, OCDBT, and Zarr.",
    ),
    (
        "StandardCheckpointHandler",
        "orbax.checkpoint.StandardCheckpointHandler",
        "class",
        "handlers",
        [
            ("save_concurrent_gb", "POSITIONAL_OR_KEYWORD", "None"),
            ("restore_concurrent_gb", "POSITIONAL_OR_KEYWORD", "None"),
            ("multiprocessing_options", "KEYWORD_ONLY", "None"),
        ],
        "Simplified PyTree handler using `StandardSave` and `StandardRestore` argument wrappers.",
    ),
    (
        "JsonCheckpointHandler",
        "orbax.checkpoint.JsonCheckpointHandler",
        "class",
        "handlers",
        [
            ("filename", "POSITIONAL_OR_KEYWORD", "None"),
            ("multiprocessing_options", "KEYWORD_ONLY", "None"),
        ],
        "Handler for saving and restoring nested JSON-serializable metadata dictionaries.",
    ),
    (
        "ArrayCheckpointHandler",
        "orbax.checkpoint.ArrayCheckpointHandler",
        "class",
        "handlers",
        [
            ("checkpoint_name", "POSITIONAL_OR_KEYWORD", "None"),
            ("multiprocessing_options", "KEYWORD_ONLY", "None"),
        ],
        "Handler for saving and restoring individual NumPy or `jax.Array` instances.",
    ),
    (
        "ProtoCheckpointHandler",
        "orbax.checkpoint.ProtoCheckpointHandler",
        "class",
        "handlers",
        [
            ("filename", "POSITIONAL_OR_KEYWORD", None),
            ("multiprocessing_options", "KEYWORD_ONLY", "None"),
        ],
        "Handler for saving and restoring Protocol Buffer messages in text format.",
    ),
    (
        "RandomKeyCheckpointHandler",
        "orbax.checkpoint.RandomKeyCheckpointHandler",
        "class",
        "handlers",
        [],
        "Handler for saving and restoring JAX PRNG keys.",
    ),
    (
        "CompositeCheckpointHandler",
        "orbax.checkpoint.CompositeCheckpointHandler",
        "class",
        "handlers",
        [
            ("item_names", "VAR_POSITIONAL", None),
            ("composite_options", "KEYWORD_ONLY", "None"),
            ("items_and_handlers", "VAR_KEYWORD", None),
        ],
        "Handler delegating multiple named checkpoint items (e.g. `state`, `metadata`, `dataset`) to specialized sub-handlers.",
    ),
    (
        "CheckpointHandlerRegistry",
        "orbax.checkpoint.CheckpointHandlerRegistry",
        "class",
        "handlers",
        [],
        "Protocol for mapping checkpoint item names and `CheckpointArgs` types to `CheckpointHandler` instances.",
    ),
    (
        "DefaultCheckpointHandlerRegistry",
        "orbax.checkpoint.DefaultCheckpointHandlerRegistry",
        "class",
        "handlers",
        [
            ("other_registry", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Default mutable registry mapping `CheckpointArgs` types to their corresponding handlers.",
    ),
    (
        "create_default_handler_registry",
        "orbax.checkpoint.create_default_handler_registry",
        "function",
        "handlers",
        [
            ("items_and_handlers", "VAR_KEYWORD", None),
        ],
        "Factory creating a `CheckpointHandlerRegistry` populated with `(item_name, handler)` pairs.",
    ),
    # Declarative Save/Restore CheckpointArgs (args)
    (
        "CheckpointArgs",
        "orbax.checkpoint.args.CheckpointArgs",
        "class",
        "args",
        [],
        "Base dataclass for declarative save and restore arguments passed to `CheckpointManager.save` and `restore`.",
    ),
    (
        "StandardSave",
        "orbax.checkpoint.args.StandardSave",
        "class",
        "args",
        [
            ("item", "POSITIONAL_OR_KEYWORD", None),
            ("save_args", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Declarative `CheckpointArgs` for saving a JAX PyTree via `StandardCheckpointHandler`.",
    ),
    (
        "StandardRestore",
        "orbax.checkpoint.args.StandardRestore",
        "class",
        "args",
        [
            ("item", "POSITIONAL_OR_KEYWORD", "None"),
            ("strict", "POSITIONAL_OR_KEYWORD", "True"),
        ],
        "Declarative `CheckpointArgs` for restoring a JAX PyTree via `StandardCheckpointHandler`.",
    ),
    (
        "PyTreeSave",
        "orbax.checkpoint.args.PyTreeSave",
        "class",
        "args",
        [
            ("item", "POSITIONAL_OR_KEYWORD", None),
            ("save_args", "POSITIONAL_OR_KEYWORD", "None"),
            ("ocdbt_target_data_file_size", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Declarative `CheckpointArgs` for saving a PyTree via `PyTreeCheckpointHandler`.",
    ),
    (
        "PyTreeRestore",
        "orbax.checkpoint.args.PyTreeRestore",
        "class",
        "args",
        [
            ("item", "POSITIONAL_OR_KEYWORD", "None"),
            ("save_args", "POSITIONAL_OR_KEYWORD", "None"),
            ("restore_args", "POSITIONAL_OR_KEYWORD", "None"),
            ("transforms", "POSITIONAL_OR_KEYWORD", "None"),
            ("transforms_default_to_original", "POSITIONAL_OR_KEYWORD", "True"),
        ],
        "Declarative `CheckpointArgs` for restoring a PyTree via `PyTreeCheckpointHandler`.",
    ),
    (
        "JsonSave",
        "orbax.checkpoint.args.JsonSave",
        "class",
        "args",
        [
            ("item", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Declarative `CheckpointArgs` for saving JSON metadata via `JsonCheckpointHandler`.",
    ),
    (
        "JsonRestore",
        "orbax.checkpoint.args.JsonRestore",
        "class",
        "args",
        [
            ("item", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Declarative `CheckpointArgs` for restoring JSON metadata via `JsonCheckpointHandler`.",
    ),
    (
        "ArraySave",
        "orbax.checkpoint.args.ArraySave",
        "class",
        "args",
        [
            ("item", "POSITIONAL_OR_KEYWORD", None),
            ("save_args", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Declarative `CheckpointArgs` for saving a single array via `ArrayCheckpointHandler`.",
    ),
    (
        "ArrayRestore",
        "orbax.checkpoint.args.ArrayRestore",
        "class",
        "args",
        [
            ("item", "POSITIONAL_OR_KEYWORD", "None"),
            ("restore_args", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Declarative `CheckpointArgs` for restoring a single array via `ArrayCheckpointHandler`.",
    ),
    (
        "ProtoSave",
        "orbax.checkpoint.args.ProtoSave",
        "class",
        "args",
        [
            ("item", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Declarative `CheckpointArgs` for saving a protobuf message via `ProtoCheckpointHandler`.",
    ),
    (
        "ProtoRestore",
        "orbax.checkpoint.args.ProtoRestore",
        "class",
        "args",
        [
            ("item", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Declarative `CheckpointArgs` for restoring a protobuf message via `ProtoCheckpointHandler`.",
    ),
    (
        "JaxRandomKeySave",
        "orbax.checkpoint.args.JaxRandomKeySave",
        "class",
        "args",
        [
            ("item", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Declarative `CheckpointArgs` for saving a JAX PRNG key.",
    ),
    (
        "JaxRandomKeyRestore",
        "orbax.checkpoint.args.JaxRandomKeyRestore",
        "class",
        "args",
        [],
        "Declarative `CheckpointArgs` for restoring a JAX PRNG key.",
    ),
    (
        "NumpyRandomKeySave",
        "orbax.checkpoint.args.NumpyRandomKeySave",
        "class",
        "args",
        [
            ("item", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Declarative `CheckpointArgs` for saving a NumPy random generator state.",
    ),
    (
        "NumpyRandomKeyRestore",
        "orbax.checkpoint.args.NumpyRandomKeyRestore",
        "class",
        "args",
        [],
        "Declarative `CheckpointArgs` for restoring a NumPy random generator state.",
    ),
    (
        "Composite",
        "orbax.checkpoint.args.Composite",
        "class",
        "args",
        [
            ("items", "VAR_KEYWORD", None),
        ],
        "Container mapping composite item names to their respective `CheckpointArgs` specs.",
    ),
    # Type Handlers, RestoreArgs, Transform & Checkpoint Utilities (type_handlers, transform_utils, checkpoint_utils, msgpack_utils)
    (
        "SaveArgs",
        "orbax.checkpoint.SaveArgs",
        "class",
        "type_handlers",
        [
            ("aggregate", "POSITIONAL_OR_KEYWORD", "False"),
            ("dtype", "POSITIONAL_OR_KEYWORD", "None"),
            ("chunk_byte_size", "POSITIONAL_OR_KEYWORD", "None"),
            ("ocdbt_target_data_file_size", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Per-leaf configuration controlling dtype casting and chunk sizes during save.",
    ),
    (
        "RestoreArgs",
        "orbax.checkpoint.RestoreArgs",
        "class",
        "type_handlers",
        [
            ("restore_type", "POSITIONAL_OR_KEYWORD", "None"),
            ("dtype", "POSITIONAL_OR_KEYWORD", "None"),
            ("strict", "POSITIONAL_OR_KEYWORD", "True"),
        ],
        "Per-leaf configuration specifying target `restore_type` and `dtype` during restore.",
    ),
    (
        "ArrayRestoreArgs",
        "orbax.checkpoint.ArrayRestoreArgs",
        "class",
        "type_handlers",
        [
            ("restore_type", "POSITIONAL_OR_KEYWORD", "None"),
            ("mesh", "POSITIONAL_OR_KEYWORD", "None"),
            ("mesh_axes", "POSITIONAL_OR_KEYWORD", "None"),
            ("sharding", "POSITIONAL_OR_KEYWORD", "None"),
            ("global_shape", "POSITIONAL_OR_KEYWORD", "None"),
            ("shape", "POSITIONAL_OR_KEYWORD", "None"),
            ("dtype", "POSITIONAL_OR_KEYWORD", "None"),
            ("strict", "POSITIONAL_OR_KEYWORD", "True"),
        ],
        "Per-leaf restore arguments specifying `jax.sharding.Sharding` and `global_shape` for distributed `jax.Array` leaves.",
    ),
    (
        "SingleReplicaArrayRestoreArgs",
        "orbax.checkpoint.SingleReplicaArrayRestoreArgs",
        "class",
        "type_handlers",
        [
            ("restore_type", "POSITIONAL_OR_KEYWORD", "None"),
            ("mesh", "POSITIONAL_OR_KEYWORD", "None"),
            ("mesh_axes", "POSITIONAL_OR_KEYWORD", "None"),
            ("sharding", "POSITIONAL_OR_KEYWORD", "None"),
            ("global_shape", "POSITIONAL_OR_KEYWORD", "None"),
            ("shape", "POSITIONAL_OR_KEYWORD", "None"),
            ("dtype", "POSITIONAL_OR_KEYWORD", "None"),
            ("strict", "POSITIONAL_OR_KEYWORD", "True"),
            ("single_replica_sharding", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Restore arguments reading array shards on a single replica and broadcasting across replicas.",
    ),
    (
        "Transform",
        "orbax.checkpoint.Transform",
        "class",
        "transform_utils",
        [
            ("original_key", "POSITIONAL_OR_KEYWORD", "None"),
            ("use_fallback", "POSITIONAL_OR_KEYWORD", "False"),
            ("value_fn", "POSITIONAL_OR_KEYWORD", "None"),
            ("multi_value_fn", "POSITIONAL_OR_KEYWORD", "None"),
            ("multi_value_fn_input_args", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Rule describing structural key renaming or value transformation between PyTrees.",
    ),
    (
        "RestoreTransform",
        "orbax.checkpoint.RestoreTransform",
        "class",
        "transform_utils",
        [
            ("original_key", "POSITIONAL_OR_KEYWORD", "None"),
            ("use_fallback", "POSITIONAL_OR_KEYWORD", "False"),
            ("value_fn", "POSITIONAL_OR_KEYWORD", "None"),
            ("multi_value_fn", "POSITIONAL_OR_KEYWORD", "None"),
            ("multi_value_fn_input_args", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Transformation rule applied during PyTree checkpoint restoration.",
    ),
    (
        "apply_transformations",
        "orbax.checkpoint.apply_transformations",
        "function",
        "transform_utils",
        [
            ("original_tree", "POSITIONAL_OR_KEYWORD", None),
            ("transformations", "POSITIONAL_OR_KEYWORD", None),
            ("new_tree", "POSITIONAL_OR_KEYWORD", None),
            ("default_to_original", "POSITIONAL_OR_KEYWORD", "True"),
        ],
        "Apply a tree of `Transform` rules to map `original_tree` into the structure of `new_tree`.",
    ),
    (
        "construct_restore_args",
        "orbax.checkpoint.checkpoint_utils.construct_restore_args",
        "function",
        "checkpoint_utils",
        [
            ("target", "POSITIONAL_OR_KEYWORD", None),
            ("sharding_tree", "POSITIONAL_OR_KEYWORD", "None"),
            ("set_global_shape", "POSITIONAL_OR_KEYWORD", "True"),
        ],
        "Construct a PyTree of `RestoreArgs` / `ArrayRestoreArgs` matching a target abstract PyTree.",
    ),
    (
        "restore_args_from_target",
        "orbax.checkpoint.checkpoint_utils.restore_args_from_target",
        "function",
        "checkpoint_utils",
        [
            ("target", "POSITIONAL_OR_KEYWORD", None),
            ("sharding_tree", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Create a PyTree of `RestoreArgs` inferred from `target` and `sharding_tree`.",
    ),
    (
        "wait_for_new_checkpoint",
        "orbax.checkpoint.checkpoint_utils.wait_for_new_checkpoint",
        "function",
        "checkpoint_utils",
        [
            ("checkpoint_dir", "POSITIONAL_OR_KEYWORD", None),
            ("until_step", "POSITIONAL_OR_KEYWORD", "None"),
            ("seconds_to_sleep", "POSITIONAL_OR_KEYWORD", "1"),
            ("timeout", "POSITIONAL_OR_KEYWORD", "None"),
            ("step_prefix", "POSITIONAL_OR_KEYWORD", "None"),
            ("step_format_fixed_length", "POSITIONAL_OR_KEYWORD", "None"),
            ("step_name_format", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Block until a new checkpoint step appears in `checkpoint_dir`.",
    ),
    (
        "msgpack_serialize",
        "orbax.checkpoint.msgpack_utils.msgpack_serialize",
        "function",
        "msgpack_utils",
        [
            ("pytree", "POSITIONAL_OR_KEYWORD", None),
            ("in_place", "POSITIONAL_OR_KEYWORD", "False"),
        ],
        "Serialize a PyTree into MsgPack bytes.",
    ),
    (
        "msgpack_restore",
        "orbax.checkpoint.msgpack_utils.msgpack_restore",
        "function",
        "msgpack_utils",
        [
            ("encoded_pytree", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Deserialize MsgPack bytes back into a Python structure.",
    ),
    # Orbax Checkpoint v1 Functional API (v1)
    (
        "save_pytree",
        "orbax.checkpoint.v1.save_pytree",
        "function",
        "v1",
        [
            ("path", "POSITIONAL_OR_KEYWORD", None),
            ("pytree", "POSITIONAL_OR_KEYWORD", None),
            ("overwrite", "KEYWORD_ONLY", "False"),
            ("custom_metadata", "KEYWORD_ONLY", "None"),
        ],
        "Synchronously save a JAX PyTree to `path` using the Orbax v1 functional API.",
    ),
    (
        "save_pytree_async",
        "orbax.checkpoint.v1.save_pytree_async",
        "function",
        "v1",
        [
            ("path", "POSITIONAL_OR_KEYWORD", None),
            ("pytree", "POSITIONAL_OR_KEYWORD", None),
            ("overwrite", "KEYWORD_ONLY", "False"),
            ("custom_metadata", "KEYWORD_ONLY", "None"),
        ],
        "Asynchronously save a JAX PyTree to `path` using the Orbax v1 functional API.",
    ),
    (
        "load_pytree",
        "orbax.checkpoint.v1.load_pytree",
        "function",
        "v1",
        [
            ("path", "POSITIONAL_OR_KEYWORD", None),
            ("abstract_pytree", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Load a JAX PyTree from `path` using optional `abstract_pytree` sharding/shape specifications.",
    ),
    (
        "save_checkpointables",
        "orbax.checkpoint.v1.save_checkpointables",
        "function",
        "v1",
        [
            ("path", "POSITIONAL_OR_KEYWORD", None),
            ("checkpointables", "POSITIONAL_OR_KEYWORD", None),
            ("overwrite", "KEYWORD_ONLY", "False"),
            ("custom_metadata", "KEYWORD_ONLY", "None"),
        ],
        "Save a dictionary of named checkpointables to `path`.",
    ),
    (
        "load_checkpointables",
        "orbax.checkpoint.v1.load_checkpointables",
        "function",
        "v1",
        [
            ("path", "POSITIONAL_OR_KEYWORD", None),
            ("abstract_checkpointables", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Load a dictionary of named checkpointables from `path`.",
    ),
    # Orbax Export (export)
    (
        "ExportManager",
        "orbax.export.ExportManager",
        "class",
        "export",
        [
            ("module", "POSITIONAL_OR_KEYWORD", None),
            ("serving_configs", "POSITIONAL_OR_KEYWORD", None),
        ],
        "Manager exporting a `JaxModule` to TensorFlow SavedModel or Orbax Model formats across `serving_configs`.",
    ),
    (
        "JaxModule",
        "orbax.export.JaxModule",
        "class",
        "export",
        [
            ("params", "POSITIONAL_OR_KEYWORD", None),
            ("apply_fn", "POSITIONAL_OR_KEYWORD", None),
            ("trainable", "POSITIONAL_OR_KEYWORD", "None"),
            ("input_polymorphic_shape", "POSITIONAL_OR_KEYWORD", "None"),
            ("jax2tf_kwargs", "POSITIONAL_OR_KEYWORD", "None"),
            ("jit_compile", "POSITIONAL_OR_KEYWORD", "True"),
            ("pspecs", "POSITIONAL_OR_KEYWORD", "None"),
            (
                "allow_multi_axis_sharding_consolidation",
                "POSITIONAL_OR_KEYWORD",
                "None",
            ),
            ("export_version", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Exportable wrapper encapsulating JAX model parameters and pure `apply_fn` callables.",
    ),
    (
        "ServingConfig",
        "orbax.export.ServingConfig",
        "class",
        "export",
        [
            ("signature_key", "POSITIONAL_OR_KEYWORD", None),
            ("input_signature", "POSITIONAL_OR_KEYWORD", "None"),
            ("tf_preprocessor", "POSITIONAL_OR_KEYWORD", "None"),
            ("tf_postprocessor", "POSITIONAL_OR_KEYWORD", "None"),
            ("extra_trackable_resources", "POSITIONAL_OR_KEYWORD", "None"),
            ("method_key", "POSITIONAL_OR_KEYWORD", "None"),
            ("obm_export_options", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Configuration binding a serving `signature_key` and `input_signature` with optional pre/postprocessors.",
    ),
    (
        "ObmExportOptions",
        "orbax.export.ObmExportOptions",
        "class",
        "export",
        [
            ("enable_auto_layout", "POSITIONAL_OR_KEYWORD", "False"),
            ("checkpoint_path", "POSITIONAL_OR_KEYWORD", "None"),
        ],
        "Options configuring native Orbax Model (OBM) export.",
    ),
]


def _build_orbax_canonical_ops(
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
    """Convert compact Orbax operation tuples into `CanonicalSubmoduleOpSpec` models.

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


CANONICAL_ORBAX_OPS: list[CanonicalSubmoduleOpSpec] = _build_orbax_canonical_ops(
    _ORBAX_OP_DEFS
)


def _get_orbax_griffe_modules() -> dict[str, GriffeCacheEntry]:
    """Statically load `orbax.checkpoint` and `orbax.export` via the custom Griffe fork.

    Returns:
        Dictionary mapping `'checkpoint'` and `'export'` to loaded Griffe AST nodes or `None`.
    """
    if is_offline_mode():
        return {"checkpoint": None, "export": None}
    return {
        "checkpoint": load_griffe_module_safely(
            "orbax.checkpoint", framework_name="orbax"
        ),
        "export": load_griffe_module_safely("orbax.export", framework_name="orbax"),
    }


def _get_orbax_export_module() -> types.ModuleType | None:
    """Safely import `orbax.export` when not in offline mode.

    Returns:
        Imported `orbax.export` module or `None` if unavailable or in offline mode.
    """
    if is_offline_mode():
        return None
    try:
        return importlib.import_module("orbax.export")
    except Exception:
        return None


def _scan_orbax_checkpoint(include_nonpublic: bool = False) -> list[GhostRef]:
    """Dynamically scan `orbax.checkpoint` or fall back to canonical specs in offline mode.

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of `GhostRef` objects representing discovered Orbax APIs.
    """
    if is_offline_mode():
        return [
            build_canonical_python_ref(
                spec,
                default_doc_prefix="Orbax",
                environment_tags=("orbax", "jax", "checkpoint"),
                domain_metadata={"submodule": spec.submodule or "checkpoint"},
            )
            for spec in CANONICAL_ORBAX_OPS
        ]

    if not ocp:
        return []

    found: list[GhostRef] = []
    try:
        for name, obj in get_all_members(ocp):
            if inspect.isclass(obj) or inspect.isfunction(obj):
                if not include_nonpublic and name.startswith("_"):
                    continue
                try:
                    found.append(
                        GhostInspector.inspect(obj, f"orbax.checkpoint.{name}")
                    )
                except Exception:
                    pass
    except Exception:
        pass

    return found


def collect_orbax_api(include_nonpublic: bool = False) -> list[GhostRef]:
    """Collect the complete Orbax Checkpoint and Orbax Export API surface.

    Combines live module introspection (`orbax.checkpoint`, `orbax.export`),
    Griffe static AST scanning via the custom Griffe fork, and canonical
    fallback specifications (`CANONICAL_ORBAX_OPS`).

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        List of `GhostRef` items covering `orbax.checkpoint` and `orbax.export`.
    """
    env_tags = ("orbax", "jax", "checkpoint")
    results: list[GhostRef] = []
    seen_paths: set[str] = set()

    if not is_offline_mode():
        for ref in _scan_orbax_checkpoint(include_nonpublic=include_nonpublic):
            if ref.api_path not in seen_paths:
                results.append(ref)
                seen_paths.add(ref.api_path)

        exp_mod = _get_orbax_export_module()
        if exp_mod is not None:
            for name, obj in get_all_members(exp_mod):
                if inspect.isclass(obj) or inspect.isfunction(obj):
                    if not include_nonpublic and name.startswith("_"):
                        continue
                    api_path = f"orbax.export.{name}"
                    try:
                        results.append(GhostInspector.inspect(obj, api_path))
                        seen_paths.add(api_path)
                    except Exception:
                        pass

        griffe_mods = _get_orbax_griffe_modules()
        for key, mod_path in (
            ("checkpoint", "orbax.checkpoint"),
            ("export", "orbax.export"),
        ):
            if griffe_mods.get(key) is not None:
                for ref in scan_griffe_module_members(
                    module_path=mod_path,
                    api_prefix=mod_path,
                    include_nonpublic=include_nonpublic,
                    environment_tags=env_tags,
                    framework_name="orbax",
                    inspector_fn=GhostInspector.inspect,
                ):
                    if ref.api_path not in seen_paths:
                        results.append(ref)
                        seen_paths.add(ref.api_path)

    for spec in CANONICAL_ORBAX_OPS:
        if spec.api_path not in seen_paths:
            results.append(
                build_canonical_python_ref(
                    spec,
                    default_doc_prefix="Orbax",
                    environment_tags=env_tags,
                    domain_metadata={"submodule": spec.submodule or "checkpoint"},
                )
            )
            seen_paths.add(spec.api_path)

    return results


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> list[GhostRef]:
    """Entrypoint to collect the Orbax Checkpoint API signature.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered for the requested category.
    """
    if category == SemanticTier.ARRAY_API:
        return _scan_orbax_checkpoint(include_nonpublic)

    return []


__all__ = [
    "CANONICAL_ORBAX_OPS",
    "_ORBAX_OP_DEFS",
    "_build_orbax_canonical_ops",
    "_get_orbax_export_module",
    "_get_orbax_griffe_modules",
    "_scan_orbax_checkpoint",
    "collect_api",
    "collect_orbax_api",
    "ocp",
]
