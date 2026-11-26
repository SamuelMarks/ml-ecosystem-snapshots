"""High-performance in-memory grounding engine for anti-hallucination verification."""

from __future__ import annotations

import logging
import os
import threading
import typing

import typing_extensions
from ml_switcheroo_ir.graph import LogicalGraph, LogicalNode
from ml_switcheroo_ir.proofs import AcyclicSortedPhase, GroundedPhase, ProvenGraph
from ml_switcheroo_ir.typeclasses.algebra import NonEmpty
from ml_switcheroo_ir.typeclasses.outcome import (
    Ok,
    Validation,
    from_error_sequence,
)
from ml_switcheroo_ir.types import TensorSpec
from ml_switcheroo_ir.validation.core import compute_levenshtein
from ml_switcheroo_ir.validation.grounding import (
    DiagnosticSeverity,
    GroundingDiagnostic,
    GroundingReport,
)

from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.models import (
    ExtendedGhostRef,
    JsonValue,
    SnapshotEnvelope,
)
from ml_ecosystem_snapshots.network_env import load_json_data
from ml_ecosystem_snapshots.proofs import EnvelopeAndHydratedPhase, ProvenSnapshot
from ml_ecosystem_snapshots.serialization.repository import SnapshotRepository

_LOGGER = logging.getLogger(__name__)

GroundingEngineInput: typing_extensions.TypeAlias = "GroundingEngine | ProvenSnapshot[EnvelopeAndHydratedPhase] | None"

_SYNTHETIC_GRAPH_OPS: typing.Final[frozenset[str]] = frozenset(
    {"Input", "Output", "Placeholder", "Constant", "Parameter"}
)


def _extract_raw_items(
    raw_data: SnapshotEnvelope | JsonValue,
) -> list[dict[str, JsonValue]]:
    """Extract raw GhostRef dictionary items from a loaded snapshot structure.

    Args:
        raw_data: Deserialized JSON list or dictionary envelope.

    Returns:
        Flat list of GhostRef item dictionaries.
    """
    raw_items: list[dict[str, JsonValue]] = []
    if isinstance(raw_data, list):
        raw_items = [dict(item) for item in raw_data if isinstance(item, dict)]
    elif isinstance(raw_data, dict):
        if "categories" in raw_data and isinstance(raw_data["categories"], dict):
            for cat_items in raw_data["categories"].values():
                if isinstance(cat_items, list):
                    raw_items.extend(dict(i) for i in cat_items if isinstance(i, dict))
        else:
            matched_list = False
            for list_key in ("items", "operations", "instructions"):
                list_val = raw_data.get(list_key)
                if isinstance(list_val, list):
                    raw_items = [dict(i) for i in list_val if isinstance(i, dict)]
                    matched_list = True
                    break
            if not matched_list:
                raw_items = [dict(raw_data)]
    return raw_items


class GroundingEngine:
    """Thread-safe in-memory cache and symbol resolution engine for anti-hallucination verification."""

    def __init__(
        self,
        base_dirs: list[str] | None = None,
        repository: SnapshotRepository | None = None,
        snapshots_dir: str | None = None,
    ) -> None:
        """Initialize the GroundingEngine.

        Args:
            base_dirs: Optional explicit directory paths to search for snapshots.
            repository: Optional shared `SnapshotRepository` instance for cache integration.
            snapshots_dir: Optional single snapshots directory path override.
        """
        default_frameworks = os.path.join(os.path.dirname(__file__), "..", "frameworks")
        default_snapshots = os.path.join(os.path.dirname(__file__), "..", "snapshots")

        explicit_dirs = base_dirs if base_dirs is not None else ([snapshots_dir] if snapshots_dir is not None else None)
        self._has_custom_dirs = explicit_dirs is not None
        self.search_dirs: list[str] = explicit_dirs or [
            os.path.abspath(default_frameworks),
            os.path.abspath(default_snapshots),
        ]
        self._snapshots_dir: str | None = snapshots_dir or (explicit_dirs[0] if explicit_dirs else None)
        self._repository = repository
        self._lock = threading.RLock()
        self._target_cache: dict[str, dict[str, ExtendedGhostRef]] = {}
        self._short_name_cache: dict[str, dict[str, ExtendedGhostRef]] = {}
        self._proven_targets: set[str] = set()

    @property
    def snapshots_dir(self) -> str | None:
        """Return the configured custom snapshots directory if set.

        Returns:
            Custom snapshots directory path or `None`.
        """
        return self._snapshots_dir

    def has_proven_target(self, target: str) -> bool:
        """Return `True` when `target` was registered from a `ProvenSnapshot`.

        Args:
            target: Framework, dialect, or ISA identifier.

        Returns:
            `True` if `register_proven_snapshot` populated `target`.
        """
        norm_target = target.lower().replace("-", "_")
        with self._lock:
            return norm_target in self._proven_targets

    def register_proven_snapshot(
        self,
        snapshot: ProvenSnapshot[EnvelopeAndHydratedPhase],
    ) -> dict[str, ExtendedGhostRef]:
        """Index a hydrated `ProvenSnapshot` into memory without disk I/O or re-hydration.

        Args:
            snapshot: `ProvenSnapshot[EnvelopeAndHydratedPhase]` (or covariant subtype).

        Returns:
            Populated symbol lookup dictionary for `snapshot.target`.
        """
        norm_target = str(snapshot.target).lower().replace("-", "_")
        with self._lock:
            index_map: dict[str, ExtendedGhostRef] = {}
            short_map: dict[str, ExtendedGhostRef] = {}
            for ref in snapshot.all_refs():
                if ref.api_path:
                    index_map[ref.api_path] = ref
                if ref.name:
                    short_map.setdefault(ref.name.lower(), ref)
                    if not ref.api_path:
                        index_map[ref.name] = ref
                mnemonic: JsonValue = (ref.model_extra or {}).get("mnemonic")
                if mnemonic:
                    index_map[str(mnemonic)] = ref
                    short_map.setdefault(str(mnemonic).lower(), ref)
                for alias in ref.aliases or ():
                    index_map[alias] = ref
                    short_map.setdefault(alias.split(".")[-1].lower(), ref)
            self._target_cache[norm_target] = index_map
            self._short_name_cache[norm_target] = short_map
            self._proven_targets.add(norm_target)
            return index_map

    @classmethod
    def from_proven_snapshot(
        cls,
        snapshot: ProvenSnapshot[EnvelopeAndHydratedPhase],
    ) -> GroundingEngine:
        """Construct an in-memory `GroundingEngine` pre-populated from `snapshot`.

        Args:
            snapshot: `ProvenSnapshot[EnvelopeAndHydratedPhase]` (or covariant subtype).

        Returns:
            Ready-to-query `GroundingEngine` instance.
        """
        eng = cls()
        eng.register_proven_snapshot(snapshot)
        return eng

    def get_loaded_targets(self) -> list[str]:
        """Retrieve the names of all currently loaded targets.

        Returns:
            List of target names.
        """
        with self._lock:
            return sorted(list(self._target_cache.keys()))

    def _discover_target_files(self, target: str) -> list[str]:
        """Locate snapshot or exhaustive files matching a target name.

        Args:
            target: The framework, dialect, or architecture name.

        Returns:
            List of matching file paths.
        """
        norm_target = target.lower().replace("-", "_")
        matched_files: list[str] = []

        for d in self.search_dirs:
            if not os.path.isdir(d):
                continue
            for fname in sorted(os.listdir(d)):
                clean_fname = fname.lower()
                is_json = clean_fname.endswith(".json") or clean_fname.endswith(".json.gz")
                if not is_json:
                    continue

                if clean_fname.startswith(f"{norm_target}_exhaustive."):
                    matched_files.append(os.path.join(d, fname))
                elif clean_fname.startswith(f"{norm_target}_v") or clean_fname == f"{norm_target}.json":
                    matched_files.append(os.path.join(d, fname))

        return matched_files

    def _index_item(
        self,
        item: dict[str, JsonValue],
        symbol_index: dict[str, ExtendedGhostRef],
        short_index: dict[str, ExtendedGhostRef],
    ) -> None:
        """Hydrate a single raw item dictionary and index it into symbol_index and short_index.

        Args:
            item: Raw GhostRef dictionary.
            symbol_index: Target full-path and name symbol index.
            short_index: Target lowercase short-name and mnemonic index.
        """
        hyd_res = GhostInspector.hydrate(item)
        if not isinstance(hyd_res, Ok):
            _LOGGER.debug("Skipping invalid GhostRef item during indexing: %s", hyd_res.error)
            return
        ref = hyd_res.value
        if ref.api_path:
            symbol_index[ref.api_path] = ref
        if ref.name:
            symbol_index[ref.name] = ref
            short_index[ref.name.lower()] = ref
        mnemonic = item.get("mnemonic")
        if mnemonic:
            symbol_index[str(mnemonic)] = ref
            short_index[str(mnemonic).lower()] = ref

    def _load_fallback_live_items(self, norm_target: str) -> list[dict[str, JsonValue]]:
        """Extract raw snapshot items via live `extract_snapshot` when disk/repo caches are empty.

        Args:
            norm_target: Normalized lowercase target framework name.

        Returns:
            List of extracted raw item dictionaries (empty on unsupported or failed extraction).
        """
        try:
            from ml_ecosystem_snapshots.api import (
                FRAMEWORK_COLLECTORS,
                extract_snapshot,
            )

            if norm_target not in FRAMEWORK_COLLECTORS:
                return []
            snap_res = extract_snapshot(norm_target)
            if isinstance(snap_res, Ok):
                return _extract_raw_items(snap_res.value.to_wire_dict())
            if isinstance(snap_res, dict):
                return _extract_raw_items(snap_res)
        except (
            AttributeError,
            TypeError,
            ValueError,
            KeyError,
            ImportError,
            OSError,
            RuntimeError,
        ) as exc:
            _LOGGER.debug("Fallback snapshot extraction failed for %s: %s", norm_target, exc)
        return []

    def load_target(self, target: str) -> dict[str, ExtendedGhostRef]:
        """Lazily load and index all symbols for a target framework or ISA.

        Args:
            target: The target framework, dialect, or hardware architecture.

        Returns:
            Dictionary mapping symbol keys to their hydrated ExtendedGhostRef objects.
        """
        norm_target = target.lower().replace("-", "_")
        with self._lock:
            if norm_target in self._target_cache:
                return self._target_cache[norm_target]

            files = self._discover_target_files(norm_target)
            symbol_index: dict[str, ExtendedGhostRef] = {}
            short_index: dict[str, ExtendedGhostRef] = {}

            for file_path in files:
                raw_res = load_json_data(file_path)
                if isinstance(raw_res, Ok):
                    for item in _extract_raw_items(raw_res.value):
                        self._index_item(item, symbol_index, short_index)

            if not symbol_index and self._repository is not None:
                for item in _extract_raw_items(self._repository.get(norm_target)):
                    self._index_item(item, symbol_index, short_index)

            if not symbol_index:
                for item in self._load_fallback_live_items(norm_target):
                    self._index_item(item, symbol_index, short_index)

            self._target_cache[norm_target] = symbol_index
            self._short_name_cache[norm_target] = short_index
            return symbol_index

    def has_symbol(self, target: str, symbol: str) -> bool:
        """Check if a symbol exists within the target ground-truth dataset.

        Args:
            target: The framework, dialect, or ISA.
            symbol: The fully-qualified path, mnemonic, or short name.

        Returns:
            True if symbol is present, False otherwise.
        """
        index = self.load_target(target)
        if symbol in index:
            return True
        norm_symbol = symbol.lower()
        short_map = self._short_name_cache.get(target.lower().replace("-", "_"), {})
        return norm_symbol in short_map

    def get_symbol(self, target: str, symbol: str) -> ExtendedGhostRef | None:
        """Retrieve the canonical ExtendedGhostRef for a symbol.

        Args:
            target: The framework, dialect, or ISA.
            symbol: The fully-qualified path, mnemonic, or short name.

        Returns:
            The matched ExtendedGhostRef if found, or None.
        """
        index = self.load_target(target)
        if symbol in index:
            return index[symbol]
        norm_symbol = symbol.lower()
        short_map = self._short_name_cache.get(target.lower().replace("-", "_"), {})
        return short_map.get(norm_symbol)

    def suggest_closest_symbol(self, target: str, typo_candidate: str, max_distance: int = 4) -> str | None:
        """Suggest the closest valid symbol using Levenshtein distance matching.

        Args:
            target: The framework, dialect, or ISA.
            typo_candidate: The ungrounded candidate string.
            max_distance: Maximum allowable edit distance for a suggestion.

        Returns:
            Suggested symbol name, or None if no close candidate exists.
        """
        index = self.load_target(target)
        if not index:
            return None

        clean_candidate = typo_candidate.lower()
        best_candidate: str | None = None
        best_distance = max_distance + 1

        all_candidates: set[str] = set()
        for ref in index.values():
            if ref.api_path:
                all_candidates.add(ref.api_path)
            if ref.name:
                all_candidates.add(ref.name)
            mnemonic: JsonValue = (ref.model_extra or {}).get("mnemonic")
            if mnemonic:
                all_candidates.add(str(mnemonic))

        for cand in sorted(all_candidates):
            dist = compute_levenshtein(clean_candidate, cand.lower())
            if dist < best_distance:
                best_distance = dist
                best_candidate = cand
            elif dist == best_distance and best_candidate is not None:
                if len(cand) < len(best_candidate):
                    best_candidate = cand

        return best_candidate if best_distance <= max_distance else None

    def fuzzy_search_symbol(self, target: str, query: str, top_k: int = 5) -> list[str]:
        """Search symbols in target using fuzzy substring and Levenshtein ranking.

        Args:
            target: The framework, dialect, or ISA.
            query: Substring or partial token to search for.
            top_k: Maximum number of ranked suggestions to return.

        Returns:
            Ranked list of matching symbol identifiers.
        """
        index = self.load_target(target)
        if not index:
            return []

        clean_query = query.lower()
        matches: list[tuple[int, str]] = []

        seen: set[str] = set()
        for ref in index.values():
            for sym in [ref.api_path, ref.name]:
                if not sym or sym in seen:
                    continue
                seen.add(sym)
                sym_lower = sym.lower()

                if clean_query == sym_lower:
                    matches.append((0, sym))
                elif sym_lower.startswith(clean_query):
                    matches.append((1, sym))
                elif clean_query in sym_lower:
                    matches.append((2, sym))
                else:
                    dist = compute_levenshtein(clean_query, sym_lower)
                    if dist <= 3:
                        matches.append((3 + dist, sym))

        matches.sort(key=lambda x: (x[0], len(x[1]), x[1]))
        return [m[1] for m in matches[:top_k]]


def coerce_grounding_engine(engine: GroundingEngineInput) -> GroundingEngine:
    """Normalize a `GroundingEngineInput` into a concrete `GroundingEngine`.

    Args:
        engine: `GroundingEngine`, `ProvenSnapshot[EnvelopeAndHydratedPhase]`, or `None`.

    Returns:
        Concrete `GroundingEngine` instance.
    """
    if isinstance(engine, GroundingEngine):
        return engine
    if isinstance(engine, ProvenSnapshot):
        return GroundingEngine.from_proven_snapshot(engine)
    return GroundingEngine()


def _collect_graph_value_specs(graph: LogicalGraph) -> dict[str, TensorSpec]:
    """Build a mapping from value/node identifiers to known `TensorSpec`s in `graph`.

    Args:
        graph: `LogicalGraph` under grounding inspection.

    Returns:
        Dictionary mapping SSA value or node IDs to `TensorSpec`.
    """
    specs: dict[str, TensorSpec] = dict(graph.input_specs)
    for nid, node in graph.nodes.items():
        if node.output_specs:
            specs.setdefault(nid, node.output_specs[0])
            for idx, out_name in enumerate(node.outputs):
                if idx < len(node.output_specs):
                    specs[out_name] = node.output_specs[idx]
    return specs


def _ground_single_node(
    node: LogicalNode,
    graph: LogicalGraph,
    value_specs: typing.Mapping[str, TensorSpec],
    target_fw: str,
    eng: GroundingEngine,
) -> list[GroundingDiagnostic]:
    """Validate a single `LogicalNode` against `eng` and return any `ERROR` diagnostics.

    Args:
        node: `LogicalNode` to verify.
        graph: Enclosing `LogicalGraph`.
        value_specs: Known `TensorSpec` map for `graph`.
        target_fw: Normalized target framework/dialect identifier.
        eng: Pre-populated `GroundingEngine`.

    Returns:
        List of `GroundingDiagnostic` entries with `severity == DiagnosticSeverity.ERROR`.
    """
    from ml_ecosystem_snapshots.grounding.compiler import validate_mlir_op, validate_stablehlo_op
    from ml_ecosystem_snapshots.grounding.domain_ops import DOMAIN_OP_CONFIGS, validate_domain_op
    from ml_ecosystem_snapshots.grounding.dtypes import (
        coerce_ir_attribute_to_json,
        extract_tensor_spec_metadata,
    )
    from ml_ecosystem_snapshots.grounding.python_fw import validate_python_call

    if node.op_type in _SYNTHETIC_GRAPH_OPS and eng.get_symbol(target_fw, node.op_type) is None:
        return []

    incoming_sources: list[str] = list(node.inputs)
    operand_count = len(incoming_sources)

    arg_dtypes: list[str] = []
    arg_ranks: list[int | str] = []
    arg_shapes: list[typing.Sequence[int | str]] = []
    ssa_operand_types: list[str] = []
    has_all_specs = bool(incoming_sources)
    for src_key in incoming_sources:
        spec = value_specs.get(src_key)
        if spec is None:
            has_all_specs = False
            ssa_operand_types.append("tensor<*xf32>")
        else:
            dt_name, rk_val, sh_tup = extract_tensor_spec_metadata(spec)
            arg_dtypes.append(dt_name)
            arg_ranks.append(rk_val)
            arg_shapes.append(sh_tup)
            dim_tokens = ["?" if not isinstance(d, int) or d < 0 else str(d) for d in sh_tup]
            ssa_prefix = "x".join(dim_tokens) + "x" if dim_tokens else ""
            ssa_operand_types.append(f"tensor<{ssa_prefix}{dt_name}>")

    json_attrs: dict[str, JsonValue] = {str(k): coerce_ir_attribute_to_json(v) for k, v in node.attributes.items()}

    domain_alias_map = {
        "wasm_simd": "wasm",
        "cpp_runtime": "cpp",
        "flash_attn": "flash_attention",
    }
    mapped_dom = domain_alias_map.get(target_fw, target_fw)

    report: GroundingReport
    if target_fw == "stablehlo":
        report = validate_stablehlo_op(
            op_name=node.op_type,
            operand_types=ssa_operand_types,
            attributes=json_attrs,
            engine=eng,
        )
    elif target_fw == "mlir":
        dialect_part = (
            node.op_type.split(".")[0]
            if "." in node.op_type
            else (node.domain if node.domain and node.domain != "ai.onnx" else "builtin")
        )
        report = validate_mlir_op(
            dialect=dialect_part,
            op_name=node.op_type,
            operand_count=operand_count,
            attributes=json_attrs,
            engine=eng,
            operand_types=ssa_operand_types if incoming_sources else None,
        )
    elif mapped_dom in DOMAIN_OP_CONFIGS:
        report = validate_domain_op(
            mapped_dom,
            node.op_type,
            inputs_count=operand_count if incoming_sources else None,
            attributes=list(json_attrs.keys()) if json_attrs else None,
            engine=eng,
        )
    else:
        call_sym = (
            node.op_type
            if "." in node.op_type or eng.get_symbol(target_fw, node.op_type) is not None
            else f"{target_fw}.{node.op_type}"
        )
        pos_placeholders: list[JsonValue] = [None] * operand_count
        report = validate_python_call(
            framework=target_fw,
            api_path=call_sym,
            args=pos_placeholders,
            kwargs=json_attrs,
            engine=eng,
            arg_dtypes=arg_dtypes if has_all_specs and arg_dtypes else None,
            arg_ranks=arg_ranks if has_all_specs and arg_ranks else None,
            arg_shapes=arg_shapes if has_all_specs and arg_shapes else None,
        )

    out_diags: list[GroundingDiagnostic] = []
    for diag in report.diagnostics:
        if isinstance(diag, GroundingDiagnostic) and diag.severity == DiagnosticSeverity.ERROR:
            out_diags.append(diag.model_copy(update={"node_id": node.id}))
    return out_diags


def _collect_subgraph_diagnostics(
    graph: LogicalGraph,
    ordered_ids: typing.Sequence[str],
    target_fw: str,
    eng: GroundingEngine,
) -> list[GroundingDiagnostic]:
    """Recursively ground `graph` nodes in `ordered_ids` order plus nested subgraphs.

    Args:
        graph: `LogicalGraph` to verify.
        ordered_ids: Deterministic node ID order for `graph`.
        target_fw: Normalized target framework/dialect identifier.
        eng: Pre-populated `GroundingEngine`.

    Returns:
        Accumulated `ERROR`-severity `GroundingDiagnostic` list.
    """
    diags: list[GroundingDiagnostic] = []
    value_specs = _collect_graph_value_specs(graph)
    for nid in ordered_ids:
        node = graph.nodes[nid]
        diags.extend(_ground_single_node(node, graph, value_specs, target_fw, eng))
        for _sub_key, sub_graph in sorted(node.subgraphs.items(), key=lambda kv: kv[0]):
            if isinstance(sub_graph, LogicalGraph):
                sub_ids = (
                    [n.id for n in sub_graph.sorted_nodes] if sub_graph.sorted_nodes else list(sub_graph.nodes.keys())
                )
                diags.extend(_collect_subgraph_diagnostics(sub_graph, sub_ids, target_fw, eng))
    return diags


def ground_proven_graph(
    proven_graph: ProvenGraph[AcyclicSortedPhase],
    snapshot: ProvenSnapshot[EnvelopeAndHydratedPhase],
) -> Validation[NonEmpty[GroundingDiagnostic], ProvenGraph[GroundedPhase]]:
    """Ground every operation in `proven_graph` against `snapshot`, accumulating all diagnostics.

    Args:
        proven_graph: Acyclic topologically sorted `ProvenGraph[AcyclicSortedPhase]` (or covariant subtype).
        snapshot: Hydrated `ProvenSnapshot[EnvelopeAndHydratedPhase]` (or covariant subtype).

    Returns:
        `Valid(ProvenGraph[GroundedPhase])` when all operations are grounded, or
        `Invalid(NonEmpty[GroundingDiagnostic])` accumulating every error diagnostic.
    """
    eng = GroundingEngine.from_proven_snapshot(snapshot)
    target_fw = str(snapshot.target).lower().replace("-", "_")
    thawed = proven_graph.thaw()
    ordered_ids = [str(nid) for nid in proven_graph.ordered_node_ids]
    error_diags = _collect_subgraph_diagnostics(thawed, ordered_ids, target_fw, eng)
    grounded_witness: ProvenGraph[GroundedPhase] = ProvenGraph(
        frozen_graph=proven_graph.frozen_graph,
        ordered_node_ids=proven_graph.ordered_node_ids,
    )
    return from_error_sequence(error_diags, grounded_witness)


__all__ = [
    "GroundingEngine",
    "GroundingEngineInput",
    "coerce_grounding_engine",
    "compute_levenshtein",
    "ground_proven_graph",
]
