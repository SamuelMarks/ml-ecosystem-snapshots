"""ML-Switcheroo IR API Snapshot Extractor.

Extracts symbols and operations from ml_switcheroo_ir.
"""

from typing import List

from ml_switcheroo_ir.schema.ghost import (
    GhostParam,
    GhostRef,
    ParameterKind,
    SemanticTier,
)


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Collect IR API symbols for the given category.

    Args:
        category: The SemanticTier category.
        include_nonpublic: Whether to include private symbols.

    Returns:
        List of GhostRef definitions.
    """
    refs: List[GhostRef] = []

    if category in (SemanticTier.ARRAY_API, SemanticTier.NEURAL, SemanticTier.UTIL):
        # 1. LogicalGraph
        refs.append(
            GhostRef(
                name="LogicalGraph",
                kind="class",
                api_path="ml_switcheroo_ir.LogicalGraph",
                docstring="Universal computation graph for Deep Learning models.",
                params=[
                    GhostParam(
                        name="name",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="'Model'",
                        annotation="str",
                    ),
                    GhostParam(
                        name="nodes",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="dict[str, LogicalNode] | None",
                    ),
                    GhostParam(
                        name="inputs",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="list[str] | None",
                    ),
                    GhostParam(
                        name="input_specs",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="dict[str, TensorSpec] | None",
                    ),
                    GhostParam(
                        name="outputs",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="list[str] | None",
                    ),
                    GhostParam(
                        name="initializers",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="dict[str, Any] | None",
                    ),
                    GhostParam(
                        name="mesh",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="LogicalMesh | None",
                    ),
                ],
            )
        )

        # 2. LogicalNode
        refs.append(
            GhostRef(
                name="LogicalNode",
                kind="class",
                api_path="ml_switcheroo_ir.LogicalNode",
                docstring="Individual operation node within a LogicalGraph.",
                params=[
                    GhostParam(
                        name="id",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="str",
                    ),
                    GhostParam(
                        name="op_type",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="''",
                        annotation="str",
                    ),
                    GhostParam(
                        name="domain",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="'ai.onnx'",
                        annotation="str",
                    ),
                    GhostParam(
                        name="version",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="1",
                        annotation="int",
                    ),
                    GhostParam(
                        name="attributes",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="dict[str, AttributeValue] | None",
                    ),
                    GhostParam(
                        name="inputs",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="list[str] | None",
                    ),
                    GhostParam(
                        name="outputs",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="list[str] | None",
                    ),
                    GhostParam(
                        name="shape_metadata",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="tuple[int | str, ...] | None",
                    ),
                    GhostParam(
                        name="source_ast_ref",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="str | None",
                    ),
                    GhostParam(
                        name="sharding",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="PartitionSpec | None",
                    ),
                    GhostParam(
                        name="dtype",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="DType | None",
                    ),
                    GhostParam(
                        name="output_specs",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="list[TensorSpec] | None",
                    ),
                    GhostParam(
                        name="subgraphs",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="dict[str, Any] | None",
                    ),
                    GhostParam(
                        name="device",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="str | None",
                    ),
                    GhostParam(
                        name="stream",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="str | None",
                    ),
                ],
            )
        )

        # 3. TensorSpec
        refs.append(
            GhostRef(
                name="TensorSpec",
                kind="class",
                api_path="ml_switcheroo_ir.TensorSpec",
                docstring="Specification for tensor values including shape, dtype, and sparsity.",
                params=[
                    GhostParam(
                        name="shape",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="tuple[int | str, ...] | Sequence[int | str] | TensorShape",
                    ),
                    GhostParam(
                        name="dtype",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="DType | str",
                    ),
                    GhostParam(
                        name="sparsity",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="str | None",
                    ),
                ],
            )
        )

        # 4. TensorShape
        refs.append(
            GhostRef(
                name="TensorShape",
                kind="class",
                api_path="ml_switcheroo_ir.TensorShape",
                docstring="Represents a tensor shape with dimension query capabilities.",
                params=[
                    GhostParam(
                        name="dims",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="tuple[int | str, ...] | Sequence[int | str] | Any",
                    ),
                ],
            )
        )

        # 5. LogicalEdge
        refs.append(
            GhostRef(
                name="LogicalEdge",
                kind="class",
                api_path="ml_switcheroo_ir.LogicalEdge",
                docstring="Represents a directed data-flow edge between two nodes in the graph.",
                params=[
                    GhostParam(
                        name="source",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="str",
                    ),
                    GhostParam(
                        name="target",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="str",
                    ),
                    GhostParam(
                        name="source_idx",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="0",
                        annotation="int",
                    ),
                    GhostParam(
                        name="target_idx",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="0",
                        annotation="int",
                    ),
                    GhostParam(
                        name="value_name",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="str | None",
                    ),
                ],
            )
        )

        # 6. LogicalMesh
        refs.append(
            GhostRef(
                name="LogicalMesh",
                kind="class",
                api_path="ml_switcheroo_ir.LogicalMesh",
                docstring="Represents a multi-dimensional grid of devices for distributed execution.",
                params=[
                    GhostParam(
                        name="shape",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="dict[str, int]",
                    ),
                ],
            )
        )

        # 7. LogicalAxis
        refs.append(
            GhostRef(
                name="LogicalAxis",
                kind="class",
                api_path="ml_switcheroo_ir.LogicalAxis",
                docstring="Represents a named dimension for tensor sizes and sharding.",
                params=[
                    GhostParam(
                        name="name",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="str",
                    ),
                    GhostParam(
                        name="size",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="int | None",
                    ),
                ],
            )
        )

        # 8. PartitionSpec
        refs.append(
            GhostRef(
                name="PartitionSpec",
                kind="class",
                api_path="ml_switcheroo_ir.PartitionSpec",
                docstring="Describes how a tensor's dimensions are mapped to a logical mesh.",
                params=[
                    GhostParam(
                        name="axes",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="tuple[str | tuple[str, ...] | None, ...]",
                    ),
                ],
            )
        )

        # 9. NodeDict
        refs.append(
            GhostRef(
                name="NodeDict",
                kind="class",
                api_path="ml_switcheroo_ir.NodeDict",
                docstring="Dictionary mapping node IDs to LogicalNodes with dual dict and sequence ergonomics.",
                params=[
                    GhostParam(
                        name="args",
                        kind=ParameterKind.VAR_POSITIONAL,
                        annotation="Any",
                    ),
                    GhostParam(
                        name="kwargs",
                        kind=ParameterKind.VAR_KEYWORD,
                        annotation="Any",
                    ),
                ],
            )
        )

        # 10. DType
        refs.append(
            GhostRef(
                name="DType",
                kind="class",
                api_path="ml_switcheroo_ir.DType",
                docstring="Data types for tensors.",
                params=[
                    GhostParam(
                        name="value",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="str",
                    ),
                ],
            )
        )

        # 11. Core Transforms & Helpers
        refs.append(
            GhostRef(
                name="eliminate_dead_nodes",
                kind="function",
                api_path="ml_switcheroo_ir.eliminate_dead_nodes",
                docstring="Eliminate non-output reachable and dead subgraphs from a LogicalGraph.",
                params=[
                    GhostParam(
                        name="graph",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="LogicalGraph",
                    ),
                ],
                returns_type="LogicalGraph",
            )
        )
        refs.append(
            GhostRef(
                name="eliminate_common_subexpressions",
                kind="function",
                api_path="ml_switcheroo_ir.eliminate_common_subexpressions",
                docstring="Eliminate common subexpressions by merging structurally identical pure operations.",
                params=[
                    GhostParam(
                        name="graph",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="LogicalGraph",
                    ),
                ],
                returns_type="LogicalGraph",
            )
        )
        refs.append(
            GhostRef(
                name="propagate_shapes_and_constants",
                kind="function",
                api_path="ml_switcheroo_ir.propagate_shapes_and_constants",
                docstring="Propagate statically computable tensor shapes and fold constant metadata.",
                params=[
                    GhostParam(
                        name="graph",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="LogicalGraph",
                    ),
                ],
                returns_type="LogicalGraph",
            )
        )
        refs.append(
            GhostRef(
                name="estimate_graph_communication_volume",
                kind="function",
                api_path="ml_switcheroo_ir.estimate_graph_communication_volume",
                docstring="Aggregate analytical communication volume across all collective operators in a graph.",
                params=[
                    GhostParam(
                        name="graph",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="LogicalGraph",
                    ),
                ],
                returns_type="dict[str, Any]",
            )
        )
        refs.append(
            GhostRef(
                name="topological_sort",
                kind="function",
                api_path="ml_switcheroo_ir.topological_sort",
                docstring="Sort graph nodes in dependency order.",
                params=[
                    GhostParam(
                        name="graph",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="LogicalGraph",
                    ),
                    GhostParam(
                        name="strict",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="False",
                        annotation="bool",
                    ),
                ],
                returns_type="list[LogicalNode]",
            )
        )

    return refs
