"""ML-Switcheroo IR API Snapshot Extractor.

Extracts symbols and operations from ml_switcheroo_ir.
"""

from __future__ import annotations

from ml_switcheroo_ir.schema.ghost import (
    GhostParam,
    GhostRef,
    ParameterKind,
    SemanticTier,
)


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> list[GhostRef]:
    """Collect IR API symbols for the given category.

    Args:
        category: The SemanticTier category.
        include_nonpublic: Whether to include private symbols.

    Returns:
        List of GhostRef definitions.
    """
    refs: list[GhostRef] = []

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
                        annotation="Mapping[str, LogicalNode] | NodeDict | None",
                    ),
                    GhostParam(
                        name="outputs",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="list[str] | None",
                    ),
                    GhostParam(
                        name="mesh",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="LogicalMesh | None",
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
                        annotation="Mapping[str, TensorSpec] | None",
                    ),
                    GhostParam(
                        name="initializers",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="Mapping[str, InitializerValue] | None",
                    ),
                    GhostParam(
                        name="edges",
                        kind=ParameterKind.KEYWORD_ONLY,
                        default="None",
                        annotation="list[LogicalEdge] | None",
                    ),
                    GhostParam(
                        name="pipeline_topology",
                        kind=ParameterKind.KEYWORD_ONLY,
                        default="None",
                        annotation="PipelineTopologyConfig | None",
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
                        annotation="Mapping[str, AttributeValue] | None",
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
                        annotation="ShapeMetadataType | Iterable[DimensionType]",
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
                        annotation="Mapping[str, LogicalGraph | AttributeValue] | None",
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
                        annotation="tuple[DimensionType, ...] | Sequence[DimensionType] | TensorShape",
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
                        default="()",
                        annotation="tuple[DimensionType, ...] | Sequence[DimensionType] | Iterable[DimensionType] | TensorShape | DimensionType",
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
                        annotation="int | None",
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
                        annotation="Mapping[str, int]",
                    ),
                    GhostParam(
                        name="webrtc_topology",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="WebRTCSignalingTopology | None",
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
                        default="()",
                        annotation="Sequence[PartitionAxisElement]",
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
                        name="mapping_or_iterable",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="Mapping[str, LogicalNode] | Iterable[tuple[str, LogicalNode]] | None",
                    ),
                    GhostParam(
                        name="kwargs",
                        kind=ParameterKind.VAR_KEYWORD,
                        annotation="LogicalNode",
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

        # 11. EdgeList & Sentinels
        refs.append(
            GhostRef(
                name="EdgeList",
                kind="class",
                api_path="ml_switcheroo_ir.EdgeList",
                docstring="Mutable list of LogicalEdge objects with validation and query helpers.",
                params=[
                    GhostParam(
                        name="iterable",
                        kind=ParameterKind.POSITIONAL_ONLY,
                        default="()",
                        annotation="Iterable[LogicalEdge]",
                    ),
                ],
            )
        )
        refs.append(
            GhostRef(
                name="ZeroTangent",
                kind="class",
                api_path="ml_switcheroo_ir.ZeroTangent",
                docstring="Autodiff sentinel node representing an analytical zero tangent tensor.",
                params=[
                    GhostParam(
                        name="id",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="str",
                    ),
                    GhostParam(
                        name="shape",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="ShapeMetadataType | Iterable[DimensionType]",
                    ),
                    GhostParam(
                        name="dtype",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="DType.float32",
                        annotation="DType | None",
                    ),
                    GhostParam(
                        name="op_type",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="'ZeroTangent'",
                        annotation="str",
                    ),
                    GhostParam(
                        name="domain",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="'ml.switcheroo.ad'",
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
                        annotation="Mapping[str, AttributeValue] | None",
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
                        annotation="ShapeMetadataType | Iterable[DimensionType]",
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
                        name="output_specs",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="list[TensorSpec] | None",
                    ),
                    GhostParam(
                        name="subgraphs",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="Mapping[str, LogicalGraph | AttributeValue] | None",
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
        refs.append(
            GhostRef(
                name="NoTangent",
                kind="class",
                api_path="ml_switcheroo_ir.NoTangent",
                docstring="Autodiff sentinel node representing a non-differentiable or disconnected tangent.",
                params=[
                    GhostParam(
                        name="id",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="str",
                    ),
                    GhostParam(
                        name="op_type",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="'NoTangent'",
                        annotation="str",
                    ),
                    GhostParam(
                        name="domain",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="'ml.switcheroo.ad'",
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
                        annotation="Mapping[str, AttributeValue] | None",
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
                        annotation="ShapeMetadataType | Iterable[DimensionType]",
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
                        annotation="Mapping[str, LogicalGraph | AttributeValue] | None",
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

        # 12. Symbolic Dimension AST & Algebra Models
        refs.append(
            GhostRef(
                name="SymInt",
                kind="class",
                api_path="ml_switcheroo_ir.SymInt",
                docstring="Symbolic integer wrapper for algebraic dimension expressions.",
                params=[
                    GhostParam(
                        name="name_or_expr",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="str | int | SymNode | SymInt",
                    ),
                ],
            )
        )
        refs.append(
            GhostRef(
                name="SymVar",
                kind="class",
                api_path="ml_switcheroo_ir.SymVar",
                docstring="Named symbolic variable node.",
                params=[
                    GhostParam(
                        name="name",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="str",
                    ),
                ],
            )
        )
        refs.append(
            GhostRef(
                name="SymConst",
                kind="class",
                api_path="ml_switcheroo_ir.SymConst",
                docstring="Constant integer symbolic node.",
                params=[
                    GhostParam(
                        name="value",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="int",
                    ),
                ],
            )
        )
        refs.append(
            GhostRef(
                name="SymBinaryOp",
                kind="class",
                api_path="ml_switcheroo_ir.SymBinaryOp",
                docstring="Binary arithmetic or relational symbolic expression node.",
                params=[
                    GhostParam(
                        name="op",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="str",
                    ),
                    GhostParam(
                        name="left",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="SymNode",
                    ),
                    GhostParam(
                        name="right",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="SymNode",
                    ),
                ],
            )
        )
        refs.append(
            GhostRef(
                name="SymUnaryOp",
                kind="class",
                api_path="ml_switcheroo_ir.SymUnaryOp",
                docstring="Unary symbolic expression node.",
                params=[
                    GhostParam(
                        name="op",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="str",
                    ),
                    GhostParam(
                        name="operand",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="SymNode",
                    ),
                ],
            )
        )
        refs.append(
            GhostRef(
                name="SymPiecewise",
                kind="class",
                api_path="ml_switcheroo_ir.SymPiecewise",
                docstring="Piecewise conditional symbolic dimension expression.",
                params=[
                    GhostParam(
                        name="cases",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="list[tuple[SymCondition, SymNode]] | None",
                    ),
                    GhostParam(
                        name="default",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="SymNode | None",
                    ),
                ],
            )
        )

        # 13. Distributed & Pipeline Topology Configuration Models
        refs.append(
            GhostRef(
                name="WebRTCSignalingTopology",
                kind="class",
                api_path="ml_switcheroo_ir.WebRTCSignalingTopology",
                docstring="WebRTC signaling and peer connectivity topology for decentralized execution.",
                params=[
                    GhostParam(
                        name="signaling_url",
                        kind=ParameterKind.KEYWORD_ONLY,
                        default="'ws://localhost:8080'",
                        annotation="str",
                    ),
                    GhostParam(
                        name="ice_servers",
                        kind=ParameterKind.KEYWORD_ONLY,
                        default="[]",
                        annotation="list[str]",
                    ),
                    GhostParam(
                        name="high_watermark_bytes",
                        kind=ParameterKind.KEYWORD_ONLY,
                        default="1048576",
                        annotation="int",
                    ),
                    GhostParam(
                        name="low_watermark_bytes",
                        kind=ParameterKind.KEYWORD_ONLY,
                        default="262144",
                        annotation="int",
                    ),
                    GhostParam(
                        name="chunk_size_bytes",
                        kind=ParameterKind.KEYWORD_ONLY,
                        default="65536",
                        annotation="int",
                    ),
                    GhostParam(
                        name="connectivity_matrix",
                        kind=ParameterKind.KEYWORD_ONLY,
                        default="[]",
                        annotation="list[list[int]]",
                    ),
                    GhostParam(
                        name="peers",
                        kind=ParameterKind.KEYWORD_ONLY,
                        default="[]",
                        annotation="list[WebRTCPeerConfig]",
                    ),
                ],
            )
        )
        refs.append(
            GhostRef(
                name="WebRTCPeerConfig",
                kind="class",
                api_path="ml_switcheroo_ir.WebRTCPeerConfig",
                docstring="Configuration for an individual WebRTC compute peer.",
                params=[
                    GhostParam(
                        name="peer_id",
                        kind=ParameterKind.KEYWORD_ONLY,
                        annotation="str",
                    ),
                    GhostParam(
                        name="device_capability",
                        kind=ParameterKind.KEYWORD_ONLY,
                        default="'webgpu'",
                        annotation="str",
                    ),
                    GhostParam(
                        name="network_transport",
                        kind=ParameterKind.KEYWORD_ONLY,
                        default="'data_channel'",
                        annotation="str",
                    ),
                    GhostParam(
                        name="ice_transport_policy",
                        kind=ParameterKind.KEYWORD_ONLY,
                        default="'all'",
                        annotation="str",
                    ),
                ],
            )
        )
        refs.append(
            GhostRef(
                name="PipelineTopologyConfig",
                kind="class",
                api_path="ml_switcheroo_ir.PipelineTopologyConfig",
                docstring="Pipeline-parallel execution topology and stage scheduling configuration.",
                params=[
                    GhostParam(
                        name="microbatch_splitting",
                        kind=ParameterKind.KEYWORD_ONLY,
                        annotation="MicrobatchSplittingConfig",
                    ),
                    GhostParam(
                        name="mesh_mapping",
                        kind=ParameterKind.KEYWORD_ONLY,
                        annotation="MeshMappingConfig",
                    ),
                    GhostParam(
                        name="stage_communication",
                        kind=ParameterKind.KEYWORD_ONLY,
                        annotation="StageCommunicationConfig",
                    ),
                    GhostParam(
                        name="dependencies",
                        kind=ParameterKind.KEYWORD_ONLY,
                        default="[]",
                        annotation="list[DependencyConfig]",
                    ),
                    GhostParam(
                        name="schedule",
                        kind=ParameterKind.KEYWORD_ONLY,
                        default="None",
                        annotation="PipelineScheduleConfig | None",
                    ),
                ],
            )
        )
        refs.append(
            GhostRef(
                name="MicrobatchSplittingConfig",
                kind="class",
                api_path="ml_switcheroo_ir.MicrobatchSplittingConfig",
                docstring="Configuration for splitting batch dimensions into pipeline microbatches.",
                params=[
                    GhostParam(
                        name="strategy",
                        kind=ParameterKind.KEYWORD_ONLY,
                        default="'uniform'",
                        annotation="str",
                    ),
                    GhostParam(
                        name="num_microbatches",
                        kind=ParameterKind.KEYWORD_ONLY,
                        default="1",
                        annotation="int",
                    ),
                ],
            )
        )
        refs.append(
            GhostRef(
                name="MeshMappingConfig",
                kind="class",
                api_path="ml_switcheroo_ir.MeshMappingConfig",
                docstring="Mapping of pipeline stages onto physical/logical device mesh slices.",
                params=[
                    GhostParam(
                        name="devices_per_stage",
                        kind=ParameterKind.KEYWORD_ONLY,
                        default="1",
                        annotation="int",
                    ),
                ],
            )
        )
        refs.append(
            GhostRef(
                name="StageCommunicationConfig",
                kind="class",
                api_path="ml_switcheroo_ir.StageCommunicationConfig",
                docstring="Inter-stage point-to-point communication protocol configuration.",
                params=[
                    GhostParam(
                        name="protocol",
                        kind=ParameterKind.KEYWORD_ONLY,
                        default="'p2p'",
                        annotation="str",
                    ),
                ],
            )
        )
        refs.append(
            GhostRef(
                name="DependencyConfig",
                kind="class",
                api_path="ml_switcheroo_ir.DependencyConfig",
                docstring="Stage-to-stage microbatch dependency specification.",
                params=[
                    GhostParam(
                        name="source_stage",
                        kind=ParameterKind.KEYWORD_ONLY,
                        annotation="str",
                    ),
                    GhostParam(
                        name="target_stage",
                        kind=ParameterKind.KEYWORD_ONLY,
                        annotation="str",
                    ),
                    GhostParam(
                        name="offset_mb",
                        kind=ParameterKind.KEYWORD_ONLY,
                        default="0",
                        annotation="int",
                    ),
                ],
            )
        )
        refs.append(
            GhostRef(
                name="PipelineScheduleConfig",
                kind="class",
                api_path="ml_switcheroo_ir.PipelineScheduleConfig",
                docstring="Pipeline schedule composed of ordered execution phases.",
                params=[
                    GhostParam(
                        name="phases",
                        kind=ParameterKind.KEYWORD_ONLY,
                        default="[]",
                        annotation="list[SchedulePhaseConfig]",
                    ),
                ],
            )
        )
        refs.append(
            GhostRef(
                name="SchedulePhaseConfig",
                kind="class",
                api_path="ml_switcheroo_ir.SchedulePhaseConfig",
                docstring="Individual phase within a pipeline execution schedule.",
                params=[
                    GhostParam(
                        name="type",
                        kind=ParameterKind.KEYWORD_ONLY,
                        default="'1F1B'",
                        annotation="str",
                    ),
                    GhostParam(
                        name="operations",
                        kind=ParameterKind.KEYWORD_ONLY,
                        default="[]",
                        annotation="list[str]",
                    ),
                    GhostParam(
                        name="count_expression",
                        kind=ParameterKind.KEYWORD_ONLY,
                        default="'1'",
                        annotation="str",
                    ),
                ],
            )
        )

        # 14. Solvers, Trackers, and Translation Engines
        refs.append(
            GhostRef(
                name="SymbolicSolver",
                kind="class",
                api_path="ml_switcheroo_ir.SymbolicSolver",
                docstring="Algebraic solver for symbolic tensor dimension expressions and constraints.",
                params=[],
            )
        )
        refs.append(
            GhostRef(
                name="SymbolicConstraintTracker",
                kind="class",
                api_path="ml_switcheroo_ir.SymbolicConstraintTracker",
                docstring="Tracks equality and relational constraints across symbolic dimensions.",
                params=[],
            )
        )
        refs.append(
            GhostRef(
                name="ShapeTracker",
                kind="class",
                api_path="ml_switcheroo_ir.ShapeTracker",
                docstring="Tracks contiguous and non-contiguous tensor views, strides, and reshape legality.",
                params=[],
            )
        )
        refs.append(
            GhostRef(
                name="ParameterTranslationEngine",
                kind="class",
                api_path="ml_switcheroo_ir.ParameterTranslationEngine",
                docstring="Translates parameter names and keyword arguments across ML frameworks without hallucinations.",
                params=[
                    GhostParam(
                        name="translations",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="TranslationMap | Mapping[str, OperationTranslationSpec | Mapping[str, dict[str, RoleFrameworkMap]]] | None",
                    ),
                    GhostParam(
                        name="concept_map_path",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="str | None",
                    ),
                ],
            )
        )

        # 15. Core Transforms, Shape Algebra, Export & Schema Helpers
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
                name="estimate_communication_volume",
                kind="function",
                api_path="ml_switcheroo_ir.estimate_communication_volume",
                docstring="Estimate communication volume in bytes for a single distributed collective node.",
                params=[
                    GhostParam(
                        name="node",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="LogicalNode",
                    ),
                    GhostParam(
                        name="mesh",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="LogicalMesh | None",
                    ),
                ],
                returns_type="int",
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
                returns_type="CommunicationVolumeReport",
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
        refs.append(
            GhostRef(
                name="export_to_python",
                kind="function",
                api_path="ml_switcheroo_ir.export_to_python",
                docstring="Export a LogicalGraph into executable Python builder code.",
                params=[
                    GhostParam(
                        name="graph",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="LogicalGraph",
                    ),
                    GhostParam(
                        name="function_name",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="'build_graph'",
                        annotation="str",
                    ),
                    GhostParam(
                        name="standalone",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="False",
                        annotation="bool",
                    ),
                ],
                returns_type="str",
            )
        )
        refs.append(
            GhostRef(
                name="export_schemas",
                kind="function",
                api_path="ml_switcheroo_ir.export_schemas",
                docstring="Export JSON Schemas and optional TypeScript definitions for IR models.",
                params=[
                    GhostParam(
                        name="out_dir",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="Path | str",
                    ),
                    GhostParam(
                        name="include_typescript",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="False",
                        annotation="bool",
                    ),
                ],
                returns_type="dict[str, Path]",
            )
        )
        refs.append(
            GhostRef(
                name="get_json_schema",
                kind="function",
                api_path="ml_switcheroo_ir.get_json_schema",
                docstring="Generate JSON Schema dictionary for a target IR model.",
                params=[
                    GhostParam(
                        name="target",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="'LogicalGraph'",
                        annotation="SchemaTargetLiteral | str",
                    ),
                ],
                returns_type="dict[str, JsonValue]",
            )
        )
        refs.append(
            GhostRef(
                name="generate_typescript_definitions",
                kind="function",
                api_path="ml_switcheroo_ir.generate_typescript_definitions",
                docstring="Generate TypeScript interface definitions for core IR schemas.",
                params=[],
                returns_type="str",
            )
        )
        refs.append(
            GhostRef(
                name="get_abstract_op",
                kind="function",
                api_path="ml_switcheroo_ir.get_abstract_op",
                docstring="Look up an abstract operator schema by name across core IR registries.",
                params=[
                    GhostParam(
                        name="name",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="str",
                    ),
                ],
                returns_type="OpSchema | None",
            )
        )
        refs.append(
            GhostRef(
                name="broadcast_shapes",
                kind="function",
                api_path="ml_switcheroo_ir.broadcast_shapes",
                docstring="Compute the broadcasted output shape of two concrete or symbolic tensor shapes.",
                params=[
                    GhostParam(
                        name="shape_a",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="ShapeType",
                    ),
                    GhostParam(
                        name="shape_b",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="ShapeType",
                    ),
                    GhostParam(
                        name="tracker",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="SymbolicConstraintTracker | None",
                    ),
                ],
                returns_type="tuple[DimensionType, ...]",
            )
        )
        refs.append(
            GhostRef(
                name="broadcast_dimension",
                kind="function",
                api_path="ml_switcheroo_ir.broadcast_dimension",
                docstring="Broadcast two concrete or symbolic tensor dimensions.",
                params=[
                    GhostParam(
                        name="dim_a",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="DimensionType",
                    ),
                    GhostParam(
                        name="dim_b",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="DimensionType",
                    ),
                    GhostParam(
                        name="tracker",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="SymbolicConstraintTracker | None",
                    ),
                ],
                returns_type="DimensionType",
            )
        )
        refs.append(
            GhostRef(
                name="matmul_shape",
                kind="function",
                api_path="ml_switcheroo_ir.matmul_shape",
                docstring="Compute the output shape of a matrix multiplication between two shapes.",
                params=[
                    GhostParam(
                        name="shape_a",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="ShapeType",
                    ),
                    GhostParam(
                        name="shape_b",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="ShapeType",
                    ),
                ],
                returns_type="tuple[DimensionType, ...]",
            )
        )
        refs.append(
            GhostRef(
                name="normalize_axis",
                kind="function",
                api_path="ml_switcheroo_ir.normalize_axis",
                docstring="Normalize negative axis or sequence of axes against a tensor rank.",
                params=[
                    GhostParam(
                        name="axis",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="int | Sequence[int]",
                    ),
                    GhostParam(
                        name="rank",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="int",
                    ),
                ],
                returns_type="int | tuple[int, ...]",
            )
        )

    return refs
